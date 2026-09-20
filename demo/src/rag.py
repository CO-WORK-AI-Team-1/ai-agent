import re
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.config import Settings, get_settings


CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
DEFAULT_K = 5
MAX_DOCUMENT_CHARS = 200_000

# Hybrid Retrieval 가중치
VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    chunk: DocumentChunk
    score: float


@dataclass
class DocumentIndex:
    chunks: list[DocumentChunk]

    # FAISS
    index: Any
    embedding_model: str

    # BM25
    bm25: Any
    tokenized_corpus: list[list[str]]


def _split_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    문서 텍스트를 일정 크기의 Chunk로 분할합니다.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size"
        )

    normalized = text.strip()

    if not normalized:
        return []

    chunks: list[str] = []

    start = 0

    while start < len(normalized):

        target_end = min(
            start + chunk_size,
            len(normalized),
        )

        end = target_end

        # 가능하면 문장/공백 경계에서 Chunk를 분리
        if target_end < len(normalized):

            search_start = start + chunk_size // 2

            candidates = [
                normalized.rfind(
                    "\n",
                    search_start,
                    target_end,
                ),
                normalized.rfind(
                    " ",
                    search_start,
                    target_end,
                ),
            ]

            boundary = max(candidates)

            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def chunk_document(
    *,
    filename: str,
    pages: tuple[str, ...],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """
    페이지별 문서를 검색 가능한 Chunk 단위로 변환합니다.
    """

    total_characters = sum(
        len(page)
        for page in pages
    )

    if total_characters > MAX_DOCUMENT_CHARS:
        raise ValueError(
            "현재 데모에서는 추출된 텍스트가 "
            f"{MAX_DOCUMENT_CHARS:,}자를 초과하는 "
            "문서를 분석할 수 없습니다. "
            "문서를 나누거나 더 짧은 문서를 업로드해 주세요."
        )

    chunks: list[DocumentChunk] = []

    for page_number, page_text in enumerate(
        pages,
        start=1,
    ):

        split_texts = _split_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, text in enumerate(
            split_texts,
            start=1,
        ):

            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"p{page_number}"
                        f"-c{chunk_number}"
                    ),
                    filename=filename,
                    page=page_number,
                    text=text,
                )
            )

    if not chunks:
        raise ValueError(
            "검색 인덱스를 만들 수 있는 "
            "문서 내용이 없습니다."
        )

    return chunks


def _tokenize(text: str) -> list[str]:
    """
    BM25 검색용 간단한 tokenizer.

    한글 / 영문 / 숫자 / 기술 용어를
    검색 가능한 단위로 분리합니다.
    """

    return re.findall(
        r"[가-힣A-Za-z0-9_.:/%-]+",
        text.lower(),
    )


def _embedding_client(
    settings: Settings,
    client: Any | None,
) -> Any:

    if client is not None:
        return client

    if not settings.api_key_configured:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다."
        )

    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=60.0,
        max_retries=2,
    )


def _embed_texts(
    *,
    texts: list[str],
    model: str,
    client: Any,
) -> np.ndarray:
    """
    OpenAI Embedding API를 사용해
    텍스트를 Vector로 변환합니다.
    """

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    vectors = np.asarray(
        [
            item.embedding
            for item in response.data
        ],
        dtype="float32",
    )

    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(texts)
    ):
        raise RuntimeError(
            "임베딩 결과의 크기가 "
            "입력 문서와 일치하지 않습니다."
        )

    # Cosine Similarity 사용을 위해 정규화
    faiss.normalize_L2(vectors)

    return vectors


def build_document_index(
    *,
    filename: str,
    pages: tuple[str, ...],
    settings: Settings | None = None,
    client: Any | None = None,
) -> DocumentIndex:
    """
    문서를 Chunk로 나눈 뒤

    1. FAISS Vector Index
    2. BM25 Keyword Index

    두 개를 동시에 생성합니다.
    """

    settings = settings or get_settings()

    chunks = chunk_document(
        filename=filename,
        pages=pages,
    )

    # -------------------------
    # FAISS Index 생성
    # -------------------------

    api = _embedding_client(
        settings,
        client,
    )

    vectors = _embed_texts(
        texts=[
            chunk.text
            for chunk in chunks
        ],
        model=settings.embedding_model,
        client=api,
    )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    # -------------------------
    # BM25 Index 생성
    # -------------------------

    tokenized_corpus = [
        _tokenize(chunk.text)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return DocumentIndex(
        chunks=chunks,
        index=index,
        embedding_model=settings.embedding_model,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
    )


def _normalize_scores(
    scores: np.ndarray,
) -> np.ndarray:
    """
    검색 점수를 0~1 범위로 정규화합니다.
    """

    scores = np.asarray(
        scores,
        dtype="float32",
    )

    if len(scores) == 0:
        return scores

    minimum = float(scores.min())
    maximum = float(scores.max())

    if maximum == minimum:
        return np.ones_like(
            scores,
            dtype="float32",
        )

    return (
        scores - minimum
    ) / (
        maximum - minimum
    )


def search_document(
    *,
    document_index: DocumentIndex,
    question: str,
    k: int = DEFAULT_K,
    settings: Settings | None = None,
    client: Any | None = None,
) -> list[SearchResult]:
    """
    FAISS + BM25 Hybrid Retrieval.

    Vector Search:
        의미적으로 비슷한 문서 검색

    BM25:
        질문과 동일한 키워드가 포함된
        문서 검색

    두 점수를 결합하여
    최종 Top-K 결과를 반환합니다.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "질문을 입력해 주세요."
        )

    settings = settings or get_settings()

    api = _embedding_client(
        settings,
        client,
    )

    # ==================================
    # 1. FAISS Vector Search
    # ==================================

    query_vector = _embed_texts(
        texts=[question],
        model=document_index.embedding_model,
        client=api,
    )

    chunk_count = len(
        document_index.chunks
    )

    vector_scores, vector_indices = (
        document_index.index.search(
            query_vector,
            chunk_count,
        )
    )

    faiss_scores = np.zeros(
        chunk_count,
        dtype="float32",
    )

    for score, index_position in zip(
        vector_scores[0],
        vector_indices[0],
        strict=True,
    ):
        if index_position >= 0:
            faiss_scores[
                int(index_position)
            ] = float(score)

    faiss_scores = _normalize_scores(
        faiss_scores
    )

    # ==================================
    # 2. BM25 Keyword Search
    # ==================================

    query_tokens = _tokenize(
        question
    )

    bm25_scores = np.asarray(
        document_index.bm25.get_scores(
            query_tokens
        ),
        dtype="float32",
    )

    bm25_scores = _normalize_scores(
        bm25_scores
    )

    # ==================================
    # 3. Hybrid Score
    # ==================================

    hybrid_scores = (
        VECTOR_WEIGHT
        * faiss_scores
        +
        BM25_WEIGHT
        * bm25_scores
    )

    # 높은 점수 순으로 정렬
    ranked_indices = np.argsort(
        hybrid_scores
    )[::-1]

    result_count = min(
        max(k, 1),
        chunk_count,
    )

    results: list[SearchResult] = []

    for rank, index_position in enumerate(
        ranked_indices[:result_count],
        start=1,
    ):

        results.append(
            SearchResult(
                source_id=f"S{rank}",
                chunk=(
                    document_index
                    .chunks[
                        int(index_position)
                    ]
                ),
                score=float(
                    hybrid_scores[
                        index_position
                    ]
                ),
            )
        )

    return results


def render_retrieved_context(
    results: list[SearchResult],
) -> str:
    """
    검색 결과를 LLM Prompt에 들어갈
    Context 문자열로 변환합니다.
    """

    if not results:
        raise ValueError(
            "질문과 관련된 "
            "문서 내용을 찾지 못했습니다."
        )

    return "\n\n".join(
        (
            f"[{result.source_id}] "
            f"파일: {result.chunk.filename} | "
            f"페이지: P{result.chunk.page}\n"
            f"{result.chunk.text}"
        )
        for result in results
    )