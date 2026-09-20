from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from openai import OpenAI

from src.config import Settings, get_settings

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
DEFAULT_K = 5
MAX_DOCUMENT_CHARS = 200_000


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
    index: Any
    embedding_model: str


def _split_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = target_end

        if target_end < len(normalized):
            search_start = start + chunk_size // 2
            candidates = [
                normalized.rfind("\n", search_start, target_end),
                normalized.rfind(" ", search_start, target_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_document(
    *,
    filename: str,
    pages: tuple[str, ...],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    total_characters = sum(len(page) for page in pages)
    if total_characters > MAX_DOCUMENT_CHARS:
        raise ValueError(
            "현재 데모에서는 추출된 텍스트가 "
            f"{MAX_DOCUMENT_CHARS:,}자를 초과하는 문서를 분석할 수 없습니다. "
            "문서를 나누거나 더 짧은 문서를 업로드해 주세요."
        )

    chunks: list[DocumentChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_number, text in enumerate(
            _split_text(page_text, chunk_size=chunk_size, overlap=overlap),
            start=1,
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page_number}-c{chunk_number}",
                    filename=filename,
                    page=page_number,
                    text=text,
                )
            )

    if not chunks:
        raise ValueError("검색 인덱스를 만들 수 있는 문서 내용이 없습니다.")
    return chunks


def _embedding_client(settings: Settings, client: Any | None) -> Any:
    if client is not None:
        return client
    if not settings.api_key_configured:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)


def _embed_texts(*, texts: list[str], model: str, client: Any) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.asarray([item.embedding for item in response.data], dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError("임베딩 결과의 크기가 입력 문서와 일치하지 않습니다.")
    faiss.normalize_L2(vectors)
    return vectors


def build_document_index(
    *,
    filename: str,
    pages: tuple[str, ...],
    settings: Settings | None = None,
    client: Any | None = None,
) -> DocumentIndex:
    settings = settings or get_settings()
    chunks = chunk_document(filename=filename, pages=pages)
    api = _embedding_client(settings, client)
    vectors = _embed_texts(
        texts=[chunk.text for chunk in chunks],
        model=settings.embedding_model,
        client=api,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return DocumentIndex(
        chunks=chunks,
        index=index,
        embedding_model=settings.embedding_model,
    )


def search_document(
    *,
    document_index: DocumentIndex,
    question: str,
    k: int = DEFAULT_K,
    settings: Settings | None = None,
    client: Any | None = None,
) -> list[SearchResult]:
    if not question.strip():
        raise ValueError("질문을 입력해 주세요.")
    settings = settings or get_settings()
    api = _embedding_client(settings, client)
    query_vector = _embed_texts(
        texts=[question.strip()],
        model=document_index.embedding_model,
        client=api,
    )
    result_count = min(max(k, 1), len(document_index.chunks))
    scores, indices = document_index.index.search(query_vector, result_count)

    results: list[SearchResult] = []
    for rank, (score, index_position) in enumerate(
        zip(scores[0], indices[0], strict=True),
        start=1,
    ):
        if index_position < 0:
            continue
        results.append(
            SearchResult(
                source_id=f"S{rank}",
                chunk=document_index.chunks[int(index_position)],
                score=float(score),
            )
        )
    return results


def render_retrieved_context(results: list[SearchResult]) -> str:
    if not results:
        raise ValueError("질문과 관련된 문서 내용을 찾지 못했습니다.")
    return "\n\n".join(
        f"[{result.source_id}] 파일: {result.chunk.filename} | 페이지: P{result.chunk.page}\n"
        f"{result.chunk.text}"
        for result in results
    )
