from types import SimpleNamespace

import pytest

from src.rag import (
    build_document_index,
    chunk_document,
    search_document,
)


class FakeEmbeddings:
    def create(self, *, model, input):
        data = []

        for text in input:
            text = text.lower()

            if "보안" in text or "tls" in text:
                vector = [1.0, 0.0]

            elif "휴가" in text:
                vector = [0.0, 1.0]

            elif "혼합" in text:
                vector = [0.8, 0.6]

            else:
                vector = [0.5, 0.5]

            data.append(
                SimpleNamespace(
                    embedding=vector
                )
            )

        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


class FakeSettings:
    embedding_model = "fake-embedding"
    api_key_configured = True


class MissingApiKeySettings:
    embedding_model = "fake-embedding"
    api_key_configured = False


def test_chunk_document():
    pages = (
        "첫 번째 페이지의 보안 요구사항입니다.",
        "두 번째 페이지의 휴가 규정입니다.",
    )

    chunks = chunk_document(
        filename="test.pdf",
        pages=pages,
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 2

    assert chunks[0].filename == "test.pdf"
    assert chunks[0].page == 1
    assert "보안" in chunks[0].text

    assert chunks[1].page == 2
    assert "휴가" in chunks[1].text


def test_missing_api_key_uses_safe_user_message():
    with pytest.raises(
        RuntimeError,
        match="AI 서비스 연결 정보가 설정되지 않았습니다",
    ):
        build_document_index(
            filename="test.pdf",
            pages=("보안 요구사항입니다.",),
            settings=MissingApiKeySettings(),
        )


def test_hybrid_search():
    pages = (
        "보안 요구사항에 따라 TLS 1.2 이상을 사용해야 합니다.",
        "직원 휴가 신청은 사내 시스템을 통해 진행합니다.",
    )

    client = FakeClient()
    settings = FakeSettings()

    document_index = build_document_index(
        filename="test.pdf",
        pages=pages,
        settings=settings,
        client=client,
    )

    results = search_document(
        document_index=document_index,
        question="TLS 1.2 보안 요구사항을 알려줘",
        k=2,
        settings=settings,
        client=client,
    )

    assert len(results) == 1

    # 보안/TLS 문서가 가장 먼저 검색되어야 함
    assert results[0].chunk.page == 1
    assert "TLS 1.2" in results[0].chunk.text

    assert results[0].score >= 0.55


def test_hybrid_search_supports_single_relevant_chunk():
    client = FakeClient()
    settings = FakeSettings()
    document_index = build_document_index(
        filename="test.pdf",
        pages=("보안 요구사항에 따라 TLS 1.2 이상을 사용해야 합니다.",),
        settings=settings,
        client=client,
    )

    results = search_document(
        document_index=document_index,
        question="TLS 보안 요구사항을 알려줘",
        settings=settings,
        client=client,
    )

    assert len(results) == 1


def test_hybrid_search_returns_no_results_for_unrelated_question():
    pages = (
        "보안 요구사항에 따라 TLS 1.2 이상을 사용해야 합니다.",
        "직원 휴가 신청은 사내 시스템을 통해 진행합니다.",
    )

    client = FakeClient()
    settings = FakeSettings()
    document_index = build_document_index(
        filename="test.pdf",
        pages=pages,
        settings=settings,
        client=client,
    )

    results = search_document(
        document_index=document_index,
        question="구내식당의 오늘 점심 메뉴는 무엇인가요?",
        settings=settings,
        client=client,
    )

    assert results == []


def test_hybrid_search_rejects_invalid_min_score():
    pages = ("보안 요구사항입니다.",)
    client = FakeClient()
    settings = FakeSettings()
    document_index = build_document_index(
        filename="test.pdf",
        pages=pages,
        settings=settings,
        client=client,
    )

    try:
        search_document(
            document_index=document_index,
            question="보안 요구사항",
            min_score=1.1,
            settings=settings,
            client=client,
        )
    except ValueError as error:
        assert "min_score" in str(error)
    else:
        raise AssertionError("invalid min_score must raise ValueError")


def test_search_preserves_raw_cosine_and_applies_absolute_threshold():
    pages = (
        "첫 번째 보안 규정입니다.",
        "두 번째 휴가 규정입니다.",
    )
    client = FakeClient()
    settings = FakeSettings()
    document_index = build_document_index(
        filename="test.pdf",
        pages=pages,
        settings=settings,
        client=client,
    )

    permissive_results = search_document(
        document_index=document_index,
        question="혼합 질문",
        min_score=0.5,
        min_vector_score=0.75,
        settings=settings,
        client=client,
    )
    strict_results = search_document(
        document_index=document_index,
        question="혼합 질문",
        min_score=0.5,
        min_vector_score=0.9,
        settings=settings,
        client=client,
    )

    assert len(permissive_results) == 1
    assert permissive_results[0].cosine_similarity == pytest.approx(0.8)
    assert strict_results == []
