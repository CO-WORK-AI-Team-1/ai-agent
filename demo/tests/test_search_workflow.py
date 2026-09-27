from types import SimpleNamespace

from src.rag import build_document_index
from src.workflow import run_document_workflow, validate_source_citations


class FakeEmbeddings:
    def create(self, *, model, input):
        data = []
        for text in input:
            normalized = text.lower()
            if "보안" in normalized or "tls" in normalized:
                vector = [1.0, 0.0]
            elif "휴가" in normalized:
                vector = [0.0, 1.0]
            else:
                vector = [0.5, 0.5]
            data.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=data)


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.call_count = 0

    def create(self, *, model, input, max_output_tokens, store):
        self.call_count += 1
        return SimpleNamespace(
            status="completed",
            output_text=self.output_text,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
            ),
            model=model,
            id="response-test",
            incomplete_details=None,
        )


class FakeClient:
    def __init__(self, output_text="TLS 1.2 이상을 사용해야 합니다. [S1]\n\n출처: [S1]"):
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses(output_text)


class FakeSettings:
    embedding_model = "fake-embedding"
    model_default = "fake-model"
    api_key_configured = True


class SequenceResponses:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.call_count = 0

    def create(self, *, model, input, max_output_tokens, store):
        self.call_count += 1
        return SimpleNamespace(
            status="completed",
            output_text=next(self.outputs),
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=2,
                total_tokens=12,
            ),
            model=model,
            id="response-sequence",
            incomplete_details=None,
        )


def build_test_index(client):
    return build_document_index(
        filename="policy.pdf",
        pages=(
            "보안 요구사항에 따라 TLS 1.2 이상을 사용해야 합니다.",
            "직원 휴가 신청은 사내 시스템을 통해 진행합니다.",
        ),
        settings=FakeSettings(),
        client=client,
    )


def test_validate_source_citations_rejects_unknown_source_id():
    is_valid, invalid_ids = validate_source_citations(
        answer="보안 정책입니다. [S1][S9]",
        allowed_source_ids={"S1", "S2"},
    )

    assert is_valid is False
    assert invalid_ids == {"S9"}


def test_document_workflow_accepts_valid_citations():
    client = FakeClient()
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="TLS 보안 요구사항을 알려줘",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert result["answer_status"] == "completed"
    assert result["sources"][0]["source_id"] == "S1"


def test_document_workflow_rejects_invalid_citation():
    client = FakeClient("TLS 정책입니다. [S99]\n\n출처: [S99]")
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="TLS 보안 요구사항을 알려줘",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is False
    assert result["answer_status"] == "citation_validation_failed"
    assert result["invalid_source_ids"] == ["S99"]


def test_document_workflow_rejects_unsupported_numeric_fact():
    client = FakeClient("TLS 9.9를 사용해야 합니다. [S1]\n\n출처: [S1]")
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="TLS 보안 요구사항을 알려줘",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is False
    assert result["answer_status"] == "fact_validation_failed"
    assert result["unsupported_facts"] == ["9.9"]


def test_document_workflow_returns_normal_not_found_without_llm_call():
    client = FakeClient()
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="오늘 구내식당 점심 메뉴는 무엇인가요?",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert result["answer_status"] == "not_found"
    assert result["sources"] == []
    assert client.responses.call_count == 0


def test_answerability_gate_rejects_context_without_direct_answer():
    client = FakeClient()
    client.responses = SequenceResponses(
        ['{"answerable": false, "reason": "직접 답할 근거가 없음"}']
    )
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="TLS 보안 요구사항을 알려줘",
        verify_answerability=True,
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert result["answer_status"] == "not_found"
    assert client.responses.call_count == 1


def test_answerability_gate_allows_grounded_answer_generation():
    client = FakeClient()
    client.responses = SequenceResponses(
        [
            '{"answerable": true, "reason": "TLS 요구사항이 명시되어 있음"}',
            "TLS 1.2 이상을 사용해야 합니다. [S1]\n\n출처: [S1]",
        ]
    )
    result = run_document_workflow(
        document_index=build_test_index(client),
        question="TLS 보안 요구사항을 알려줘",
        verify_answerability=True,
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert result["answer_status"] == "completed"
    assert client.responses.call_count == 2
