from types import SimpleNamespace

from scripts.evaluate_retrieval import EvaluationConfig, evaluate_config


class FakeEmbeddings:
    def create(self, *, model, input):
        data = []
        for text in input:
            normalized = text.lower()
            if "tls" in normalized or "보안" in normalized:
                vector = [1.0, 0.0]
            elif "휴가" in normalized:
                vector = [0.0, 1.0]
            else:
                vector = [0.5, 0.5]
            data.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=data)


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses()


class FakeResponses:
    def create(self, *, model, input, max_output_tokens, store):
        is_verification = "답변 가능성을 판정" in input
        answerable = "구내식당" not in input
        if is_verification:
            output_text = (
                '{"answerable": true, "reason": "근거 있음"}'
                if answerable
                else '{"answerable": false, "reason": "근거 없음"}'
            )
        else:
            output_text = "TLS 1.2 이상을 사용합니다. [S1]\n\n출처: [S1]"
        return SimpleNamespace(
            status="completed",
            output_text=output_text,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            model=model,
            id="evaluation-response",
            incomplete_details=None,
        )


class FakeSettings:
    embedding_model = "fake-embedding"
    model_default = "fake-model"
    api_key_configured = True


def test_evaluate_config_calculates_retrieval_metrics():
    dataset = {
        "documents": [
            {
                "id": "policy",
                "filename": "policy.pdf",
                "pages": [
                    "보안 정책에 따라 TLS 1.2 이상을 사용합니다.",
                    "직원 휴가는 사내 시스템에서 신청합니다.",
                ],
            }
        ],
        "questions": [
            {
                "id": "q1",
                "document_id": "policy",
                "question": "TLS 보안 정책은 무엇인가요?",
                "answerable": True,
                "expected_pages": [1],
            },
            {
                "id": "q2",
                "document_id": "policy",
                "question": "오늘 구내식당 메뉴는 무엇인가요?",
                "answerable": False,
                "expected_pages": [],
            },
        ],
    }
    config = EvaluationConfig(
        chunk_size=100,
        overlap=10,
        top_k=3,
        min_score=0.55,
        min_vector_score=None,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    metrics, details = evaluate_config(
        dataset=dataset,
        config=config,
        settings=FakeSettings(),
        client=FakeClient(),
    )

    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.false_rejection_rate == 0.0
    assert metrics.not_found_accuracy == 1.0
    assert details[0]["retrieved_pages"] == [1]
    assert details[1]["not_found"] is True


def test_evaluate_config_includes_answerability_gate_metrics():
    dataset = {
        "documents": [
            {
                "id": "policy",
                "filename": "policy.pdf",
                "pages": [
                    "보안 정책에 따라 TLS 1.2 이상을 사용합니다.",
                    "직원 휴가는 사내 시스템에서 신청합니다.",
                ],
            }
        ],
        "questions": [
            {
                "id": "q1",
                "document_id": "policy",
                "question": "TLS 보안 정책은 무엇인가요?",
                "answerable": True,
                "expected_pages": [1],
            },
            {
                "id": "q2",
                "document_id": "policy",
                "question": "오늘 구내식당 메뉴는 무엇인가요?",
                "answerable": False,
                "expected_pages": [],
            },
        ],
    }
    config = EvaluationConfig(
        chunk_size=100,
        overlap=10,
        top_k=3,
        min_score=0.0,
        min_vector_score=None,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    metrics, details = evaluate_config(
        dataset=dataset,
        config=config,
        settings=FakeSettings(),
        client=FakeClient(),
        evaluate_answerability=True,
        fact_check_question_ids={"q1"},
    )

    assert metrics.answerability_accuracy == 1.0
    assert metrics.end_to_end_balanced_accuracy == 1.0
    assert metrics.end_to_end_not_found_accuracy == 1.0
    assert details[0]["answerability_decision"] is True
    assert details[1]["answerability_decision"] is False
    assert metrics.fact_validation_checked_count == 1
    assert metrics.fact_validation_pass_rate == 1.0
    assert details[0]["fact_validation_passed"] is True
