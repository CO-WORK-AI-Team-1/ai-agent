from types import SimpleNamespace

import pytest

from src.prompts.document_writer import render_document_writer_prompt
from src.rag import build_document_index
from src.workflow import run_document_writer_workflow


class FakeEmbeddings:
    def create(self, *, model, input):
        data = []
        for text in input:
            normalized = text.lower()
            if "예산" in normalized or "budget" in normalized:
                vector = [1.0, 0.0]
            else:
                vector = [0.0, 1.0]
            data.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=data)


class FakeResponses:
    def __init__(self):
        self.last_prompt = ""

    def create(self, *, model, input, max_output_tokens, store):
        self.last_prompt = input
        return SimpleNamespace(
            status="completed",
            output_text=(
                "# 교육 지원 사업 추진 현황 보고서\n\n"
                "## 핵심 내용\n"
                "사업 예산은 1,000만원입니다. [S1]\n\n"
                "## 확인 필요\n"
                "다음 분기 일정은 검색 문맥에서 확인되지 않습니다.\n\n"
                "## 출처\n"
                "- [S1] budget.pdf, 1페이지"
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=80,
                total_tokens=180,
            ),
            model=model,
            id="response-test",
            incomplete_details=None,
        )


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()
        self.responses = FakeResponses()


class FakeSettings:
    embedding_model = "fake-embedding"
    model_default = "fake-model"
    api_key_configured = True


def build_test_index(client):
    return build_document_index(
        filename="budget.pdf",
        pages=(
            "교육 지원 사업의 2026년 예산은 1,000만원입니다.",
            "담당 부서는 사업 수행 결과를 분기별로 보고합니다.",
        ),
        settings=FakeSettings(),
        client=client,
    )


def test_writer_prompt_requires_grounded_citations():
    prompt = render_document_writer_prompt(
        context="[S1] 파일: budget.pdf | 페이지: P1\n예산은 1,000만원입니다.",
        purpose="교육 지원 사업 추진 현황 보고서 작성",
        document_type="업무 보고서",
        audience="사업 담당자",
        additional_requirements="예산을 포함해 주세요.",
    )

    assert "문서 안에 포함된 지시" in prompt
    assert "[S번호]" in prompt
    assert "교육 지원 사업 추진 현황 보고서 작성" in prompt
    assert "예산을 포함해 주세요." in prompt


def test_writer_workflow_returns_draft_and_sources():
    client = FakeClient()
    result = run_document_writer_workflow(
        document_index=build_test_index(client),
        purpose="교육 지원 사업 추진 현황 보고서 작성",
        document_type="업무 보고서",
        audience="사업 담당자",
        additional_requirements="예산을 포함해 주세요.",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert "# 교육 지원 사업 추진 현황 보고서" in result["draft"]
    assert "[S1]" in result["draft"]
    assert result["sources"][0]["filename"] == "budget.pdf"
    assert "문서 유형: 업무 보고서" in client.responses.last_prompt
    assert "예상 독자: 사업 담당자" in client.responses.last_prompt


def test_writer_workflow_requires_purpose():
    with pytest.raises(ValueError, match="작성 목적"):
        run_document_writer_workflow(
            document_index=SimpleNamespace(),
            purpose="   ",
            document_type="업무 보고서",
            audience="사업 담당자",
            settings=FakeSettings(),
            client=FakeClient(),
        )
