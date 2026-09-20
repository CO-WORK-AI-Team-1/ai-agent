from typing import Any

from src.config import Settings, get_settings
from src.prompts.week2_rag import render_week2_prompt
from src.rag import DocumentIndex, render_retrieved_context, search_document
from src.services.llm import invoke_response


def run_document_workflow(
    *,
    document_index: DocumentIndex,
    question: str,
    settings: Settings | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Search the indexed document and answer using only retrieved chunks."""
    if not question.strip():
        raise ValueError("질문을 입력해 주세요.")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError("AI 서비스 연결 정보가 설정되지 않았습니다.")

    search_results = search_document(
        document_index=document_index,
        question=question,
        settings=settings,
        client=client,
    )
    context = render_retrieved_context(search_results)
    prompt = render_week2_prompt(context=context, question=question)
    result = invoke_response(
        prompt=prompt,
        model=settings.model_default,
        settings=settings,
        client=client,
    )
    common = {
        "sources": [
            {
                "source_id": result.source_id,
                "filename": result.chunk.filename,
                "page": result.chunk.page,
                "text": result.chunk.text,
                "score": result.score,
            }
            for result in search_results
        ],
        "response_status": result.status,
        "incomplete_reason": result.incomplete_reason,
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
    }

    if not result.success:
        return {
            "success": False,
            "answer": "",
            "error": "AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            **common,
        }

    return {
        "success": True,
        "answer": result.output_text,
        "error": None,
        "is_incomplete": result.status == "incomplete",
        **common,
    }
