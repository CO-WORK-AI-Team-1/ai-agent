from typing import Any

from src.config import Settings, get_settings
from src.prompts.document_writer import render_document_writer_prompt
from src.prompts.meeting_minutes import render_meeting_minutes_prompt
from src.prompts.week2_rag import render_week2_prompt
from src.rag import DocumentIndex, render_retrieved_context, search_document
from src.services.llm import invoke_response
from src.services.transcription import transcribe_meeting_audio


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


def run_document_writer_workflow(
    *,
    document_index: DocumentIndex,
    purpose: str,
    document_type: str,
    audience: str,
    additional_requirements: str = "",
    settings: Settings | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Create a source-grounded internal-document draft from retrieved chunks."""
    if not purpose.strip():
        raise ValueError("문서 작성 목적을 입력해 주세요.")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError("AI 서비스 연결 정보가 설정되지 않았습니다.")

    retrieval_query = "\n".join(
        value.strip()
        for value in (
            purpose,
            document_type,
            audience,
            additional_requirements,
        )
        if value.strip()
    )
    search_results = search_document(
        document_index=document_index,
        question=retrieval_query,
        settings=settings,
        client=client,
    )
    context = render_retrieved_context(search_results)
    prompt = render_document_writer_prompt(
        context=context,
        purpose=purpose,
        document_type=document_type,
        audience=audience,
        additional_requirements=additional_requirements,
    )
    result = invoke_response(
        prompt=prompt,
        model=settings.model_default,
        settings=settings,
        client=client,
    )
    common = {
        "sources": [
            {
                "source_id": search_result.source_id,
                "filename": search_result.chunk.filename,
                "page": search_result.chunk.page,
                "text": search_result.chunk.text,
                "score": search_result.score,
            }
            for search_result in search_results
        ],
        "response_status": result.status,
        "incomplete_reason": result.incomplete_reason,
        "latency_ms": result.latency_ms,
        "total_tokens": result.total_tokens,
    }

    if not result.success:
        return {
            "success": False,
            "draft": "",
            "error": "문서 초안 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            **common,
        }

    return {
        "success": True,
        "draft": result.output_text,
        "error": None,
        "is_incomplete": result.status == "incomplete",
        **common,
    }


def run_meeting_workflow(
    *,
    filename: str,
    audio_data: bytes,
    meeting_title: str,
    focus: str = "",
    settings: Settings | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Transcribe a recording and turn the verified transcript into meeting minutes."""
    if not meeting_title.strip():
        raise ValueError("회의 제목을 입력해 주세요.")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError("AI 서비스 연결 정보가 설정되지 않았습니다.")

    transcription = transcribe_meeting_audio(
        filename=filename,
        data=audio_data,
        settings=settings,
        client=client,
    )
    common = {
        "transcript": transcription.text,
        "transcription_latency_ms": transcription.latency_ms,
    }
    if not transcription.success:
        return {
            "success": False,
            "minutes": "",
            "error": transcription.error_message,
            **common,
        }

    prompt = render_meeting_minutes_prompt(
        transcript=transcription.text,
        meeting_title=meeting_title,
        focus=focus,
    )
    result = invoke_response(
        prompt=prompt,
        model=settings.model_default,
        settings=settings,
        client=client,
    )
    common.update(
        {
            "response_status": result.status,
            "incomplete_reason": result.incomplete_reason,
            "summary_latency_ms": result.latency_ms,
            "total_tokens": result.total_tokens,
        }
    )
    if not result.success:
        return {
            "success": False,
            "minutes": "",
            "error": "회의록 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            **common,
        }

    return {
        "success": True,
        "minutes": result.output_text,
        "error": None,
        "is_incomplete": result.status == "incomplete",
        **common,
    }
