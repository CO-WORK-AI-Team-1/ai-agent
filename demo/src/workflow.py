import re
from typing import Any

from src.config import AI_CONNECTION_ERROR_MESSAGE, Settings, get_settings
from src.grounding import assess_answerability, validate_numeric_date_facts
from src.prompts.document_writer import render_document_writer_prompt
from src.prompts.meeting_minutes import render_meeting_minutes_prompt
from src.prompts.week2_rag import render_week2_prompt
from src.rag import DocumentIndex, render_retrieved_context, search_document
from src.services.llm import invoke_response
from src.services.transcription import transcribe_meeting_audio


SOURCE_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def validate_source_citations(
    *,
    answer: str,
    allowed_source_ids: set[str],
) -> tuple[bool, set[str]]:
    """답변의 모든 [S번호]가 실제 검색 결과에 존재하는지 확인합니다."""
    cited_source_ids = {
        f"S{match}"
        for match in SOURCE_CITATION_PATTERN.findall(answer)
    }
    invalid_source_ids = cited_source_ids - allowed_source_ids
    is_valid = bool(cited_source_ids) and not invalid_source_ids
    return is_valid, invalid_source_ids


def run_document_workflow(
    *,
    document_index: DocumentIndex,
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
    min_vector_score: float | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    verify_answerability: bool = False,
    settings: Settings | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Search the indexed document and answer using only retrieved chunks."""
    if not question.strip():
        raise ValueError("질문을 입력해 주세요.")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError(AI_CONNECTION_ERROR_MESSAGE)

    search_options: dict[str, Any] = {}
    if top_k is not None:
        search_options["k"] = top_k
    if min_score is not None:
        search_options["min_score"] = min_score
    if min_vector_score is not None:
        search_options["min_vector_score"] = min_vector_score
    if vector_weight is not None:
        search_options["vector_weight"] = vector_weight
    if bm25_weight is not None:
        search_options["bm25_weight"] = bm25_weight

    search_results = search_document(
        document_index=document_index,
        question=question,
        settings=settings,
        client=client,
        **search_options,
    )
    if not search_results:
        return {
            "success": True,
            "answer_status": "not_found",
            "answer": (
                "업로드된 문서에서 질문과 관련된 충분한 근거를 "
                "찾지 못했습니다. 질문을 구체화하거나 관련 문서를 "
                "확인해 주세요."
            ),
            "error": None,
            "sources": [],
            "response_status": "not_found",
            "incomplete_reason": None,
            "latency_ms": 0,
            "total_tokens": 0,
            "is_incomplete": False,
        }
    context = render_retrieved_context(search_results)
    answerability_reason = None
    if verify_answerability:
        try:
            verification = assess_answerability(
                context=context,
                question=question,
                settings=settings,
                client=client,
            )
        except (RuntimeError, ValueError):
            return {
                "success": False,
                "answer_status": "answerability_check_failed",
                "answer": "",
                "error": (
                    "검색 근거의 답변 가능성을 확인하는 중 오류가 발생했습니다. "
                    "잠시 후 다시 시도해 주세요."
                ),
                "sources": [],
                "response_status": "verification_error",
                "incomplete_reason": None,
                "latency_ms": 0,
                "total_tokens": None,
            }

        answerability_reason = verification.reason
        if not verification.answerable:
            return {
                "success": True,
                "answer_status": "not_found",
                "answer": (
                    "검색된 문서 내용만으로는 질문에 답할 수 있는 "
                    "충분한 근거를 확인하지 못했습니다."
                ),
                "error": None,
                "sources": [],
                "response_status": "not_found",
                "incomplete_reason": None,
                "latency_ms": verification.latency_ms,
                "total_tokens": verification.total_tokens,
                "answerability_reason": answerability_reason,
                "is_incomplete": False,
            }

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
                "cosine_similarity": result.cosine_similarity,
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

    allowed_source_ids = {
        search_result.source_id
        for search_result in search_results
    }
    citations_valid, invalid_source_ids = validate_source_citations(
        answer=result.output_text,
        allowed_source_ids=allowed_source_ids,
    )
    if not citations_valid:
        return {
            "success": False,
            "answer_status": "citation_validation_failed",
            "answer": "",
            "error": (
                "답변은 생성되었지만 출처 연결을 확인하지 못했습니다. "
                "잠시 후 다시 시도해 주세요."
            ),
            "invalid_source_ids": sorted(invalid_source_ids),
            **common,
        }

    cited_source_ids = {
        f"S{match}"
        for match in SOURCE_CITATION_PATTERN.findall(result.output_text)
    }
    cited_source_texts = [
        search_result.chunk.text
        for search_result in search_results
        if search_result.source_id in cited_source_ids
    ]
    facts_valid, unsupported_facts = validate_numeric_date_facts(
        answer=result.output_text,
        cited_source_texts=cited_source_texts,
    )
    if not facts_valid:
        return {
            "success": False,
            "answer_status": "fact_validation_failed",
            "answer": "",
            "error": (
                "답변의 수치 또는 날짜를 검색 근거에서 확인하지 못했습니다. "
                "질문을 다시 시도해 주세요."
            ),
            "unsupported_facts": unsupported_facts,
            **common,
        }

    return {
        "success": True,
        "answer_status": "completed",
        "answer": result.output_text,
        "error": None,
        "is_incomplete": result.status == "incomplete",
        "answerability_reason": answerability_reason,
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
        raise RuntimeError(AI_CONNECTION_ERROR_MESSAGE)

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
        raise RuntimeError(AI_CONNECTION_ERROR_MESSAGE)

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
