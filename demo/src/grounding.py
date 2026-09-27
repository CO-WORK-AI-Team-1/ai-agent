import json
import re
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.prompts.week2_rag import render_answerability_prompt
from src.services.llm import invoke_response


CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
SOURCE_CITATION_PATTERN = re.compile(r"\[S\d+\]")
FACT_PATTERN = re.compile(
    r"(?<![A-Za-z가-힣])"
    r"(?:\d{4}년(?:\s*\d{1,2}월(?:\s*\d{1,2}일)?)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|명|개국|개|원|톤|헥타르|"
    r"킬로그램|제곱미터|곳|대|년|월|일)?)"
)


@dataclass(frozen=True)
class AnswerabilityDecision:
    answerable: bool
    reason: str
    latency_ms: int
    total_tokens: int | None


def parse_answerability_json(text: str) -> tuple[bool, str]:
    cleaned = CODE_FENCE_PATTERN.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("answerability response is not valid JSON") from error

    if not isinstance(payload, dict) or type(payload.get("answerable")) is not bool:
        raise ValueError("answerability response must contain a boolean answerable field")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("answerability reason must be a string")
    return payload["answerable"], reason.strip()


def assess_answerability(
    *,
    context: str,
    question: str,
    settings: Settings,
    client: Any | None = None,
) -> AnswerabilityDecision:
    result = invoke_response(
        prompt=render_answerability_prompt(context=context, question=question),
        model=settings.model_default,
        max_output_tokens=2000,
        settings=settings,
        client=client,
    )
    if not result.success:
        raise RuntimeError("answerability model call failed")
    answerable, reason = parse_answerability_json(result.output_text)
    return AnswerabilityDecision(
        answerable=answerable,
        reason=reason,
        latency_ms=result.latency_ms,
        total_tokens=result.total_tokens,
    )


def _normalize_fact(value: str) -> str:
    return re.sub(r"[\s,]", "", value).lower()


def validate_numeric_date_facts(
    *,
    answer: str,
    cited_source_texts: list[str],
) -> tuple[bool, list[str]]:
    """답변의 숫자·날짜가 인용된 검색 근거에 실제 존재하는지 확인합니다."""
    answer_without_citations = SOURCE_CITATION_PATTERN.sub("", answer)
    answer_without_source_list = "\n".join(
        line
        for line in answer_without_citations.splitlines()
        if not line.strip().startswith("출처:")
    )
    facts = {
        match.group(0).strip()
        for match in FACT_PATTERN.finditer(answer_without_source_list)
    }
    normalized_context = _normalize_fact("\n".join(cited_source_texts))
    unsupported = sorted(
        fact
        for fact in facts
        if _normalize_fact(fact) not in normalized_context
    )
    return not unsupported, unsupported
