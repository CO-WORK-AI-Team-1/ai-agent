
import logging
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

from openai import OpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    success: bool
    status: str
    incomplete_reason: str | None
    model_requested: str
    model_returned: str | None
    response_id: str | None
    output_text: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if value is not None else None


def _incomplete_reason(response: Any) -> str | None:
    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    return str(reason) if reason is not None else None


def invoke_response(
    *,
    prompt: str,
    model: str,
    settings: Settings | None = None,
    client: Any | None = None,
) -> LLMResult:
    settings = settings or get_settings()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if client is None and not settings.api_key_configured:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    api = client or OpenAI(api_key=settings.openai_api_key, timeout=60.0, max_retries=2)
    started = perf_counter()
    try:
        response = api.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=4000,
            store=False,
        )
        latency_ms = round((perf_counter() - started) * 1000)
        usage = getattr(response, "usage", None)
        status = str(getattr(response, "status", "completed"))
        output_text = getattr(response, "output_text", "") or ""
        incomplete_reason = _incomplete_reason(response)

        if status == "incomplete":
            logger.warning(
                "LLM response incomplete: model=%s reason=%s latency_ms=%s "
                "input_tokens=%s output_tokens=%s output_length=%s",
                model,
                incomplete_reason or "unknown",
                latency_ms,
                _usage_value(usage, "input_tokens"),
                _usage_value(usage, "output_tokens"),
                len(output_text),
            )
        elif status != "completed":
            logger.error(
                "LLM response failed: model=%s status=%s latency_ms=%s",
                model,
                status,
                latency_ms,
            )

        return LLMResult(
            success=status == "completed" or (status == "incomplete" and bool(output_text)),
            status=status,
            incomplete_reason=incomplete_reason,
            model_requested=model,
            model_returned=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            output_text=output_text,
            latency_ms=latency_ms,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
        )
    except Exception as error:
        latency_ms = round((perf_counter() - started) * 1000)
        status_code = getattr(error, "status_code", None)
        error_code = getattr(error, "code", None)
        logger.error(
            "LLM API error: model=%s error_type=%s status_code=%s "
            "error_code=%s latency_ms=%s message=%s",
            model,
            type(error).__name__,
            status_code,
            error_code,
            latency_ms,
            str(error).replace("\n", " ")[:500],
        )
        return LLMResult(
            success=False,
            status="error",
            incomplete_reason=None,
            model_requested=model,
            model_returned=None,
            response_id=None,
            output_text="",
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            error_type=type(error).__name__,
            error_message=str(error)[:500],
        )
