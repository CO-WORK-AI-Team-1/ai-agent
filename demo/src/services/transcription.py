import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_AUDIO_SUFFIXES = (
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
)


@dataclass(frozen=True)
class TranscriptionResult:
    success: bool
    text: str
    latency_ms: int
    error_message: str | None = None


def transcribe_meeting_audio(
    *,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
    client: Any | None = None,
) -> TranscriptionResult:
    """Transcribe one supported meeting recording without persisting it to disk."""
    if not data:
        raise ValueError("빈 음성 파일은 업로드할 수 없습니다.")
    if len(data) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError("음성 파일 크기는 25MB를 초과할 수 없습니다.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_AUDIO_SUFFIXES
        )
        raise ValueError(f"지원하지 않는 음성 형식입니다. 지원 형식: {supported_formats}")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError("AI 서비스 연결 정보가 설정되지 않았습니다.")

    api = client or OpenAI(
        api_key=settings.openai_api_key,
        timeout=120.0,
        max_retries=2,
    )
    audio_file = BytesIO(data)
    audio_file.name = filename
    started = perf_counter()

    try:
        response = api.audio.transcriptions.create(
            model=settings.transcribe_model,
            file=audio_file,
        )
        latency_ms = round((perf_counter() - started) * 1000)
        transcript = str(getattr(response, "text", "")).strip()
        if not transcript:
            logger.warning(
                "Audio transcription returned empty text: filename=%s latency_ms=%s",
                filename,
                latency_ms,
            )
            return TranscriptionResult(
                success=False,
                text="",
                latency_ms=latency_ms,
                error_message="음성에서 텍스트를 추출하지 못했습니다. 녹음 상태를 확인해 주세요.",
            )

        return TranscriptionResult(
            success=True,
            text=transcript,
            latency_ms=latency_ms,
        )
    except Exception as error:
        latency_ms = round((perf_counter() - started) * 1000)
        logger.error(
            "Audio transcription error: filename=%s error_type=%s latency_ms=%s message=%s",
            filename,
            type(error).__name__,
            latency_ms,
            str(error).replace("\n", " ")[:500],
        )
        return TranscriptionResult(
            success=False,
            text="",
            latency_ms=latency_ms,
            error_message="음성 전사 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
