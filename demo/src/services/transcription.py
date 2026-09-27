import logging
import shutil
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from openai import OpenAI

from src.config import AI_SERVICE_CONFIGURATION_ERROR, Settings, get_settings

logger = logging.getLogger(__name__)

# 사용자가 업로드할 수 있는 회의 녹음의 최대 크기입니다.
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024
# OpenAI 전사 API의 파일 단위 제한(25MB)보다 여유를 둔 안전한 청크 크기입니다.
MAX_TRANSCRIPTION_REQUEST_BYTES = 24 * 1024 * 1024
CHUNK_DURATION_SECONDS = 10 * 60
CHUNK_AUDIO_BITRATE = "64k"
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
    chunk_count: int = 1
    error_message: str | None = None


def _split_large_audio_for_transcription(
    *,
    filename: str,
    data: bytes,
) -> list[tuple[str, bytes]]:
    """Create API-safe MP3 chunks from one large upload and remove temp files.

    OpenAI accepts 25MB per transcription request. Large uploads are converted
    to mono 16kHz MP3 chunks so the browser can accept a meeting recording up
    to 100MB while every API request remains below the provider limit.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "대용량 녹음 파일을 처리할 준비가 되지 않았습니다. "
            "잠시 후 다시 시도하거나 25MB 이하 파일을 업로드해 주세요."
        )

    suffix = Path(filename).suffix.lower()
    # Use the app working directory rather than a system temp directory.
    # Some managed Windows/Cloud environments deny writes to their global temp path.
    directory = Path.cwd() / ".worldvision_audio_tmp"
    directory.mkdir(parents=True, exist_ok=True)
    job_prefix = uuid4().hex
    source_path = directory / f"{job_prefix}_source{suffix}"
    output_pattern = directory / f"{job_prefix}_part_%03d.mp3"
    try:
        source_path.write_bytes(data)

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            CHUNK_AUDIO_BITRATE,
            "-f",
            "segment",
            "-segment_time",
            str(CHUNK_DURATION_SECONDS),
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if process.returncode != 0:
            logger.error(
                "Large audio conversion failed: filename=%s return_code=%s error=%s",
                filename,
                process.returncode,
                process.stderr.replace("\n", " ")[:500],
            )
            raise RuntimeError(
                "음성 파일을 전사 가능한 형식으로 변환하지 못했습니다. "
                "지원 형식과 파일 상태를 확인해 주세요."
            )

        chunks = [
            (f"{Path(filename).stem}_part_{position:03d}.mp3", part_path.read_bytes())
            for position, part_path in enumerate(
                sorted(directory.glob(f"{job_prefix}_part_*.mp3")),
                start=1,
            )
        ]
    finally:
        # Files are transient. Ignore a Windows file-lock cleanup race; a later
        # OS cleanup is safer than failing an otherwise completed upload.
        for temporary_path in directory.glob(f"{job_prefix}_*"):
            temporary_path.unlink(missing_ok=True)

    if not chunks:
        raise RuntimeError("음성 파일에서 전사할 구간을 만들지 못했습니다.")
    if any(len(chunk_data) > MAX_TRANSCRIPTION_REQUEST_BYTES for _, chunk_data in chunks):
        raise RuntimeError(
            "대용량 음성 파일을 안전한 전사 구간으로 나누지 못했습니다. "
            "파일을 더 짧게 나눈 뒤 다시 업로드해 주세요."
        )
    return chunks


def _prepare_transcription_chunks(
    *,
    filename: str,
    data: bytes,
) -> list[tuple[str, bytes]]:
    if len(data) <= MAX_TRANSCRIPTION_REQUEST_BYTES:
        return [(filename, data)]
    return _split_large_audio_for_transcription(filename=filename, data=data)


def transcribe_meeting_audio(
    *,
    filename: str,
    data: bytes,
    settings: Settings | None = None,
    client: Any | None = None,
) -> TranscriptionResult:
    """Transcribe one supported recording; large uploads are split temporarily."""
    if not data:
        raise ValueError("빈 음성 파일은 업로드할 수 없습니다.")
    if len(data) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError("음성 파일 크기는 100MB를 초과할 수 없습니다.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        supported_formats = ", ".join(
            extension.removeprefix(".").upper()
            for extension in SUPPORTED_AUDIO_SUFFIXES
        )
        raise ValueError(f"지원하지 않는 음성 형식입니다. 지원 형식: {supported_formats}")

    settings = settings or get_settings()
    if not settings.api_key_configured and client is None:
        raise RuntimeError(AI_SERVICE_CONFIGURATION_ERROR)

    api = client or OpenAI(
        api_key=settings.openai_api_key,
        timeout=120.0,
        max_retries=2,
    )
    started = perf_counter()

    try:
        chunks = _prepare_transcription_chunks(filename=filename, data=data)
        transcript_parts: list[str] = []
        for chunk_filename, chunk_data in chunks:
            audio_file = BytesIO(chunk_data)
            audio_file.name = chunk_filename
            response = api.audio.transcriptions.create(
                model=settings.transcribe_model,
                file=audio_file,
            )
            transcript = str(getattr(response, "text", "")).strip()
            if not transcript:
                raise RuntimeError("음성에서 텍스트를 추출하지 못했습니다.")
            transcript_parts.append(transcript)

        latency_ms = round((perf_counter() - started) * 1000)

        return TranscriptionResult(
            success=True,
            text="\n\n".join(transcript_parts),
            latency_ms=latency_ms,
            chunk_count=len(chunks),
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
            chunk_count=0,
            error_message="음성 전사 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )
