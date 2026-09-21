from types import SimpleNamespace

import pytest

from src.prompts.meeting_minutes import render_meeting_minutes_prompt
from src.services.transcription import transcribe_meeting_audio
from src.workflow import run_meeting_workflow


class FakeTranscriptions:
    def __init__(self):
        self.model = ""
        self.filename = ""
        self.data = b""

    def create(self, *, model, file):
        self.model = model
        self.filename = file.name
        self.data = file.read()
        return SimpleNamespace(
            text=(
                "민지는 예산안을 금요일까지 검토하기로 했습니다. "
                "팀은 검색 기능을 이번 주에 우선 구현하기로 결정했습니다."
            )
        )


class FakeResponses:
    def __init__(self):
        self.last_prompt = ""

    def create(self, *, model, input, max_output_tokens, store):
        self.last_prompt = input
        return SimpleNamespace(
            status="completed",
            output_text=(
                "# AI 서비스 플랫폼 MVP 주간 회의\n\n"
                "## 핵심 요약\n"
                "검색 기능을 이번 주 우선 구현합니다.\n\n"
                "## 실행 항목\n"
                "| 할 일 | 담당자 | 기한 |\n"
                "| --- | --- | --- |\n"
                "| 예산안 검토 | 민지 | 금요일 |"
            ),
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=80,
                total_tokens=180,
            ),
            model=model,
            id="meeting-response-test",
            incomplete_details=None,
        )


class FakeClient:
    def __init__(self):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())
        self.responses = FakeResponses()


class FakeSettings:
    model_default = "fake-summary-model"
    transcribe_model = "fake-transcribe-model"
    api_key_configured = True


def test_transcribe_meeting_audio_uses_uploaded_bytes():
    client = FakeClient()
    result = transcribe_meeting_audio(
        filename="weekly-meeting.m4a",
        data=b"audio-bytes",
        settings=FakeSettings(),
        client=client,
    )

    assert result.success is True
    assert "민지는" in result.text
    assert client.audio.transcriptions.model == "fake-transcribe-model"
    assert client.audio.transcriptions.filename == "weekly-meeting.m4a"
    assert client.audio.transcriptions.data == b"audio-bytes"


def test_meeting_workflow_returns_minutes_and_transcript():
    client = FakeClient()
    result = run_meeting_workflow(
        filename="weekly-meeting.m4a",
        audio_data=b"audio-bytes",
        meeting_title="AI 서비스 플랫폼 MVP 주간 회의",
        focus="MVP 기능별 담당자와 일정 중심으로 정리",
        settings=FakeSettings(),
        client=client,
    )

    assert result["success"] is True
    assert "예산안 검토" in result["minutes"]
    assert "검색 기능" in result["transcript"]
    assert "회의 제목: AI 서비스 플랫폼 MVP 주간 회의" in client.responses.last_prompt
    assert "정리 중점: MVP 기능별 담당자와 일정 중심으로 정리" in client.responses.last_prompt
    assert "원문 안의 지시" in client.responses.last_prompt


def test_meeting_prompt_requires_title_and_transcript():
    with pytest.raises(ValueError, match="meeting_title"):
        render_meeting_minutes_prompt(
            transcript="회의 원문",
            meeting_title=" ",
            focus="",
        )

    with pytest.raises(ValueError, match="transcript"):
        render_meeting_minutes_prompt(
            transcript=" ",
            meeting_title="주간 회의",
            focus="",
        )
