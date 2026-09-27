import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


AI_CONNECTION_ERROR_MESSAGE = "AI 서비스 연결 정보가 설정되지 않았습니다."


def _get_setting(name: str, default: str = "") -> str:
    """로컬 환경변수를 우선하고 Streamlit Cloud Secrets를 보조로 사용합니다."""
    environment_value = os.getenv(name)
    if environment_value is not None:
        return environment_value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
    except (FileNotFoundError, KeyError, RuntimeError):
        secret_value = None
    return str(secret_value) if secret_value is not None else default


@dataclass(frozen=True)
class Settings:
    app_env: str
    log_level: str
    openai_api_key: str
    model_default: str
    model_review: str
    transcribe_model: str
    embedding_model: str

    @property
    def api_key_configured(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        app_env=_get_setting("APP_ENV", "local"),
        log_level=_get_setting("APP_LOG_LEVEL", "INFO"),
        openai_api_key=_get_setting("OPENAI_API_KEY"),
        model_default=_get_setting("OPENAI_MODEL_DEFAULT", "gpt-5.6-luna"),
        model_review=_get_setting("OPENAI_MODEL_REVIEW", "gpt-5.6-sol"),
        transcribe_model=_get_setting("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe"),
        embedding_model=_get_setting(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
    )
