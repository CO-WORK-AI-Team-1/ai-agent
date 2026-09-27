import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


# 화면과 백엔드가 동일하게 노출하는 안전한 설정 오류 메시지입니다.
# API 키 이름이나 내부 설정값을 사용자에게 보여 주지 않습니다.
AI_SERVICE_CONFIGURATION_ERROR = "AI 서비스 연결 정보가 설정되지 않았습니다."


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
        app_env=os.getenv("APP_ENV", "local"),
        log_level=os.getenv("APP_LOG_LEVEL", "INFO"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model_default=os.getenv("OPENAI_MODEL_DEFAULT", "gpt-5.6-luna"),
        model_review=os.getenv("OPENAI_MODEL_REVIEW", "gpt-5.6-sol"),
        transcribe_model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-transcribe"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
