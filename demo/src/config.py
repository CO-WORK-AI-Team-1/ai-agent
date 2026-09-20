import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


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
