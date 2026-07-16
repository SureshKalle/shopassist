# api/config.py
"""
Typed access to the env vars that control this API's own behaviour (app
metadata, logging, CORS). services/clients-layer vars (OLLAMA_*,
DATABASE_URL, OPENAI_API_KEY) are read directly where they're used instead
- see .env.example for the full list.

Calls load_dotenv() explicitly so env loading doesn't depend on import
order (services/llm_inference.py also calls it, as an import side effect).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CORS_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_version: str
    log_level: str
    # Comma-separated in the env; must be explicit origins, not "*" -
    # CORSMiddleware is wired up with allow_credentials=True in main.py,
    # and browsers reject a wildcard origin combined with credentials.
    cors_allowed_origins: list[str]
    # Max chars of request/response body logged per line when
    # log_level=DEBUG - see api/middleware.py.
    log_body_max_chars: int


def get_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "AI Agentic Customer Support Platform"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        cors_allowed_origins=_split_csv(os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_CORS_ORIGINS)),
        log_body_max_chars=int(os.getenv("LOG_BODY_MAX_CHARS", "2000")),
    )


settings = get_settings()
