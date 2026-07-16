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


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


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
    # Dummy API-key auth - see api/security.py. Empty api_key disables the
    # check entirely regardless of api_key_enforce.
    api_key: str
    api_key_enforce: bool
    # Wall-clock cap on the orchestrator call in routers/chat.py - protects
    # the request/response cycle from a hung LLM call. Doesn't reclaim the
    # worker thread itself (Python threads can't be force-cancelled); see
    # that file's comment.
    chat_request_timeout_seconds: float
    # <= 0 disables rate limiting. In-memory, per-process only - see
    # api/rate_limit.py.
    rate_limit_per_minute: int


def get_settings() -> Settings:
    return Settings(
        app_title=os.getenv("APP_TITLE", "AI Agentic Customer Support Platform"),
        app_version=os.getenv("APP_VERSION", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        cors_allowed_origins=_split_csv(os.getenv("CORS_ALLOWED_ORIGINS", _DEFAULT_CORS_ORIGINS)),
        log_body_max_chars=int(os.getenv("LOG_BODY_MAX_CHARS", "2000")),
        api_key=os.getenv("API_KEY", ""),
        api_key_enforce=_env_bool("API_KEY_ENFORCE", False),
        chat_request_timeout_seconds=float(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "60")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
    )


settings = get_settings()
