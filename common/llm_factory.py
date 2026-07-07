# common/llm_factory.py
"""
Provider-agnostic LLM factory.

Single place that decides which chat model backend to instantiate, based on
`LLM_PROVIDER` in .env. Every other file (services/llm_inference.py chains,
services/agents/base_agent.py tool-calling agents) calls `build_chat_model()`
instead of importing a specific provider directly — swapping providers is a
one-line .env change, no code changes anywhere else.

Supported providers:
  "groq"      -> ChatGroq       (public API, fast inference, needs GROQ_API_KEY)
  "ollama"    -> ChatOllama     (local inference, needs a running Ollama server)
  "anthropic" -> ChatAnthropic  (Anthropic API, needs ANTHROPIC_API_KEY)

Returns None if the selected provider's credentials/config are missing, so
callers can fall back to deterministic mock logic (see each service's
`_mock_*` methods) rather than crashing.
"""

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from common.config import (
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


def build_chat_model(
    max_tokens: int = 1024,
    temperature: float = 0.2,
    provider: Optional[str] = None,
) -> Optional[BaseChatModel]:
    """
    Build a LangChain chat model for the configured (or explicitly passed) provider.

    Returns None if the provider's required config/credentials are absent —
    this is intentional so the rest of the pipeline can run in offline/mock
    mode without crashing.
    """
    provider = (provider or LLM_PROVIDER).lower()

    if provider == "groq":
        return _build_groq(max_tokens, temperature)
    if provider == "ollama":
        return _build_ollama(max_tokens, temperature)
    if provider == "anthropic":
        return _build_anthropic(max_tokens, temperature)

    logger.warning("[LLMFactory] Unknown LLM_PROVIDER '%s'. No model built.", provider)
    return None


def _build_groq(max_tokens: int, temperature: float) -> Optional[BaseChatModel]:
    if not GROQ_API_KEY:
        logger.warning("[LLMFactory] GROQ_API_KEY not set — Groq model unavailable.")
        return None
    from langchain_groq import ChatGroq
    from pydantic import SecretStr

    return ChatGroq(
        model=GROQ_MODEL,
        api_key=SecretStr(GROQ_API_KEY),
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _build_ollama(max_tokens: int, temperature: float) -> Optional[BaseChatModel]:
    """
    Ollama runs locally — no API key needed, but the server must be running
    (`ollama serve`) with the model already pulled (`ollama pull llama3`).
    We do a lightweight reachability check so callers get a clean None
    (→ mock fallback) instead of a hung connection.
    """
    try:
        import httpx

        httpx.get(OLLAMA_BASE_URL, timeout=1.5)
    except Exception:
        logger.warning(
            "[LLMFactory] Ollama server not reachable at %s — falling back to mock.",
            OLLAMA_BASE_URL,
        )
        return None

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        stop=None,
    )


def _build_anthropic(max_tokens: int, temperature: float) -> Optional[BaseChatModel]:
    if not ANTHROPIC_API_KEY:
        logger.warning("[LLMFactory] ANTHROPIC_API_KEY not set — Anthropic model unavailable.")
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens_to_sample=max_tokens,
        temperature=temperature,
        timeout=None,
        stop=None,
    )
