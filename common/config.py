# common/config.py
"""
Central configuration loader.
Reads from .env file (or real environment variables in production).
All services import from here — never import os.environ directly elsewhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root if present

# ── LLM ──────────────────────────────────────────────────────────────────────
# LLM_PROVIDER controls which chat model backend is used across the whole
# platform (llm_inference.py chains AND base_agent.py tool-calling agents).
#   "groq"      -> public, fast inference via Groq API (needs GROQ_API_KEY)
#   "ollama"    -> local inference via a running Ollama server (no API key)
#   "anthropic" -> Anthropic API (needs ANTHROPIC_API_KEY)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

LLM_ROUTER_MAX_TOKENS: int = int(os.getenv("LLM_ROUTER_MAX_TOKENS", "512"))
LLM_GENERATIVE_MAX_TOKENS: int = int(os.getenv("LLM_GENERATIVE_MAX_TOKENS", "1024"))
LLM_AGENT_REASON_MAX_TOKENS: int = int(os.getenv("LLM_AGENT_REASON_MAX_TOKENS", "1024"))
LLM_AGENT_INTERPRET_MAX_TOKENS: int = int(os.getenv("LLM_AGENT_INTERPRET_MAX_TOKENS", "1024"))

# ── Embeddings ────────────────────────────────────────────────────────────────
# EMBEDDING_PROVIDER:
#   "huggingface"   -> local sentence-transformers model (default, no API key)
#   "deterministic" -> zero-dependency placeholder (offline dev/testing only)
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ── Vector Store ──────────────────────────────────────────────────────────────
# VECTOR_STORE: "chroma" (default) | "faiss"
VECTOR_STORE: str = os.getenv("VECTOR_STORE", "chroma")
FAISS_PERSIST_DIR: str = os.getenv("FAISS_PERSIST_DIR", "./faiss_index")

# ── ChromaDB / RAG ────────────────────────────────────────────────────────────
CHROMA_MODE: str = os.getenv("CHROMA_MODE", "local")          # "local" | "http"
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "ecommerce_support_kb")

RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))

# ── PII Masking ───────────────────────────────────────────────────────────────
PII_MASKER_MODE: str = os.getenv("PII_MASKER_MODE", "regex")   # "regex" | "presidio"

# ── Evaluation Framework (see services/evaluation.py) ─────────────────────────
# Targets mirror the project's "9. Evaluation Framework" spec.
EVAL_CONTEXT_RECALL_TARGET: float = float(os.getenv("EVAL_CONTEXT_RECALL_TARGET", "0.75"))
EVAL_CONTEXT_PRECISION_TARGET: float = float(os.getenv("EVAL_CONTEXT_PRECISION_TARGET", "0.75"))
EVAL_FAITHFULNESS_TARGET: float = float(os.getenv("EVAL_FAITHFULNESS_TARGET", "0.80"))
EVAL_ANSWER_RELEVANCE_TARGET: float = float(os.getenv("EVAL_ANSWER_RELEVANCE_TARGET", "0.75"))
EVAL_BERTSCORE_F1_TARGET: float = float(os.getenv("EVAL_BERTSCORE_F1_TARGET", "0.75"))
EVAL_MAX_LATENCY_SECONDS: float = float(os.getenv("EVAL_MAX_LATENCY_SECONDS", "5.0"))
EVAL_MAX_HALLUCINATION_RATE: float = float(os.getenv("EVAL_MAX_HALLUCINATION_RATE", "0.15"))
EVAL_MIN_WORKFLOW_COMPLETION: float = float(os.getenv("EVAL_MIN_WORKFLOW_COMPLETION", "0.85"))

# ── Observability ─────────────────────────────────────────────────────────────
# Setting LANGCHAIN_TRACING_V2=true (+ LANGCHAIN_API_KEY) in .env enables
# LangSmith tracing with zero code changes, since all LLM calls already go
# through LangChain's ChatModel interface.
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
