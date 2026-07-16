# ShopAssist

Multi-agent customer support backend: PII masking -> LLM-based intent routing -> a specialised agent (order tracking, product recommendation, or a general Q&A fallback) -> an LLM turns the agent's structured result into a reply. Any unhandled failure falls back to `EscalationAgent` instead of dropping the message.

Two entry points, one shared `services/` layer underneath:

- `main_simulation.py` - CLI walkthrough, runs one sample interaction against a small seeded dataset.
- `api/` - FastAPI service (see `api/main.py`).

## Request flow

Same steps for both entry points (`services/orchestrator.py`, `AgentOrchestratorService.handle_customer_query`):

1. `PIIMasker.mask_text()` replaces a small set of known sample names/emails with `[NAME]`/`[EMAIL]` before anything reaches the LLM. It's pattern matching against hardcoded sample strings, not a real NER/regex PII detector - fine for the demo data, not production-ready as-is.
2. The message is appended to that session's in-memory conversation history (a plain dict keyed by `session_id`, not backed by any external store - lost on process restart).
3. `LLMInferenceService.call_router()` asks the LLM which agent should handle it (`OrderTrackingAgent`, `ProductRecommendationAgent`, or `GeneralPurposeAgent`), with a confidence score. Recent identical queries reuse the last routing decision instead of asking again.
4. The chosen agent runs its own logic:
   - `OrderTrackingAgent` asks the LLM to plan a tool call, then actually queries the order DB (`clients/ecommerce_api_client.py` - SQLite locally, Postgres via `DATABASE_URL`). This is the only agent that touches a real data store.
   - `ProductRecommendationAgent` / `GeneralPurposeAgent` search `RAGService`, an in-memory dict of ingested text chunks matched by substring/keyword - not a vector DB, and `call_embeddings()` is a deterministic stub, not a real embedding model.
   - Any unhandled exception from an agent, or a routing failure, falls back to `EscalationAgent`, which packages the reason and last 3 turns for a human handover (it doesn't file a ticket anywhere - that's a stub too).
5. `LLMInferenceService.call_generative()` turns the agent's structured result into a reply.
6. The reply is appended to conversation history, and a response object goes back to the caller (`ChatbotResponse` internally, `ChatResponse` over HTTP).

## Running it

### CLI simulation

```bash
pip install -r requirements.txt
cp .env.example .env        # see .env.example for what each value does
python main_simulation.py
```

Without a working LLM configured, agents still run via rule-based/error fallbacks, so you can see the shape of the system with zero setup.

### API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Docker

```bash
docker compose up -d
docker compose down
```

Builds the image and runs it with a healthcheck-gated start and a graceful stop. Reaches Ollama on the host via `host.docker.internal`; see `docker-compose.yml`'s comments for pointing it at Postgres instead of the bundled SQLite dev DB.

## Configuration

Everything is documented in `.env.example` (copy to `.env`). Roughly:

| Concern | Vars | Notes |
|---|---|---|
| API behaviour | `APP_TITLE`, `APP_VERSION`, `LOG_LEVEL`, `CORS_ALLOWED_ORIGINS`, `LOG_BODY_MAX_CHARS` | Read via `api/config.py` |
| Auth | `API_KEY`, `API_KEY_ENFORCE` | Dummy shared-secret auth on `/api/v1/chat` - see [Auth](#auth) below |
| Request handling | `CHAT_REQUEST_TIMEOUT_SECONDS`, `RATE_LIMIT_PER_MINUTE` | Per-process only, not shared across replicas |
| LLM provider | `OLLAMA_API_BASE_URL`, `OLLAMA_ROUTER_MODEL`, `OLLAMA_AGENT_REASON_MODEL`, `OLLAMA_AGENT_INTERPRET_MODEL`, `OLLAMA_GENERATIVE_MODEL`, `OLLAMA_EMBEDDING_MODEL` | Each model must actually be pulled into that Ollama instance (`ollama list`), or calls 404 |
| Order/customer DB | `DATABASE_URL` | Unset -> local SQLite (`db/shopassist.db`, built by `python db/init_db.py`) |

## Auth

`/api/v1/chat` is gated by `api/security.py`'s `verify_api_key` - a shared `X-API-Key` header, checked with a constant-time comparison. It's a placeholder for real JWT bearer auth, not a security boundary you'd trust in production.

- `API_KEY` unset: auth off entirely, no header required.
- `API_KEY` set, `API_KEY_ENFORCE=false` (default): missing/wrong key is logged (`api.auth`) but the request still goes through - lets you roll this out before every caller sends the header.
- `API_KEY_ENFORCE=true`: missing/wrong key gets `401`.
- `API_KEY_ENFORCE=true` with no `API_KEY` configured: treated as a misconfiguration, fails closed with `500` rather than silently letting everyone in.

The current enforcement mode is logged once at startup (`WARNING` level) and reported by `/api/v1/health` (`api_key_enforced`), so it's never just buried in a log stream. `/api/v1/health` and `/` are intentionally not gated - health checks shouldn't need credentials.

## Observability

- Every request gets a correlation ID (`api/request_context.py`) - reused from an incoming `X-Request-ID` header or generated, present in every log line for that request and echoed back on the response.
- `api/middleware.py` logs a one-line summary per request at `INFO`; `LOG_LEVEL=DEBUG` also logs full request/response bodies (unmasked - don't leave this on outside local dev).
- `GET /api/v1/health` reports registered agents, RAG doc count, DB reachability, and Ollama reachability - each downstream check degrades to `false` rather than throwing, so one dependency being down doesn't take the health check down with it.

## Project layout

| To understand... | Open |
|---|---|
| The FastAPI app, middleware, and router wiring | `api/main.py`, `api/middleware.py` |
| Auth, rate limiting, request correlation | `api/security.py`, `api/rate_limit.py`, `api/request_context.py` |
| Singleton service construction (DI) | `api/dependencies.py` |
| The one method that runs every customer interaction | `services/orchestrator.py` -> `handle_customer_query()` |
| How PII gets masked | `services/pii_masker.py` |
| How the LLM router/agent-reasoning/generative calls work | `services/llm_inference.py` |
| Each specialist's logic | `services/agents/*.py` |
| The in-memory "RAG" store | `services/rag.py` |
| How raw sample data becomes searchable chunks | `services/data_pipeline.py` |
| The CLI entry point | `main_simulation.py` |

## Known gaps

- `ChatResponse.confidence_score` always reads `1.0` - `services/orchestrator.py` doesn't pass the router's actual confidence through to the response object.
- `GET /api/v1/chat/{session_id}/history` and `DELETE /api/v1/chat/{session_id}` are mentioned in `api/routers/chat.py`'s docstring but not implemented.
- `OrderTrackingAgent` tells the LLM its tool is called `ECommerceAPI.getOrderDetails`, then checks for exactly `ECommerceAPI` - a correct tool-call decision from the LLM can still fail this check and silently skip the real lookup.
- Rate limiting and the request-routing cache are in-memory and per-process - fine for one instance, not for multiple replicas.
