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
   - `OrderTrackingAgent` asks the LLM to plan a tool call, then actually queries the order DB via `EcommerceClient` (`clients/ecommerce_api_client.py` - SQLite locally, Postgres via `DATABASE_URL`).
   - `ProductRecommendationAgent` also goes through `EcommerceClient` (customer purchase history), then searches `RAGService` for a matching product.
   - `GeneralPurposeAgent` only searches `RAGService`, an in-memory dict of ingested text chunks matched by substring/keyword - not a vector DB, and `call_embeddings()` is a deterministic stub, not a real embedding model.
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
docker compose up -d --build
docker compose down
```

Runs it with a healthcheck-gated start and a graceful stop. Reaches Ollama on the host via `host.docker.internal`; see `docker-compose.yml`'s comments for pointing it at Postgres instead of the bundled SQLite dev DB.

Always include `--build`: the image is a one-time snapshot of the source tree, not a live mount, so plain `docker compose up -d` silently keeps running whatever was last built and won't pick up newer code, committed or not.

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
| The HTTP contract for a chatbot client integrating against this gateway | [`api/README.md`](api/README.md) |
| Auth, rate limiting, request correlation | `api/security.py`, `api/rate_limit.py`, `api/request_context.py` |
| Singleton service construction (DI) | `api/dependencies.py` |
| The one method that runs every customer interaction | `services/orchestrator.py` -> `handle_customer_query()` |
| How PII gets masked | `services/pii_masker.py` |
| How the LLM router/agent-reasoning/generative calls work | `services/llm_inference.py` |
| Each specialist's logic | `services/agents/*.py` |
| Order/customer/item data access (the one place every module goes through) | `clients/ecommerce_api_client.py` -> `EcommerceClient` |
| The in-memory "RAG" store | `services/rag.py` |
| How raw sample data becomes searchable chunks | `services/data_pipeline.py` |
| The CLI entry point | `main_simulation.py` |

## Where to start reading

For a first pass at the codebase, in this order:

1. `services/orchestrator.py` -> `handle_customer_query()` - the whole request lifecycle in one method, with the numbered steps below matching [Request flow](#request-flow) above.
2. `common/models.py` - every request/response shape passed between the pieces above; skim top to bottom to see one message's data shape evolve end to end.
3. `services/llm_inference.py` - the five LLM entry points (`call_router`, `call_agent_reason`, `call_agent_interpret`, `call_generative`, `call_embeddings`), each with a docstring explaining its contract and fallback behaviour.
4. `services/agents/*.py` - one file per specialist; `order_tracking_agent.py` is the most complete example of the reason -> tool-call -> interpret pattern.
5. `clients/ecommerce_api_client.py` and `db/README.md` - the data layer, local SQLite dev DB, and the switch to Postgres later.

## Known gaps

- `GET /api/v1/chat/{session_id}/history` and `DELETE /api/v1/chat/{session_id}` are mentioned in `api/routers/chat.py`'s docstring but not implemented.
- Rate limiting and the request-routing cache are in-memory and per-process - fine for one instance, not for multiple replicas.
- `ProductRecommendationAgent` hardcodes the recommended `product_id`/`name`/`price` (see its docstring) - only the RAG-matched description snippet and the customer-history framing are real. Wiring the recommendation itself to `EcommerceClient.get_item()`/`search_items()` is a natural next step.
- `MockRAGService.query_knowledge_base()` accepts a `query_embedding` parameter but matches purely by substring/keyword against `query_text` - the embedding is computed by every caller but never actually used. `LLMInferenceService.call_embeddings()` is a deterministic stub (16-dimensional at most), not a real embedding model, so this only matters once a real vector store replaces `MockRAGService`.
- `LLMInferenceService.call_agent_generate()` is fully implemented but not called by any agent or by the orchestrator today - every agent returns a structured result and lets `call_generative()` do the one customer-facing synthesis step instead.
- See `db/README.md`'s own Known Gaps for data-layer specifics (order-ID format, the disabled order-ownership check).
