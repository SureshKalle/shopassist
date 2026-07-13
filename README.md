# shopassist — AI Agentic Customer Support Platform

FastAPI backend for a multi-agent e-commerce support chatbot: an LLM router
picks a specialist agent (order tracking, product recommendation, general
Q&A) per message, agents call out to a real order database and a knowledge
base, and a final reply gets returned to whatever's calling it — right now
that's the `shopassist-streamlit` storefront's chat widget, over plain HTTP.

This README describes what's actually implemented, not the target
architecture. Where something is a placeholder rather than the real thing,
it says so — see **Known issues / limitations** below before you assume a
piece of this is more finished than it is.

## What's actually here

- **4 agents**, not 5: `OrderTrackingAgent`, `ProductRecommendationAgent`,
  `GeneralPurposeAgent`, `EscalationAgent`. A `ReturnsAgent` was planned
  (the router's own prompt still lists it as a valid target — see Known
  issues) but was never built; return/refund questions fall through to
  `GeneralPurposeAgent` or escalate.
- **Routing and per-agent tool-planning are real LLM calls** — via an
  OpenAI-compatible client pointed at a local Ollama server
  (`services/llm_inference.py::call_router` / `call_agent_reason`).
- **Order lookups are real** — `services/ecommerce_client.py` runs actual
  SQL (SQLAlchemy) against a database: SQLite locally
  (`db/shopassist.db`), or Postgres if you set `DATABASE_URL`.
- **The knowledge base is not real RAG yet** — `services/rag.py` is an
  in-memory dict matched by substring (`if query_text in doc.content`),
  and `call_embeddings()` returns character codes, not a real embedding.
  It's there so the agent shapes are right; the retrieval quality isn't.
- **The final customer-facing reply is not LLM-generated** —
  `call_generative()` builds it with plain string templates. The two real
  LLM calls above happen earlier in the pipeline (routing, tool planning);
  by the time a reply is assembled, it's back to templating.
- **PII masking is literal string replacement** for a handful of
  known-fixture values (`"John Doe"`, one hardcoded email, etc.) — not
  regex, not NER. Treat it as a placeholder for a real masking pass, not
  an actual privacy guarantee.

## API

Two endpoints:

```
POST /api/v1/chat
GET  /api/v1/health
```

### `POST /api/v1/chat`

```json
{ "session_id": null, "user_id": "1", "text": "Where is order 12345?", "source_channel": "web_chat" }
```

`session_id` is optional — omit it to start a new conversation; the
response echoes back the one you should reuse for the next turn.

```json
{ "session_id": "session_a1b2c3d4e5f6", "response_text": "...",
  "agent_invoked": "OrderTrackingAgent", "confidence_score": 1.0,
  "timestamp": "2026-07-13T20:06:50.766424" }
```

`confidence_score` is currently always `1.0` regardless of what the router
actually returned — see Known issues.

### `GET /api/v1/health`

```json
{ "status": "ok",
  "registered_agents": ["OrderTrackingAgent", "ProductRecommendationAgent", "GeneralPurposeAgent", "EscalationAgent"],
  "rag_collection_size": 0,
  "database_reachable": true }
```

Doesn't throw on a downstream outage — a dead database shows up as
`database_reachable: false` in a 200, not a 500 that takes the health
check out along with it.

There used to be `ingestion`, `evaluation`, and chat history/clear
endpoints too; they called into modules (`services/evaluation.py`,
`services/agents/returns_agent.py`, a couple of `common/` files) that
never actually got written, so the app couldn't start. They were removed
rather than stubbed out — see `git log` around "removed unused codebase"
if you want the story. `/health` here is new, re-added because a
deployed service without one is a pain to run behind anything.

## Running locally

You need Python 3.10+, and Ollama running locally with two models pulled
(everything degrades to rule-based fallbacks without it, but routing and
tool-planning stop being AI-driven):

```bash
ollama pull llama3:8b-instruct
ollama pull nomic-embed-text
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # defaults match a local Ollama install
python db/init_db.py             # builds db/shopassist.db from schema + seed data
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs at `http://localhost:8000/docs`. Try it:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1", "text": "Where is order 12345?"}'
```

Or run the standalone simulation instead of the API — same services, no
HTTP:

```bash
python main_simulation.py
```

It only runs one interaction by default; four more are written but
commented out at the bottom of the file (product recommendation, a
returns-policy question, a second order lookup with PII in it, and a
general question with no good match). Uncomment them to see more of the
routing in action — the returns one will demonstrate the `ReturnsAgent`
gap described in Known issues rather than a real returns flow.

## Configuration

`.env` (gitignored — `.env.example` is the checked-in template):

```env
OLLAMA_API_BASE_URL=http://localhost:11434
OLLAMA_ROUTER_MODEL=llama3:8b-instruct
OLLAMA_AGENT_REASON_MODEL=llama3:8b-instruct
OLLAMA_AGENT_INTERPRET_MODEL=llama3:8b-instruct
OLLAMA_GENERATIVE_MODEL=llama3:8b-instruct
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

`DATABASE_URL` isn't in that list because it's optional — unset, it falls
back to the local SQLite file. Set it to point at Postgres instead (e.g.
`shopassist-database`'s instance):

```
DATABASE_URL=postgresql://shopassist:shopassist123@localhost:5432/shopassist
```

For a server deployment, `.env.production.example` documents the same
variables with production-shaped placeholders (a real Postgres URL, an
Ollama endpoint that isn't `localhost`) — copy it to `.env.production` and
fill in real values through your platform's secret manager, don't commit
the filled-in version.

## Known issues / limitations

Worth knowing before you build on top of this:

- **The router can recommend an agent that doesn't exist.** The system
  prompt in `call_router` lists `ReturnsAgent` as a valid target
  (`services/llm_inference.py`), but it's not registered in
  `api/dependencies.py::get_agents()` or `main_simulation.py`. When the
  router picks it, `orchestrator.py` falls back to `EscalationAgent` —
  so returns questions work, technically, they just always escalate to a
  human instead of getting handled.
- **`confidence_score` in the API response is always `1.0`.** The router
  computes a real confidence value, but `orchestrator.py`'s
  `handle_customer_query()` never passes it into the `ChatbotResponse` it
  builds, so the Pydantic default (`1.0`) is what actually ships. The
  number the router computed is discarded.
- ~~The chat reply's greeting derives a "name" from `session_id`~~ **fixed.**
  `call_generative()` now greets with `NLGRequest.customer_name`, looked up
  by `user_id` via `ECommerceAPIClient.get_customer_name()` in
  `orchestrator.py`, once per session (first turn only — it doesn't repeat
  on every reply). Falls back to a plain "Hello!" when `user_id` isn't a
  real numeric customer ID, rather than fabricating a name.
- **`user_id` from the chat request and the database's numeric
  `customer_id` aren't the same namespace.** `get_order_details()` only
  enforces ownership when `user_id` happens to parse as an int; otherwise
  it skips the check rather than reject the request. Degrades gracefully,
  but isn't real authorization.
- **This service and `shopassist-database` maintain separate schemas.**
  `db/schema.sql` here (`customers` / `items` / `sessions` / `orders` /
  `order_items`) isn't the same shape as `shopassist-database`'s
  (`customers` / `products` / `orders` / `order_items`), despite that repo
  being meant as the platform's shared source of truth. Pointing
  `DATABASE_URL` at it today would mean rewriting the queries in
  `services/ecommerce_client.py` to match its column names.
- **Only Ollama is actually wired up as an LLM provider.** `openai` is
  the client library, but `services/llm_inference.py` hardcodes
  `api_key="ollama"` — there's no path to a hosted provider (OpenAI,
  Anthropic) without a code change, regardless of what you set in `.env`.
- **Session/conversation state is in-process memory** on the
  `AgentOrchestratorService` singleton — gone on restart, and won't work
  correctly if you ever run more than one instance behind a load balancer.

## Project structure

```
shopassist/
├── api/
│   ├── main.py              app instance, lifespan, router wiring
│   ├── schemas.py            HTTP request/response models
│   ├── dependencies.py       singleton service construction (DI via lru_cache)
│   └── routers/
│       ├── chat.py           POST /api/v1/chat
│       └── health.py         GET  /api/v1/health
├── services/
│   ├── orchestrator.py       routes each message through PII mask → agent → reply
│   ├── llm_inference.py      LLM calls (router, tool-planning) + the templated reply
│   ├── rag.py                in-memory keyword-matched "knowledge base"
│   ├── pii_masker.py         literal-string PII redaction
│   ├── ecommerce_client.py   real SQL against the order/customer database
│   ├── data_pipeline.py      cleans/masks/chunks raw text before it hits rag.py
│   └── agents/               one file per specialist agent
├── common/
│   └── models.py             internal Pydantic domain models shared across services/
├── db/
│   ├── schema.sql             Postgres/Supabase schema (source of truth for prod)
│   ├── schema_sqlite.sql       same schema, adapted for local SQLite
│   ├── seed_data.sql           demo customers/items/orders
│   ├── init_db.py              rebuilds db/shopassist.db from the two files above
│   └── shopassist.db           generated locally, gitignored
├── main_simulation.py         CLI run of the same services, no HTTP
├── requirements.txt
├── .env.example
└── .env.production.example
```

## Not done yet

- A `ReturnsAgent` that actually exists (see Known issues)
- Real RAG: real embeddings, a real vector store, real chunking
- The reply-generation step actually calling an LLM
- Regex/NER-based PII masking instead of hardcoded literals
- Session state in Redis or Postgres instead of process memory
- Auth on the API (currently wide open; CORS is `allow_origins=["*"]`)
- Reconciling this repo's DB schema with `shopassist-database`'s
- A Dockerfile — there isn't one yet, despite `shopassist-database`
  already having a docker-compose setup this could sit alongside
