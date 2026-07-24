# API Gateway — Client Integration Guide

Reference for teams integrating a chatbot client (e.g. `shopassist-client`)
against this FastAPI service. Covers the HTTP contract only — for how a
request is processed once it's inside the gateway, see the root
[`README.md`](../README.md)'s "Request flow".

Base URL: `http://<host>:8000` locally (`uvicorn api.main:app --port 8000`
or the bundled `docker-compose.yml`, which publishes the same port).
Interactive schema/try-it-out: `GET /docs` (Swagger UI) or `GET /redoc`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/chat` | Send a customer message, get the orchestrated reply |
| GET | `/api/v1/health` | Liveness + downstream-dependency status |
| GET | `/` | Basic root/landing check |

`GET /api/v1/chat/{session_id}/history` and `DELETE /api/v1/chat/{session_id}`
are referenced in the router's docstring but **not implemented yet** — do not
build against them.

### POST /api/v1/chat

Request body (`ChatRequest`, `api/schemas.py`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | no | Omit to start a new session — the gateway generates one and returns it in the response. Send the same value back on every subsequent turn of that conversation. |
| `user_id` | string | **yes** | The logged-in identifier for this customer, e.g. `alum-1001`. Sent as-is to the DB layer for order/customer lookups — send the real logged-in value, not a display name. |
| `text` | string | **yes** | The customer's message. 1–2000 characters. |
| `source_channel` | string | no | Defaults to `"web_chat"`. Free-form label (`mobile_app`, `twitter`, ...) — not validated against a fixed list. |

Example:

```json
{
  "session_id": "abc123",
  "user_id": "alum-1001",
  "text": "Where is my order ord-1001?",
  "source_channel": "web_chat"
}
```

Response body (`ChatResponse`, `200 OK`):

| Field | Type | Notes |
|---|---|---|
| `session_id` | string | Echoes the request's `session_id`, or the newly generated one if omitted — persist this for the next turn. |
| `response_text` | string | The reply to show the customer. |
| `agent_invoked` | string \| null | Which specialist handled it: `OrderTrackingAgent`, `ProductRecommendationAgent`, `GeneralPurposeAgent`, or `EscalationAgent`. A single message can decompose into sub-tasks handled by more than one agent (e.g. an order-status question plus a policy question in one turn) — when that happens this is the literal string `"Multi-Agent Orchestrator"` instead of one of the four names above. `EscalationAgent` means the gateway couldn't resolve the request automatically — there is no ticket/CRM concept at this API layer, so if the client needs a support-ticket UI flow, it must synthesize that client-side when it sees this value. |
| `confidence_score` | float | 0.0–1.0, from the LLM's own self-reported confidence — not independently validated. |
| `timestamp` | string (ISO-8601) | Server-side response time. |

Example:

```json
{
  "session_id": "abc123",
  "response_text": "Your order ord-1001 has been delivered...",
  "agent_invoked": "OrderTrackingAgent",
  "confidence_score": 1.0,
  "timestamp": "2026-07-13T10:15:00.000000"
}
```

### GET /api/v1/health

No auth, no rate limiting — always callable, intended for container
healthchecks and manual checks. Every downstream check degrades to `false`
instead of raising, so one dependency being down never turns into a 500 here.

```json
{
  "status": "ok",
  "registered_agents": ["OrderTrackingAgent", "ProductRecommendationAgent", "GeneralPurposeAgent", "EscalationAgent"],
  "rag_documents_indexed": 5,
  "database_reachable": true,
  "llm_reachable": true,
  "api_key_enforced": false
}
```

## Auth

Header: `X-API-Key: <key>`.

| `API_KEY` set? | `API_KEY_ENFORCE` | Behaviour |
|---|---|---|
| no | (n/a) | Auth off entirely — no header needed. |
| yes | `false` (default) | Missing/wrong key is logged but the request still succeeds — safe to integrate before every caller sends the header. |
| yes | `true` | Missing/wrong key → `401`. |

Check `GET /api/v1/health`'s `api_key_enforced` field to see which mode is
currently active rather than assuming — the client should always know
whether a `401` is expected on a bad/missing key.

## Rate limiting

`RATE_LIMIT_PER_MINUTE` (default 30) per API key (or per client IP if no key
is sent), sliding 60-second window. Exceeding it returns `429` with
`{"detail": "Rate limit exceeded. Try again shortly."}`. In-memory and
per-process — with multiple gateway replicas the effective limit is
`RATE_LIMIT_PER_MINUTE × replica count`, not a global cap.

## Timeouts

`CHAT_REQUEST_TIMEOUT_SECONDS` (default 60) bounds how long `/api/v1/chat`
waits on the orchestrator before returning `504` with
`{"detail": "Request timed out while processing your message."}`. Set the
client's own HTTP timeout at least a few seconds *above* this value — a
client timeout shorter than the gateway's own means the client gives up
before the gateway would have, defeating the point of the gateway timeout.

## Errors

| Status | When |
|---|---|
| `401` | `API_KEY_ENFORCE=true` and the key is missing/wrong |
| `422` | Request body fails schema validation (e.g. `text` empty or over 2000 chars, `user_id` missing) — standard FastAPI validation error body |
| `429` | Rate limit exceeded |
| `500` | Unhandled error in the orchestrator, or `API_KEY_ENFORCE=true` with no `API_KEY` configured (server misconfiguration) |
| `504` | Orchestrator call exceeded `CHAT_REQUEST_TIMEOUT_SECONDS` |

All error bodies are `{"detail": "<message>"}` (FastAPI's default `HTTPException` shape), except `422` which uses FastAPI's standard validation-error array format.

## CORS

`CORS_ALLOWED_ORIGINS` (comma-separated, default `http://localhost:8501,http://127.0.0.1:8501`
— `shopassist-client`'s Streamlit dev port) must list the client's exact
origin; add it before pointing a new client deployment at this gateway, since
a wildcard is not permitted here (`allow_credentials=True` is set, and
browsers reject `*` combined with credentials).

## Request correlation

Every response carries an `X-Request-ID` header — echoes an incoming
`X-Request-ID` if the client sends one, otherwise the gateway generates one.
Every server-side log line for that request is tagged with the same ID.
Sending your own `X-Request-ID` (e.g. a client-generated UUID per request)
makes cross-service debugging easier — pass it along if filing an issue about
a specific call.

## Integration checklist

- [ ] Send the real logged-in `user_id` on every request — not a display name (see root `README.md` for why this matters to order/customer lookups).
- [ ] Persist and resend `session_id` across turns of the same conversation.
- [ ] Set the client's HTTP timeout comfortably above `CHAT_REQUEST_TIMEOUT_SECONDS`.
- [ ] Treat `agent_invoked == "EscalationAgent"` as this gateway's signal to hand off to a human — build any ticket/CRM UI client-side.
- [ ] Send an `X-API-Key` header on every request.
- [ ] Confirm the client's origin is in `CORS_ALLOWED_ORIGINS` before pointing a new deployment at this gateway.
