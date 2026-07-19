# Changes — feature/ordertracking session log

Summary of everything done in this working session, committed to
`feature/ordertracking` at `1d6d7e4` (pushed to origin) unless noted otherwise.

## 1. Database schema

**`db/schema_sqlite.sql`**
- Extended `items` to match an Amazon-style product CSV: added
  `discount_percentage`, `rating`, `rating_count`, `img_link`,
  `product_link` (all nullable, CHECK-bounded). Widened `category` from
  `VARCHAR(100)` to `TEXT` (pipe-delimited category paths exceed 100 chars).
- Added a new `item_reviews` table: `review_id` (PK), `item_id` (FK →
  `items`), `user_id` (FK → `customers`), `review_title`, `review_content`,
  `created_at`. `user_id` is a real FK to `customers` (not free-text), so
  every table in the schema sits in one connected FK/PK graph:
  `item_reviews` → `items` and → `customers`; `sessions`/`orders` →
  `customers`; `order_items` → `orders` and → `items`.

**`db/seed_sqlite.sql`**
- Added 10 sample `item_reviews` rows against existing seeded items/customers.

**`db/README.md`**
- Documented the six-table schema, the CSV → column mapping, and the
  FK/PK connectivity.

## 2. PostgreSQL migration

- **`db/schema_postgres.sql`** (new) — Postgres port of the schema: same
  tables/columns/FKs, with `TIMESTAMPTZ`/`now()` instead of `TEXT`/
  `CURRENT_TIMESTAMP`, `TRUE`/`FALSE` instead of `1`/`0`, and no SQLite-only
  `PRAGMA` line.
- **`db/seed_postgres.sql`** (new) — same seed data, ported.
- **`docker-compose.yml`** — added a `postgres` service (`postgres:16-alpine`,
  creds matching `.env.example`) that auto-loads the two files above via
  Postgres's `docker-entrypoint-initdb.d` on first start, plus a named
  volume and healthcheck.
- **`db/README.md`** — added a "Switching to PostgreSQL" walkthrough
  (start command, `DATABASE_URL` for host vs. in-network use, how to force
  a reload since init scripts only run once per fresh volume).
- **`.env`** (local only, gitignored, not in this list's commit) —
  `DATABASE_URL` pointed at the local Postgres container.

Verified live: brought the container up, confirmed schema/seed loaded,
and ran `EcommerceClient` directly against it with zero code changes.

## 3. `clients/ecommerce_api_client.py` — new methods

Six new methods, each paired with the other half of its CRUD lifecycle and
guarded the same way `cancel_order` already guards on order status:

| Pair | Guard |
|---|---|
| `create_order` / `delete_order` | create checks customer exists + per-item stock; delete only allowed while `pending` or `cancelled` — never `delivered`/`confirmed`/`shipped`/`returned` |
| `add_item` / `remove_item` | add blocks duplicate `item_id`; remove is a soft-delete (`is_active=FALSE`) and blocks while the item is in any unfulfilled (`pending`/`confirmed`/`shipped`) order |
| `add_review` / `remove_review` | add requires a real, non-cancelled purchase of that item by that `user_id` (verified-purchase); remove only lets the review's own author delete it |

All six verified directly against live Postgres (happy paths and guard
rejections), then the DB was reset to its clean seeded state afterward.

**`Dockerfile`** — fixed a stale reference to a nonexistent
`db/seed_data.sql` (should've been `seed_sqlite.sql`) that would have broken
the image build entirely.

## 4. Chatbot wiring (customer-facing actions only)

- **`services/agents/order_tracking_agent.py`** — added `DELETE_ORDER_TOOL`
  to the tool registry alongside `GET_ORDER_DETAILS_TOOL`/`CANCEL_ORDER_TOOL`;
  its guard error surfaces verbatim the same way `cancel_order`'s does.
- **`services/agents/product_recommendation_agent.py`** — added
  `ADD_REVIEW_TOOL`/`REMOVE_REVIEW_TOOL`, following the same
  `tool_params`-extraction pattern `search_items` already uses; `user_id`
  always comes from the authenticated task, never from LLM-extracted params.
- Deliberately **not** wired into chat: `add_item`/`remove_item` (catalog
  admin ops — no admin/customer role separation exists in this app) and
  `create_order` (placing a new order needs its own checkout intent/product-
  resolution flow, not just a tool-registry entry — see the "why" discussion
  in the conversation this log came from).

Verified via a live smoke test using a fake LLM stand-in for just the
reasoning/interpretation step (Ollama wasn't installed yet at that point) —
everything downstream (tool dispatch, `EcommerceClient`, Postgres,
`StructuredOrderSummary` construction) was real.

## 5. Local environment setup

- Installed **Docker Desktop** (already done by the user) and stood up the
  `postgres` service above.
- Installed **Ollama** (v0.32.1, via `winget install Ollama.Ollama`) and
  pulled `llama3.1:8b` (4.9GB) and `nomic-embed-text` (274MB) — the two
  models `services/llm_inference.py` is configured to use.
- Installed the project's Python dependencies (`sqlalchemy`, `psycopg2-binary`,
  `python-dotenv`, `fastapi`, etc. via `pip install -r requirements.txt`)
  into the active environment, which didn't have them yet.
- Started the FastAPI app (`uvicorn api.main:app`) and confirmed
  `GET /api/v1/health` reports `database_reachable: true` and
  `llm_reachable: true` against the real Postgres + Ollama instances.

## 6. Bug found, not fixed (documented instead)

While testing `deleteOrder`/`addReview` through real chat (not simulated),
found that `services/llm_inference.py`'s `call_router()` misroutes
cancel/delete/review intents to `GeneralPurposeAgent` instead of
`OrderTrackingAgent`/`ProductRecommendationAgent` — its system prompt lists
a phantom `'ReturnsAgent'` that doesn't exist anywhere in the codebase, and
gives no explicit guidance on which agent owns those intents.

Worse: `GeneralPurposeAgent`'s generative fallback then **fabricates a
plausible success message** for the action it never performed (confirmed —
a "review" it claimed to add was never written to the DB).

This bug **predates this session** (reproduced with the pre-existing
`cancelOrder` tool, not just the new tools added here) — it was left
unfixed per an explicit decision to document rather than fix in-session.
Full detail saved to this machine's Claude memory at:

```
C:\Users\rosha\.claude\projects\c--Users-rosha-Documents-Course-GenaiTalentSprint-ShopAssitV2-shopassist\memory\router_misroute_hallucination.md
```

## 7. Git

- Committed as `1d6d7e4` — *"Add item_reviews table, Postgres schema, and
  order/review CRUD to EcommerceClient"* — and pushed to
  `origin/feature/ordertracking`.
