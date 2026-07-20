# ShopAssist SQLite (local development)

Zero-infrastructure database for running and testing `shopassist` locally —
no Docker, no server, no credentials. `clients/ecommerce_api_client.py` uses
this by default whenever `DATABASE_URL` is unset.

For a real demo or anything production-like, this is replaced by a real,
externally hosted PostgreSQL instance — see [Switching to PostgreSQL](#switching-to-postgresql)
below.

## Layout

```text
db/
├── init_db.py          # (re)builds shopassist.db from the two files below
├── schema_sqlite.sql    # tables, indexes, CHECK constraints
├── seed_sqlite.sql       # 10 customers, 75 items, 10 sessions, 25 orders, 10 item_reviews
├── shopassist.db        # generated, git-ignored
├── schema_postgres.sql  # same schema, ported for Postgres - see "Switching to PostgreSQL"
└── seed_postgres.sql     # same seed data, ported for Postgres
```

## Usage

```bash
python db/init_db.py
```

Deletes any existing `db/shopassist.db` and rebuilds it from scratch — safe
to run as many times as you like, including on a fresh clone where the file
doesn't exist yet.

## Connecting from Python

```python
import sqlite3

conn = sqlite3.connect("db/shopassist.db")
conn.execute("PRAGMA foreign_keys = ON")

cur = conn.execute("SELECT item_id, name, price FROM items LIMIT 5")
for row in cur.fetchall():
    print(row)
```

`clients/ecommerce_api_client.py` connects the same way via SQLAlchemy —
`sqlite:///db/shopassist.db` is the default when `DATABASE_URL` isn't set.

Every other module reads/writes this DB exclusively through `EcommerceClient`
— never import `sqlite3`/SQLAlchemy directly in `services/` or `api/` code.
Its full method surface:

| Method | Returns |
|---|---|
| `is_reachable()` | `bool` - used by `GET /api/v1/health` |
| `get_order_details(user_id, order_id)` | order + line items, or `{"error": ...}` |
| `create_order(user_id, line_items, shipping_address=None)` | new order, or `{"error": ...}` if a customer/item/stock check fails |
| `cancel_order(user_id, order_id)` | order with `status="Cancelled"`, or `{"error": ...}` if not cancellable |
| `delete_order(user_id, order_id)` | `{"order_id", "deleted": True}`, or `{"error": ...}` if not `pending`/`cancelled` |
| `get_customer_history(user_id)` | last purchase + favorite category, or `{"error": ...}` |
| `get_customer(user_id)` | customer profile dict, or `None` |
| `get_item(item_id)` | item dict, or `None` |
| `add_item(item_id, name, price, ...)` | `{"item_id", "added": True}`, or `{"error": ...}` if `item_id` exists |
| `remove_item(item_id)` | `{"item_id", "removed": True}`, or `{"error": ...}` if unfulfilled orders reference it |
| `search_items(category=None, keyword=None, limit=10)` | list of matching items |
| `list_orders_for_customer(user_id, limit=10)` | that customer's orders, newest first |
| `add_review(item_id, user_id, review_title, review_content)` | new review, or `{"error": ...}` if not a verified purchase |
| `remove_review(review_id, user_id)` | `{"review_id", "deleted": True}`, or `{"error": ...}` if not the review's own author |

## Schema

Six tables: `customers`, `items`, `item_reviews`, `sessions`, `orders`,
`order_items`. `user_id` / `item_id` / `order_id` are human-readable
business keys (`alum-1001`, `item-1001`, `ord-1001`), not autoincrementing
integers — supplied explicitly in `seed_sqlite.sql`, same idea `session_id`
already used. SQLite-specific adaptations (`TEXT`/`NUMERIC` in place of
Postgres types, inline `CHECK` constraints instead of separate constraint
files) are documented in `schema_sqlite.sql`'s own header comment.

`items` and `item_reviews` are shaped to match the Amazon-style product CSV
(`product_id`, `product_name`, `category`, `discounted_price`,
`actual_price`, `discount_percentage`, `rating`, `rating_count`,
`about_product`, `img_link`, `product_link`, plus per-review `user_id` /
`user_name` / `review_id` / `review_title` / `review_content`) that
shopassist's RAG ingestion service is built against — see
`schema_sqlite.sql`'s header comment for the exact column mapping.
`item_reviews.user_id` is a real FK to `customers(user_id)` (unlike the
source CSV's anonymous reviewer IDs), so every table sits in one FK/PK-
connected graph: `item_reviews` → `items` and → `customers`; `sessions` /
`orders` → `customers`; `order_items` → `orders` and → `items`.
`author_name` isn't stored on `item_reviews` — it's derived via that
customers join (`first_name`/`last_name`) instead of risking drift from the
customer's actual profile name. That RAG service re-joins `items` with
`item_reviews` (and `customers` for reviewer name) to reconstruct the same
CSV shape before handing it to the vector DB.

`user_id` is shopassist's single identifier end to end: the same value
`shopassist-client` sends at login (`CustomerQuery.user_id`,
`common/models.py`), used directly as `customers.user_id` for every DB
lookup. The seeded values use an `alum-1001` style prefix - that's what to
type as the "User ID" at `shopassist-client`'s login for a given seeded
customer's data to resolve.

## Switching to PostgreSQL

`docker-compose.yml`'s `postgres` service spins up a local Postgres 17
instance (`pgvector/pgvector:pg17` - Postgres 17 with the pgvector
extension pre-built, needed for `document_chunks`' `VECTOR(768)` column;
see that table's own header comment in `schema_postgres.sql` - infra-ready
for a real vector-search RAG, not read/written by `services/rag.py` yet)
and loads `schema_postgres.sql` then `seed_postgres.sql` automatically via
Postgres's `docker-entrypoint-initdb.d` mechanism — those two files are a
straight port of the SQLite schema/seed above (same tables/columns/FKs,
see `schema_postgres.sql`'s own header for the exact SQLite → Postgres
adaptations).

```bash
docker compose up -d postgres
```

Then point `DATABASE_URL` at it - from your host machine (running `api`
outside Docker):

```bash
DATABASE_URL=postgresql://shopassist:shopassist123@localhost:5432/shopassist
```

or, if `api` also runs via this same `docker-compose.yml`, uncomment the
`DATABASE_URL` line already in that service's `environment:` block instead
(it uses `postgres` - the service name - as the host, not `localhost`).

**Note:** `docker-entrypoint-initdb.d` only runs on a brand-new (empty) data
volume — editing `schema_postgres.sql`/`seed_postgres.sql` after the
container's first start won't reapply them, and neither does a Postgres
major-version image bump (e.g. the 16 → 17 upgrade above) - old data
directories from a different major version won't even start under a newer
Postgres binary. To force a reload (also required once, right after that
upgrade, for anyone with a pre-existing volume):

```bash
docker compose down postgres
docker volume rm shopassist_shopassist-postgres-data  # project-prefixed - see `docker compose config --volumes` if this repo is ever renamed/moved
docker compose up -d postgres
```

This repo's Postgres is independent of `shopassist-devops`'s own Postgres
container — pick one or the other, don't point `DATABASE_URL` at both at
once. Whichever Postgres you use, its schema must use matching table/column
names to this one (see `.env.example`) — that's exactly what
`schema_postgres.sql` gives you here.

Note `services/rag.py`'s RAG store has no database backing either way,
SQLite or Postgres — it's an in-memory store regardless of `DATABASE_URL`.

## Known gaps

- ~~`services/agents/order_tracking_agent.py`'s order-ID extraction regex
  only matched purely numeric IDs~~ - fixed: `ORDER_ID_PATTERN` now matches
  the full `ord-<n>` business-key format this schema actually uses, and a
  new `_normalize_order_id()` step recovers that full form even when the
  LLM router's own extraction strips the `ord-` prefix (its prompt's
  example happens to be a bare number, which biased it that way - see that
  method's docstring). This was the root cause of the intermittent
  "sometimes can't find my order" symptom.
- ~~`EcommerceClient.get_order_details()`'s ownership check was commented
  out~~ - re-enabled, and extended to `cancel_order()`/`delete_order()`
  (which never had it at all) as part of this project's guardrails - see
  the main README's "Guardrails" section.

Both fixes are in `services/agents/`/`clients/`, not this folder's
schema/seed data.
