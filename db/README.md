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
├── seed_sqlite.sql       # 10 customers, 75 items, 10 sessions, 25 orders
└── shopassist.db        # generated, git-ignored
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
| `get_customer_history(user_id)` | last purchase + favorite category, or `{"error": ...}` |
| `get_customer(user_id)` | customer profile dict, or `None` |
| `get_item(item_id)` | item dict, or `None` |
| `search_items(category=None, keyword=None, limit=10)` | list of matching items |
| `list_orders_for_customer(user_id, limit=10)` | that customer's orders, newest first |

## Schema

Five tables: `customers`, `items`, `sessions`, `orders`, `order_items`.
`user_id` / `item_id` / `order_id` are human-readable business keys
(`alum-1001`, `item-1001`, `ord-1001`), not autoincrementing integers —
supplied explicitly in `seed_sqlite.sql`, same idea `session_id` already
used. SQLite-specific adaptations (`TEXT`/`NUMERIC` in place of Postgres
types, inline `CHECK` constraints instead of separate constraint files) are
documented in `schema_sqlite.sql`'s own header comment.

`user_id` is shopassist's single identifier end to end: the same value
`shopassist-client` sends at login (`CustomerQuery.user_id`,
`common/models.py`), used directly as `customers.user_id` for every DB
lookup. The seeded values use an `alum-1001` style prefix - that's what to
type as the "User ID" at `shopassist-client`'s login for a given seeded
customer's data to resolve.

## Switching to PostgreSQL

Point `DATABASE_URL` at a real Postgres instance once its schema uses
matching table/column names to this one (see `.env.example`):

```bash
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/shopassist
```

Note `services/rag.py`'s RAG store has no database backing either way,
SQLite or Postgres — it's an in-memory store regardless of `DATABASE_URL`.

## Known gaps

- `main_simulation.py`'s sample queries and
  `services/agents/order_tracking_agent.py`'s order-ID extraction regex
  (`\b(\d{4,})\b`) only match purely numeric IDs (`"order 12345"`), not
  business keys like `ord-1001`. A numeric-looking ID won't match any
  seeded row.
- `EcommerceClient.get_order_details()`'s check that the requesting
  `user_id` actually owns the order is commented out (see the `TODO` in
  that method) - any `user_id` can fetch any `order_id` until it's
  re-enabled.

Both are in `services/agents/` or the caller side of `clients/`, not this
folder's schema/seed data.
