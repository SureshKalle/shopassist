-- ShopAssist local dev schema (SQLite)
-- Adapted from schema.sql (PostgreSQL/Supabase, the source of truth) so the
-- order tracking client has a real DB to query locally without Supabase credentials.
--
-- Adaptations vs schema.sql:
--   - ENUM types      -> TEXT + CHECK constraint
--   - GENERATED IDENTITY -> INTEGER PRIMARY KEY AUTOINCREMENT
--   - UUID / gen_random_uuid() -> TEXT (uuid4 hex generated in Python on insert)
--   - INET            -> TEXT
--   - TIMESTAMPTZ     -> TEXT (ISO-8601), DEFAULT CURRENT_TIMESTAMP
--   - line_total generated column -> plain column, computed on insert
--     (SQLite generated-column support varies by build; kept explicit for portability)

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name      VARCHAR(100)  NOT NULL,
    last_name       VARCHAR(100)  NOT NULL,
    email           VARCHAR(255)  NOT NULL UNIQUE,
    phone           VARCHAR(20),
    address_line1   VARCHAR(255),
    address_line2   VARCHAR(255),
    city            VARCHAR(100),
    state           VARCHAR(100),
    postal_code     VARCHAR(20),
    country         VARCHAR(100)  DEFAULT 'India',
    is_active       BOOLEAN       NOT NULL DEFAULT 1,
    created_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE items (
    item_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sku             VARCHAR(50)   NOT NULL UNIQUE,
    name            VARCHAR(255)  NOT NULL,
    description     TEXT,
    category        VARCHAR(100),
    price           NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    mrp             NUMERIC(10,2) CHECK (mrp >= price),
    stock_quantity  INTEGER       NOT NULL DEFAULT 0,
    is_active       BOOLEAN       NOT NULL DEFAULT 1,
    created_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    session_id      TEXT PRIMARY KEY,
    customer_id     INTEGER       REFERENCES customers(customer_id) ON DELETE SET NULL,
    ip_address      TEXT,
    user_agent      TEXT,
    device_type     VARCHAR(50),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'terminated')),
    started_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_seen_at    TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at        TEXT,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE orders (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER       NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    session_id      TEXT          REFERENCES sessions(session_id) ON DELETE SET NULL,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned')),
    subtotal        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    discount        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (discount >= 0),
    shipping_fee    NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (shipping_fee >= 0),
    total_amount    NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
    shipping_address TEXT,
    placed_at       TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT          NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    order_id        INTEGER       NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    item_id         INTEGER       NOT NULL REFERENCES items(item_id)  ON DELETE RESTRICT,
    quantity        INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0),
    line_total      NUMERIC(12,2) NOT NULL CHECK (line_total >= 0),
    PRIMARY KEY (order_id, item_id)
);
