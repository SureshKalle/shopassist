-- ShopAssist Database Schema (for PostgreSQL)
-- Source of truth, as finalized by Vikas (task #9). Run this against Supabase/Postgres.
-- Do not hand-edit for local dev — see schema_sqlite.sql for the local SQLite adaptation.

-- Tables:
-- 1. customers,
-- 2. items,
-- 3. sessions,
-- 4. orders
-- 5. order_items

-- Drop in dependency order - safe for re-runs during development
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS items CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

DROP TYPE IF EXISTS order_status_enum;
DROP TYPE IF EXISTS session_status_enum;

CREATE TYPE order_status_enum AS ENUM (
    'pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned'
);

CREATE TYPE session_status_enum AS ENUM (
    'active', 'expired', 'terminated'
);

CREATE TABLE customers (
    customer_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
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
    is_active       BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE items (
    item_id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sku             VARCHAR(50)   NOT NULL UNIQUE,
    name            VARCHAR(255)  NOT NULL,
    description     TEXT,
    category        VARCHAR(100),
    price           NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    mrp             NUMERIC(10,2) CHECK (mrp >= price),
    stock_quantity  INTEGER       NOT NULL DEFAULT 0,
    is_active       BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE sessions (
    session_id      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     BIGINT        REFERENCES customers(customer_id) ON DELETE SET NULL,
    ip_address      INET,
    user_agent      TEXT,
    device_type     VARCHAR(50),
    status          session_status_enum NOT NULL DEFAULT 'active',
    started_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE orders (
    order_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id     BIGINT        NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
    session_id      UUID          REFERENCES sessions(session_id) ON DELETE SET NULL,
    status          order_status_enum NOT NULL DEFAULT 'pending',
    subtotal        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    discount        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (discount >= 0),
    shipping_fee    NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (shipping_fee >= 0),
    total_amount    NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
    shipping_address TEXT,
    placed_at       TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE order_items (
    order_id        BIGINT        NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    item_id         BIGINT        NOT NULL REFERENCES items(item_id)  ON DELETE RESTRICT,
    quantity        INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0),
    line_total      NUMERIC(12,2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    PRIMARY KEY (order_id, item_id)
);
