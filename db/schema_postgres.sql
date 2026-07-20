-- ShopAssist :: PostgreSQL schema
-- Target: PostgreSQL 17 (docker-compose.yml's `postgres` service runs the
-- pgvector/pgvector:pg17 image - Postgres 17 with the pgvector extension
-- pre-built, needed for document_chunks' embedding column below; the bare
-- postgres image doesn't bundle it). Every table but document_chunks would
-- still run fine on plain PostgreSQL 13+.
-- Purpose: production-like database, swapped in for the SQLite dev DB via
-- DATABASE_URL (see .env.example, db/README.md).
--
-- This is a straight port of db/schema_sqlite.sql - same tables, columns,
-- CHECK constraints, and FK/PK wiring (every table sits in one connected
-- graph: item_reviews -> items and -> customers; sessions/orders ->
-- customers; order_items -> orders and -> items). Table/column names are
-- unchanged so clients/ecommerce_api_client.py's SQL runs against either
-- database without modification - see db/README.md.
--
-- Adaptations vs. schema_sqlite.sql (see that file's own header for the
-- SQLite-side reasoning being reversed here):
--   - No PRAGMA foreign_keys line - Postgres enforces FKs by default, always.
--   - TEXT timestamp columns -> TIMESTAMPTZ, DEFAULT CURRENT_TIMESTAMP -> now().
--   - BOOLEAN DEFAULT 1/0 -> DEFAULT TRUE/FALSE (Postgres booleans aren't
--     implicitly cast from integer literals).
--   - CREATE TABLE/INDEX IF NOT EXISTS carries over unchanged (Postgres has
--     supported both since 9.1/9.5).
--
-- items / item_reviews column choices mirror the Amazon-style product CSV
-- shopassist's RAG ingestion service is built against - see
-- schema_sqlite.sql's header comment for the full CSV -> column mapping and
-- rationale (kept there as the canonical copy, not duplicated here).
--
-- Loaded automatically on first container start via
-- docker-entrypoint-initdb.d (see docker-compose.yml's postgres service) -
-- alongside seed_postgres.sql, which runs after this file.

-- Needed for document_chunks' embedding column near the end of this file -
-- see that table's own header comment.
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- customers
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    user_id         TEXT PRIMARY KEY,
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
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- items
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS items (
    item_id             TEXT PRIMARY KEY,
    name                VARCHAR(255)  NOT NULL,
    description         TEXT,
    -- Full pipe-delimited category path as the source CSV ships it, e.g.
    -- 'Computers&Accessories|Accessories&Peripherals|Cables&Accessories|
    -- Cables|USBCables' - TEXT rather than VARCHAR(100) because that path
    -- routinely exceeds 100 chars. Existing seed data's plain single-word
    -- categories ('Apparel', 'Drinkware', ...) fit the same column fine.
    category            TEXT,
    price               NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    mrp                 NUMERIC(10,2) CHECK (mrp >= price),
    -- CSV's "64%" stored as the numeric percentage (64.00), not a fraction.
    discount_percentage NUMERIC(5,2)  CHECK (discount_percentage IS NULL OR (discount_percentage >= 0 AND discount_percentage <= 100)),
    -- Product-level aggregates from the CSV (see file header) - nullable
    -- since shopassist's own seed items predate this dataset and don't have
    -- them.
    rating              NUMERIC(2,1)  CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5)),
    rating_count        INTEGER       CHECK (rating_count IS NULL OR rating_count >= 0),
    img_link            TEXT,
    product_link        TEXT,
    stock_quantity      INTEGER       NOT NULL DEFAULT 0,
    is_active           BOOLEAN       NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- item_reviews
-- ---------------------------------------------------------------------------
-- One row per review - the source CSV's pipe-separated user_id/user_name/
-- review_id/review_title/review_content columns (parallel arrays, one
-- product row fanning out to several reviews) normalized into their own
-- rows here. user_id ties the review to a real customers row (see
-- schema_sqlite.sql's header) - author_name isn't stored, join customers to
-- get it.
CREATE TABLE IF NOT EXISTS item_reviews (
    review_id       TEXT PRIMARY KEY,
    item_id         TEXT NOT NULL REFERENCES items(item_id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES customers(user_id) ON DELETE CASCADE,
    review_title    VARCHAR(255),
    review_content  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- sessions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT          REFERENCES customers(user_id) ON DELETE SET NULL,
    ip_address      TEXT,
    user_agent      TEXT,
    device_type     VARCHAR(50),
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'expired', 'terminated')),
    started_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ   NOT NULL DEFAULT now(),
    ended_at        TIMESTAMPTZ,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

-- ---------------------------------------------------------------------------
-- orders
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    user_id         TEXT          NOT NULL REFERENCES customers(user_id) ON DELETE RESTRICT,
    session_id      TEXT          REFERENCES sessions(session_id) ON DELETE SET NULL,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned')),
    subtotal        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (subtotal >= 0),
    discount        NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (discount >= 0),
    shipping_fee    NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (shipping_fee >= 0),
    total_amount    NUMERIC(12,2) NOT NULL DEFAULT 0 CHECK (total_amount >= 0),
    shipping_address TEXT,
    placed_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- order_items
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    order_id        TEXT          NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    item_id         TEXT          NOT NULL REFERENCES items(item_id)  ON DELETE RESTRICT,
    quantity        INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price      NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0),
    line_total      NUMERIC(12,2) NOT NULL CHECK (line_total >= 0),
    PRIMARY KEY (order_id, item_id)
);

-- ---------------------------------------------------------------------------
-- document_chunks
-- ---------------------------------------------------------------------------
-- Backing store for a real vector-search-based RAG, matching
-- shopassist-database's own document_chunks table 1:1 (same columns/types)
-- so both projects agree on shape. Infra-ready only, added alongside the
-- Postgres 16 -> 17/pgvector upgrade above: `services/rag.py`'s
-- MockRAGService still does in-memory substring matching today and doesn't
-- read or write this table yet - see README.md's Known Gaps. Column shape
-- mirrors common/models.py's ChunkedDocument 1:1 (doc_id/content/embedding/
-- source_type/metadata) so a future real RAG service needs no ID/field
-- translation at this boundary.
--
-- embedding is a fixed VECTOR(768) - nomic-embed-text's output dimension
-- (the embedding model already bundled via the `ollama` service) - pgvector
-- rejects any INSERT whose vector isn't exactly this width.
CREATE TABLE IF NOT EXISTS document_chunks (
    doc_id          VARCHAR(255)  PRIMARY KEY,
    content         TEXT          NOT NULL,
    embedding       VECTOR(768)   NOT NULL,
    source_type     VARCHAR(50)   NOT NULL,
    metadata        JSONB         NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT now()
);

COMMENT ON TABLE document_chunks IS 'Chunked text + embeddings for a future vector-search RAG (product catalog, support policy, conversation history) - not yet read/written by services/rag.py.';
COMMENT ON COLUMN document_chunks.embedding IS 'nomic-embed-text output dimension (768) via the bundled ollama service - see docker-compose.yml.';

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_items_category         ON items(category);
CREATE INDEX IF NOT EXISTS idx_item_reviews_item_id    ON item_reviews(item_id);
CREATE INDEX IF NOT EXISTS idx_item_reviews_user_id    ON item_reviews(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id        ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_id          ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status           ON orders(status);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id    ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_item_id     ON order_items(item_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_source_type ON document_chunks(source_type);

-- HNSW over IVFFlat: no list-count "training" step needed, good recall/
-- latency out of the box at this project's scale. Cosine ops since
-- nomic-embed-text is designed to be compared by cosine similarity - same
-- choice shopassist-database's own indexes.sql makes for this table.
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
