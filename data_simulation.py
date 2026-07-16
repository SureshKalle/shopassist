# data_simulation.py
"""Connects to both ShopAssist databases - SQLite (local dev) and
PostgreSQL (the capstone demo instance) - one after the other, and prints
the first few rows of every table in each.

This is a reference for how to point the app at either database: both are
reached through the exact same SQLAlchemy engine/query code (see
services/ecommerce_client.py), because shopassist-database keeps their
schemas identical. Swapping SQLite for Postgres in your own code is just a
matter of the connection string - nothing else changes.

Usage:
    python3 data_simulation.py
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from services.ecommerce_client import DEFAULT_DB_URL

ROW_PREVIEW_LIMIT = 5

# Same SQLite build shopassist-database/sqlite/scripts/create_db.py produces
# - see services/ecommerce_client.py for why this lives in a sibling repo.
SQLITE_URL = DEFAULT_DB_URL

# DATABASE_URL, when set, is how the app itself switches to Postgres (see
# README.md > Configuration). Unset, this falls back to shopassist-database's
# docker-compose defaults so this script still works out of the box against
# `docker compose up -d` there.
POSTGRES_URL = os.environ.get(
    "DATABASE_URL", "postgresql://shopassist:shopassist123@localhost:5432/shopassist"
)


def _missing_sqlite_file(database_url: str) -> Path | None:
    """SQLAlchemy/sqlite3 silently creates an empty database file the
    moment you connect to a path that doesn't exist yet, which would make
    a missing dev DB look like an empty one instead of a clear error.
    Catch that case explicitly before connecting."""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    path = Path(database_url[len(prefix):])
    return path if not path.exists() else None


def preview_database(label: str, database_url: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{database_url}\n{'=' * 72}")

    missing_file = _missing_sqlite_file(database_url)
    if missing_file is not None:
        print(f"No database file at {missing_file}.")
        print("Build it with: python3 ../shopassist-database/sqlite/scripts/create_db.py")
        return

    engine: Engine | None = None
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            metadata = MetaData()
            metadata.reflect(bind=engine)

            if not metadata.tables:
                print("Connected, but no tables found - has the schema been applied yet?")
                return

            for table_name in sorted(metadata.tables):
                table = metadata.tables[table_name]
                print(f"\n--- {table_name} (top {ROW_PREVIEW_LIMIT} rows) ---")

                query = select(table).limit(ROW_PREVIEW_LIMIT)
                if table.primary_key.columns:
                    query = query.order_by(*table.primary_key.columns)

                rows = conn.execute(query).mappings().all()
                if not rows:
                    print("(empty table)")
                    continue

                headers = list(rows[0].keys())
                print(" | ".join(headers))
                for row in rows:
                    print(" | ".join(str(row[h]) for h in headers))
    except SQLAlchemyError as exc:
        print(f"Could not connect to / query this database: {exc}")
        print("Skipping - see shopassist-database/README.md for how to start it.")
    finally:
        if engine is not None:
            engine.dispose()


def main() -> None:
    preview_database("SQLite (local dev)", SQLITE_URL)
    preview_database("PostgreSQL (capstone demo)", POSTGRES_URL)


if __name__ == "__main__":
    main()
