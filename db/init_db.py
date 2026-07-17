"""Builds db/shopassist.db from schema_sqlite.sql + seed_sqlite.sql.

Usage:
    python db/init_db.py

Safe to re-run: deletes any existing db/shopassist.db first, then rebuilds
from scratch. schema_sqlite.sql itself uses CREATE TABLE IF NOT EXISTS (no
DROP TABLE - see its own comments), so re-running the schema and seed
scripts against an already-populated file would otherwise fail with a
UNIQUE constraint violation on the seed data.
"""
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "shopassist.db"
SCHEMA_PATH = DB_DIR / "schema_sqlite.sql"
SEED_PATH = DB_DIR / "seed_sqlite.sql"


def build_db() -> None:
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.executescript(SEED_PATH.read_text())
        conn.commit()
    finally:
        conn.close()
    print(f"Rebuilt {DB_PATH} from {SCHEMA_PATH.name} + {SEED_PATH.name}")


if __name__ == "__main__":
    build_db()
