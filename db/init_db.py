"""Builds db/shopassist.db from schema_sqlite.sql + seed_data.sql.

Usage:
    python db/init_db.py

Safe to re-run: drops and recreates all tables each time (per schema_sqlite.sql).
"""
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).parent
DB_PATH = DB_DIR / "shopassist.db"
SCHEMA_PATH = DB_DIR / "schema_sqlite.sql"
SEED_PATH = DB_DIR / "seed_data.sql"


def build_db() -> None:
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
