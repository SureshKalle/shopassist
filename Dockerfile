# Builds the shopassist FastAPI backend image.
#
# COPY lines are explicit (no `COPY . .`) so local cruft (.venv/, .git/,
# .env, db/shopassist.db) never ends up in an image layer even if
# .dockerignore falls out of sync.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY services ./services
COPY clients ./clients
COPY common ./common
COPY db/init_db.py db/schema_sqlite.sql db/seed_sqlite.sql ./db/

EXPOSE 8000

# db/init_db.py + schema_sqlite.sql/seed_sqlite.sql are still copied in
# above (untouched, for local/CLI testing - `python db/init_db.py` on the
# host still builds db/shopassist.db exactly as before) but no longer run
# automatically here: docker-compose.yml always sets DATABASE_URL to the
# bundled Postgres, so rebuilding a SQLite file this container never reads
# was pure wasted startup time, not a real fallback.
#
# uvicorn runs directly as PID 1 (no shell wrapper) so `docker stop`'s
# SIGTERM reaches it for a graceful shutdown.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
