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

# Rebuilds the SQLite dev DB fresh on every start (safe to re-run, no-op
# once DATABASE_URL points at real Postgres), then `exec`s uvicorn so it
# becomes PID 1 - that way `docker stop`'s SIGTERM reaches uvicorn
# directly for a graceful shutdown instead of being swallowed by the shell.
CMD ["sh", "-c", "python db/init_db.py && exec uvicorn api.main:app --host 0.0.0.0 --port 8000"]
