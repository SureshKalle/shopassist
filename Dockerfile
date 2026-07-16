# Builds the shopassist FastAPI backend image.
#
# COPY lines are explicit (no `COPY . .`) so a local .venv/, .git/, .env,
# .env.production don't end up in an image layer even if .dockerignore
# ever falls out of sync.
#
# No db/ here on purpose: schema, seed data, and the SQLite/Postgres build
# tooling live in the sibling shopassist-database repo (see
# services/ecommerce_client.py and that repo's docs/database-design.md).
# That means this image has no bundled local database - set DATABASE_URL
# to a reachable Postgres (shopassist-database's, most commonly) for order
# lookups to work. Without it, the app still starts fine; DB-backed
# endpoints just report `database_reachable: false` rather than crash.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY common ./common
COPY services ./services

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
