# Builds the shopassist FastAPI backend image.
#
# COPY lines are explicit (no `COPY . .`) so local cruft (.venv/, .git/,
# .env, db/shopassist.db) never ends up in an image layer even if
# .dockerignore falls out of sync.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root - defense in depth: if a dependency vulnerability ever allowed
# code execution inside this container, a non-root user limits the blast
# radius versus the default root. Fixed UID/GID (not auto-assigned) so the
# --chown below is deterministic.
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser -m appuser

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser services ./services
COPY --chown=appuser:appuser clients ./clients
COPY --chown=appuser:appuser common ./common
COPY --chown=appuser:appuser db/init_db.py db/schema_sqlite.sql db/seed_sqlite.sql ./db/
# Policy PDFs ingested into RAG at startup - see api/dependencies.py's
# warm_up_services() / services/data_pipeline.py's ingest_pdf_documents().
# --chown (not root-owned): services/rag.py creates docs/rag_data/ at
# startup (the generated FAISS index + JSONL store) - appuser needs write
# access here to do that.
COPY --chown=appuser:appuser docs ./docs

USER appuser

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
