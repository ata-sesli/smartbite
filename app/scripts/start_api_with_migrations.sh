#!/bin/sh
set -eu

MAX_RETRIES="${SMARTBITE_MIGRATION_MAX_RETRIES:-30}"
RETRY_SLEEP_SECONDS="${SMARTBITE_MIGRATION_RETRY_SECONDS:-2}"
ATTEMPT=1

echo "[startup] Ensuring SmartBite model assets are present..."
smartbite-prefetch-models || echo "[startup] Warning: model prefetch could not reach Hugging Face; continuing with existing assets..."

echo "[startup] Running Alembic migrations..."
while ! alembic upgrade head; do
  if [ "$ATTEMPT" -ge "$MAX_RETRIES" ]; then
    echo "[startup] Migration failed after ${ATTEMPT} attempts; aborting startup."
    exit 1
  fi
  echo "[startup] Migration attempt ${ATTEMPT} failed; retrying in ${RETRY_SLEEP_SECONDS}s..."
  ATTEMPT=$((ATTEMPT + 1))
  sleep "$RETRY_SLEEP_SECONDS"
done

echo "[startup] Migrations are up to date. Starting API..."
cd /app
exec python -c "from app.main import run; run()"
