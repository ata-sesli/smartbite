# SmartBite V1 Backend

Backend service for expiry-date scan ingestion and async AI processing.

## Stack
- Litestar API
- PostgreSQL + SQLAlchemy + Alembic
- Redis + arq worker
- YOLO detector + device-aware OCR routing

## Device-aware OCR routing
- `cpu` runtime: ONNX lite OCR model (`SMARTBITE_OCR_ONNX_MODEL_PATH`)
- `mps` or `cuda` runtime: full PaddleOCR
- routing selected by `SMARTBITE_OCR_DEVICE_MODE=auto|cpu|mps|cuda`

## Quick start
1. Copy `.env.example` to `.env` and adjust values.
2. Start infrastructure:
   - `docker compose up -d postgres redis`
3. Install package:
   - `pip install -e .[dev]`
4. Run migrations:
   - `alembic upgrade head`
5. Start API:
   - `smartbite-api`
6. Start worker:
   - `smartbite-worker`

## Endpoints
- `POST /scans`
- `GET /scans/{scan_id}`
- `PATCH /scans/{scan_id}`
- `GET /expiry`
- `POST /alerts/process`
- `GET /health`
- `GET /admin/metrics`

## Running all services with Docker
- `docker compose --profile docker-worker up --build`

## Hybrid runtime (recommended for native AI)
1. Start non-AI services in Docker:
   - `docker compose up -d postgres redis api`
2. In a host shell (macOS/Windows), run the worker natively:
   - macOS/Linux:
     - `export SMARTBITE_DATABASE_URL=postgresql+asyncpg://smartbite:smartbite@localhost:5432/smartbite`
     - `export SMARTBITE_REDIS_URL=redis://localhost:6379/0`
     - `export SMARTBITE_STORAGE_ROOT=<absolute-path-to-repo>/data/storage`
     - `export SMARTBITE_OCR_DEVICE_MODE=auto`
     - `smartbite-worker`
   - Windows PowerShell:
     - `$env:SMARTBITE_DATABASE_URL='postgresql+asyncpg://smartbite:smartbite@localhost:5432/smartbite'`
     - `$env:SMARTBITE_REDIS_URL='redis://localhost:6379/0'`
     - `$env:SMARTBITE_STORAGE_ROOT='C:\\path\\to\\smartbite\\data\\storage'`
     - `$env:SMARTBITE_OCR_DEVICE_MODE='auto'`
     - `smartbite-worker`

Notes:
- `SMARTBITE_STORAGE_ROOT` must point to the same host directory used by the Docker API bind mount.
- To run worker in Docker instead, enable the profile: `docker compose --profile docker-worker up -d worker`.

## Minimal SvelteKit web console
- `cd web`
- `pnpm install`
- `pnpm dev`
- Set backend target: `SMARTBITE_BACKEND_URL=http://localhost:8000`
