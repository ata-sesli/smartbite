# SmartBite V1

SmartBite is an expiry-date scanning backend with async AI processing, plus a minimal SvelteKit web console.

## Current status
- Backend API: Litestar
- Database: PostgreSQL with SQLAlchemy + Alembic
- Queue/worker: Redis + `arq`
- Detection: Ultralytics YOLO adapter (default model: `yolo26n.pt`)
- OCR runtime: PP-OCRv5-first (`ppocrv5_main`)
- Web: SvelteKit + TypeScript (PNPM)

## OCR runtime behavior
- Active default on all devices (`cpu|mps|cuda`): `ppocrv5_main`
- Device selection env: `SMARTBITE_OCR_DEVICE_MODE=auto|cpu|mps|cuda`
- Substitute model metadata lives in `app/ai/ocr_substitute_config.json`
- Substitute model is disabled by default (`SMARTBITE_OCR_ENABLE_SUBSTITUTE_MODEL=false`)
- No silent fallback: missing model/config surfaces explicit failure reasons

## API endpoints
- `POST /scans`
- `GET /scans/{scan_id}`
- `PATCH /scans/{scan_id}`
- `GET /expiry`
- `POST /alerts/process`
- `GET /health`
- `GET /admin/metrics`

## Quick start (hybrid recommended)
Hybrid means Docker for infra/API and native host worker for AI.

1. Create env file:
   - `cp .env.example .env`
2. Install backend package:
   - `pip install -e .[dev]`
3. Start infrastructure + API:
   - `docker compose up -d postgres redis api`
4. Run migrations:
   - `alembic upgrade head`
5. Start worker natively:
   - `smartbite-worker`

Notes:
- Default published API port is `8001` (`SMARTBITE_API_PORT=8001`).
- `SMARTBITE_STORAGE_ROOT` in `.env` must match the same host folder that Docker bind-mounts (`data/storage` by default).

## Full Docker mode
Run API + worker in Docker:

- `docker compose --profile docker-worker up --build`

## Web console (PNPM only)
1. Create frontend env:
   - `cp web/.env.example web/.env`
2. Start web app:
   - `cd web`
   - `pnpm install`
   - `pnpm sync`
   - `pnpm check`
   - `pnpm dev`

Default backend target in `web/.env.example` is `http://localhost:8001`.

## Fine-tuning PP-OCRv5 recognition
CLI script:
- `app/scripts/finetune_ppocrv5_rec.py`
- Entry point: `smartbite-finetune-ppocrv5-rec`

Notebook:
- `notebooks/ppocrv5_finetune_rec.ipynb`

Expected dataset layout:
- `train_images/`
- `val_images/`
- `train_label.txt`
- `val_label.txt`

Each label line format:
- `relative/image/path.jpg<TAB>text label`

## More commands
For explicit operational commands and troubleshooting, see `COMMANDS.md`.
