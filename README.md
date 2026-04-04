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
- Default recognition model path: `models/fine-tuned-models/best_model_inference`
- Default char dict path: `ppocr/utils/dict/ppocrv5_dict.txt`
- Device selection env: `SMARTBITE_OCR_DEVICE_MODE=auto|cpu|mps|cuda`
- Substitute model metadata lives in `app/ai/ocr_substitute_config.json`
- Substitute model is disabled by default (`SMARTBITE_OCR_ENABLE_SUBSTITUTE_MODEL=false`)
- No silent fallback: missing model/config surfaces explicit failure reasons
- Easiest switch point: set `SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR` in `.env` and restart the worker

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
2. Sync backend dependencies:
   - `uv sync --extra dev`
   - `uv sync --extra ai` (required for native worker: Ultralytics + PaddleOCR runtime)
3. Start infrastructure + API:
   - `docker compose up -d postgres redis api`
4. Start worker natively:
   - `uv run smartbite-worker`

Notes:
- Default published API port is `8005` (`SMARTBITE_API_PORT=8005`).
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

Default backend target in `web/.env.example` is `http://localhost:8005`.

## Fine-tuning PP-OCRv5 recognition
CLI script:
- `app/scripts/finetune_ppocrv5_rec.py`
- Entry point: `smartbite-finetune-ppocrv5-rec`

Dataset exporter for PP-OCRv5 recognition format:
- `app/scripts/build_ppocrv5_rec_dataset.py`
- Entry point: `smartbite-build-ppocrv5-rec-dataset`
- Exports `train_images/`, `val_images/`, `train_label.txt`, `val_label.txt`, and JSONL metadata sidecars.
- Supports optional DMY component crops and bbox-jitter candidate crops for review/curation.
- Use this exporter before running `smartbite-finetune-ppocrv5-rec`.

Date-only image converter for PP-OCRv5 recognition format:
- `app/scripts/build_ppocrv5_rec_date_dataset.py`
- Entry point: `smartbite-build-ppocrv5-rec-date-dataset`
- Converts pre-cropped date datasets with multiple label-source formats (JSON/TXT/CSV/JSONL/filename parser) into `train_images/`, `val_images/`, `train_label.txt`, and `val_label.txt`.
- Use this converter when source data is already cropped to date regions and ready for recognition fine-tuning.

Semi-auto candidate generator for review-first recognition data:
- `app/scripts/generate_ppocrv5_rec_candidates.py`
- Entry point: `smartbite-generate-ppocrv5-rec-candidates`
- Uses PP-OCRv5 line extraction to propose candidate text crops from full product images or product crops.
- Outputs review artifacts under `candidates/` (`images/`, `manifest.jsonl`, optional `review.csv`, optional overlays) with deterministic heuristic scoring and suggested classes.

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
