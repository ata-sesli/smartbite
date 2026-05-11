# SmartBite V1

SmartBite is an expiry-date scanning backend with asynchronous AI processing and a lightweight SvelteKit testing console.

## Stack
- API: Litestar
- DB: PostgreSQL + SQLAlchemy + Alembic
- Queue: Redis + arq
- Detector: Ultralytics YOLO (`models/yolo20n/yolo26s/yolo26s-best.pt` by default)
- Expiry OCR lane: `YOLO -> PP-OCRv5 text detection -> ParSeq recognition -> parser/decision`
- General text OCR lane: PP-OCRv5 visible text extraction on separate synchronous endpoint
- Web console: SvelteKit + TypeScript (PNPM)

## Implemented API
- `POST /scans`
- `POST /scans/oneshot`
- `GET /scans/{scan_id}`
- `PATCH /scans/{scan_id}`
- `POST /ocr/general-text`
- `GET /expiry`
- `POST /alerts/process`
- `GET /health`
- `GET /admin/metrics`

### `POST /scans` vs `POST /scans/oneshot`
- `POST /scans` requires `image`, `qr_code`, and `user_id`, then enqueues async processing.
- `POST /scans/oneshot` requires only `image` (optional `metadata`) and waits for a completed result by polling scan status.
- One-shot is still queue-backed. Redis + worker must be available, otherwise one-shot returns `503` or `504`.

## OCR runtime behavior
- ParSeq model directory: `models/parseq-small` (`SMARTBITE_PARSEQ_MODEL_DIR`)
- PP-OCRv5 text detection model directory (optional): `SMARTBITE_OCR_PPOCRV5_TEXT_DET_MODEL_DIR`
- Device settings:
  - `SMARTBITE_OCR_DEVICE_MODE=auto|cpu|mps|cuda` (PP text detector)
  - `SMARTBITE_PARSEQ_DEVICE_MODE=auto|cpu|mps|cuda`
- Current safe default: `SMARTBITE_PARSEQ_DEVICE_MODE=cpu` (MPS auto-fallback remains enabled at runtime).
- Prefetch command: `uv run smartbite-prefetch-models`
- Startup enforces `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip remote host checks
- Startup validates local model assets and fails fast when strict validation is enabled.

## Run modes
### 1) Hybrid mode (recommended)
Docker for Postgres/Redis/API, native worker on host:

```bash
cp .env.example .env
uv sync --extra dev
uv sync --extra ai
docker compose up -d postgres redis api
uv run smartbite-worker
```

Notes:
- Default API port is `8005`.
- Docker API startup auto-runs migrations via `app/scripts/start_api_with_migrations.sh`.
- `SMARTBITE_STORAGE_ROOT` must point to the same bind-mounted folder (`data/storage` by default).

### 2) Full Docker mode
```bash
docker compose --profile docker-worker up --build
```

## Web console
```bash
cp web/.env.example web/.env
cd web
pnpm install
pnpm dev
```

Default proxied backend URL is `http://localhost:8005`.

## Dataset and training CLIs
- `smartbite-build-ppocrv5-rec-dataset`
  - JSON bbox annotations to PP-OCR recognition dataset (+ metadata JSONL, optional component and candidate crops)
- `smartbite-build-ppocrv5-rec-date-dataset`
  - Date-only cropped image datasets (JSON/TXT/CSV/JSONL/filename-label modes) to PP-OCR recognition dataset
- `smartbite-generate-ppocrv5-rec-candidates`
  - Semi-automatic proposal generation from product images/crops for review-first curation
- `smartbite-finetune-ppocrv5-rec`
  - Dataset validation + PP-OCRv5 recognition fine-tuning launch helper

## Verification snapshot
- Current automated test run in this repository: `63 passed, 1 failed`.
- Known failing case: one-shot API test expects success without queue, but runtime now requires Redis/worker availability.

## More commands
Use [COMMANDS.md](COMMANDS.md) for full operational command chains.
