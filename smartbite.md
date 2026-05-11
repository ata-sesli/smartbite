# SMARTBITE V1 - As-Built Technical Specification

## 1) Project Scope
SmartBite V1 is a backend-focused expiry-date analysis system. It accepts product images, runs server-side AI inference, persists intermediate/final results, and exposes operational endpoints for retrieval, corrections, and alerts.

V1 includes:
- Queue-backed scan processing
- One-shot analysis endpoint for immediate testing
- Rule-based expiry decisioning
- DB persistence for full scan lifecycle
- Minimal web console for internal manual testing
- Dataset and fine-tuning tooling for PP-OCRv5 recognition workflows

V1 excludes:
- End-user mobile app
- Production auth/tenant model
- Live video streaming pipeline
- Non-date freshness estimation

## 2) Runtime Architecture

```text
Client/Web Console
  -> Litestar API
  -> Scan row + image persisted
  -> Redis enqueue
  -> arq worker
  -> AI pipeline:
       decode
       detect (YOLO)
       ROI preprocess variants
       OCR (PP-OCRv5)
       parse candidates
       expiry decision
  -> DB upserts (OCR / parsed / expiry / logs)
  -> scan finalized
  -> optional alert event creation
```

## 3) Technology Stack
- API: Litestar
- Persistence: PostgreSQL (default), SQLAlchemy 2.x async, Alembic
- Queue: Redis + arq
- Detector: Ultralytics YOLO
- OCR: PaddleOCR / PP-OCRv5 recognition-first flow
- Frontend: SvelteKit + TypeScript (internal console)
- Packaging/runtime: `uv` + `pyproject.toml` entry points

## 4) Core API Contract

### `POST /scans`
Purpose:
- Standard async scan creation.

Input:
- multipart `image`
- `qr_code` (required)
- `user_id` (required)
- optional `metadata` JSON string

Output:
- `scan_id`, `status=queued`

### `POST /scans/oneshot`
Purpose:
- Immediate analysis endpoint for internal testing.

Input:
- multipart `image`
- optional `metadata`

Behavior:
- Creates a real scan row using generated `qr_code` and user `oneshot`
- Requires queue availability (`redis` in app state)
- Polls scan status until `done` or timeout

Failure modes:
- `503` if queue unavailable
- `504` on timeout

### `GET /scans/{scan_id}`
Purpose:
- Retrieve current status and final payload.

### `PATCH /scans/{scan_id}`
Purpose:
- Manual correction override (`parsed_date`, `reason`).

### `GET /expiry`
Purpose:
- Filterable list of processed expiry outcomes.

### `POST /alerts/process`
Purpose:
- Create idempotent alert events and attempt delivery.

### `GET /health`
Purpose:
- Basic service heartbeat.

### `GET /admin/metrics`
Purpose:
- Aggregate operational counters and latency.

## 5) Domain Model (Database)

Primary tables:
- `products`
- `scans`
- `ocr_results` (unique by `scan_id`)
- `parsed_date_results` (unique by `scan_id`)
- `expiry_statuses` (unique by `scan_id`)
- `alert_events` (unique by `scan_id + alert_type`)
- `processing_logs`

Enumerations:
- `ScanStatus`: `queued|processing|done|failed`
- `FinalResultStatus`: includes `parsed_success`, `ocr_failed`, `parser_failed`, `manual_review_required`, etc.
- `ExpiryClassification`: `safe|expiring_soon|expired|manual_review_required`
- `AlertType`: `expiring_soon|expired|manual_review`
- `DeliveryStatus`: `pending|sent|failed`

## 6) Worker and Processing Lifecycle
The worker entry point is `smartbite-worker` (`app/worker_main.py`), which:
- forces `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`
- validates AI dependencies (`ultralytics`, `paddle`, `paddleocr`) on startup
- builds `ExpiryPipeline` from environment settings

Per scan job:
1. Move scan to `processing`
2. Load raw image bytes
3. Run pipeline
4. Save ROI image if available
5. Upsert OCR/parse/expiry rows
6. Finalize scan row
7. Write stage timing logs
8. Create alert event if needed

## 7) AI Pipeline (As Implemented)

### Detection (`app/ai/detector.py`)
- YOLO-based region proposal
- confidence thresholding
- explicit error reason propagation for load/inference failures

### ROI preprocessing (`app/ai/preprocess.py`)
- grayscale conversion
- CLAHE
- adaptive Gaussian thresholding
- morphological close + dilation
- denoising
- conditional upscaling for small crops
- multi-variant output for OCR robustness

### OCR router (`app/ai/ocr.py`)
- PP-OCRv5 main runner selected by default
- explicit substitute-model path support (disabled by default)
- runtime device detection (`auto|cpu|mps|cuda`)
- effective runtime reported as `cuda` or `cpu` in output
- recognition path supports rotation search when angle-cls enabled
- local model directory checks and explicit startup errors

### Parsing (`app/ai/parser.py`)
- deterministic regex candidate extraction for common date formats
- confidence scoring and candidate ranking
- plausibility filters (year and time-window bounds)
- de-prioritization using low-priority tokens
- fuzzy month correction for common OCR confusion (`DD/MM/YY_FUZZY_MONTH`)

### Decision engine (`app/ai/decision.py`)
- maps parsed date into:
  - `expired`
  - `expiring_soon`
  - `safe`
  - `manual_review_required`
- uses configurable alert threshold
- generates review flags for low confidence cases

### Pipeline orchestration (`app/ai/pipeline.py`)
- staged timings: decode, detect, preprocess, ocr, parse, decision
- OCR candidate selection across raw + preprocessed variants
- parse-aware variant selection
- fallback full-image OCR/parse path when ROI route fails
- normalized output object for persistence/API

## 8) Operational Configuration
Primary config is environment-driven through `app/infra/settings.py`.

Key controls:
- `SMARTBITE_API_PORT` (default `8005`)
- `SMARTBITE_DATABASE_URL`
- `SMARTBITE_REDIS_URL`
- `SMARTBITE_STORAGE_ROOT`
- `SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR`
- `SMARTBITE_OCR_PPOCRV5_MAIN_CHAR_DICT_PATH`
- `SMARTBITE_OCR_DEVICE_MODE`
- `SMARTBITE_OCR_PPOCRV5_DET_DB_THRESH`
- `SMARTBITE_OCR_PPOCRV5_DET_DB_BOX_THRESH`
- `SMARTBITE_OCR_ENABLE_SUBSTITUTE_MODEL`

Docker API startup runs migrations automatically via:
- `app/scripts/start_api_with_migrations.sh`

## 9) Web Console
SvelteKit web console (`web/src/routes/+page.svelte`) is designed for internal testing:
- Analyze section:
  - upload image
  - camera capture via `getUserMedia`
- One-shot request path:
  - UI -> SvelteKit proxy -> `POST /scans/oneshot`
- Result panel:
  - status, scan id, parsed value summary, raw JSON collapsible
- Secondary tools:
  - health/metrics
  - scan lookup
  - manual correction
  - expiry/alert processing

## 10) Dataset and Training Tooling

### Trusted recognition dataset exporter
- CLI: `smartbite-build-ppocrv5-rec-dataset`
- Source: JSON annotations with bboxes/transcriptions
- Output:
  - `train_images/`, `val_images/`
  - `train_label.txt`, `val_label.txt`
  - metadata JSONL sidecars
  - optional component crops
  - optional bbox-jitter review candidates

### Date-only recognition dataset converter
- CLI: `smartbite-build-ppocrv5-rec-date-dataset`
- Label modes:
  - JSON, TXT, CSV/TSV, JSONL, filename regex
- Deterministic split/export with validation and summary output

### Semi-auto candidate proposal generator
- CLI: `smartbite-generate-ppocrv5-rec-candidates`
- Input modes:
  - `full_images` (optional YOLO detections JSONL)
  - `product_crops`
- Output:
  - `candidates/images`
  - `manifest.jsonl`
  - optional `review.csv`
  - optional overlays
  - deterministic heuristic ranking metadata

### Fine-tune launcher
- CLI: `smartbite-finetune-ppocrv5-rec`
- Validates PP-OCR dataset layout and prints/launches PaddleOCR training command
- Supports dry-run diagnostics and device resolution logic

## 11) Verification Snapshot
Latest local automated run:
- `UV_CACHE_DIR=.uv-cache-local uv run pytest -q`
- Result: `63 passed, 1 failed`

Known failing test:
- `app/tests/api/test_api_validation.py::test_create_one_shot_scan_accepts_valid_upload_without_qr_or_user`
- Current implementation requires queue availability for one-shot; the test assumes success without Redis.

## 12) Known Constraints and Notes
- One-shot endpoint is queue-backed, not direct synchronous in-process inference.
- Native worker requires AI extras (`uv sync --extra ai`).
- OCR model path must point to exported inference directory, not raw training checkpoint.
- The code disables Paddle model-hoster connectivity checks to avoid startup delays.
- `items.py` route area is reserved and not part of active API routing.
