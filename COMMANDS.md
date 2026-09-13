# SmartBite Commands Guide

This guide lists the exact commands for the most common workflows.

## 1) Hybrid mode (recommended)
Use Docker for PostgreSQL + Redis + API, and run AI worker natively on host.

### Step 0: Create `.env` once (no repeated exports)
```bash
cp .env.example .env
```

`smartbite-worker` reads `.env` automatically via backend settings.
This project defaults API published port to `8005` via `.env` (`SMARTBITE_API_PORT=8005`).

### Step 1: Start Docker services (no worker)
```bash
docker compose up -d postgres redis api
```

### Step 2: Run native worker on host
```bash
uv sync --extra ai --extra dev
uv run smartbite-worker
```
If you previously saw a `RequestsDependencyWarning` (`chardet` mismatch), run the same `uv sync` once to apply pinned versions.

DB migration note:
- `docker compose up -d ... api` now auto-runs `alembic upgrade head` before starting API.
- Manual `alembic upgrade head` is only needed when you run API on host (outside Docker).

Why this works:
- API container writes scans into the bind-mounted repo path.
- Native worker reads the same files from `data/storage` on host.
- Postgres/Redis are reachable from host via `localhost` mapped ports.
- `.env` already points host runtime to `localhost` services.

## 2) Full Docker mode (including worker)
```bash
docker compose --profile docker-worker up --build
```

## 3) Start/stop/check status
Start:
```bash
docker compose up -d postgres redis api
```

Stop:
```bash
docker compose down
```

Logs:
```bash
docker compose logs -f api
```

Service list (default):
```bash
docker compose config --services
```

Service list (with docker worker profile):
```bash
docker compose --profile docker-worker config --services
```

## 4) Backend local commands (host)
Install:
```bash
uv sync --extra dev
```

Run API:
```bash
smartbite-api
```

Run worker:
```bash
uv sync --extra ai
uv run smartbite-worker
```

Fine-tune PP-OCRv5 recognition:
```bash
smartbite-finetune-ppocrv5-rec \
  --data-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/output \
  --pretrained-model /absolute/path/to/pretrained/model \
  --base-config /absolute/path/to/rec_ppocr_v5_train.yml \
  --dry-run
```

Launch real training (remove `--dry-run`):
```bash
smartbite-finetune-ppocrv5-rec \
  --data-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/output \
  --pretrained-model /absolute/path/to/pretrained/model \
  --base-config /absolute/path/to/rec_ppocr_v5_train.yml \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --device auto
```

Build PP-OCRv5 recognition dataset from JSON annotations:
```bash
smartbite-build-ppocrv5-rec-dataset \
  --images-dir /absolute/path/to/images \
  --annotation-json /absolute/path/to/annotations.json \
  --output-dir /absolute/path/to/ppocrv5_dataset
```

Build with DMY component crops:
```bash
smartbite-build-ppocrv5-rec-dataset \
  --images-dir /absolute/path/to/images \
  --annotation-json /absolute/path/to/annotations.json \
  --output-dir /absolute/path/to/ppocrv5_dataset \
  --generate-component-crops true
```

Build with semi-auto bbox-jitter review candidates:
```bash
smartbite-build-ppocrv5-rec-dataset \
  --images-dir /absolute/path/to/images \
  --annotation-json /absolute/path/to/annotations.json \
  --output-dir /absolute/path/to/ppocrv5_dataset \
  --generate-candidate-crops true \
  --candidate-jitter-count 2 \
  --candidate-max-pad-px 12
```

Dry-run (compute/report only, no file writes):
```bash
smartbite-build-ppocrv5-rec-dataset \
  --images-dir /absolute/path/to/images \
  --annotation-json /absolute/path/to/annotations.json \
  --output-dir /absolute/path/to/ppocrv5_dataset \
  --dry-run
```

Build PP-OCRv5 date-recognition dataset from date-only images + JSON labels:
```bash
smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir /absolute/path/to/date_images \
  --labels-json /absolute/path/to/date_labels.json \
  --output-dir /absolute/path/to/ppocrv5_date_dataset
```

Build date dataset from CSV labels:
```bash
smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir /absolute/path/to/date_images \
  --labels-csv /absolute/path/to/date_labels.csv \
  --path-column image_path \
  --text-column transcription \
  --output-dir /absolute/path/to/ppocrv5_date_dataset
```

Build date dataset using filename parser mode:
```bash
smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir /absolute/path/to/date_images \
  --labels-from-filename true \
  --filename-label-regex '(?P<label>\d{4}-\d{2}-\d{2})' \
  --output-dir /absolute/path/to/ppocrv5_date_dataset
```

Build date dataset with explicit split file:
```bash
smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir /absolute/path/to/date_images \
  --labels-txt /absolute/path/to/date_labels.txt \
  --split-file /absolute/path/to/split.txt \
  --output-dir /absolute/path/to/ppocrv5_date_dataset
```

Dry-run date dataset conversion (compute/report only):
```bash
smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir /absolute/path/to/date_images \
  --labels-jsonl /absolute/path/to/date_labels.jsonl \
  --output-dir /absolute/path/to/ppocrv5_date_dataset \
  --dry-run
```

Generate semi-auto PP-OCRv5 recognition candidates (review set):
```bash
smartbite-generate-ppocrv5-rec-candidates \
  --images-dir /absolute/path/to/product_images \
  --output-dir /absolute/path/to/candidates_out \
  --input-mode full_images
```

Generate candidates with YOLO product detections:
```bash
smartbite-generate-ppocrv5-rec-candidates \
  --images-dir /absolute/path/to/product_images \
  --output-dir /absolute/path/to/candidates_out \
  --input-mode full_images \
  --yolo-detections-jsonl /absolute/path/to/yolo_detections.jsonl
```

YOLO JSONL canonical record fields:
- `image_path`
- `product_id`
- `bbox_xyxy` (`[x1, y1, x2, y2]`)
- `score`
- `class_name`

Generate candidates from pre-cropped product images:
```bash
smartbite-generate-ppocrv5-rec-candidates \
  --images-dir /absolute/path/to/product_crops \
  --output-dir /absolute/path/to/candidates_out \
  --input-mode product_crops \
  --export-csv true \
  --export-overlays true
```

Dry-run candidate generation (compute/report only):
```bash
smartbite-generate-ppocrv5-rec-candidates \
  --images-dir /absolute/path/to/product_images \
  --output-dir /absolute/path/to/candidates_out \
  --dry-run
```

Run tests:
```bash
python3 -m pytest -q
```

## 5) SvelteKit web (Bun)
Create `web/.env` once:
```bash
cp web/.env.example web/.env
```

`web/.env.example` already points to `http://localhost:8005`.

```bash
cd web
bun install
bun run sync
bun run check
bun run dev
```

## 6) If you see this Docker build error
Error:
```text
Multiple top-level packages discovered in a flat-layout: ['app', 'alembic']
```

Meaning:
- setuptools tried to package both `app` and `alembic`.

Fix (already applied in this repo):
- `pyproject.toml` now explicitly includes only `app*` packages.

After pulling latest changes, rebuild:
```bash
docker compose build --no-cache api
docker compose up -d postgres redis api
```

## 7) If API port is already allocated
Error:
```text
Bind for 0.0.0.0:<port> failed: port is already allocated
```

Option A: stop the process/container using your chosen port (default `8005`), then retry:
```bash
docker ps --filter publish=8005
lsof -nP -iTCP:8005 -sTCP:LISTEN
docker compose up -d postgres redis api
```

Option B: change API published port in `docker-compose.yml`:
- edit `.env` and set `SMARTBITE_API_PORT`:
```bash
SMARTBITE_API_PORT=8005
```

Then restart compose:
```bash
docker compose down
docker compose up -d postgres redis api
```

## 8) PP-OCRv5 model artifacts and substitute model
Detector model default:
- `SMARTBITE_DETECTOR_MODEL_PATH=models/yolo20n/yolo26s/yolo26s-best.pt`

Main model path config (active by default):
- `SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR`
- `SMARTBITE_OCR_PPOCRV5_MAIN_CHAR_DICT_PATH`

Current repo defaults:
- `SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR=models/fine-tuned-models/best_model_inference`
- `SMARTBITE_OCR_PPOCRV5_MAIN_CHAR_DICT_PATH=ppocr/utils/dict/ppocrv5_dict.txt`

Easy model switch (single setting):
```bash
# in .env
SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR=models/fine-tuned-models/best_model_inference

# restart worker to apply
smartbite-worker
```

Substitute model metadata file:
- `SMARTBITE_OCR_SUBSTITUTE_CONFIG_PATH`
- example file in repo: `app/ai/ocr_substitute_config.json`

Default behavior:
- `SMARTBITE_OCR_ENABLE_SUBSTITUTE_MODEL=false`
- Runtime always uses `ppocrv5_main`.

To explicitly switch to substitute model (non-default):
```bash
# in .env
SMARTBITE_OCR_ENABLE_SUBSTITUTE_MODEL=true

# then restart worker
smartbite-worker
```

Expected PaddleOCR-native dataset layout for fine-tuning:
- `train_images/`
- `val_images/`
- `train_label.txt`
- `val_label.txt`

Each label line format:
```text
relative/image/path.jpg<TAB>text label
```

Example one-shot scan invocation:
```bash
curl -sS -X POST \
  -F image=@path/to/sample.jpg \
  http://localhost:8005/scans/oneshot
```
