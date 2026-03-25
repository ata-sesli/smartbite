# SmartBite Commands Guide

This guide lists the exact commands for the most common workflows.

## 1) Hybrid mode (recommended)
Use Docker for PostgreSQL + Redis + API, and run AI worker natively on host.

### Step 0: Create `.env` once (no repeated exports)
```bash
cp .env.example .env
```

`smartbite-worker` reads `.env` automatically via backend settings.
This project defaults API published port to `8001` via `.env` (`SMARTBITE_API_PORT=8001`).

### Step 1: Start Docker services (no worker)
```bash
docker compose up -d postgres redis api
```

### Step 2: Run DB migrations (host shell)
```bash
alembic upgrade head
```

### Step 3: Run native worker on host
```bash
smartbite-worker
```

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
pip install -e .[dev]
```

Run API:
```bash
smartbite-api
```

Run worker:
```bash
smartbite-worker
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

Run tests:
```bash
python3 -m pytest -q
```

## 5) SvelteKit web (PNPM only)
Create `web/.env` once:
```bash
cp web/.env.example web/.env
```

`web/.env.example` already points to `http://localhost:8001`.

```bash
cd web
pnpm install
pnpm sync
pnpm check
pnpm dev
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

## 7) If you see port 8000 already allocated
Error:
```text
Bind for 0.0.0.0:8000 failed: port is already allocated
```

Option A: stop the process/container using port 8000, then retry:
```bash
docker ps --filter publish=8000
lsof -nP -iTCP:8000 -sTCP:LISTEN
docker compose up -d postgres redis api
```

Option B: change API published port in `docker-compose.yml`:
- edit `.env` and set `SMARTBITE_API_PORT`:
```bash
SMARTBITE_API_PORT=8001
```

Then restart compose:
```bash
docker compose down
docker compose up -d postgres redis api
```

## 8) PP-OCRv5 model artifacts and substitute model
Detector model default:
- `SMARTBITE_DETECTOR_MODEL_PATH=yolo26n.pt`

Main model path config (active by default):
- `SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR`
- `SMARTBITE_OCR_PPOCRV5_MAIN_CHAR_DICT_PATH`

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
