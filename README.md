# SmartBite V1

> **Note:** This repository is an unmaintained, read-only archive preserved for reference. No active development is occurring, and issues or pull requests are not monitored.

SmartBite is an expiry-date scanning backend with asynchronous AI processing and a lightweight SvelteKit testing console.

## Stack
- API: Litestar
- DB: PostgreSQL + SQLAlchemy + Alembic
- Queue: Redis + arq
- Detector: Ultralytics YOLO (`models/yolo20n/yolo26s/yolo26s-best.pt` by default)
- Expiry OCR lane: `YOLO26s-OBB -> SVTRv2 recognition -> parser/decision`
- Web console: SvelteKit + TypeScript (Bun)

## Implemented API
- `POST /scans`
- `POST /scans/oneshot`
- `GET /scans/{scan_id}`
- `PATCH /scans/{scan_id}`
- `POST /mobile/expiry-scans`
- `PATCH /mobile/expiry-scans/{id}`
- `GET /expiry`
- `POST /alerts/process`
- `GET /health`
- `GET /admin/metrics`

### `POST /scans` vs `POST /scans/oneshot`
- `POST /scans` requires `image`, `qr_code`, and `user_id`, then enqueues async processing.
- `POST /scans/oneshot` requires only `image` (optional `metadata`) and waits for a completed result by polling scan status.
- One-shot is still queue-backed. Redis + worker must be available, otherwise one-shot returns `503` or `504`.

## OCR runtime behavior & Model Weights
- Pretrained & fine-tuned model weights are hosted on Hugging Face: [atasesli/smartbite-models](https://huggingface.co/atasesli/smartbite-models)
- Expiry detector model: `models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt`
- SVTRv2 recognition model: `models/svtrv2/smartbite_svtrv2_expdate_rec`
- Automatic download: Docker containers and worker automatically download missing model assets on startup via `SMARTBITE_MODELS_AUTO_DOWNLOAD=true`.
- Manual prefetch command:
  ```bash
  uv run smartbite-prefetch-models
  ```
- Runtime backends default to the verified fallback path:
  - `SMARTBITE_MOBILE_EXPIRY_DETECTOR_BACKEND=ultralytics`
  - `SMARTBITE_SVTRV2_REC_BACKEND=paddle`
- Device setting: `SMARTBITE_SVTRV2_DEVICE_MODE=cpu|cuda`
- Startup enforces `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` to skip remote host checks

## ONNX conversion and parity
```bash
uv run --extra ai --extra export smartbite-export-onnx-models
uv run smartbite-yolo-onnx-parity --ultralytics-report path/to/pt_report.json --onnx-report path/to/onnx_report.json
uv run smartbite-svtr-onnx-parity --paddle-report path/to/paddle_report.json --onnx-report path/to/onnx_report.json
```

Switch `SMARTBITE_MOBILE_EXPIRY_DETECTOR_BACKEND=onnx` or `SMARTBITE_SVTRV2_REC_BACKEND=onnx` only after the parity reports are acceptable.

## Run modes
### Full Docker mode
```bash
cp .env.example .env
docker compose up -d --build
```

Notes:
- Default API port is `8005`.
- Docker API startup auto-runs migrations via `app/scripts/start_api_with_migrations.sh`.
- The API, worker, Postgres, and Redis all start with plain `docker compose up -d`.
- `SMARTBITE_STORAGE_ROOT` points to the shared bind-mounted folder (`data/storage` by default).

## Web console
```bash
cp web/.env.example web/.env
cd web
bun install
bun run dev
```

Default proxied backend URL is `http://localhost:8005`.

## More commands
Use [COMMANDS.md](COMMANDS.md) for full operational command chains.
