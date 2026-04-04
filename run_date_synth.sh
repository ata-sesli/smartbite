#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is not installed or not on PATH." >&2
  exit 1
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRIPT_DIR/.uv-cache}"
export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR"

# Install project + optional AI dependencies into uv-managed environment.
uv sync --extra ai

uv run python - <<'PY'
import importlib.util
import platform
import sys

print("[INFO] Runtime preflight")
print(f"  python={sys.version.split()[0]} exe={sys.executable}")
print(f"  platform={platform.platform()} machine={platform.machine()}")

if importlib.util.find_spec("torch") is not None:
    import torch

    print(
        "  torch="
        f"{torch.__version__} mps_built={torch.backends.mps.is_built()} "
        f"mps_available={torch.backends.mps.is_available()}"
    )
else:
    print("  torch=not-installed")

if importlib.util.find_spec("paddle") is not None:
    import paddle

    print(f"  paddle={paddle.__version__} cuda_compiled={paddle.is_compiled_with_cuda()}")
else:
    print("  paddle=not-installed")
    print("[ERROR] Missing `paddle` runtime. Install it with uv, then re-run:")
    print("  uv pip install --python .venv/bin/python paddlepaddle")
    raise SystemExit(1)
PY

DATE_SYNTH_ROOT="${DATE_SYNTH_ROOT:-$SCRIPT_DIR/data/SMARTBITE-DATASET/Date-Synth}"
DATASET_OUT="${DATASET_OUT:-$SCRIPT_DIR/data/ppocrv5_date_synth_dataset}"
TRAIN_OUT="${TRAIN_OUT:-$SCRIPT_DIR/artifacts/ppocrv5_date_synth_run1}"
DEVICE_MODE="${DEVICE_MODE:-auto}"

# Defaults point to files downloaded into this repo; export to override.
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$SCRIPT_DIR/models/ppocrv5/pretrained/PP-OCRv5_server_rec_pretrained.pdparams}"
BASE_CONFIG="${BASE_CONFIG:-$SCRIPT_DIR/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml}"

if [[ ! -d "$DATE_SYNTH_ROOT/images" ]]; then
  echo "[ERROR] Missing images directory: $DATE_SYNTH_ROOT/images" >&2
  exit 1
fi
if [[ ! -f "$DATE_SYNTH_ROOT/annotations.json" ]]; then
  echo "[ERROR] Missing labels file: $DATE_SYNTH_ROOT/annotations.json" >&2
  exit 1
fi
if [[ ! -f "$PRETRAINED_MODEL" ]]; then
  echo "[ERROR] Missing pretrained model: $PRETRAINED_MODEL" >&2
  exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "[ERROR] Missing base config: $BASE_CONFIG" >&2
  exit 1
fi

uv run smartbite-build-ppocrv5-rec-date-dataset \
  --images-dir "$DATE_SYNTH_ROOT/images" \
  --labels-json "$DATE_SYNTH_ROOT/annotations.json" \
  --output-dir "$DATASET_OUT" \
  --train-ratio 0.9 \
  --seed 42 \
  --copy-mode symlink

uv run python - <<'PY'
from pathlib import Path

root = Path("data/ppocrv5_date_synth_dataset")
train = [x for x in (root / "train_label.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
val = [x for x in (root / "val_label.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
print(f"train={len(train)} val={len(val)} total={len(train)+len(val)}")
PY

uv run smartbite-finetune-ppocrv5-rec \
  --data-root "$DATASET_OUT" \
  --output-dir "$TRAIN_OUT" \
  --pretrained-model "$PRETRAINED_MODEL" \
  --base-config "$BASE_CONFIG" \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --device "$DEVICE_MODE" \
  --dry-run

# Set RUN_REAL_TRAINING=true to launch real training.
RUN_REAL_TRAINING="${RUN_REAL_TRAINING:-false}"
if [[ "$RUN_REAL_TRAINING" == "true" ]]; then
  uv run smartbite-finetune-ppocrv5-rec \
    --data-root "$DATASET_OUT" \
    --output-dir "$TRAIN_OUT" \
    --pretrained-model "$PRETRAINED_MODEL" \
    --base-config "$BASE_CONFIG" \
    --epochs 30 \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --device "$DEVICE_MODE"
else
  echo "[INFO] Skipping real training. Export RUN_REAL_TRAINING=true to run it."
fi
