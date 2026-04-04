#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/zip_dataset_with_real_files.sh \
    --source-dir /absolute/path/to/dataset \
    --output-zip /absolute/path/to/output.zip

Description:
  Creates a zip archive from a dataset directory while resolving symlinks
  into real copied files. Useful for uploading to Colab.

Notes:
  - The script stages a fully de-symlinked copy under /tmp.
  - It validates there are no symlinks in the staged copy before zipping.
EOF
}

SOURCE_DIR=""
OUTPUT_ZIP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --output-zip)
      OUTPUT_ZIP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_DIR" || -z "$OUTPUT_ZIP" ]]; then
  echo "[ERROR] --source-dir and --output-zip are required." >&2
  usage
  exit 2
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "[ERROR] Source dataset directory does not exist: $SOURCE_DIR" >&2
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync is required but not found on PATH." >&2
  exit 2
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "[ERROR] zip is required but not found on PATH." >&2
  exit 2
fi

SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"
OUTPUT_ZIP="$(python3 - <<'PY' "$OUTPUT_ZIP"
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"

TMP_ROOT="$(mktemp -d /tmp/smartbite_zip_stage.XXXXXX)"
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

STAGE_DIR="$TMP_ROOT/$(basename "$SOURCE_DIR")"
mkdir -p "$STAGE_DIR"

echo "[INFO] Copying dataset with symlink resolution..."
rsync -aL --delete "$SOURCE_DIR"/ "$STAGE_DIR"/

SYMLINK_COUNT="$(find "$STAGE_DIR" -type l | wc -l | tr -d ' ')"
if [[ "$SYMLINK_COUNT" != "0" ]]; then
  echo "[ERROR] Staged directory still contains symlinks: $SYMLINK_COUNT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_ZIP")"
rm -f "$OUTPUT_ZIP"

echo "[INFO] Creating zip archive..."
(
  cd "$TMP_ROOT"
  zip -qr "$OUTPUT_ZIP" "$(basename "$SOURCE_DIR")"
)

echo "[INFO] Done."
echo "[INFO] Output zip: $OUTPUT_ZIP"
du -sh "$OUTPUT_ZIP"
