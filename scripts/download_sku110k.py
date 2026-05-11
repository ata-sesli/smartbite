#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare SKU-110K into ./data/sku110k using Ultralytics dataset YAML flow."
    )
    parser.add_argument(
        "--yaml",
        default="SKU-110k.yaml",
        help="Source dataset YAML file (default: SKU-110k.yaml).",
    )
    parser.add_argument(
        "--target-dir",
        default="data/sku110k",
        help="Dataset output directory (default: data/sku110k).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing target directory and download again.",
    )
    return parser.parse_args()


def _is_ready(target_dir: Path) -> bool:
    required = (
        target_dir / "train.txt",
        target_dir / "val.txt",
        target_dir / "test.txt",
        target_dir / "images",
        target_dir / "labels",
    )
    return all(p.exists() for p in required)


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    source_yaml = (project_root / args.yaml).resolve()
    target_dir = (project_root / args.target_dir).resolve()
    work_yaml = project_root / "data" / "sku110k_download.yaml"

    if not source_yaml.exists():
        print(f"Source YAML not found: {source_yaml}", file=sys.stderr)
        return 1

    if args.force and target_dir.exists():
        print(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    work_yaml.parent.mkdir(parents=True, exist_ok=True)

    if _is_ready(target_dir):
        print(f"Dataset already prepared: {target_dir}")
        return 0

    try:
        import yaml  # type: ignore
        from ultralytics.data.utils import check_det_dataset  # type: ignore
    except Exception as exc:
        print(
            "Missing dependencies. Install first:\n"
            "  pip install -U ultralytics polars pyyaml\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    cfg = yaml.safe_load(source_yaml.read_text())
    cfg["path"] = str(target_dir)
    work_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"Downloading/preparing SKU-110K to: {target_dir}")
    info = check_det_dataset(str(work_yaml), autodownload=True)
    print("Resolved dataset path:", info.get("path"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
