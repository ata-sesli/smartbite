"""Join expiry-date Roboflow YOLOv8-OBB exports for YOLO26s-OBB training."""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"}
DEFAULT_SOURCES = (
    Path("data/expiry-date-detection.yolov8-obb"),
    Path("data/Expiry-Date-Detection-SingleClass.yolov8-obb"),
)


@dataclass(frozen=True)
class DatasetItem:
    source_prefix: str
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/expiry-date-detection-joined-yolo26s-obb"),
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/expiry-date-detection-joined-yolo26s-obb.zip"),
    )
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_prefix(source: Path) -> str:
    return (
        source.name.lower()
        .replace(".yolov8-obb", "")
        .replace("-", "_")
        .replace(".", "_")
    )


def collect_items(source: Path) -> list[DatasetItem]:
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    prefix = source_prefix(source)
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)
    items = []
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image {image_path.name}: {label_path}")
        items.append(DatasetItem(prefix, image_path, label_path))
    return items


def split_items(
    items: list[DatasetItem],
    *,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> dict[str, list[DatasetItem]]:
    ratio_total = train_ratio + valid_ratio + test_ratio
    if abs(ratio_total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_total}")
    if len(items) < 3:
        raise ValueError(f"Need at least 3 images to split, got {len(items)}")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, round(len(shuffled) * valid_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    train_count = len(shuffled) - valid_count - test_count
    if train_count <= 0:
        raise ValueError(f"Not enough images for requested split: {len(items)}")
    return {
        "train": sorted(shuffled[:train_count], key=lambda item: item.image_path.name),
        "valid": sorted(shuffled[train_count : train_count + valid_count], key=lambda item: item.image_path.name),
        "test": sorted(shuffled[train_count + valid_count :], key=lambda item: item.image_path.name),
    }


def normalize_obb_label(label_path: Path) -> tuple[list[str], int]:
    rows: list[str] = []
    raw_text = label_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return rows, 1

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        parts = raw_line.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_no} expected 9 OBB tokens, got {len(parts)}")
        coords = [max(0.0, min(1.0, float(value))) for value in parts[1:]]
        rows.append("0 " + " ".join(f"{value:.6f}" for value in coords))
    return rows, 0


def unique_stem(item: DatasetItem) -> str:
    return f"{item.source_prefix}__{item.image_path.stem}"


def write_dataset_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"""# Joined expiry-date YOLO26s-OBB dataset
path: {output.resolve()}
train: train/images
val: valid/images
test: test/images
names:
  0: expiry_date
""",
        encoding="utf-8",
    )


def copy_split(output: Path, splits: dict[str, list[DatasetItem]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    seen_names: set[str] = set()
    for split, items in splits.items():
        image_out = output / split / "images"
        label_out = output / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        objects = 0
        empty_labels = 0
        for idx, item in enumerate(items, start=1):
            stem = unique_stem(item)
            out_image_name = f"{stem}{item.image_path.suffix.lower()}"
            out_label_name = f"{stem}.txt"
            if out_image_name in seen_names:
                raise ValueError(f"Duplicate output image name: {out_image_name}")
            seen_names.add(out_image_name)

            rows, empty = normalize_obb_label(item.label_path)
            objects += len(rows)
            empty_labels += empty
            shutil.copy2(item.image_path, image_out / out_image_name)
            (label_out / out_label_name).write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")
            print(f"[{split}] finished item {idx}/{len(items)}: {out_image_name} objects={len(rows)}")

        summary[split] = {
            "images": len(items),
            "labels": len(items),
            "objects": objects,
            "empty_label_files": empty_labels,
        }
    return summary


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir.parent))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists, pass --force to replace: {args.output}")
        shutil.rmtree(args.output)
    if args.zip.exists():
        if not args.force:
            raise FileExistsError(f"Zip exists, pass --force to replace: {args.zip}")
        args.zip.unlink()

    items: list[DatasetItem] = []
    for source in args.sources:
        source_items = collect_items(source)
        print(f"Source {source}: {len(source_items)} images")
        items.extend(source_items)
    if not items:
        raise ValueError("No source images found")

    splits = split_items(
        items,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_dataset_yaml(args.output)
    summary = copy_split(args.output, splits)
    zip_dir(args.output, args.zip)

    print("Joined dataset summary:")
    for split, values in summary.items():
        print(f"  {split}: {values}")
    print(f"Dataset directory: {args.output}")
    print(f"Zip: {args.zip}")


if __name__ == "__main__":
    main()
