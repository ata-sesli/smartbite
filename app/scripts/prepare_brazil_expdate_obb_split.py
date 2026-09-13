"""Prepare the Brazilian product expiry OBB dataset for Colab fine-tuning.

The source export currently contains all images under train/. This script
creates a deterministic train/valid/test split and preserves the original
four-class OBB labels so Colab notebooks can choose their own class mapping.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CLASS_NAMES = {
    0: "code",
    1: "date",
    2: "due",
    3: "prod",
}


@dataclass(frozen=True)
class SplitCounts:
    train: int = 544
    valid: int = 61
    test: int = 63

    @property
    def total(self) -> int:
        return self.train + self.valid + self.test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/product-expdates-brazil"),
        help="Source YOLO OBB export with train/images and train/labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/product-expdates-brazil-obb-split"),
        help="Output split dataset directory.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/product-expdates-brazil-obb-split.zip"),
        help="Output zip path for Colab/Drive upload.",
    )
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--train-count", type=int, default=544)
    parser.add_argument("--valid-count", type=int, default=61)
    parser.add_argument("--test-count", type=int, default=63)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory/zip before writing.",
    )
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing source image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing source label directory: {label_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    missing_labels = [p for p in images if not (label_dir / f"{p.stem}.txt").exists()]
    if missing_labels:
        preview = ", ".join(p.name for p in missing_labels[:5])
        raise FileNotFoundError(f"{len(missing_labels)} images are missing labels: {preview}")
    return images


def split_images(images: list[Path], counts: SplitCounts, seed: int) -> dict[str, list[Path]]:
    if len(images) != counts.total:
        raise ValueError(f"Expected {counts.total} images, found {len(images)}")

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    train_end = counts.train
    valid_end = train_end + counts.valid
    return {
        "train": sorted(shuffled[:train_end]),
        "valid": sorted(shuffled[train_end:valid_end]),
        "test": sorted(shuffled[valid_end:]),
    }


def validate_obb_label(label_path: Path) -> tuple[int, int]:
    rows = 0
    empty = 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return rows, 1

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_no} expected 9 OBB tokens, got {len(parts)}")
        class_id = int(parts[0])
        if class_id not in DEFAULT_CLASS_NAMES:
            raise ValueError(f"{label_path}:{line_no} unknown class id {class_id}")
        coords = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{label_path}:{line_no} has coordinate outside [0, 1]")
        rows += 1
    return rows, empty


def write_data_yaml(output: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in DEFAULT_CLASS_NAMES.items())
    (output / "data.yaml").write_text(
        f"""# Brazilian product expiry-date OBB dataset, deterministic local split
train: train/images
val: valid/images
test: test/images

names:
{names}
""",
        encoding="utf-8",
    )


def copy_split(source: Path, output: Path, splits: dict[str, list[Path]]) -> dict[str, dict[str, int]]:
    source_label_dir = source / "train" / "labels"
    summary: dict[str, dict[str, int]] = {}
    for split, images in splits.items():
        image_out = output / split / "images"
        label_out = output / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        rows = 0
        empty_labels = 0
        for idx, image_path in enumerate(images, start=1):
            label_path = source_label_dir / f"{image_path.stem}.txt"
            label_rows, label_empty = validate_obb_label(label_path)
            rows += label_rows
            empty_labels += label_empty
            shutil.copy2(image_path, image_out / image_path.name)
            shutil.copy2(label_path, label_out / label_path.name)
            print(f"[{split}] finished item {idx}/{len(images)}: {image_path.name}")

        summary[split] = {
            "images": len(images),
            "labels": len(images),
            "objects": rows,
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
    counts = SplitCounts(args.train_count, args.valid_count, args.test_count)

    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists, pass --force to replace: {args.output}")
        shutil.rmtree(args.output)
    if args.zip.exists():
        if not args.force:
            raise FileExistsError(f"Zip exists, pass --force to replace: {args.zip}")
        args.zip.unlink()

    images = collect_images(args.source)
    splits = split_images(images, counts, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output)
    summary = copy_split(args.source, args.output, splits)
    zip_dir(args.output, args.zip)

    print("Split summary:")
    for split, values in summary.items():
        print(f"  {split}: {values}")
    print(f"Dataset directory: {args.output}")
    print(f"Zip: {args.zip}")


if __name__ == "__main__":
    main()
