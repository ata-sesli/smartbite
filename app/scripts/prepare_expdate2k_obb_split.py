"""Prepare the ExpDate-2K YOLO OBB dataset for Colab fine-tuning.

The source export contains all images under train/. This script removes empty
label files, creates a deterministic train/valid/test split, keeps duplicate
Roboflow-export variants of the same original image in the same split, and
zips the resulting single-class OBB dataset for Drive/Colab upload.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAME = "expiry_date"


@dataclass(frozen=True)
class SplitCounts:
    train: int = 1339
    valid: int = 363
    test: int = 196

    @property
    def total(self) -> int:
        return self.train + self.valid + self.test


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    group_key: str
    objects: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/expdate-2k-yolo"),
        help="Source YOLO OBB export with train/images and train/labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/expdate-2k-yolo-obb-split"),
        help="Output split dataset directory.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/expdate-2k-yolo-obb-split.zip"),
        help="Output zip path for Colab/Drive upload.",
    )
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--train-count", type=int, default=1339)
    parser.add_argument("--valid-count", type=int, default=363)
    parser.add_argument("--test-count", type=int, default=196)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory/zip before writing.",
    )
    return parser.parse_args()


def original_group_key(image_path: Path) -> str:
    return image_path.stem.split(".rf.")[0]


def validate_and_count_obb_label(label_path: Path) -> int:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0

    rows = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_no} expected 9 OBB tokens, got {len(parts)}")
        class_id = int(parts[0])
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_no} expected class 0, got {class_id}")
        coords = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{label_path}:{line_no} has coordinate outside [0, 1]")
        rows += 1
    return rows


def collect_records(source: Path) -> tuple[list[ImageRecord], list[Path]]:
    image_dir = source / "train" / "images"
    label_dir = source / "train" / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing source image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing source label directory: {label_dir}")

    records: list[ImageRecord] = []
    removed_empty: list[Path] = []
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        objects = validate_and_count_obb_label(label_path)
        if objects == 0:
            removed_empty.append(image_path)
            continue
        records.append(
            ImageRecord(
                image_path=image_path,
                label_path=label_path,
                group_key=original_group_key(image_path),
                objects=objects,
            )
        )
    return records, removed_empty


def group_records(records: list[ImageRecord]) -> list[list[ImageRecord]]:
    groups_by_key: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups_by_key.setdefault(record.group_key, []).append(record)
    return [sorted(group, key=lambda r: r.image_path.name) for group in groups_by_key.values()]


def choose_groups_for_count(groups: list[list[ImageRecord]], target: int) -> tuple[list[list[ImageRecord]], list[list[ImageRecord]]]:
    """Choose a deterministic subset of groups whose image count equals target."""

    # Store only backpointers per reachable sum to keep memory modest.
    back: dict[int, tuple[int, int] | None] = {0: None}
    for idx, group in enumerate(groups):
        size = len(group)
        for existing in sorted(list(back.keys()), reverse=True):
            new_sum = existing + size
            if new_sum > target or new_sum in back:
                continue
            back[new_sum] = (existing, idx)
        if target in back:
            break

    if target not in back:
        raise ValueError(f"Could not choose duplicate-safe groups totaling {target} images")

    selected_indices: set[int] = set()
    current = target
    while current:
        previous, idx = back[current]  # type: ignore[misc]
        selected_indices.add(idx)
        current = previous

    selected = [group for idx, group in enumerate(groups) if idx in selected_indices]
    remaining = [group for idx, group in enumerate(groups) if idx not in selected_indices]
    return selected, remaining


def split_records(records: list[ImageRecord], counts: SplitCounts, seed: int) -> dict[str, list[ImageRecord]]:
    if len(records) != counts.total:
        raise ValueError(f"Expected {counts.total} non-empty-label images, found {len(records)}")

    groups = group_records(records)
    random.Random(seed).shuffle(groups)

    test_groups, remaining = choose_groups_for_count(groups, counts.test)
    valid_groups, train_groups = choose_groups_for_count(remaining, counts.valid)

    splits = {
        "train": [record for group in train_groups for record in group],
        "valid": [record for group in valid_groups for record in group],
        "test": [record for group in test_groups for record in group],
    }
    for split, expected in [("train", counts.train), ("valid", counts.valid), ("test", counts.test)]:
        actual = len(splits[split])
        if actual != expected:
            raise AssertionError(f"{split} split has {actual} images, expected {expected}")
    return {split: sorted(values, key=lambda r: r.image_path.name) for split, values in splits.items()}


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"""# ExpDate-2K single-class expiry-date OBB dataset, deterministic local split
train: train/images
val: valid/images
test: test/images

names:
  0: {CLASS_NAME}
""",
        encoding="utf-8",
    )


def copy_split(output: Path, splits: dict[str, list[ImageRecord]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split, records in splits.items():
        image_out = output / split / "images"
        label_out = output / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        objects = 0
        duplicate_group_keys: set[str] = set()
        seen_group_keys: set[str] = set()
        for idx, record in enumerate(records, start=1):
            objects += record.objects
            if record.group_key in seen_group_keys:
                duplicate_group_keys.add(record.group_key)
            seen_group_keys.add(record.group_key)
            shutil.copy2(record.image_path, image_out / record.image_path.name)
            shutil.copy2(record.label_path, label_out / record.label_path.name)
            print(f"[{split}] finished item {idx}/{len(records)}: {record.image_path.name} objects={record.objects}")

        summary[split] = {
            "images": len(records),
            "labels": len(records),
            "objects": objects,
            "duplicate_group_keys_inside_split": len(duplicate_group_keys),
        }
    return summary


def validate_no_group_leakage(splits: dict[str, list[ImageRecord]]) -> None:
    owner: dict[str, str] = {}
    leaks: list[tuple[str, str, str]] = []
    for split, records in splits.items():
        for record in records:
            previous = owner.setdefault(record.group_key, split)
            if previous != split:
                leaks.append((record.group_key, previous, split))
    if leaks:
        preview = ", ".join(f"{key}:{a}->{b}" for key, a, b in leaks[:10])
        raise AssertionError(f"Duplicate source image groups leaked across splits: {preview}")


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    file_paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, path in enumerate(file_paths, start=1):
            zf.write(path, path.relative_to(source_dir.parent))
            if idx % 250 == 0 or idx == len(file_paths):
                print(f"[zip] finished item {idx}/{len(file_paths)}: {path.relative_to(source_dir.parent)}")


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

    records, removed_empty = collect_records(args.source)
    print(f"Source images with non-empty labels: {len(records)}")
    print(f"Removed empty-label images: {len(removed_empty)}")
    for idx, image_path in enumerate(removed_empty, start=1):
        print(f"[removed-empty] finished item {idx}/{len(removed_empty)}: {image_path.name}")

    splits = split_records(records, counts, args.seed)
    validate_no_group_leakage(splits)

    args.output.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output)
    summary = copy_split(args.output, splits)

    manifest = {
        "source": str(args.source),
        "seed": args.seed,
        "removed_empty_images": [path.name for path in removed_empty],
        "summary": summary,
    }
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_dir(args.output, args.zip)

    print("Split summary:")
    for split, values in summary.items():
        print(f"  {split}: {values}")
    print(f"Dataset directory: {args.output}")
    print(f"Zip: {args.zip}")


if __name__ == "__main__":
    main()
