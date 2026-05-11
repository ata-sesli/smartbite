#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAIN_STORES = {"store1", "store2", "store3", "store4"}
TEST_STORE = "store5"


@dataclass(frozen=True)
class YoloBox:
    line: str


@dataclass(frozen=True)
class Sample:
    store: str
    stem: str
    source_image: Path
    output_name: str
    boxes: list[YoloBox]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Grocery_products shelf annotations to a single-class YOLO dataset."
    )
    p.add_argument("--source-dir", default="data/Grocery_products")
    p.add_argument("--output-dir", default="data/grocery-products-yolo")
    p.add_argument("--zip-out", default="data/grocery-products-yolo.zip")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        default=True,
        help="Delete output dir/zip before writing. Enabled by default.",
    )
    p.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        help="Refuse to overwrite an existing output dir or zip.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing YOLO output directory instead of converting.",
    )
    return p.parse_args(argv)


def _parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing {key}")
    return float(value)


def row_to_yolo_box(row: dict[str, str]) -> YoloBox:
    left_x = _parse_float(row, "left_x")
    right_x = _parse_float(row, "right_x")
    top_y = _parse_float(row, "top_y")
    bottom_y = _parse_float(row, "bottom_y")

    if not (0.0 <= left_x < right_x <= 1.0 and 0.0 <= top_y < bottom_y <= 1.0):
        raise ValueError("bbox coordinates are out of range or unordered")

    x_center = (left_x + right_x) / 2.0
    y_center = (top_y + bottom_y) / 2.0
    width = right_x - left_x
    height = bottom_y - top_y
    return YoloBox(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = images_dir / f"{stem}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    return None


def collect_samples(source_dir: Path) -> tuple[list[Sample], dict[str, object]]:
    testing_dir = source_dir / "Testing"
    if not testing_dir.exists():
        raise FileNotFoundError(f"Missing Testing directory: {testing_dir}")

    samples: list[Sample] = []
    skipped_images: list[dict[str, str]] = []
    invalid_bbox_rows = 0
    missing_images = 0
    total_rows = 0

    for store_dir in sorted(p for p in testing_dir.glob("store*") if p.is_dir()):
        ann_dir = store_dir / "annotation"
        images_dir = store_dir / "images"
        for csv_path in sorted(ann_dir.glob("*.csv")):
            image_path = _find_image(images_dir, csv_path.stem)
            if image_path is None:
                missing_images += 1
                skipped_images.append(
                    {
                        "store": store_dir.name,
                        "stem": csv_path.stem,
                        "reason": "missing_image",
                    }
                )
                continue

            boxes: list[YoloBox] = []
            with csv_path.open(newline="", encoding="utf-8", errors="ignore") as f:
                for row in csv.DictReader(f):
                    total_rows += 1
                    try:
                        boxes.append(row_to_yolo_box(row))
                    except (TypeError, ValueError):
                        invalid_bbox_rows += 1

            if not boxes:
                skipped_images.append(
                    {
                        "store": store_dir.name,
                        "stem": csv_path.stem,
                        "reason": "no_valid_boxes",
                    }
                )
                continue

            samples.append(
                Sample(
                    store=store_dir.name,
                    stem=csv_path.stem,
                    source_image=image_path,
                    output_name=f"{store_dir.name}_{csv_path.stem}{image_path.suffix.lower()}",
                    boxes=boxes,
                )
            )

    skipped = {
        "total_annotation_rows": total_rows,
        "invalid_bbox_rows": invalid_bbox_rows,
        "missing_images": missing_images,
        "skipped_images": skipped_images,
    }
    return samples, skipped


def split_samples(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    train = [s for s in samples if s.store in TRAIN_STORES]
    store5 = [s for s in samples if s.store == TEST_STORE]
    unsupported = [s for s in samples if s.store not in TRAIN_STORES and s.store != TEST_STORE]
    if unsupported:
        names = ", ".join(sorted({s.store for s in unsupported}))
        raise ValueError(f"Unsupported store directories for fixed split: {names}")

    rnd = random.Random(seed)
    store5 = list(store5)
    rnd.shuffle(store5)
    midpoint = len(store5) // 2
    return {
        "train": train,
        "val": store5[:midpoint],
        "test": store5[midpoint:],
    }


def _safe_prepare_output(output_dir: Path, zip_out: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output dir exists: {output_dir}")
        shutil.rmtree(output_dir)
    if zip_out.exists():
        if not force:
            raise FileExistsError(f"Zip already exists: {zip_out}")
        zip_out.unlink()

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    zip_out.parent.mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(output_dir: Path) -> None:
    (output_dir / "dataset.yaml").write_text(
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: product\n",
        encoding="utf-8",
    )


def _write_split(output_dir: Path, split: str, samples: list[Sample]) -> None:
    for sample in samples:
        image_dst = output_dir / "images" / split / sample.output_name
        label_dst = output_dir / "labels" / split / f"{Path(sample.output_name).stem}.txt"
        shutil.copy2(sample.source_image, image_dst)
        label_dst.write_text("\n".join(box.line for box in sample.boxes) + "\n", encoding="utf-8")


def _split_summary(split_samples_map: dict[str, list[Sample]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "images": len(items),
            "boxes": sum(len(sample.boxes) for sample in items),
        }
        for split, items in split_samples_map.items()
    }


def _write_manifest(
    output_dir: Path,
    source_dir: Path,
    seed: int,
    split_samples_map: dict[str, list[Sample]],
    skipped: dict[str, object],
) -> dict[str, object]:
    split_counts = _split_summary(split_samples_map)
    total_usable_images = sum(v["images"] for v in split_counts.values())
    total_usable_boxes = sum(v["boxes"] for v in split_counts.values())
    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "seed": seed,
        "class_mode": "single_class_generic_product",
        "names": {"0": "product"},
        "split_policy": {
            "train": "store1-store4",
            "val": "first half of shuffled store5",
            "test": "second half of shuffled store5",
        },
        "counts": {
            "total_annotation_rows": skipped["total_annotation_rows"],
            "total_usable_images": total_usable_images,
            "total_usable_boxes": total_usable_boxes,
            "invalid_bbox_rows_skipped": skipped["invalid_bbox_rows"],
            "missing_images_skipped": skipped["missing_images"],
            "images_without_valid_boxes_skipped": len(
                [
                    image
                    for image in skipped["skipped_images"]  # type: ignore[index]
                    if image["reason"] == "no_valid_boxes"
                ]
            ),
            "splits": split_counts,
        },
        "skipped_images": skipped["skipped_images"],
        "splits": {
            split: [
                {
                    "store": sample.store,
                    "stem": sample.stem,
                    "source_image": str(sample.source_image),
                    "output_image": f"images/{split}/{sample.output_name}",
                    "output_label": f"labels/{split}/{Path(sample.output_name).stem}.txt",
                    "boxes": len(sample.boxes),
                }
                for sample in items
            ]
            for split, items in split_samples_map.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _zip_dataset(output_dir: Path, zip_out: Path) -> None:
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(output_dir))


def validate_output(output_dir: Path) -> list[str]:
    errors: list[str] = []
    for required in [
        output_dir / "dataset.yaml",
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "images" / "test",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
        output_dir / "labels" / "test",
    ]:
        if not required.exists():
            errors.append(f"missing required path: {required}")

    for label_path in sorted((output_dir / "labels").glob("*/*.txt")):
        for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{idx}: expected 5 columns")
                continue
            if parts[0] != "0":
                errors.append(f"{label_path}:{idx}: expected class 0")
            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{idx}: non-numeric YOLO coordinate")
                continue
            if any(v < 0.0 or v > 1.0 for v in values):
                errors.append(f"{label_path}:{idx}: coordinate outside 0..1")
            if values[2] <= 0.0 or values[3] <= 0.0:
                errors.append(f"{label_path}:{idx}: width/height must be positive")
    return errors


def convert(source_dir: Path, output_dir: Path, zip_out: Path, seed: int, force: bool) -> dict[str, object]:
    samples, skipped = collect_samples(source_dir)
    split_samples_map = split_samples(samples, seed)
    _safe_prepare_output(output_dir, zip_out, force)
    _write_dataset_yaml(output_dir)
    for split, items in split_samples_map.items():
        _write_split(output_dir, split, items)
    manifest = _write_manifest(output_dir, source_dir, seed, split_samples_map, skipped)

    errors = validate_output(output_dir)
    if errors:
        raise ValueError("Generated dataset failed validation:\n" + "\n".join(errors))

    _zip_dataset(output_dir, zip_out)
    return manifest


def _print_summary(manifest: dict[str, object], output_dir: Path, zip_out: Path) -> None:
    counts = manifest["counts"]  # type: ignore[index]
    splits = counts["splits"]  # type: ignore[index]
    for split in ("train", "val", "test"):
        item = splits[split]
        print(f"{split} images: {item['images']} | {split} boxes: {item['boxes']}")
    print(f"total usable images: {counts['total_usable_images']}")
    print(f"total usable boxes: {counts['total_usable_boxes']}")
    print(f"invalid bbox rows skipped: {counts['invalid_bbox_rows_skipped']}")
    print(f"missing images skipped: {counts['missing_images_skipped']}")
    print(f"output dir: {output_dir}")
    print(f"zip out: {zip_out}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    zip_out = Path(args.zip_out).resolve()

    try:
        if args.check_only:
            errors = validate_output(output_dir)
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 2
            print(f"Dataset validation passed: {output_dir}")
            return 0

        manifest = convert(source_dir, output_dir, zip_out, args.seed, args.force)
        _print_summary(manifest, output_dir, zip_out)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
