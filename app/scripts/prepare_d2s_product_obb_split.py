"""Convert MVTec D2S COCO masks into a YOLO26s-OBB product dataset.

The D2S test JSON files contain image metadata only, so this script uses the
full labeled train + augmented sets for training and deterministically splits
the labeled validation set into valid/test. All D2S categories are collapsed
to a single `product` class because the downstream model is only responsible
for product cropping.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError as exc:  # pragma: no cover - exercised in local setup, not unit tests
    raise SystemExit(
        "Missing dependency: pycocotools. Install it before running this converter, "
        "for example: uv pip install pycocotools"
    ) from exc


CLASS_NAME = "product"
TRAIN_JSONS = ("D2S_training.json", "D2S_augmented.json")
VALIDATION_JSON = "D2S_validation.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ConvertedImage:
    source_image: Path
    output_name: str
    labels: tuple[str, ...]
    source_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/d2s_images_v1"),
        help="D2S image directory containing D2S_*.jpg files.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/d2s_annotations_v1.1/annotations"),
        help="Directory containing D2S COCO annotation JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/d2s-product-obb-yolo26s"),
        help="Output YOLO OBB dataset directory.",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("data/d2s-product-obb-yolo26s.zip"),
        help="Output zip path for Google Drive / Colab upload.",
    )
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.8,
        help="Fraction of D2S_validation.json images assigned to valid; the rest become test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing output directory/zip before writing.",
    )
    return parser.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation JSON: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"{path} is missing COCO key: {key}")
    return data


def image_path_for(images_root: Path, file_name: str) -> Path:
    candidate = images_root / file_name
    if candidate.exists():
        return candidate
    stem = Path(file_name).stem
    for ext in IMAGE_EXTS:
        fallback = images_root / f"{stem}{ext}"
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Missing D2S image file for {file_name} under {images_root}")


def normalize_box_points(points: np.ndarray, width: int, height: int) -> list[float]:
    clipped = points.astype(np.float64).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    normalized = clipped / np.array([width, height], dtype=np.float64)
    return [float(value) for value in normalized.reshape(-1)]


def annotation_to_obb_line(annotation: dict[str, Any], width: int, height: int) -> str | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, dict):
        return None

    decoded = coco_mask.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    ys, xs = np.where(decoded > 0)
    if len(xs) < 3:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    rect_width, rect_height = rect[1]
    if rect_width < 1.0 or rect_height < 1.0:
        return None

    box_points = cv2.boxPoints(rect)
    coords = normalize_box_points(box_points, width, height)
    if any(value < 0.0 or value > 1.0 for value in coords):
        raise ValueError(f"Normalized OBB coordinate outside [0, 1]: {coords}")
    return "0 " + " ".join(f"{value:.6f}" for value in coords)


def convert_json(images_root: Path, json_path: Path) -> list[ConvertedImage]:
    data = load_coco(json_path)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    converted: list[ConvertedImage] = []
    skipped_no_labels = 0
    images = sorted(data["images"], key=lambda item: item["file_name"])
    for idx, image in enumerate(images, start=1):
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        labels: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            label = annotation_to_obb_line(annotation, width, height)
            if label is not None:
                labels.append(label)

        if not labels:
            skipped_no_labels += 1
            print(
                f"[{json_path.name}] skipped item {idx}/{len(images)}: "
                f"{image['file_name']} reason=no_valid_obb"
            )
            continue

        converted.append(
            ConvertedImage(
                source_image=image_path_for(images_root, image["file_name"]),
                output_name=image["file_name"],
                labels=tuple(labels),
                source_json=json_path.name,
            )
        )
        print(
            f"[{json_path.name}] finished item {idx}/{len(images)}: "
            f"{image['file_name']} objects={len(labels)}"
        )

    if skipped_no_labels:
        print(f"[{json_path.name}] skipped no-label images: {skipped_no_labels}")
    return converted


def split_validation(records: list[ConvertedImage], valid_ratio: float, seed: int) -> tuple[list[ConvertedImage], list[ConvertedImage]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("--valid-ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    valid_count = round(len(shuffled) * valid_ratio)
    valid = sorted(shuffled[:valid_count], key=lambda item: item.output_name)
    test = sorted(shuffled[valid_count:], key=lambda item: item.output_name)
    return valid, test


def prepare_output(output: Path, zip_path: Path, force: bool) -> None:
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists, pass --force to replace: {output}")
        shutil.rmtree(output)
    if zip_path.exists():
        if not force:
            raise FileExistsError(f"Zip exists, pass --force to replace: {zip_path}")
        zip_path.unlink()
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"""# D2S single-class product OBB dataset converted from COCO masks
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: {CLASS_NAME}
""",
        encoding="utf-8",
    )


def validate_label_text(label_text: str, label_path: Path) -> int:
    rows = 0
    for line_no, line in enumerate(label_text.strip().splitlines(), start=1):
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_no} expected 9 OBB tokens, got {len(parts)}")
        if int(parts[0]) != 0:
            raise ValueError(f"{label_path}:{line_no} expected class 0, got {parts[0]}")
        coords = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{label_path}:{line_no} coordinate outside [0, 1]")
        rows += 1
    if rows == 0:
        raise ValueError(f"{label_path} would be empty")
    return rows


def copy_split(output: Path, split: str, records: list[ConvertedImage]) -> dict[str, int]:
    image_out = output / split / "images"
    label_out = output / split / "labels"
    seen_names: set[str] = set()
    objects = 0

    for idx, record in enumerate(records, start=1):
        if record.output_name in seen_names:
            raise ValueError(f"Duplicate output image name in {split}: {record.output_name}")
        seen_names.add(record.output_name)

        image_dst = image_out / record.output_name
        label_dst = label_out / f"{Path(record.output_name).stem}.txt"
        label_text = "\n".join(record.labels) + "\n"
        objects += validate_label_text(label_text, label_dst)
        shutil.copy2(record.source_image, image_dst)
        label_dst.write_text(label_text, encoding="utf-8")
        print(f"[{split}] finished item {idx}/{len(records)}: {record.output_name} objects={len(record.labels)}")

    return {"images": len(records), "labels": len(records), "objects": objects}


def validate_dataset(output: Path, summary: dict[str, dict[str, int]]) -> None:
    for split, counts in summary.items():
        image_files = sorted((output / split / "images").glob("*"))
        label_files = sorted((output / split / "labels").glob("*.txt"))
        if len(image_files) != counts["images"]:
            raise AssertionError(f"{split} image count mismatch: {len(image_files)} != {counts['images']}")
        if len(label_files) != counts["labels"]:
            raise AssertionError(f"{split} label count mismatch: {len(label_files)} != {counts['labels']}")
        missing = [path for path in image_files if not (output / split / "labels" / f"{path.stem}.txt").exists()]
        if missing:
            preview = ", ".join(path.name for path in missing[:5])
            raise AssertionError(f"{split} images missing labels: {preview}")
        for label_path in label_files:
            validate_label_text(label_path.read_text(encoding="utf-8"), label_path)

    data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
    if f"0: {CLASS_NAME}" not in data_yaml:
        raise AssertionError("data.yaml does not expose the single product class")
    if "nc: 1" not in data_yaml:
        raise AssertionError("data.yaml does not declare nc: 1")


def write_manifest(
    output: Path,
    args: argparse.Namespace,
    summary: dict[str, dict[str, int]],
    train_sources: dict[str, int],
    validation_source_images: int,
) -> None:
    manifest = {
        "source_images": str(args.images),
        "source_annotations": str(args.annotations),
        "class_mapping": {"all_d2s_categories": CLASS_NAME},
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "train_jsons": list(TRAIN_JSONS),
        "validation_json": VALIDATION_JSON,
        "ignored_test_info_jsons": "D2S test-info files contain no annotations",
        "train_source_images": train_sources,
        "validation_source_images": validation_source_images,
        "summary": summary,
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    file_paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, path in enumerate(file_paths, start=1):
            zf.write(path, path.relative_to(source_dir.parent))
            if idx % 500 == 0 or idx == len(file_paths):
                print(f"[zip] finished item {idx}/{len(file_paths)}: {path.relative_to(source_dir.parent)}")


def main() -> None:
    args = parse_args()
    prepare_output(args.output, args.zip, args.force)

    train_records: list[ConvertedImage] = []
    train_sources: dict[str, int] = {}
    for json_name in TRAIN_JSONS:
        records = convert_json(args.images, args.annotations / json_name)
        train_sources[json_name] = len(records)
        train_records.extend(records)
    train_records = sorted(train_records, key=lambda item: item.output_name)

    validation_records = convert_json(args.images, args.annotations / VALIDATION_JSON)
    valid_records, test_records = split_validation(validation_records, args.valid_ratio, args.seed)

    write_data_yaml(args.output)
    summary = {
        "train": copy_split(args.output, "train", train_records),
        "valid": copy_split(args.output, "valid", valid_records),
        "test": copy_split(args.output, "test", test_records),
    }
    validate_dataset(args.output, summary)
    write_manifest(args.output, args, summary, train_sources, len(validation_records))
    zip_dir(args.output, args.zip)

    print("Conversion summary:")
    for split, values in summary.items():
        print(f"  {split}: {values}")
    print(f"Dataset directory: {args.output}")
    print(f"Zip: {args.zip}")


if __name__ == "__main__":
    main()
