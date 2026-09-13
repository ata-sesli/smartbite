from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class DetectionImage:
    source: str
    split: str
    image_name: str
    boxes: list[BBox]


@dataclass(slots=True)
class Counters:
    real_train_images: int = 0
    real_val_images: int = 0
    real_train_images_with_dates: int = 0
    real_val_images_with_dates: int = 0
    synth_train_images_available: int = 0
    synth_train_images_sampled: int = 0
    real_test_images_with_dates: int = 0
    exported_train_images: int = 0
    exported_val_images: int = 0
    exported_test_images: int = 0
    exported_train_boxes: int = 0
    exported_val_boxes: int = 0
    exported_test_boxes: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced Products real/synth date detection dataset")
    parser.add_argument("--products-real-root", type=Path, default=Path("data/SMARTBITE-DATASET/Products-Real"))
    parser.add_argument("--products-synth-root", type=Path, default=Path("data/SMARTBITE-DATASET/Products-Synth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.85)
    parser.add_argument("--synth-ratio", type=float, default=1.0)
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--min-box-width", type=int, default=4)
    parser.add_argument("--min-box-height", type=int, default=4)
    return parser.parse_args(argv)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"annotation file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in annotation file: {path} (line {exc.lineno}, col {exc.colno})") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"annotation file must be an object keyed by image name: {path}")
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def parse_bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    coords: list[int] = []
    for raw in value:
        if not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not math.isfinite(number):
            return None
        coords.append(int(round(number)))
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def split_image_keys(image_keys: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(image_keys)
    rng.shuffle(shuffled)
    cutoff = int(len(shuffled) * train_ratio)
    return set(shuffled[:cutoff]), set(shuffled[cutoff:])


def collect_detection_images(
    *,
    annotations: dict[str, dict[str, Any]],
    source: str,
    split: str,
    allowed_classes: set[str],
    min_box_width: int,
    min_box_height: int,
    image_filter: set[str] | None = None,
) -> list[DetectionImage]:
    images: list[DetectionImage] = []
    for image_name in sorted(annotations):
        if image_filter is not None and image_name not in image_filter:
            continue
        anns = annotations[image_name].get("ann")
        if not isinstance(anns, list):
            continue
        boxes: list[BBox] = []
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            ann_cls = str(ann.get("cls", "")).strip().lower()
            if ann_cls not in allowed_classes:
                continue
            bbox = parse_bbox(ann.get("bbox"))
            if bbox is None:
                continue
            if bbox[2] - bbox[0] < min_box_width or bbox[3] - bbox[1] < min_box_height:
                continue
            boxes.append(bbox)
        if boxes:
            images.append(DetectionImage(source=source, split=split, image_name=image_name, boxes=boxes))
    return images


def source_image_path(item: DetectionImage, products_real_root: Path, products_synth_root: Path) -> Path:
    if item.source == "real_train":
        return products_real_root / "train" / "images" / item.image_name
    if item.source == "real_test":
        return products_real_root / "evaluation" / "images" / item.image_name
    if item.source == "synth":
        return products_synth_root / "images" / item.image_name
    raise ValueError(f"unknown source: {item.source}")


def output_filename(item: DetectionImage) -> str:
    return f"{item.source}__{Path(item.image_name).name}"


def points_for_bbox(bbox: BBox) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]


def ensure_output_layout(output_dir: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("train_images", "val_images", "test_images"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    return {
        "train": (output_dir / "train_det_label.txt").open("w", encoding="utf-8"),
        "val": (output_dir / "val_det_label.txt").open("w", encoding="utf-8"),
        "test": (output_dir / "test_det_label.txt").open("w", encoding="utf-8"),
        "metadata": (output_dir / "metadata.jsonl").open("w", encoding="utf-8"),
    }


def close_files(files: dict[str, Any]) -> None:
    for handle in files.values():
        handle.close()


def copy_or_symlink(source: Path, destination: Path, copy_mode: str) -> str:
    if copy_mode == "symlink":
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        rel_src = os.path.relpath(source, destination.parent)
        os.symlink(rel_src, destination)
        return "symlink"
    with Image.open(source) as image_raw:
        image = ImageOps.exif_transpose(image_raw).convert("RGB")
        image.save(destination, format="JPEG")
    return "copy"


def export_detection_images(
    *,
    items: list[DetectionImage],
    products_real_root: Path,
    products_synth_root: Path,
    output_dir: Path,
    files: dict[str, Any],
    counters: Counters,
    copy_mode: str,
    dry_run: bool,
) -> None:
    for item in items:
        image_path = source_image_path(item, products_real_root, products_synth_root)
        if not image_path.exists():
            counters.skip("missing_source_image")
            warn(f"missing source image: {image_path}")
            continue

        rel_dir = f"{item.split}_images"
        rel_path = f"{rel_dir}/{output_filename(item)}"
        if not dry_run:
            copy_or_symlink(image_path, output_dir / rel_path, copy_mode)
            records = [{"transcription": "date", "points": points_for_bbox(box)} for box in item.boxes]
            files[item.split].write(f"{rel_path}\t{json.dumps(records, ensure_ascii=False)}\n")
            files["metadata"].write(
                json.dumps(
                    {
                        "task": "detection",
                        "split": item.split,
                        "source": item.source,
                        "source_image": item.image_name,
                        "output_path": rel_path,
                        "boxes": [list(box) for box in item.boxes],
                        "box_count": len(item.boxes),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if item.split == "train":
            counters.exported_train_images += 1
            counters.exported_train_boxes += len(item.boxes)
        elif item.split == "val":
            counters.exported_val_images += 1
            counters.exported_val_boxes += len(item.boxes)
        else:
            counters.exported_test_images += 1
            counters.exported_test_boxes += len(item.boxes)


def write_zip(output_dir: Path, zip_output: Path, dry_run: bool) -> None:
    if dry_run:
        return
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_output.resolve():
                archive.write(path, path.relative_to(output_dir))


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.real_train_ratio < 1.0:
        raise ValueError("--real-train-ratio must be between 0 and 1")
    if args.synth_ratio < 0:
        raise ValueError("--synth-ratio must be >= 0")
    if args.min_box_width <= 0 or args.min_box_height <= 0:
        raise ValueError("--min-box-width and --min-box-height must be > 0")
    for path in (
        args.products_real_root / "train" / "annotations.json",
        args.products_real_root / "evaluation" / "annotations.json",
        args.products_synth_root / "annotations.json",
    ):
        if not path.exists():
            raise ValueError(f"missing required annotation file: {path}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        real_train_annotations = load_annotations(args.products_real_root / "train" / "annotations.json")
        real_eval_annotations = load_annotations(args.products_real_root / "evaluation" / "annotations.json")
        synth_annotations = load_annotations(args.products_synth_root / "annotations.json")
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    counters = Counters()
    real_train_keys, real_val_keys = split_image_keys(sorted(real_train_annotations), args.real_train_ratio, args.seed)
    counters.real_train_images = len(real_train_keys)
    counters.real_val_images = len(real_val_keys)

    real_train_items = collect_detection_images(
        annotations=real_train_annotations,
        source="real_train",
        split="train",
        allowed_classes={"date"},
        image_filter=real_train_keys,
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
    )
    real_val_items = collect_detection_images(
        annotations=real_train_annotations,
        source="real_train",
        split="val",
        allowed_classes={"date"},
        image_filter=real_val_keys,
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
    )
    synth_available = collect_detection_images(
        annotations=synth_annotations,
        source="synth",
        split="train",
        allowed_classes={"date"},
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
    )
    real_test_items = collect_detection_images(
        annotations=real_eval_annotations,
        source="real_test",
        split="test",
        allowed_classes={"date", "exp"},
        min_box_width=args.min_box_width,
        min_box_height=args.min_box_height,
    )

    counters.real_train_images_with_dates = len(real_train_items)
    counters.real_val_images_with_dates = len(real_val_items)
    counters.synth_train_images_available = len(synth_available)
    counters.real_test_images_with_dates = len(real_test_items)

    synth_target = min(len(synth_available), int(round(len(real_train_items) * args.synth_ratio)))
    rng = random.Random(args.seed)
    synth_train_items = rng.sample(synth_available, synth_target) if synth_target else []
    synth_train_items.sort(key=lambda item: item.image_name)
    counters.synth_train_images_sampled = len(synth_train_items)

    all_items = real_train_items + synth_train_items + real_val_items + real_test_items
    files = ensure_output_layout(args.output_dir, args.dry_run)
    try:
        export_detection_images(
            items=all_items,
            products_real_root=args.products_real_root,
            products_synth_root=args.products_synth_root,
            output_dir=args.output_dir,
            files=files,
            counters=counters,
            copy_mode=args.copy_mode,
            dry_run=args.dry_run,
        )
    finally:
        close_files(files)

    summary = {
        "dataset": "products_date_detection_balanced",
        "config": {
            "products_real_root": str(args.products_real_root),
            "products_synth_root": str(args.products_synth_root),
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "real_train_ratio": args.real_train_ratio,
            "synth_ratio": args.synth_ratio,
            "copy_mode": args.copy_mode,
            "min_box_width": args.min_box_width,
            "min_box_height": args.min_box_height,
            "zip_output": str(args.zip_output) if args.zip_output else None,
            "dry_run": args.dry_run,
        },
        "counts": {
            "real_train_images": counters.real_train_images,
            "real_val_images": counters.real_val_images,
            "real_train_images_with_dates": counters.real_train_images_with_dates,
            "real_val_images_with_dates": counters.real_val_images_with_dates,
            "synth_train_images_available": counters.synth_train_images_available,
            "synth_train_images_sampled": counters.synth_train_images_sampled,
            "real_test_images_with_dates": counters.real_test_images_with_dates,
            "exported_train_images": counters.exported_train_images,
            "exported_val_images": counters.exported_val_images,
            "exported_test_images": counters.exported_test_images,
            "exported_train_boxes": counters.exported_train_boxes,
            "exported_val_boxes": counters.exported_val_boxes,
            "exported_test_boxes": counters.exported_test_boxes,
            "skipped_total": counters.skipped_total,
            "skipped_by_reason": counters.skipped_by_reason,
        },
    }
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.zip_output is not None:
            write_zip(args.output_dir, args.zip_output, args.dry_run)

    print("Products date detection dataset export summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
