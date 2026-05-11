from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class CropSample:
    source: str
    split: str
    image_name: str
    annotation_index: int
    bbox: BBox
    transcription: str


@dataclass(slots=True)
class Counters:
    real_train_images: int = 0
    real_val_images: int = 0
    real_train_crops: int = 0
    real_val_crops: int = 0
    synth_train_crops_available: int = 0
    synth_train_crops_sampled: int = 0
    real_test_crops: int = 0
    exported_train: int = 0
    exported_val: int = 0
    exported_test: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced Products real/synth date recognition dataset")
    parser.add_argument("--products-real-root", type=Path, default=Path("data/SMARTBITE-DATASET/Products-Real"))
    parser.add_argument("--products-synth-root", type=Path, default=Path("data/SMARTBITE-DATASET/Products-Synth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-train-ratio", type=float, default=0.85)
    parser.add_argument("--synth-ratio", type=float, default=1.0)
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pad-x-ratio", type=float, default=0.08)
    parser.add_argument("--pad-y-ratio", type=float, default=0.20)
    parser.add_argument("--min-crop-width", type=int, default=8)
    parser.add_argument("--min-crop-height", type=int, default=8)
    return parser.parse_args(argv)


def info(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr, flush=True)


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


def padded_bbox(
    bbox: BBox,
    *,
    image_width: int,
    image_height: int,
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> BBox | None:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = int(round(width * pad_x_ratio))
    pad_y = int(round(height * pad_y_ratio))
    clamped = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


def split_image_keys(image_keys: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    shuffled = sorted(image_keys)
    rng.shuffle(shuffled)
    cutoff = int(len(shuffled) * train_ratio)
    return set(shuffled[:cutoff]), set(shuffled[cutoff:])


def collect_samples(
    *,
    annotations: dict[str, dict[str, Any]],
    source: str,
    split: str,
    allowed_classes: set[str],
    image_filter: set[str] | None = None,
) -> list[CropSample]:
    samples: list[CropSample] = []
    for image_name in sorted(annotations):
        if image_filter is not None and image_name not in image_filter:
            continue
        anns = annotations[image_name].get("ann")
        if not isinstance(anns, list):
            continue
        for ann_index, ann in enumerate(anns):
            if not isinstance(ann, dict):
                continue
            ann_cls = str(ann.get("cls", "")).strip().lower()
            if ann_cls not in allowed_classes:
                continue
            transcription_raw = ann.get("transcription")
            if transcription_raw is None or not str(transcription_raw).strip():
                continue
            bbox = parse_bbox(ann.get("bbox"))
            if bbox is None:
                continue
            samples.append(
                CropSample(
                    source=source,
                    split=split,
                    image_name=image_name,
                    annotation_index=ann_index,
                    bbox=bbox,
                    transcription=str(transcription_raw),
                )
            )
    return samples


def source_image_path(sample: CropSample, products_real_root: Path, products_synth_root: Path) -> Path:
    if sample.source == "real_train":
        return products_real_root / "train" / "images" / sample.image_name
    if sample.source == "real_test":
        return products_real_root / "evaluation" / "images" / sample.image_name
    if sample.source == "synth":
        return products_synth_root / "images" / sample.image_name
    raise ValueError(f"unknown sample source: {sample.source}")


def output_filename(sample: CropSample) -> str:
    stem = Path(sample.image_name).stem
    return f"{sample.source}__{stem}__ann{sample.annotation_index}.jpg"


def open_label_files(output_dir: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("train_images", "val_images", "test_images"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)
    return {
        "train": (output_dir / "train_label.txt").open("w", encoding="utf-8"),
        "val": (output_dir / "val_label.txt").open("w", encoding="utf-8"),
        "test": (output_dir / "test_label.txt").open("w", encoding="utf-8"),
        "metadata": (output_dir / "metadata.jsonl").open("w", encoding="utf-8"),
    }


def close_files(files: dict[str, Any]) -> None:
    for handle in files.values():
        handle.close()


def export_samples(
    *,
    samples: list[CropSample],
    products_real_root: Path,
    products_synth_root: Path,
    output_dir: Path,
    label_files: dict[str, Any],
    counters: Counters,
    pad_x_ratio: float,
    pad_y_ratio: float,
    min_crop_width: int,
    min_crop_height: int,
    dry_run: bool,
) -> None:
    for sample in samples:
        image_path = source_image_path(sample, products_real_root, products_synth_root)
        if not image_path.exists():
            counters.skip("missing_source_image")
            warn(f"missing source image: {image_path}")
            continue
        try:
            with Image.open(image_path) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
                crop_bbox = padded_bbox(
                    sample.bbox,
                    image_width=image.width,
                    image_height=image.height,
                    pad_x_ratio=pad_x_ratio,
                    pad_y_ratio=pad_y_ratio,
                )
                if crop_bbox is None:
                    counters.skip("invalid_padded_bbox")
                    continue
                crop_w = crop_bbox[2] - crop_bbox[0]
                crop_h = crop_bbox[3] - crop_bbox[1]
                if crop_w < min_crop_width or crop_h < min_crop_height:
                    counters.skip("crop_too_small")
                    continue
                rel_dir = f"{sample.split}_images"
                rel_path = f"{rel_dir}/{output_filename(sample)}"
                if not dry_run:
                    crop = image.crop(crop_bbox)
                    crop.save(output_dir / rel_path, format="JPEG")
                    label_files[sample.split].write(f"{rel_path}\t{sample.transcription}\n")
                    label_files["metadata"].write(
                        json.dumps(
                            {
                                "task": "recognition",
                                "split": sample.split,
                                "source": sample.source,
                                "source_image": sample.image_name,
                                "annotation_index": sample.annotation_index,
                                "bbox_original": list(sample.bbox),
                                "bbox_used": list(crop_bbox),
                                "output_path": rel_path,
                                "transcription": sample.transcription,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if sample.split == "train":
                    counters.exported_train += 1
                elif sample.split == "val":
                    counters.exported_val += 1
                else:
                    counters.exported_test += 1
        except Exception as exc:
            counters.skip("image_open_or_crop_failed")
            warn(f"failed to export {image_path}: {exc}")


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
    if args.pad_x_ratio < 0 or args.pad_y_ratio < 0:
        raise ValueError("--pad-x-ratio and --pad-y-ratio must be >= 0")
    if args.min_crop_width <= 0 or args.min_crop_height <= 0:
        raise ValueError("--min-crop-width and --min-crop-height must be > 0")
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

    real_train_samples = collect_samples(
        annotations=real_train_annotations,
        source="real_train",
        split="train",
        allowed_classes={"date"},
        image_filter=real_train_keys,
    )
    real_val_samples = collect_samples(
        annotations=real_train_annotations,
        source="real_train",
        split="val",
        allowed_classes={"date"},
        image_filter=real_val_keys,
    )
    synth_available = collect_samples(
        annotations=synth_annotations,
        source="synth",
        split="train",
        allowed_classes={"date"},
    )
    real_test_samples = collect_samples(
        annotations=real_eval_annotations,
        source="real_test",
        split="test",
        allowed_classes={"date", "exp"},
    )

    counters.real_train_crops = len(real_train_samples)
    counters.real_val_crops = len(real_val_samples)
    counters.synth_train_crops_available = len(synth_available)
    counters.real_test_crops = len(real_test_samples)

    synth_target = min(len(synth_available), int(round(len(real_train_samples) * args.synth_ratio)))
    rng = random.Random(args.seed)
    synth_train_samples = rng.sample(synth_available, synth_target) if synth_target else []
    synth_train_samples.sort(key=lambda item: (item.image_name, item.annotation_index))
    counters.synth_train_crops_sampled = len(synth_train_samples)

    all_samples = real_train_samples + synth_train_samples + real_val_samples + real_test_samples
    label_files = open_label_files(args.output_dir, args.dry_run)
    try:
        export_samples(
            samples=all_samples,
            products_real_root=args.products_real_root,
            products_synth_root=args.products_synth_root,
            output_dir=args.output_dir,
            label_files=label_files,
            counters=counters,
            pad_x_ratio=args.pad_x_ratio,
            pad_y_ratio=args.pad_y_ratio,
            min_crop_width=args.min_crop_width,
            min_crop_height=args.min_crop_height,
            dry_run=args.dry_run,
        )
    finally:
        close_files(label_files)

    summary = {
        "dataset": "products_date_recognition_balanced",
        "config": {
            "products_real_root": str(args.products_real_root),
            "products_synth_root": str(args.products_synth_root),
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "real_train_ratio": args.real_train_ratio,
            "synth_ratio": args.synth_ratio,
            "pad_x_ratio": args.pad_x_ratio,
            "pad_y_ratio": args.pad_y_ratio,
            "min_crop_width": args.min_crop_width,
            "min_crop_height": args.min_crop_height,
            "zip_output": str(args.zip_output) if args.zip_output else None,
            "dry_run": args.dry_run,
        },
        "counts": {
            "real_train_images": counters.real_train_images,
            "real_val_images": counters.real_val_images,
            "real_train_crops": counters.real_train_crops,
            "real_val_crops": counters.real_val_crops,
            "synth_train_crops_available": counters.synth_train_crops_available,
            "synth_train_crops_sampled": counters.synth_train_crops_sampled,
            "real_test_crops": counters.real_test_crops,
            "exported_train": counters.exported_train,
            "exported_val": counters.exported_val,
            "exported_test": counters.exported_test,
            "skipped_total": counters.skipped_total,
            "skipped_by_reason": counters.skipped_by_reason,
        },
    }
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.zip_output is not None:
            write_zip(args.output_dir, args.zip_output, args.dry_run)

    print("Products date recognition dataset export summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
