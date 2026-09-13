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

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class CropTruthItem:
    index: int
    image_path: Path
    label: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ProductDateSample:
    split: str
    image_path: Path
    image_name: str
    annotation_index: int
    bbox: BBox
    label: str


@dataclass(slots=True)
class Counters:
    crop_truth_items: int = 0
    expdate_products_available: int = 0
    expdate_products_train_sampled: int = 0
    expdate_products_val_sampled: int = 0
    exported_train: int = 0
    exported_val: int = 0
    exported_test: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SVTRv2 hard-case ExpDate recognition dataset")
    parser.add_argument("--crop-truth-dir", type=Path, default=Path("artifacts/svtr_hardcase_crop_truth_20_selected"))
    parser.add_argument("--expdate-root", type=Path, default=Path("data/SMARTBITE-DATASET"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-truth-variants-per-crop", type=int, default=25)
    parser.add_argument("--expdate-train-count", type=int, default=650)
    parser.add_argument("--expdate-val-count", type=int, default=50)
    parser.add_argument("--pad-x-ratio", type=float, default=0.08)
    parser.add_argument("--pad-y-ratio", type=float, default=0.20)
    parser.add_argument("--min-crop-width", type=int, default=8)
    parser.add_argument("--min-crop-height", type=int, default=8)
    return parser.parse_args(argv)


def info(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr, flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr, flush=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in {path} at line {exc.lineno}, col {exc.colno}") from exc


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


def load_crop_truth_items(crop_truth_dir: Path) -> list[CropTruthItem]:
    manifest = load_json(crop_truth_dir / "manifest.json")
    raw_items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(raw_items, list):
        raise ValueError(f"crop truth manifest must contain an items list: {crop_truth_dir / 'manifest.json'}")

    items: list[CropTruthItem] = []
    for position, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        rel_path = raw.get("crop_path") or raw.get("output_path")
        label = str(raw.get("label", "")).strip()
        if not rel_path or not label:
            continue
        image_path = crop_truth_dir / str(rel_path)
        if not image_path.exists():
            raise ValueError(f"crop truth image listed in manifest is missing: {image_path}")
        items.append(
            CropTruthItem(
                index=int(raw.get("index") or position),
                image_path=image_path,
                label=label,
                metadata=raw,
            )
        )
    if not items:
        raise ValueError(f"no usable crop truth items found in {crop_truth_dir / 'manifest.json'}")
    return items


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"annotation file must be an object keyed by image name: {path}")
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def collect_product_date_samples(expdate_root: Path) -> list[ProductDateSample]:
    products_root = expdate_root / "Products-Real"
    sources = [
        (products_root / "train" / "annotations.json", products_root / "train" / "images"),
        (products_root / "evaluation" / "annotations.json", products_root / "evaluation" / "images"),
    ]
    samples: list[ProductDateSample] = []
    for annotation_path, image_dir in sources:
        annotations = load_annotations(annotation_path)
        for image_name in sorted(annotations):
            anns = annotations[image_name].get("ann")
            if not isinstance(anns, list):
                continue
            for annotation_index, ann in enumerate(anns):
                if not isinstance(ann, dict):
                    continue
                cls = str(ann.get("cls", "")).strip().lower()
                if cls not in {"date", "exp"}:
                    continue
                label = str(ann.get("transcription", "")).strip()
                if not label or not any(ch.isdigit() for ch in label):
                    continue
                bbox = parse_bbox(ann.get("bbox"))
                if bbox is None:
                    continue
                image_path = image_dir / image_name
                samples.append(
                    ProductDateSample(
                        split="train",
                        image_path=image_path,
                        image_name=image_name,
                        annotation_index=annotation_index,
                        bbox=bbox,
                        label=label,
                    )
                )
    return samples


def ensure_output_dirs(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for dirname in ("train_images", "val_images", "test_images"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)


def label_files(output_dir: Path) -> dict[str, Any]:
    return {
        "train": (output_dir / "train_label.txt").open("w", encoding="utf-8"),
        "val": (output_dir / "val_label.txt").open("w", encoding="utf-8"),
        "test": (output_dir / "test_label.txt").open("w", encoding="utf-8"),
        "metadata": (output_dir / "metadata.jsonl").open("w", encoding="utf-8"),
    }


def close_files(files: dict[str, Any]) -> None:
    for handle in files.values():
        handle.close()


def write_record(
    *,
    files: dict[str, Any],
    split: str,
    rel_path: str,
    label: str,
    metadata: dict[str, Any],
) -> None:
    files[split].write(f"{rel_path}\t{label}\n")
    files["metadata"].write(json.dumps({**metadata, "output_path": rel_path, "label": label}, ensure_ascii=False) + "\n")


def augment_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    crop = image.convert("RGB")
    if crop.width > 8 and crop.height > 8:
        jitter_x = max(1, int(round(crop.width * 0.03)))
        jitter_y = max(1, int(round(crop.height * 0.08)))
        left = rng.randint(0, jitter_x)
        upper = rng.randint(0, jitter_y)
        right = crop.width - rng.randint(0, jitter_x)
        lower = crop.height - rng.randint(0, jitter_y)
        if right - left >= 8 and lower - upper >= 8:
            crop = crop.crop((left, upper, right, lower))

    angle = rng.uniform(-4.5, 4.5)
    fill = tuple(int(channel) for channel in ImageStatLite.mean_rgb(crop))
    crop = crop.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)

    crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.78, 1.22))
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.70, 1.35))
    crop = ImageEnhance.Sharpness(crop).enhance(rng.uniform(0.65, 1.45))
    if rng.random() < 0.35:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.65)))
    if rng.random() < 0.25:
        pad_x = rng.randint(1, max(1, int(crop.width * 0.03)))
        pad_y = rng.randint(1, max(1, int(crop.height * 0.08)))
        crop = ImageOps.expand(crop, border=(pad_x, pad_y, pad_x, pad_y), fill=fill)
    return crop


class ImageStatLite:
    @staticmethod
    def mean_rgb(image: Image.Image) -> tuple[int, int, int]:
        tiny = image.resize((1, 1), Image.Resampling.BILINEAR)
        return tiny.getpixel((0, 0))[:3]


def save_jpeg(image: Image.Image, path: Path, rng: random.Random | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = 92 if rng is None else rng.randint(68, 94)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def export_crop_truth(
    *,
    items: list[CropTruthItem],
    output_dir: Path,
    files: dict[str, Any],
    variants_per_crop: int,
    seed: int,
    counters: Counters,
) -> None:
    for item in items:
        try:
            with Image.open(item.image_path) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
                original_rel = f"test_images/crop_truth_{item.index:02d}__original.jpg"
                save_jpeg(image, output_dir / original_rel)
                write_record(
                    files=files,
                    split="test",
                    rel_path=original_rel,
                    label=item.label,
                    metadata={
                        "split": "test",
                        "source": "crop_truth_original",
                        "crop_truth_index": item.index,
                        "source_image": str(item.image_path),
                        "source_metadata": item.metadata,
                    },
                )
                counters.exported_test += 1

                for variant_idx in range(variants_per_crop):
                    rng = random.Random(f"{seed}:{item.index}:{variant_idx}")
                    variant = augment_crop(image, rng)
                    rel_path = f"train_images/crop_truth_{item.index:02d}__aug_{variant_idx:03d}.jpg"
                    save_jpeg(variant, output_dir / rel_path, rng)
                    write_record(
                        files=files,
                        split="train",
                        rel_path=rel_path,
                        label=item.label,
                        metadata={
                            "split": "train",
                            "source": "crop_truth_aug",
                            "crop_truth_index": item.index,
                            "variant_index": variant_idx,
                            "source_image": str(item.image_path),
                            "source_metadata": item.metadata,
                        },
                    )
                    counters.exported_train += 1
        except Exception as exc:
            counters.skip("crop_truth_export_failed")
            warn(f"failed to export crop truth {item.image_path}: {exc}")


def export_product_samples(
    *,
    samples: list[ProductDateSample],
    output_dir: Path,
    files: dict[str, Any],
    counters: Counters,
    pad_x_ratio: float,
    pad_y_ratio: float,
    min_crop_width: int,
    min_crop_height: int,
) -> None:
    for sample in samples:
        if not sample.image_path.exists():
            counters.skip("missing_product_image")
            warn(f"missing product image: {sample.image_path}")
            continue
        try:
            with Image.open(sample.image_path) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
                crop_bbox = padded_bbox(
                    sample.bbox,
                    image_width=image.width,
                    image_height=image.height,
                    pad_x_ratio=pad_x_ratio,
                    pad_y_ratio=pad_y_ratio,
                )
                if crop_bbox is None:
                    counters.skip("invalid_product_bbox")
                    continue
                crop_w = crop_bbox[2] - crop_bbox[0]
                crop_h = crop_bbox[3] - crop_bbox[1]
                if crop_w < min_crop_width or crop_h < min_crop_height:
                    counters.skip("product_crop_too_small")
                    continue
                stem = Path(sample.image_name).stem
                rel_path = f"{sample.split}_images/expdate_{stem}__ann{sample.annotation_index}.jpg"
                save_jpeg(image.crop(crop_bbox), output_dir / rel_path)
                write_record(
                    files=files,
                    split=sample.split,
                    rel_path=rel_path,
                    label=sample.label,
                    metadata={
                        "split": sample.split,
                        "source": "expdate_products_real",
                        "source_image": str(sample.image_path),
                        "annotation_index": sample.annotation_index,
                        "bbox_original": list(sample.bbox),
                        "bbox_used": list(crop_bbox),
                    },
                )
                if sample.split == "train":
                    counters.exported_train += 1
                else:
                    counters.exported_val += 1
        except Exception as exc:
            counters.skip("product_export_failed")
            warn(f"failed to export product sample {sample.image_path}: {exc}")


def sample_product_splits(
    samples: list[ProductDateSample],
    *,
    train_count: int,
    val_count: int,
    seed: int,
) -> tuple[list[ProductDateSample], list[ProductDateSample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val = shuffled[: min(val_count, len(shuffled))]
    train_pool = shuffled[len(val) :]
    train = train_pool[: min(train_count, len(train_pool))]
    for sample in train:
        sample.split = "train"
    for sample in val:
        sample.split = "val"
    return sorted(train, key=lambda item: (item.image_name, item.annotation_index)), sorted(
        val, key=lambda item: (item.image_name, item.annotation_index)
    )


def write_contact_sheet(output_dir: Path) -> None:
    label_lines = []
    for label_file in ("test_label.txt", "train_label.txt", "val_label.txt"):
        label_lines.extend((output_dir / label_file).read_text(encoding="utf-8").splitlines())
    rows = []
    for line in label_lines[:40]:
        if "\t" not in line:
            continue
        rel_path, label = line.split("\t", 1)
        path = output_dir / rel_path
        if path.exists():
            rows.append((path, label))
    if not rows:
        return

    thumb_w, thumb_h = 220, 70
    margin = 16
    text_h = 24
    columns = 4
    width = columns * thumb_w + (columns + 1) * margin
    height = math.ceil(len(rows) / columns) * (thumb_h + text_h + margin) + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for idx, (path, label) in enumerate(rows):
        with Image.open(path) as image_raw:
            image = ImageOps.exif_transpose(image_raw).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = margin + (idx % columns) * (thumb_w + margin)
            y = margin + (idx // columns) * (thumb_h + text_h + margin)
            sheet.paste(image, (x, y))
            draw.text((x, y + thumb_h + 4), label[:32], fill=(20, 20, 20), font=font)
    sheet.save(output_dir / "contact_sheet.jpg", format="JPEG", quality=92)


def write_zip(output_dir: Path, zip_output: Path) -> None:
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.resolve() != zip_output.resolve():
                archive.write(path, path.relative_to(output_dir))


def validate_args(args: argparse.Namespace) -> None:
    if args.crop_truth_variants_per_crop <= 0:
        raise ValueError("--crop-truth-variants-per-crop must be > 0")
    if args.expdate_train_count < 0 or args.expdate_val_count < 0:
        raise ValueError("--expdate-train-count and --expdate-val-count must be >= 0")
    if args.pad_x_ratio < 0 or args.pad_y_ratio < 0:
        raise ValueError("--pad-x-ratio and --pad-y-ratio must be >= 0")
    if args.min_crop_width <= 0 or args.min_crop_height <= 0:
        raise ValueError("--min-crop-width and --min-crop-height must be > 0")
    required = [
        args.crop_truth_dir / "manifest.json",
        args.expdate_root / "Products-Real" / "train" / "annotations.json",
        args.expdate_root / "Products-Real" / "evaluation" / "annotations.json",
    ]
    for path in required:
        if not path.exists():
            raise ValueError(f"missing required input: {path}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        crop_truth_items = load_crop_truth_items(args.crop_truth_dir)
        product_samples = collect_product_date_samples(args.expdate_root)
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    counters = Counters(crop_truth_items=len(crop_truth_items), expdate_products_available=len(product_samples))
    product_train, product_val = sample_product_splits(
        product_samples,
        train_count=args.expdate_train_count,
        val_count=args.expdate_val_count,
        seed=args.seed,
    )
    counters.expdate_products_train_sampled = len(product_train)
    counters.expdate_products_val_sampled = len(product_val)

    ensure_output_dirs(args.output_dir)
    files = label_files(args.output_dir)
    try:
        export_crop_truth(
            items=crop_truth_items,
            output_dir=args.output_dir,
            files=files,
            variants_per_crop=args.crop_truth_variants_per_crop,
            seed=args.seed,
            counters=counters,
        )
        export_product_samples(
            samples=product_train + product_val,
            output_dir=args.output_dir,
            files=files,
            counters=counters,
            pad_x_ratio=args.pad_x_ratio,
            pad_y_ratio=args.pad_y_ratio,
            min_crop_width=args.min_crop_width,
            min_crop_height=args.min_crop_height,
        )
    finally:
        close_files(files)

    summary = {
        "dataset": "svtrv2_hardcase_expdate_rec",
        "config": {
            "crop_truth_dir": str(args.crop_truth_dir),
            "expdate_root": str(args.expdate_root),
            "output_dir": str(args.output_dir),
            "zip_output": str(args.zip_output) if args.zip_output else None,
            "seed": args.seed,
            "crop_truth_variants_per_crop": args.crop_truth_variants_per_crop,
            "expdate_train_count": args.expdate_train_count,
            "expdate_val_count": args.expdate_val_count,
            "pad_x_ratio": args.pad_x_ratio,
            "pad_y_ratio": args.pad_y_ratio,
        },
        "counts": {
            "crop_truth_items": counters.crop_truth_items,
            "expdate_products_available": counters.expdate_products_available,
            "expdate_products_train_sampled": counters.expdate_products_train_sampled,
            "expdate_products_val_sampled": counters.expdate_products_val_sampled,
            "exported_train": counters.exported_train,
            "exported_val": counters.exported_val,
            "exported_test": counters.exported_test,
            "skipped_total": counters.skipped_total,
            "skipped_by_reason": counters.skipped_by_reason,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_contact_sheet(args.output_dir)
    if args.zip_output is not None:
        write_zip(args.output_dir, args.zip_output)

    print("SVTRv2 hard-case ExpDate recognition dataset export summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
