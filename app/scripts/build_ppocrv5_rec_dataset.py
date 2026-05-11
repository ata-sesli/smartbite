from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class Counters:
    images_seen: int = 0
    annotations_seen: int = 0
    exported_main_crops: int = 0
    exported_component_crops: int = 0
    exported_candidate_crops: int = 0
    train_count: int = 0
    val_count: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1


@dataclass(slots=True)
class DatasetWriters:
    train_label_file: Any | None = None
    val_label_file: Any | None = None
    train_metadata_file: Any | None = None
    val_metadata_file: Any | None = None
    components_label_file: Any | None = None
    candidates_metadata_file: Any | None = None

    def close(self) -> None:
        for handle in (
            self.train_label_file,
            self.val_label_file,
            self.train_metadata_file,
            self.val_metadata_file,
            self.components_label_file,
            self.candidates_metadata_file,
        ):
            if handle is not None:
                handle.close()


@dataclass(slots=True)
class ProgressTracker:
    phase: str
    total: int
    step: int = field(init=False)
    next_mark: int = field(init=False)

    def __post_init__(self) -> None:
        self.step = max(1, self.total // 20) if self.total > 0 else 1
        self.next_mark = self.step
        info(f"{self.phase}: 0/{self.total}")

    def tick(self, current: int) -> None:
        if self.total <= 0:
            return
        if current >= self.next_mark or current == self.total:
            pct = (current / self.total) * 100.0
            info(f"{self.phase}: {current}/{self.total} ({pct:.1f}%)")
            while current >= self.next_mark:
                self.next_mark += self.step


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PP-OCRv5 recognition dataset from annotation JSON")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-dmy-metadata", type=parse_bool, default=True)
    parser.add_argument("--generate-component-crops", type=parse_bool, default=False)
    parser.add_argument("--generate-candidate-crops", type=parse_bool, default=False)
    parser.add_argument("--candidate-jitter-count", type=int, default=2)
    parser.add_argument("--candidate-max-pad-px", type=int, default=12)
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--pad-bbox", type=int, default=0)
    parser.add_argument("--min-crop-width", type=int, default=8)
    parser.add_argument("--min-crop-height", type=int, default=8)
    parser.add_argument("--normalize-transcription", choices=["none", "strip", "unicode"], default="none")
    parser.add_argument("--allow-cross-image-split", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def normalize_transcription(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "unicode":
        return unicodedata.normalize("NFKC", text).strip()
    raise ValueError(f"unsupported normalize mode: {mode}")


def parse_bbox(value: Any) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be a list/tuple with 4 numeric values")

    coords: list[int] = []
    for raw in value:
        if not isinstance(raw, (int, float)):
            raise ValueError("bbox coordinates must be numeric")
        num = float(raw)
        if not math.isfinite(num):
            raise ValueError("bbox coordinates must be finite")
        coords.append(int(round(num)))

    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width/height")
    return x1, y1, x2, y2


def clamp_bbox(bbox: BBox, *, pad: int, image_width: int, image_height: int) -> BBox | None:
    x1, y1, x2, y2 = bbox
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    clamped = (
        max(0, min(image_width, x1)),
        max(0, min(image_height, y1)),
        max(0, min(image_width, x2)),
        max(0, min(image_height, y2)),
    )

    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


def bbox_size(bbox: BBox) -> tuple[int, int]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def build_main_crop_filename(image_filename: str, ann_index: int) -> str:
    stem = Path(image_filename).stem
    return f"{stem}__ann{ann_index}.jpg"


def build_component_crop_filename(image_filename: str, ann_index: int, component_index: int, component_cls: str) -> str:
    stem = Path(image_filename).stem
    safe_cls = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in component_cls) or "component"
    return f"{stem}__ann{ann_index}__{safe_cls}{component_index}.jpg"


def build_candidate_crop_filename(image_filename: str, ann_index: int, candidate_index: int) -> str:
    stem = Path(image_filename).stem
    return f"{stem}__ann{ann_index}__cand{candidate_index}.jpg"


def assign_image_level_splits(image_keys: list[str], train_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = image_keys[:]
    rng.shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)

    mapping: dict[str, str] = {}
    for idx, image_key in enumerate(shuffled):
        mapping[image_key] = "train" if idx < train_cutoff else "val"
    return mapping


def assign_crop_level_splits(annotation_keys: list[tuple[str, int]], train_ratio: float, seed: int) -> dict[tuple[str, int], str]:
    rng = random.Random(seed)
    shuffled = annotation_keys[:]
    rng.shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)

    mapping: dict[tuple[str, int], str] = {}
    for idx, ann_key in enumerate(shuffled):
        mapping[ann_key] = "train" if idx < train_cutoff else "val"
    return mapping


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr, flush=True)


def ensure_positive_int(value: int, arg_name: str) -> None:
    if value < 0:
        raise ValueError(f"{arg_name} must be >= 0")


def load_annotation_payload(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"annotation file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in annotation file: {path} (line {exc.lineno}, col {exc.colno})") from exc

    if not isinstance(payload, dict):
        raise ValueError("annotation JSON top-level must be an object keyed by image filename")

    normalized: dict[str, dict[str, Any]] = {}
    for image_filename, image_obj in payload.items():
        if not isinstance(image_filename, str) or not image_filename.strip():
            raise ValueError("annotation JSON has non-string/empty image filename key")
        if not isinstance(image_obj, dict):
            raise ValueError(f"annotation entry for {image_filename!r} must be an object")
        normalized[image_filename] = image_obj

    return normalized


def open_writers(output_dir: Path, generate_component_crops: bool, generate_candidate_crops: bool, dry_run: bool) -> DatasetWriters:
    writers = DatasetWriters()
    if dry_run:
        return writers

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_images").mkdir(parents=True, exist_ok=True)
    (output_dir / "val_images").mkdir(parents=True, exist_ok=True)

    writers.train_label_file = (output_dir / "train_label.txt").open("w", encoding="utf-8")
    writers.val_label_file = (output_dir / "val_label.txt").open("w", encoding="utf-8")
    writers.train_metadata_file = (output_dir / "train_metadata.jsonl").open("w", encoding="utf-8")
    writers.val_metadata_file = (output_dir / "val_metadata.jsonl").open("w", encoding="utf-8")

    if generate_component_crops:
        (output_dir / "components_images").mkdir(parents=True, exist_ok=True)
        writers.components_label_file = (output_dir / "components_label.txt").open("w", encoding="utf-8")

    if generate_candidate_crops:
        (output_dir / "candidates_images").mkdir(parents=True, exist_ok=True)
        writers.candidates_metadata_file = (output_dir / "candidates_metadata.jsonl").open("w", encoding="utf-8")

    return writers


def write_label(writers: DatasetWriters, split: str, rel_path: str, transcription: str, dry_run: bool) -> None:
    if dry_run:
        return
    line = f"{rel_path}\t{transcription}\n"
    if split == "train":
        assert writers.train_label_file is not None
        writers.train_label_file.write(line)
    else:
        assert writers.val_label_file is not None
        writers.val_label_file.write(line)


def write_split_metadata(writers: DatasetWriters, split: str, record: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if split == "train":
        assert writers.train_metadata_file is not None
        writers.train_metadata_file.write(line)
    else:
        assert writers.val_metadata_file is not None
        writers.val_metadata_file.write(line)


def write_candidate_metadata(writers: DatasetWriters, record: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    assert writers.candidates_metadata_file is not None
    writers.candidates_metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_symlink_or_save_crop(
    crop: Image.Image,
    source_image_path: Path,
    destination_path: Path,
    *,
    copy_mode: str,
    crop_bbox: BBox,
    image_width: int,
    image_height: int,
    dry_run: bool,
) -> str:
    if dry_run:
        return copy_mode

    if copy_mode == "symlink":
        full_bbox = (0, 0, image_width, image_height)
        if crop_bbox == full_bbox and source_image_path.suffix.lower() in {".jpg", ".jpeg"}:
            if destination_path.exists() or destination_path.is_symlink():
                destination_path.unlink()
            rel_src = os.path.relpath(source_image_path, destination_path.parent)
            os.symlink(rel_src, destination_path)
            return "symlink"

    crop.save(destination_path, format="JPEG")
    return "copy"


def extract_dmy_components(dmy_ann: Any) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []

    if isinstance(dmy_ann, list):
        for idx, item in enumerate(dmy_ann):
            if isinstance(item, dict):
                components.append(
                    {
                        "component_index": idx,
                        "cls": str(item.get("cls", "component")),
                        "bbox": item.get("bbox"),
                        "transcription": item.get("transcription"),
                    }
                )
        return components

    if isinstance(dmy_ann, dict):
        ordered_keys = ["day", "month", "year"] + [k for k in dmy_ann.keys() if k not in {"day", "month", "year"}]
        index = 0
        for key in ordered_keys:
            value = dmy_ann.get(key)
            if not isinstance(value, dict):
                continue
            components.append(
                {
                    "component_index": index,
                    "cls": str(value.get("cls", key)),
                    "bbox": value.get("bbox"),
                    "transcription": value.get("transcription"),
                }
            )
            index += 1

    return components


def make_base_metadata_record(
    *,
    image_filename: str,
    image_width: int,
    image_height: int,
    ann_index: int,
    ann_cls: str | None,
    raw_bbox: Any,
    clamped_bbox: BBox | None,
    transcription_raw: str | None,
    transcription_normalized: str | None,
    output_crop_path: str | None,
    split: str,
    warnings: list[str],
    include_dmy_metadata: bool,
    dmy_ann: Any,
    is_component_crop: bool,
    is_candidate_crop: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_image_file": image_filename,
        "source_image_size": {"width": image_width, "height": image_height},
        "annotation_index": ann_index,
        "annotation_class": ann_cls,
        "bbox_original": raw_bbox,
        "bbox_used": list(clamped_bbox) if clamped_bbox is not None else None,
        "transcription_raw": transcription_raw,
        "transcription_normalized": transcription_normalized,
        "output_crop_path": output_crop_path,
        "split": split,
        "warnings": warnings,
        "is_component_crop": is_component_crop,
        "is_candidate_crop": is_candidate_crop,
    }
    if include_dmy_metadata and dmy_ann is not None:
        record["dmy_ann"] = dmy_ann
    return record


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError("--train-ratio must be between 0 and 1")

        ensure_positive_int(args.pad_bbox, "--pad-bbox")
        ensure_positive_int(args.min_crop_width, "--min-crop-width")
        ensure_positive_int(args.min_crop_height, "--min-crop-height")
        ensure_positive_int(args.candidate_jitter_count, "--candidate-jitter-count")
        ensure_positive_int(args.candidate_max_pad_px, "--candidate-max-pad-px")

        if not args.images_dir.exists():
            raise ValueError(f"images directory not found: {args.images_dir}")

        payload = load_annotation_payload(args.annotation_json)
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    image_keys = sorted(payload.keys())
    info(f"Loaded annotations for {len(image_keys)} images from {args.annotation_json}")
    all_annotation_keys: list[tuple[str, int]] = []
    for image_key in image_keys:
        anns = payload[image_key].get("ann")
        if isinstance(anns, list):
            for ann_index in range(len(anns)):
                all_annotation_keys.append((image_key, ann_index))
    info(f"Total annotation entries discovered: {len(all_annotation_keys)}")

    image_split_map = assign_image_level_splits(image_keys, args.train_ratio, args.seed)
    crop_split_map = assign_crop_level_splits(all_annotation_keys, args.train_ratio, args.seed)

    counters = Counters()
    rng_candidates = random.Random(args.seed)
    writers = open_writers(args.output_dir, args.generate_component_crops, args.generate_candidate_crops, args.dry_run)

    try:
        image_progress = ProgressTracker("Processing source images", len(image_keys))
        for image_idx, image_filename in enumerate(image_keys, start=1):
            counters.images_seen += 1
            image_progress.tick(image_idx)
            image_obj = payload[image_filename]

            source_width = image_obj.get("width")
            source_height = image_obj.get("height")
            if not isinstance(source_width, (int, float)) or not isinstance(source_height, (int, float)):
                warn(f"image={image_filename} missing/invalid width/height metadata; using loaded image size")
                source_width = None
                source_height = None
            else:
                source_width = int(source_width)
                source_height = int(source_height)

            anns = image_obj.get("ann")
            if not isinstance(anns, list):
                warn(f"image={image_filename} has non-list ann field; skipping image annotations")
                continue

            source_image_path = args.images_dir / image_filename
            image_exists = source_image_path.exists()
            pil_image: Image.Image | None = None

            if image_exists:
                try:
                    pil_image = ImageOps.exif_transpose(Image.open(source_image_path))
                    pil_image.load()
                    if source_width is None or source_height is None:
                        source_width, source_height = pil_image.size
                except Exception as exc:
                    image_exists = False
                    warn(f"image={image_filename} failed to open: {exc}")

            if source_width is None or source_height is None:
                source_width = 0
                source_height = 0

            for ann_index, ann in enumerate(anns):
                counters.annotations_seen += 1
                split = crop_split_map[(image_filename, ann_index)] if args.allow_cross_image_split else image_split_map[image_filename]
                ann_cls = None
                raw_bbox = None
                dmy_ann = None

                if not isinstance(ann, dict):
                    reason = "invalid_annotation_entry"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=None,
                        transcription_normalized=None,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                ann_cls = str(ann.get("cls", ann.get("class", ""))) or None
                raw_bbox = ann.get("bbox")
                dmy_ann = ann.get("dmy_ann")

                transcription_value = ann.get("transcription")
                if transcription_value is None or not str(transcription_value).strip():
                    reason = "missing_or_empty_transcription"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=str(transcription_value) if transcription_value is not None else None,
                        transcription_normalized=None,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                transcription_raw = str(transcription_value)
                transcription_normalized = normalize_transcription(transcription_raw, args.normalize_transcription)
                if not transcription_normalized:
                    reason = "empty_transcription_after_normalization"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=transcription_raw,
                        transcription_normalized=transcription_normalized,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                if not image_exists or pil_image is None:
                    reason = "missing_source_image"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=transcription_raw,
                        transcription_normalized=transcription_normalized,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                try:
                    parsed_bbox = parse_bbox(raw_bbox)
                except ValueError as exc:
                    reason = f"invalid_bbox:{exc}"
                    counters.skip("invalid_bbox")
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=transcription_raw,
                        transcription_normalized=transcription_normalized,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                clamped_bbox = clamp_bbox(
                    parsed_bbox,
                    pad=args.pad_bbox,
                    image_width=pil_image.width,
                    image_height=pil_image.height,
                )
                if clamped_bbox is None:
                    reason = "clamped_bbox_invalid"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason}")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=None,
                        transcription_raw=transcription_raw,
                        transcription_normalized=transcription_normalized,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                crop_w, crop_h = bbox_size(clamped_bbox)
                if crop_w < args.min_crop_width or crop_h < args.min_crop_height:
                    reason = "crop_too_small"
                    counters.skip(reason)
                    warn(f"image={image_filename} ann={ann_index} reason={reason} size=({crop_w}x{crop_h})")
                    metadata = make_base_metadata_record(
                        image_filename=image_filename,
                        image_width=source_width,
                        image_height=source_height,
                        ann_index=ann_index,
                        ann_cls=ann_cls,
                        raw_bbox=raw_bbox,
                        clamped_bbox=clamped_bbox,
                        transcription_raw=transcription_raw,
                        transcription_normalized=transcription_normalized,
                        output_crop_path=None,
                        split=split,
                        warnings=[reason],
                        include_dmy_metadata=args.include_dmy_metadata,
                        dmy_ann=dmy_ann,
                        is_component_crop=False,
                        is_candidate_crop=False,
                    )
                    write_split_metadata(writers, split, metadata, args.dry_run)
                    continue

                split_dirname = "train_images" if split == "train" else "val_images"
                main_filename = build_main_crop_filename(image_filename, ann_index)
                rel_crop_path = f"{split_dirname}/{main_filename}"
                abs_crop_path = args.output_dir / rel_crop_path

                main_crop = pil_image.crop(clamped_bbox)
                maybe_symlink_or_save_crop(
                    main_crop,
                    source_image_path,
                    abs_crop_path,
                    copy_mode=args.copy_mode,
                    crop_bbox=clamped_bbox,
                    image_width=pil_image.width,
                    image_height=pil_image.height,
                    dry_run=args.dry_run,
                )

                write_label(writers, split, rel_crop_path, transcription_normalized, args.dry_run)

                metadata = make_base_metadata_record(
                    image_filename=image_filename,
                    image_width=source_width,
                    image_height=source_height,
                    ann_index=ann_index,
                    ann_cls=ann_cls,
                    raw_bbox=raw_bbox,
                    clamped_bbox=clamped_bbox,
                    transcription_raw=transcription_raw,
                    transcription_normalized=transcription_normalized,
                    output_crop_path=rel_crop_path,
                    split=split,
                    warnings=[],
                    include_dmy_metadata=args.include_dmy_metadata,
                    dmy_ann=dmy_ann,
                    is_component_crop=False,
                    is_candidate_crop=False,
                )
                write_split_metadata(writers, split, metadata, args.dry_run)

                counters.exported_main_crops += 1
                if split == "train":
                    counters.train_count += 1
                else:
                    counters.val_count += 1

                if args.generate_component_crops:
                    components = extract_dmy_components(dmy_ann)
                    for component in components:
                        component_bbox_raw = component.get("bbox")
                        component_text_raw = component.get("transcription")
                        component_cls = str(component.get("cls", "component"))
                        component_idx = int(component.get("component_index", 0))

                        if component_text_raw is None or not str(component_text_raw).strip():
                            counters.skip("component_missing_or_empty_transcription")
                            warn(
                                f"image={image_filename} ann={ann_index} component={component_idx} "
                                "reason=component_missing_or_empty_transcription"
                            )
                            continue

                        component_text = normalize_transcription(str(component_text_raw), args.normalize_transcription)
                        if not component_text:
                            counters.skip("component_empty_after_normalization")
                            warn(
                                f"image={image_filename} ann={ann_index} component={component_idx} "
                                "reason=component_empty_after_normalization"
                            )
                            continue

                        try:
                            parsed_component_bbox = parse_bbox(component_bbox_raw)
                        except ValueError:
                            counters.skip("invalid_component_bbox")
                            warn(
                                f"image={image_filename} ann={ann_index} component={component_idx} "
                                "reason=invalid_component_bbox"
                            )
                            continue

                        clamped_component_bbox = clamp_bbox(
                            parsed_component_bbox,
                            pad=args.pad_bbox,
                            image_width=pil_image.width,
                            image_height=pil_image.height,
                        )
                        if clamped_component_bbox is None:
                            counters.skip("component_bbox_invalid_after_clamp")
                            warn(
                                f"image={image_filename} ann={ann_index} component={component_idx} "
                                "reason=component_bbox_invalid_after_clamp"
                            )
                            continue

                        comp_w, comp_h = bbox_size(clamped_component_bbox)
                        if comp_w < args.min_crop_width or comp_h < args.min_crop_height:
                            counters.skip("component_crop_too_small")
                            warn(
                                f"image={image_filename} ann={ann_index} component={component_idx} "
                                "reason=component_crop_too_small"
                            )
                            continue

                        component_filename = build_component_crop_filename(
                            image_filename, ann_index, component_idx, component_cls
                        )
                        component_rel_path = f"components_images/{component_filename}"
                        component_abs_path = args.output_dir / component_rel_path

                        component_crop = pil_image.crop(clamped_component_bbox)
                        maybe_symlink_or_save_crop(
                            component_crop,
                            source_image_path,
                            component_abs_path,
                            copy_mode=args.copy_mode,
                            crop_bbox=clamped_component_bbox,
                            image_width=pil_image.width,
                            image_height=pil_image.height,
                            dry_run=args.dry_run,
                        )

                        if not args.dry_run:
                            assert writers.components_label_file is not None
                            writers.components_label_file.write(f"{component_rel_path}\t{component_text}\n")

                        counters.exported_component_crops += 1

                if args.generate_candidate_crops and args.candidate_jitter_count > 0:
                    for candidate_index in range(args.candidate_jitter_count):
                        jittered_bbox = (
                            parsed_bbox[0] + rng_candidates.randint(-args.candidate_max_pad_px, args.candidate_max_pad_px),
                            parsed_bbox[1] + rng_candidates.randint(-args.candidate_max_pad_px, args.candidate_max_pad_px),
                            parsed_bbox[2] + rng_candidates.randint(-args.candidate_max_pad_px, args.candidate_max_pad_px),
                            parsed_bbox[3] + rng_candidates.randint(-args.candidate_max_pad_px, args.candidate_max_pad_px),
                        )
                        if jittered_bbox[2] <= jittered_bbox[0] or jittered_bbox[3] <= jittered_bbox[1]:
                            counters.skip("candidate_jitter_invalid")
                            continue

                        clamped_candidate_bbox = clamp_bbox(
                            jittered_bbox,
                            pad=0,
                            image_width=pil_image.width,
                            image_height=pil_image.height,
                        )
                        if clamped_candidate_bbox is None:
                            counters.skip("candidate_bbox_invalid_after_clamp")
                            continue

                        cand_w, cand_h = bbox_size(clamped_candidate_bbox)
                        if cand_w < args.min_crop_width or cand_h < args.min_crop_height:
                            counters.skip("candidate_crop_too_small")
                            continue

                        candidate_filename = build_candidate_crop_filename(image_filename, ann_index, candidate_index)
                        candidate_rel_path = f"candidates_images/{candidate_filename}"
                        candidate_abs_path = args.output_dir / candidate_rel_path

                        candidate_crop = pil_image.crop(clamped_candidate_bbox)
                        maybe_symlink_or_save_crop(
                            candidate_crop,
                            source_image_path,
                            candidate_abs_path,
                            copy_mode=args.copy_mode,
                            crop_bbox=clamped_candidate_bbox,
                            image_width=pil_image.width,
                            image_height=pil_image.height,
                            dry_run=args.dry_run,
                        )

                        candidate_metadata = make_base_metadata_record(
                            image_filename=image_filename,
                            image_width=source_width,
                            image_height=source_height,
                            ann_index=ann_index,
                            ann_cls=ann_cls,
                            raw_bbox=raw_bbox,
                            clamped_bbox=clamped_candidate_bbox,
                            transcription_raw=transcription_raw,
                            transcription_normalized=transcription_normalized,
                            output_crop_path=candidate_rel_path,
                            split="candidate",
                            warnings=[],
                            include_dmy_metadata=args.include_dmy_metadata,
                            dmy_ann=dmy_ann,
                            is_component_crop=False,
                            is_candidate_crop=True,
                        )
                        write_candidate_metadata(writers, candidate_metadata, args.dry_run)
                        counters.exported_candidate_crops += 1

            if pil_image is not None:
                pil_image.close()

        summary = {
            "images_seen": counters.images_seen,
            "annotations_seen": counters.annotations_seen,
            "exported_main_crops": counters.exported_main_crops,
            "exported_component_crops": counters.exported_component_crops,
            "exported_candidate_crops": counters.exported_candidate_crops,
            "train_count": counters.train_count,
            "val_count": counters.val_count,
            "skipped_total": counters.skipped_total,
            "skipped_by_reason": counters.skipped_by_reason,
            "config": {
                "images_dir": str(args.images_dir),
                "annotation_json": str(args.annotation_json),
                "output_dir": str(args.output_dir),
                "train_ratio": args.train_ratio,
                "seed": args.seed,
                "include_dmy_metadata": args.include_dmy_metadata,
                "generate_component_crops": args.generate_component_crops,
                "generate_candidate_crops": args.generate_candidate_crops,
                "candidate_jitter_count": args.candidate_jitter_count,
                "candidate_max_pad_px": args.candidate_max_pad_px,
                "copy_mode": args.copy_mode,
                "pad_bbox": args.pad_bbox,
                "min_crop_width": args.min_crop_width,
                "min_crop_height": args.min_crop_height,
                "normalize_transcription": args.normalize_transcription,
                "allow_cross_image_split": args.allow_cross_image_split,
                "dry_run": args.dry_run,
            },
        }

        if not args.dry_run:
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("PP-OCRv5 dataset export summary")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        writers.close()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
