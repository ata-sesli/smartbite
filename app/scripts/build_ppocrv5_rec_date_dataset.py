from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class Counters:
    images_discovered: int = 0
    labels_loaded: int = 0
    labeled_samples: int = 0
    exported_train: int = 0
    exported_val: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped_total += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1


@dataclass(slots=True)
class LabelEntry:
    image_key: str
    transcription: str
    source: str


@dataclass(slots=True)
class SampleRecord:
    image_rel: str
    image_abs: Path
    transcription_raw: str
    transcription_normalized: str


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
    parser = argparse.ArgumentParser(description="Build PP-OCRv5 recognition dataset from date-only cropped images")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--labels-txt", type=Path)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--labels-jsonl", type=Path)
    parser.add_argument("--labels-from-filename", type=parse_bool, default=False)
    parser.add_argument("--filename-label-regex", type=str, default=r"(?P<label>[^/]+?)(?:\.[^.]+)?$")
    parser.add_argument("--date-joiner", type=str, default="/")
    parser.add_argument("--path-column", type=str, default="image_path")
    parser.add_argument("--text-column", type=str, default="transcription")

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path)

    parser.add_argument("--normalize-transcription", choices=["none", "strip", "unicode"], default="none")

    parser.add_argument("--min-text-length", type=int, default=1)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--min-image-width", type=int, default=1)
    parser.add_argument("--min-image-height", type=int, default=1)

    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr, flush=True)


def normalize_transcription(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "unicode":
        return unicodedata.normalize("NFKC", text).strip()
    raise ValueError(f"unsupported normalize mode: {mode}")


def safe_stem(path_str: str) -> str:
    stem = Path(path_str).stem
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return sanitized or "image"


def build_export_filename(image_rel: str) -> str:
    stem = safe_stem(image_rel)
    short_hash = hashlib.sha1(image_rel.encode("utf-8")).hexdigest()[:10]
    return f"{stem}__{short_hash}.jpg"


def ensure_positive(value: int, arg_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{arg_name} must be > 0")


def validate_args(args: argparse.Namespace) -> None:
    if not args.images_dir.exists() or not args.images_dir.is_dir():
        raise ValueError(f"images directory not found: {args.images_dir}")

    source_count = sum(
        [
            1 if args.labels_json is not None else 0,
            1 if args.labels_txt is not None else 0,
            1 if args.labels_csv is not None else 0,
            1 if args.labels_jsonl is not None else 0,
            1 if args.labels_from_filename else 0,
        ]
    )
    if source_count != 1:
        raise ValueError(
            "exactly one label source is required: --labels-json | --labels-txt | --labels-csv | "
            "--labels-jsonl | --labels-from-filename true"
        )

    if not 0.0 <= args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be within [0.0, 1.0]")

    ensure_positive(args.min_text_length, "--min-text-length")
    ensure_positive(args.max_text_length, "--max-text-length")
    ensure_positive(args.min_image_width, "--min-image-width")
    ensure_positive(args.min_image_height, "--min-image-height")
    if args.max_text_length < args.min_text_length:
        raise ValueError("--max-text-length must be >= --min-text-length")

    if args.labels_from_filename:
        try:
            pattern = re.compile(args.filename_label_regex)
        except re.error as exc:
            raise ValueError(f"invalid --filename-label-regex: {exc}") from exc
        if "label" not in pattern.groupindex:
            raise ValueError("--filename-label-regex must include a named capture group 'label'")


def discover_images(images_dir: Path) -> tuple[list[str], dict[str, Path], dict[str, list[str]]]:
    rel_paths: list[str] = []
    rel_to_abs: dict[str, Path] = {}
    basename_to_rel: dict[str, list[str]] = {}

    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        rel = path.relative_to(images_dir).as_posix()
        rel_paths.append(rel)
        rel_to_abs[rel] = path
        basename_to_rel.setdefault(path.name, []).append(rel)

    for values in basename_to_rel.values():
        values.sort()

    return rel_paths, rel_to_abs, basename_to_rel


def parse_component_x1(value: Any) -> int | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    raw_x1 = value[0]
    if not isinstance(raw_x1, (int, float)):
        return None
    return int(round(float(raw_x1)))


def compose_date_from_ann(ann: Any, date_joiner: str) -> str | None:
    if not isinstance(ann, list):
        return None

    parsed: list[tuple[int, int | None, int, str]] = []
    # tuple fields: class_priority, x1, original_index, text
    class_order = {"year": 0, "month": 1, "day": 2}

    for idx, item in enumerate(ann):
        if not isinstance(item, dict):
            continue
        transcription_raw = item.get("transcription")
        if transcription_raw is None:
            continue
        text = str(transcription_raw).strip()
        if not text:
            continue

        cls_raw = item.get("cls")
        cls = str(cls_raw).strip().lower() if cls_raw is not None else ""
        cls_priority = class_order.get(cls, 99)
        x1 = parse_component_x1(item.get("bbox"))
        parsed.append((cls_priority, x1, idx, text))

    if not parsed:
        return None

    has_all_x1 = all(item[1] is not None for item in parsed)
    if has_all_x1:
        ordered = sorted(parsed, key=lambda item: (item[1], item[2]))
    else:
        ordered = sorted(parsed, key=lambda item: (item[0], item[2]))

    return date_joiner.join(item[3] for item in ordered)


def parse_json_labels(path: Path, date_joiner: str) -> tuple[list[LabelEntry], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"label file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in label file: {path} (line {exc.lineno}, col {exc.colno})") from exc

    if not isinstance(payload, dict):
        raise ValueError("JSON label file must be an object keyed by image")

    entries: list[LabelEntry] = []
    skipped = 0
    for key, value in payload.items():
        image_key = str(key)
        text: str | None = None

        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            if "transcription" in value:
                text = "" if value["transcription"] is None else str(value["transcription"])
            elif "ann" in value:
                text = compose_date_from_ann(value.get("ann"), date_joiner=date_joiner)
            else:
                skipped += 1
                continue
        else:
            skipped += 1
            continue

        if text is None:
            skipped += 1
            continue

        entries.append(LabelEntry(image_key=image_key, transcription=text, source=f"json:{image_key}"))

    return entries, skipped


def parse_txt_labels(path: Path) -> tuple[list[LabelEntry], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"label file not found: {path}") from exc

    entries: list[LabelEntry] = []
    skipped = 0

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "\t" in raw_line:
            left, right = raw_line.split("\t", 1)
        else:
            parts = raw_line.split(None, 1)
            if len(parts) != 2:
                skipped += 1
                continue
            left, right = parts

        image_key = left.strip()
        if not image_key:
            skipped += 1
            continue

        entries.append(LabelEntry(image_key=image_key, transcription=right, source=f"txt:{lineno}"))

    return entries, skipped


def detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        return "\t" if path.suffix.lower() == ".tsv" else ","


def parse_csv_labels(path: Path, path_column: str, text_column: str) -> tuple[list[LabelEntry], int]:
    if not path.exists():
        raise ValueError(f"label file not found: {path}")

    delimiter = detect_csv_delimiter(path)
    entries: list[LabelEntry] = []
    skipped = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV/TSV label file has no header: {path}")
        if path_column not in reader.fieldnames:
            raise ValueError(f"missing path column {path_column!r} in {path}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"missing text column {text_column!r} in {path}")

        for row_index, row in enumerate(reader, start=2):
            raw_key = row.get(path_column)
            if raw_key is None or not str(raw_key).strip():
                skipped += 1
                continue
            text = row.get(text_column)
            if text is None:
                skipped += 1
                continue
            entries.append(LabelEntry(image_key=str(raw_key), transcription=str(text), source=f"csv:{row_index}"))

    return entries, skipped


def parse_jsonl_labels(path: Path) -> tuple[list[LabelEntry], int]:
    if not path.exists():
        raise ValueError(f"label file not found: {path}")

    entries: list[LabelEntry] = []
    skipped = 0

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue

        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSONL in {path} at line {lineno}: {exc}") from exc

        if not isinstance(record, dict):
            skipped += 1
            continue

        key = record.get("image_path")
        text = record.get("transcription")
        if key is None or text is None or not str(key).strip():
            skipped += 1
            continue

        entries.append(LabelEntry(image_key=str(key), transcription=str(text), source=f"jsonl:{lineno}"))

    return entries, skipped


def parse_filename_labels(image_rel_paths: list[str], regex: str) -> tuple[list[LabelEntry], int]:
    pattern = re.compile(regex)
    entries: list[LabelEntry] = []
    skipped = 0

    for rel_path in image_rel_paths:
        match = pattern.search(rel_path)
        if match is None:
            skipped += 1
            continue

        if "label" not in match.groupdict():
            skipped += 1
            continue

        label = match.group("label")
        if label is None or not str(label).strip():
            skipped += 1
            continue

        entries.append(LabelEntry(image_key=rel_path, transcription=str(label), source=f"filename:{rel_path}"))

    return entries, skipped


def normalize_image_key(value: str) -> str:
    key = value.strip().replace("\\", "/")
    while key.startswith("./"):
        key = key[2:]
    return key


def resolve_image_key(
    image_key: str,
    *,
    images_dir: Path,
    rel_to_abs: dict[str, Path],
    basename_to_rel: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    normalized = normalize_image_key(image_key)
    if not normalized:
        return None, "empty_image_key"

    if normalized in rel_to_abs:
        return normalized, None

    maybe_abs = Path(normalized)
    if maybe_abs.is_absolute():
        try:
            rel = maybe_abs.resolve(strict=False).relative_to(images_dir.resolve(strict=False)).as_posix()
            if rel in rel_to_abs:
                return rel, None
        except ValueError:
            pass

    basename = Path(normalized).name
    if basename in basename_to_rel:
        matches = basename_to_rel[basename]
        if len(matches) == 1:
            return matches[0], None
        return None, "ambiguous_basename"

    return None, "image_not_found"


def assign_image_level_splits(image_keys: list[str], train_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = image_keys[:]
    rng.shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)

    mapping: dict[str, str] = {}
    for idx, image_key in enumerate(shuffled):
        mapping[image_key] = "train" if idx < train_cutoff else "val"
    return mapping


def parse_split_file(
    split_file: Path,
    *,
    images_dir: Path,
    rel_to_abs: dict[str, Path],
    basename_to_rel: dict[str, list[str]],
    counters: Counters,
) -> dict[str, str]:
    if not split_file.exists():
        raise ValueError(f"split file not found: {split_file}")

    mapping: dict[str, str] = {}
    for lineno, raw_line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if "\t" in raw_line:
            left, right = raw_line.split("\t", 1)
        else:
            parts = raw_line.split(None, 1)
            if len(parts) != 2:
                counters.skip("split_line_invalid")
                warn(f"split_file={split_file} line={lineno} reason=split_line_invalid")
                continue
            left, right = parts

        split_value = right.strip().lower()
        if split_value not in {"train", "val"}:
            counters.skip("split_value_invalid")
            warn(f"split_file={split_file} line={lineno} reason=split_value_invalid value={split_value!r}")
            continue

        image_rel, reason = resolve_image_key(
            left,
            images_dir=images_dir,
            rel_to_abs=rel_to_abs,
            basename_to_rel=basename_to_rel,
        )
        if image_rel is None:
            assert reason is not None
            counters.skip(f"split_{reason}")
            warn(f"split_file={split_file} line={lineno} reason=split_{reason} image_key={left!r}")
            continue

        mapping[image_rel] = split_value

    return mapping


def open_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def export_image(source_path: Path, destination_path: Path, copy_mode: str, dry_run: bool) -> None:
    if dry_run:
        return

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()

    if copy_mode == "symlink":
        destination_path.symlink_to(source_path.resolve(strict=False))
        return

    with Image.open(source_path) as image:
        if image.mode not in {"RGB", "L"}:
            converted = image.convert("RGB")
            converted.save(destination_path, format="JPEG", quality=95)
            converted.close()
        else:
            image.save(destination_path, format="JPEG", quality=95)


def build_summary(args: argparse.Namespace, counters: Counters, label_source: str) -> dict[str, Any]:
    return {
        "images_discovered": counters.images_discovered,
        "labels_loaded": counters.labels_loaded,
        "labeled_samples": counters.labeled_samples,
        "exported_train": counters.exported_train,
        "exported_val": counters.exported_val,
        "exported_total": counters.exported_train + counters.exported_val,
        "skipped_total": counters.skipped_total,
        "skipped_by_reason": counters.skipped_by_reason,
        "config": {
            "images_dir": str(args.images_dir),
            "output_dir": str(args.output_dir),
            "label_source": label_source,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "split_file": str(args.split_file) if args.split_file else None,
            "normalize_transcription": args.normalize_transcription,
            "min_text_length": args.min_text_length,
            "max_text_length": args.max_text_length,
            "min_image_width": args.min_image_width,
            "min_image_height": args.min_image_height,
            "copy_mode": args.copy_mode,
            "dry_run": args.dry_run,
        },
    }


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    counters = Counters()

    image_rel_paths, rel_to_abs, basename_to_rel = discover_images(args.images_dir)
    counters.images_discovered = len(image_rel_paths)
    info(f"Discovered {counters.images_discovered} images under {args.images_dir}")

    if not image_rel_paths:
        print("[ERROR] no images found under --images-dir", file=sys.stderr)
        return 2

    label_source_name = ""
    try:
        if args.labels_json is not None:
            label_source_name = "json"
            label_entries, loader_skipped = parse_json_labels(args.labels_json, date_joiner=args.date_joiner)
        elif args.labels_txt is not None:
            label_source_name = "txt"
            label_entries, loader_skipped = parse_txt_labels(args.labels_txt)
        elif args.labels_csv is not None:
            label_source_name = "csv"
            label_entries, loader_skipped = parse_csv_labels(args.labels_csv, args.path_column, args.text_column)
        elif args.labels_jsonl is not None:
            label_source_name = "jsonl"
            label_entries, loader_skipped = parse_jsonl_labels(args.labels_jsonl)
        else:
            label_source_name = "filename"
            label_entries, loader_skipped = parse_filename_labels(image_rel_paths, args.filename_label_regex)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    counters.labels_loaded = len(label_entries)
    info(f"Loaded {counters.labels_loaded} label entries from source={label_source_name}")
    if loader_skipped:
        counters.skipped_by_reason["label_loader_skipped"] = loader_skipped
        counters.skipped_total += loader_skipped
        info(f"Label loader skipped {loader_skipped} malformed/unsupported entries")

    resolved_labels: dict[str, str] = {}
    resolve_progress = ProgressTracker("Resolving labels to images", len(label_entries))
    for idx, entry in enumerate(label_entries, start=1):
        image_rel, reason = resolve_image_key(
            entry.image_key,
            images_dir=args.images_dir,
            rel_to_abs=rel_to_abs,
            basename_to_rel=basename_to_rel,
        )
        if image_rel is None:
            assert reason is not None
            reason_key = f"label_{reason}"
            counters.skip(reason_key)
            warn(f"source={entry.source} reason={reason_key} image_key={entry.image_key!r}")
            resolve_progress.tick(idx)
            continue

        if image_rel in resolved_labels:
            counters.skip("duplicate_label_for_image")
            warn(f"source={entry.source} reason=duplicate_label_for_image image={image_rel}")
            resolve_progress.tick(idx)
            continue

        resolved_labels[image_rel] = entry.transcription
        resolve_progress.tick(idx)

    info(f"Resolved {len(resolved_labels)} unique labels to discovered images")

    missing_label_progress = ProgressTracker("Checking images missing labels", len(image_rel_paths))
    for idx, image_rel in enumerate(image_rel_paths, start=1):
        if image_rel not in resolved_labels:
            counters.skip("image_missing_label")
        missing_label_progress.tick(idx)

    counters.labeled_samples = len(resolved_labels)

    size_cache: dict[str, tuple[int, int]] = {}
    validated_samples: list[SampleRecord] = []

    resolved_sorted = sorted(resolved_labels)
    validate_progress = ProgressTracker("Validating labeled samples", len(resolved_sorted))
    for idx, image_rel in enumerate(resolved_sorted, start=1):
        raw_text = resolved_labels[image_rel]
        normalized = normalize_transcription(str(raw_text), args.normalize_transcription)

        if "\n" in normalized or "\r" in normalized:
            counters.skip("transcription_contains_newline")
            warn(f"image={image_rel} reason=transcription_contains_newline")
            validate_progress.tick(idx)
            continue

        text_length = len(normalized)
        if text_length < args.min_text_length:
            counters.skip("text_too_short")
            warn(f"image={image_rel} reason=text_too_short length={text_length}")
            validate_progress.tick(idx)
            continue
        if text_length > args.max_text_length:
            counters.skip("text_too_long")
            warn(f"image={image_rel} reason=text_too_long length={text_length}")
            validate_progress.tick(idx)
            continue

        image_abs = rel_to_abs[image_rel]
        try:
            if image_rel not in size_cache:
                size_cache[image_rel] = open_image_size(image_abs)
            width, height = size_cache[image_rel]
        except Exception as exc:  # pragma: no cover - PIL failure path
            counters.skip("image_unreadable")
            warn(f"image={image_rel} reason=image_unreadable error={exc}")
            validate_progress.tick(idx)
            continue

        if width < args.min_image_width or height < args.min_image_height:
            counters.skip("image_too_small")
            warn(f"image={image_rel} reason=image_too_small size={width}x{height}")
            validate_progress.tick(idx)
            continue

        validated_samples.append(
            SampleRecord(
                image_rel=image_rel,
                image_abs=image_abs,
                transcription_raw=str(raw_text),
                transcription_normalized=normalized,
            )
        )
        validate_progress.tick(idx)

    info(f"Validated samples: {len(validated_samples)}")

    split_mapping: dict[str, str]
    if args.split_file is not None:
        try:
            split_mapping = parse_split_file(
                args.split_file,
                images_dir=args.images_dir,
                rel_to_abs=rel_to_abs,
                basename_to_rel=basename_to_rel,
                counters=counters,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
    else:
        split_mapping = assign_image_level_splits([sample.image_rel for sample in validated_samples], args.train_ratio, args.seed)

    train_file = None
    val_file = None
    try:
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "train_images").mkdir(parents=True, exist_ok=True)
            (args.output_dir / "val_images").mkdir(parents=True, exist_ok=True)
            train_file = (args.output_dir / "train_label.txt").open("w", encoding="utf-8")
            val_file = (args.output_dir / "val_label.txt").open("w", encoding="utf-8")

        ordered_samples = sorted(validated_samples, key=lambda item: item.image_rel)
        export_progress = ProgressTracker("Exporting train/val dataset", len(ordered_samples))
        for idx, sample in enumerate(ordered_samples, start=1):
            split = split_mapping.get(sample.image_rel)
            if split not in {"train", "val"}:
                counters.skip("split_unassigned")
                warn(f"image={sample.image_rel} reason=split_unassigned")
                export_progress.tick(idx)
                continue

            out_name = build_export_filename(sample.image_rel)
            rel_out_path = f"{split}_images/{out_name}"
            abs_out_path = args.output_dir / rel_out_path

            try:
                export_image(sample.image_abs, abs_out_path, args.copy_mode, args.dry_run)
            except Exception as exc:
                counters.skip("image_export_failed")
                warn(f"image={sample.image_rel} reason=image_export_failed error={exc}")
                export_progress.tick(idx)
                continue

            line = f"{rel_out_path}\t{sample.transcription_normalized}\n"
            if not args.dry_run:
                if split == "train":
                    assert train_file is not None
                    train_file.write(line)
                else:
                    assert val_file is not None
                    val_file.write(line)

            if split == "train":
                counters.exported_train += 1
            else:
                counters.exported_val += 1
            export_progress.tick(idx)

        summary = build_summary(args, counters, label_source=label_source_name)

        if not args.dry_run:
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print("PP-OCRv5 date-recognition dataset export summary")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        if train_file is not None:
            train_file.close()
        if val_file is not None:
            val_file.close()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
