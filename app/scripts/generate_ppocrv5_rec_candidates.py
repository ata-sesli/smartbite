from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.ai.ocr import detect_runtime_device
from app.infra.settings import get_settings


BBox = tuple[int, int, int, int]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

EXPIRY_KEYWORDS = [
    "skt",
    "tett",
    "son tuketim tarihi",
    "son tüketim tarihi",
    "best before",
    "use by",
    "exp",
    "expiry",
    "expiration",
]
PRODUCTION_KEYWORDS = [
    "uretim tarihi",
    "üretim tarihi",
    "production date",
    "mfg",
    "packed on",
]
LOT_KEYWORDS = ["lot", "batch", "parti", "seri"]
CONFUSER_KEYWORDS = [
    *PRODUCTION_KEYWORDS,
    *LOT_KEYWORDS,
    "gram",
    " ml",
    " g",
    "kg",
    "barcode",
]

DATE_PATTERN = re.compile(
    r"(\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b)|(\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b)",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(r"\b\d+\s?(g|kg|ml|l)\b", re.IGNORECASE)
BARCODE_LIKE_PATTERN = re.compile(r"\b\d{8,14}\b")


@dataclass(slots=True)
class Counters:
    images_processed: int = 0
    product_regions_processed: int = 0
    candidate_crops_generated_raw: int = 0
    candidate_crops_generated: int = 0
    top_ranked_expiry_like_candidates: int = 0
    confuser_candidates: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str, count: int = 1) -> None:
        self.skipped_total += count
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + count


@dataclass(slots=True)
class YOLODetection:
    image_path: Path
    image_id: str
    product_id: str
    bbox_xyxy: BBox
    score: float
    class_name: str


@dataclass(slots=True)
class ProductRegion:
    source_image_path: Path
    source_image_id: str
    product_id: str
    bbox_xyxy: BBox
    score: float
    class_name: str


@dataclass(slots=True)
class OCRLine:
    bbox_xyxy: BBox
    text: str
    confidence: float


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    source_image: str
    source_image_path: Path
    source_product_id: str
    source_product_bbox: BBox
    source_product_score: float
    source_product_class: str
    text_region_bbox_in_product: BBox
    text_region_bbox_in_source: BBox
    ocr_prefill: str
    ocr_confidence: float
    crop_width: int
    crop_height: int
    expiry_score: float
    confuser_score: float
    hardness_score: float
    priority_score: float
    heuristic_tags: list[str]
    suggested_class: str
    tie_break: int
    crop_image: Image.Image


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


class PPOCRLineExtractor:
    def __init__(
        self,
        *,
        model_dir: Path,
        char_dict_path: Path | None,
        paddle_lang: str,
        device_mode: str,
        use_angle_cls: bool,
        det_db_thresh: float,
    ) -> None:
        self.model_dir = model_dir
        self.char_dict_path = char_dict_path
        self.paddle_lang = paddle_lang
        self.runtime_device = detect_runtime_device(device_mode)
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self._model: Any | None = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            self._load_error = f"ppocrv5 model directory not found: {self.model_dir}"
            return

        if self.char_dict_path is not None and not self.char_dict_path.exists():
            self._load_error = f"ppocrv5 char dict file not found: {self.char_dict_path}"
            return

        try:
            from paddleocr import PaddleOCR

            kwargs: dict[str, Any] = {
                "lang": self.paddle_lang,
                "use_angle_cls": self.use_angle_cls,
                "det_db_thresh": self.det_db_thresh,
                "rec_model_dir": str(self.model_dir),
                "use_gpu": self.runtime_device == "cuda",
            }
            if self.char_dict_path is not None:
                kwargs["rec_char_dict_path"] = str(self.char_dict_path)

            self._model = PaddleOCR(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime dependency
            self._load_error = f"failed to initialize ppocrv5 extractor: {exc}"

    def extract_lines(self, image_np: np.ndarray) -> tuple[list[OCRLine], str | None]:
        self._ensure_loaded()

        if self._load_error is not None:
            return [], self._load_error
        if self._model is None:
            return [], "ocr extractor unavailable"

        height, width = image_np.shape[:2]
        try:
            raw_result = self._model.ocr(image_np, cls=self.use_angle_cls)
        except Exception as exc:  # pragma: no cover - runtime inference
            return [], f"ocr inference error: {exc}"

        lines: list[OCRLine] = []
        ocr_lines = raw_result[0] if raw_result else []
        for line in ocr_lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue

            points = line[0]
            text_conf = line[1]
            if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                continue

            text = str(text_conf[0])
            try:
                confidence = float(text_conf[1])
            except (TypeError, ValueError):
                continue

            bbox = quad_points_to_bbox(points, width, height)
            if bbox is None:
                continue

            lines.append(OCRLine(bbox_xyxy=bbox, text=text, confidence=confidence))

        return lines, None


class StaticFailureExtractor:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def extract_lines(self, image_np: np.ndarray) -> tuple[list[OCRLine], str | None]:
        _ = image_np
        return [], self.reason


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate semi-auto PP-OCRv5 recognition candidate dataset")

    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--input-mode", choices=["full_images", "product_crops"], default="full_images")
    parser.add_argument("--yolo-detections-jsonl", type=Path)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-crop-width", type=int, default=8)
    parser.add_argument("--min-crop-height", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=64)

    parser.add_argument("--max-candidates-per-source", type=int, default=40)
    parser.add_argument("--max-expiry-per-source", type=int, default=15)
    parser.add_argument("--max-confuser-per-source", type=int, default=15)
    parser.add_argument("--max-other-per-source", type=int, default=10)

    parser.add_argument("--export-csv", type=parse_bool, default=True)
    parser.add_argument("--export-overlays", type=parse_bool, default=False)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--ppocrv5-model-dir", type=Path)
    parser.add_argument("--ppocrv5-char-dict", type=Path)
    parser.add_argument("--paddle-lang", type=str)
    parser.add_argument("--device-mode", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--ppocrv5-use-angle-cls", type=parse_bool)
    parser.add_argument("--ppocrv5-det-db-thresh", type=float)

    return parser.parse_args(argv)


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr, flush=True)


def ensure_non_negative(value: int | float, arg_name: str) -> None:
    if value < 0:
        raise ValueError(f"{arg_name} must be >= 0")


def discover_images(images_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def canonical_image_id(image_path: Path, images_dir: Path) -> str:
    image_resolved = image_path.resolve(strict=False)
    images_resolved = images_dir.resolve(strict=False)
    try:
        return image_resolved.relative_to(images_resolved).as_posix()
    except ValueError:
        return image_resolved.as_posix()


def parse_bbox_xyxy(raw: Any) -> BBox:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("bbox_xyxy must have exactly 4 numeric values")

    nums: list[int] = []
    for value in raw:
        if not isinstance(value, (int, float)):
            raise ValueError("bbox_xyxy values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("bbox_xyxy values must be finite")
        nums.append(int(round(number)))

    x1, y1, x2, y2 = nums
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_xyxy must have positive width and height")
    return x1, y1, x2, y2


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox | None:
    x1, y1, x2, y2 = bbox
    clamped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None
    return clamped


def quad_points_to_bbox(points: Any, width: int, height: int) -> BBox | None:
    if not isinstance(points, (list, tuple)) or not points:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            px = float(point[0])
            py = float(point[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(px) or not math.isfinite(py):
            continue
        xs.append(px)
        ys.append(py)

    if not xs or not ys:
        return None

    bbox = (int(math.floor(min(xs))), int(math.floor(min(ys))), int(math.ceil(max(xs))), int(math.ceil(max(ys))))
    return clamp_bbox(bbox, width, height)


def bbox_size(bbox: BBox) -> tuple[int, int]:
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def offset_bbox(local_bbox: BBox, parent_bbox: BBox) -> BBox:
    return (
        parent_bbox[0] + local_bbox[0],
        parent_bbox[1] + local_bbox[1],
        parent_bbox[0] + local_bbox[2],
        parent_bbox[1] + local_bbox[3],
    )


def normalize_for_matching(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def find_keyword_hits(text_norm: str, keywords: list[str]) -> list[str]:
    hits: list[str] = []
    for keyword in keywords:
        if normalize_for_matching(keyword) in text_norm:
            hits.append(keyword)
    return hits


def evaluate_candidate(
    text: str,
    confidence: float,
    crop_width: int,
    crop_height: int,
    crop_image: Image.Image,
) -> tuple[float, float, float, float, list[str], str]:
    text_norm = normalize_for_matching(text)
    text_len = len(text.strip())

    has_digits = any(char.isdigit() for char in text)
    date_like = DATE_PATTERN.search(text) is not None
    expiry_hits = find_keyword_hits(text_norm, EXPIRY_KEYWORDS)
    production_hits = find_keyword_hits(text_norm, PRODUCTION_KEYWORDS)
    lot_hits = find_keyword_hits(text_norm, LOT_KEYWORDS)
    confuser_hits = find_keyword_hits(text_norm, CONFUSER_KEYWORDS)
    unit_like = UNIT_PATTERN.search(text_norm) is not None
    barcode_like = BARCODE_LIKE_PATTERN.search(text_norm) is not None

    grayscale = np.asarray(crop_image.convert("L"), dtype=np.float32)
    contrast = float(grayscale.std()) if grayscale.size else 0.0
    low_contrast = contrast < 22.0
    very_small_crop = crop_width < 16 or crop_height < 16

    tags: list[str] = []
    if has_digits:
        tags.append("has_digits")
    if date_like:
        tags.append("date_like")
    if expiry_hits:
        tags.append("expiry_keyword")
    if production_hits:
        tags.append("production_keyword")
    if lot_hits:
        tags.append("lot_keyword")
    if confuser_hits:
        tags.append("confuser_keyword")
    if unit_like:
        tags.append("unit_like")
    if barcode_like:
        tags.append("barcode_like")
    if low_contrast:
        tags.append("low_contrast")
    if very_small_crop:
        tags.append("very_small_crop")

    if text_len <= 5:
        tags.append("length_short")
    elif text_len <= 24:
        tags.append("length_medium")
    else:
        tags.append("length_long")

    expiry_score = 0.0
    confuser_score = 0.0
    hardness_score = 0.0

    if has_digits:
        expiry_score += 0.8
        confuser_score += 0.2
    if date_like:
        expiry_score += 2.2
    if expiry_hits:
        expiry_score += min(3.2, 1.2 * len(expiry_hits))
    if text_len >= 4 and text_len <= 24:
        expiry_score += 0.4

    if production_hits:
        confuser_score += min(3.0, 1.5 * len(production_hits))
    if lot_hits:
        confuser_score += min(2.8, 1.4 * len(lot_hits))
    if unit_like:
        confuser_score += 0.9
    if barcode_like:
        confuser_score += 1.1

    conf_for_score = max(0.0, min(1.0, confidence))
    expiry_score += conf_for_score
    confuser_score += 0.7 * conf_for_score

    if confidence < 0.6:
        hardness_score += 1.0
        tags.append("low_confidence")
    if very_small_crop:
        hardness_score += 1.0
    if low_contrast:
        hardness_score += 0.8
    if text_len <= 2:
        hardness_score += 0.3

    priority_score = max(expiry_score, confuser_score) + (0.3 * hardness_score)

    alpha_only = any(ch.isalpha() for ch in text) and not has_digits
    if expiry_hits or (date_like and expiry_score >= confuser_score):
        suggested_class = "expiry"
    elif production_hits:
        suggested_class = "production_date"
    elif lot_hits:
        suggested_class = "lot"
    elif alpha_only:
        token_count = len([t for t in text.strip().split(" ") if t])
        suggested_class = "brand" if token_count <= 2 and text_len <= 16 else "product_name"
    else:
        suggested_class = "other"

    tags = sorted(set(tags))
    return (
        round(expiry_score, 4),
        round(confuser_score, 4),
        round(hardness_score, 4),
        round(priority_score, 4),
        tags,
        suggested_class,
    )


def bucket_for_class(suggested_class: str) -> str:
    if suggested_class == "expiry":
        return "expiry"
    if suggested_class in {"production_date", "lot"}:
        return "confuser"
    return "other"


def build_candidate_id(
    *,
    seed: int,
    source_image: str,
    source_product_id: str,
    source_product_bbox: BBox,
    text_region_bbox_in_source: BBox,
    text: str,
    line_index: int,
) -> str:
    payload = {
        "seed": seed,
        "source_image": source_image,
        "source_product_id": source_product_id,
        "source_product_bbox": list(source_product_bbox),
        "text_region_bbox_in_source": list(text_region_bbox_in_source),
        "text": text,
        "line_index": line_index,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"cand_{digest[:16]}"


def seeded_tie_break(candidate_id: str, seed: int) -> int:
    digest = hashlib.sha1(f"{seed}:{candidate_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def limit_candidates_for_source(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-c.priority_score, c.tie_break, c.candidate_id),
    )

    selected: list[Candidate] = []
    expiry_count = 0
    confuser_count = 0
    other_count = 0

    for candidate in sorted_candidates:
        if len(selected) >= args.max_candidates_per_source:
            break

        bucket = bucket_for_class(candidate.suggested_class)
        if bucket == "expiry":
            if expiry_count >= args.max_expiry_per_source:
                continue
            expiry_count += 1
        elif bucket == "confuser":
            if confuser_count >= args.max_confuser_per_source:
                continue
            confuser_count += 1
        else:
            if other_count >= args.max_other_per_source:
                continue
            other_count += 1

        selected.append(candidate)

    return selected


def resolve_image_path(image_path_field: str, images_dir: Path) -> Path:
    raw = Path(image_path_field)
    if raw.is_absolute():
        return raw
    return images_dir / raw


def load_yolo_detections_jsonl(path: Path, images_dir: Path) -> dict[str, list[YOLODetection]]:
    if not path.exists():
        raise ValueError(f"YOLO detections JSONL file not found: {path}")

    out: dict[str, list[YOLODetection]] = defaultdict(list)
    required_fields = {"image_path", "product_id", "bbox_xyxy", "score", "class_name"}

    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid YOLO JSONL line {idx}: malformed JSON ({exc})") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"invalid YOLO JSONL line {idx}: record must be an object")

        missing_fields = required_fields - set(payload.keys())
        if missing_fields:
            raise ValueError(f"invalid YOLO JSONL line {idx}: missing fields {sorted(missing_fields)}")

        image_path_field = payload["image_path"]
        product_id = payload["product_id"]
        class_name = payload["class_name"]
        score = payload["score"]

        if not isinstance(image_path_field, str) or not image_path_field.strip():
            raise ValueError(f"invalid YOLO JSONL line {idx}: image_path must be a non-empty string")
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError(f"invalid YOLO JSONL line {idx}: product_id must be a non-empty string")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"invalid YOLO JSONL line {idx}: class_name must be a non-empty string")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError(f"invalid YOLO JSONL line {idx}: score must be a finite number")

        bbox_xyxy = parse_bbox_xyxy(payload["bbox_xyxy"])
        resolved_image_path = resolve_image_path(image_path_field, images_dir)
        image_id = canonical_image_id(resolved_image_path, images_dir)

        out[image_id].append(
            YOLODetection(
                image_path=resolved_image_path,
                image_id=image_id,
                product_id=product_id,
                bbox_xyxy=bbox_xyxy,
                score=float(score),
                class_name=class_name,
            )
        )

    for image_id in out:
        out[image_id].sort(
            key=lambda d: (
                d.product_id,
                d.bbox_xyxy[1],
                d.bbox_xyxy[0],
                -d.score,
                d.class_name,
            )
        )

    return dict(out)


def create_ocr_extractor(args: argparse.Namespace) -> PPOCRLineExtractor | StaticFailureExtractor:
    settings = get_settings()

    model_dir = args.ppocrv5_model_dir or settings.ocr_ppocrv5_main_model_dir
    char_dict = args.ppocrv5_char_dict if args.ppocrv5_char_dict is not None else settings.ocr_ppocrv5_main_char_dict_path
    paddle_lang = args.paddle_lang or settings.ocr_paddle_lang
    device_mode = args.device_mode or settings.ocr_device_mode
    use_angle_cls = settings.ocr_ppocrv5_use_angle_cls if args.ppocrv5_use_angle_cls is None else args.ppocrv5_use_angle_cls
    det_db_thresh = settings.ocr_ppocrv5_det_db_thresh if args.ppocrv5_det_db_thresh is None else args.ppocrv5_det_db_thresh

    if not model_dir:
        return StaticFailureExtractor("ppocrv5 model path is empty")

    return PPOCRLineExtractor(
        model_dir=Path(model_dir),
        char_dict_path=Path(char_dict) if char_dict else None,
        paddle_lang=paddle_lang,
        device_mode=device_mode,
        use_angle_cls=bool(use_angle_cls),
        det_db_thresh=float(det_db_thresh),
    )


def build_manifest_record(candidate: Candidate, crop_rel_path: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source_image": candidate.source_image,
        "source_product_id": candidate.source_product_id,
        "source_product_bbox": list(candidate.source_product_bbox),
        "source_product_score": candidate.source_product_score,
        "source_product_class": candidate.source_product_class,
        "text_region_bbox_in_product": list(candidate.text_region_bbox_in_product),
        "text_region_bbox_in_source": list(candidate.text_region_bbox_in_source),
        "crop_path": crop_rel_path,
        "ocr_prefill": candidate.ocr_prefill,
        "ocr_confidence": candidate.ocr_confidence,
        "crop_width": candidate.crop_width,
        "crop_height": candidate.crop_height,
        "expiry_score": candidate.expiry_score,
        "confuser_score": candidate.confuser_score,
        "hardness_score": candidate.hardness_score,
        "priority_score": candidate.priority_score,
        "heuristic_tags": candidate.heuristic_tags,
        "suggested_class": candidate.suggested_class,
        "reviewer_decision": None,
        "final_text": None,
        "accepted": None,
        "final_tag": None,
    }


def render_overlays(candidates: list[Candidate], overlays_dir: Path) -> None:
    by_image: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_image[candidate.source_image].append(candidate)

    color_map = {
        "expiry": "green",
        "production_date": "orange",
        "lot": "red",
        "product_name": "blue",
        "brand": "purple",
        "other": "gray",
    }

    overlays_dir.mkdir(parents=True, exist_ok=True)
    overlay_progress = ProgressTracker("Rendering overlays", len(by_image))
    for index, (source_image, grouped) in enumerate(by_image.items(), start=1):
        first = grouped[0]
        base = Image.open(first.source_image_path).convert("RGB")
        draw = ImageDraw.Draw(base)

        for candidate in grouped:
            color = color_map.get(candidate.suggested_class, "white")
            draw.rectangle(candidate.source_product_bbox, outline="yellow", width=2)
            draw.rectangle(candidate.text_region_bbox_in_source, outline=color, width=2)
            label = f"{candidate.suggested_class}:{candidate.priority_score:.2f}"
            draw.text((candidate.text_region_bbox_in_source[0], max(0, candidate.text_region_bbox_in_source[1] - 10)), label, fill=color)

        overlay_name = f"{Path(source_image).stem}__overlay.jpg"
        base.save(overlays_dir / overlay_name, format="JPEG")
        base.close()
        overlay_progress.tick(index)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        ensure_non_negative(args.min_confidence, "--min-confidence")
        ensure_non_negative(args.min_crop_width, "--min-crop-width")
        ensure_non_negative(args.min_crop_height, "--min-crop-height")
        ensure_non_negative(args.max_text_length, "--max-text-length")
        ensure_non_negative(args.max_candidates_per_source, "--max-candidates-per-source")
        ensure_non_negative(args.max_expiry_per_source, "--max-expiry-per-source")
        ensure_non_negative(args.max_confuser_per_source, "--max-confuser-per-source")
        ensure_non_negative(args.max_other_per_source, "--max-other-per-source")

        if not args.images_dir.exists() or not args.images_dir.is_dir():
            raise ValueError(f"images directory not found: {args.images_dir}")

        if args.input_mode == "product_crops" and args.yolo_detections_jsonl is not None:
            warn("--yolo-detections-jsonl is ignored when --input-mode=product_crops")

        yolo_detections = (
            load_yolo_detections_jsonl(args.yolo_detections_jsonl, args.images_dir)
            if args.yolo_detections_jsonl is not None and args.input_mode == "full_images"
            else {}
        )
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    image_paths = discover_images(args.images_dir)
    image_map: dict[str, Path] = {canonical_image_id(path, args.images_dir): path for path in image_paths}
    info(f"Discovered {len(image_map)} images under {args.images_dir}")

    counters = Counters()
    extractor = create_ocr_extractor(args)
    selected_candidates: list[Candidate] = []

    ordered_image_ids = sorted(image_map.keys())
    image_progress = ProgressTracker("Processing images for OCR proposals", len(ordered_image_ids))
    for image_index, image_id in enumerate(ordered_image_ids, start=1):
        source_image_path = image_map[image_id]

        counters.images_processed += 1
        image_progress.tick(image_index)

        try:
            source_image = Image.open(source_image_path).convert("RGB")
        except Exception as exc:
            counters.skip("image_open_failed")
            warn(f"image={image_id} reason=image_open_failed detail={exc}")
            continue

        source_width, source_height = source_image.size

        regions: list[ProductRegion] = []
        if args.input_mode == "product_crops":
            regions.append(
                ProductRegion(
                    source_image_path=source_image_path,
                    source_image_id=image_id,
                    product_id=f"{Path(image_id).stem}__product0",
                    bbox_xyxy=(0, 0, source_width, source_height),
                    score=1.0,
                    class_name="product_crop",
                )
            )
        else:
            detections = yolo_detections.get(image_id, [])
            if detections:
                for det_index, det in enumerate(detections):
                    clamped_bbox = clamp_bbox(det.bbox_xyxy, source_width, source_height)
                    if clamped_bbox is None:
                        counters.skip("invalid_yolo_bbox")
                        warn(f"image={image_id} product_id={det.product_id} reason=invalid_yolo_bbox")
                        continue

                    regions.append(
                        ProductRegion(
                            source_image_path=source_image_path,
                            source_image_id=image_id,
                            product_id=det.product_id or f"{Path(image_id).stem}__det{det_index}",
                            bbox_xyxy=clamped_bbox,
                            score=det.score,
                            class_name=det.class_name,
                        )
                    )
            else:
                regions.append(
                    ProductRegion(
                        source_image_path=source_image_path,
                        source_image_id=image_id,
                        product_id=f"{Path(image_id).stem}__whole",
                        bbox_xyxy=(0, 0, source_width, source_height),
                        score=1.0,
                        class_name="whole_image_fallback",
                    )
                )

        if not regions:
            source_image.close()
            continue

        for region in regions:
            counters.product_regions_processed += 1

            product_crop = source_image.crop(region.bbox_xyxy)
            product_np = np.asarray(product_crop)

            ocr_lines, ocr_error = extractor.extract_lines(product_np)
            if ocr_error is not None:
                counters.skip("ocr_error")
                warn(f"image={image_id} product_id={region.product_id} reason=ocr_error detail={ocr_error}")
                product_crop.close()
                continue

            source_candidates: list[Candidate] = []
            for line_index, line in enumerate(ocr_lines):
                text = str(line.text or "").strip()
                confidence = float(line.confidence)

                if not text:
                    counters.skip("empty_text")
                    continue
                if confidence < args.min_confidence:
                    counters.skip("below_min_confidence")
                    continue
                if len(text) > args.max_text_length:
                    counters.skip("text_too_long")
                    continue

                line_bbox = clamp_bbox(line.bbox_xyxy, product_crop.width, product_crop.height)
                if line_bbox is None:
                    counters.skip("invalid_text_bbox")
                    continue

                crop_width, crop_height = bbox_size(line_bbox)
                if crop_width < args.min_crop_width or crop_height < args.min_crop_height:
                    counters.skip("candidate_crop_too_small")
                    continue

                text_crop = product_crop.crop(line_bbox)
                line_bbox_in_source = offset_bbox(line_bbox, region.bbox_xyxy)

                (
                    expiry_score,
                    confuser_score,
                    hardness_score,
                    priority_score,
                    heuristic_tags,
                    suggested_class,
                ) = evaluate_candidate(text, confidence, crop_width, crop_height, text_crop)

                candidate_id = build_candidate_id(
                    seed=args.seed,
                    source_image=image_id,
                    source_product_id=region.product_id,
                    source_product_bbox=region.bbox_xyxy,
                    text_region_bbox_in_source=line_bbox_in_source,
                    text=text,
                    line_index=line_index,
                )

                source_candidates.append(
                    Candidate(
                        candidate_id=candidate_id,
                        source_image=image_id,
                        source_image_path=source_image_path,
                        source_product_id=region.product_id,
                        source_product_bbox=region.bbox_xyxy,
                        source_product_score=round(region.score, 6),
                        source_product_class=region.class_name,
                        text_region_bbox_in_product=line_bbox,
                        text_region_bbox_in_source=line_bbox_in_source,
                        ocr_prefill=text,
                        ocr_confidence=round(confidence, 6),
                        crop_width=crop_width,
                        crop_height=crop_height,
                        expiry_score=expiry_score,
                        confuser_score=confuser_score,
                        hardness_score=hardness_score,
                        priority_score=priority_score,
                        heuristic_tags=heuristic_tags,
                        suggested_class=suggested_class,
                        tie_break=seeded_tie_break(candidate_id, args.seed),
                        crop_image=text_crop,
                    )
                )

            counters.candidate_crops_generated_raw += len(source_candidates)

            selected_for_region = limit_candidates_for_source(source_candidates, args)
            selected_candidates.extend(selected_for_region)
            selected_ids = {candidate.candidate_id for candidate in selected_for_region}

            for discarded in source_candidates:
                if discarded.candidate_id not in selected_ids:
                    discarded.crop_image.close()

            product_crop.close()

        source_image.close()

    if args.input_mode == "full_images" and yolo_detections:
        unmatched_ids = sorted(set(yolo_detections.keys()) - set(image_map.keys()))
        if unmatched_ids:
            missing_count = sum(len(yolo_detections[key]) for key in unmatched_ids)
            counters.skip("yolo_image_not_found", count=missing_count)
            for image_id in unmatched_ids:
                warn(f"image={image_id} reason=yolo_image_not_found")

    selected_candidates.sort(
        key=lambda c: (
            c.source_image,
            c.source_product_id,
            -c.priority_score,
            c.tie_break,
            c.candidate_id,
        )
    )

    counters.candidate_crops_generated = len(selected_candidates)
    counters.top_ranked_expiry_like_candidates = sum(1 for c in selected_candidates if c.suggested_class == "expiry")
    counters.confuser_candidates = sum(1 for c in selected_candidates if c.suggested_class in {"production_date", "lot"})

    summary = {
        "images_processed": counters.images_processed,
        "product_regions_processed": counters.product_regions_processed,
        "candidate_crops_generated_raw": counters.candidate_crops_generated_raw,
        "candidate_crops_generated": counters.candidate_crops_generated,
        "top_ranked_expiry_like_candidates": counters.top_ranked_expiry_like_candidates,
        "confuser_candidates": counters.confuser_candidates,
        "skipped_total": counters.skipped_total,
        "skipped_by_reason": counters.skipped_by_reason,
        "config": {
            "images_dir": str(args.images_dir),
            "output_dir": str(args.output_dir),
            "input_mode": args.input_mode,
            "yolo_detections_jsonl": str(args.yolo_detections_jsonl) if args.yolo_detections_jsonl else None,
            "seed": args.seed,
            "min_confidence": args.min_confidence,
            "min_crop_width": args.min_crop_width,
            "min_crop_height": args.min_crop_height,
            "max_text_length": args.max_text_length,
            "max_candidates_per_source": args.max_candidates_per_source,
            "max_expiry_per_source": args.max_expiry_per_source,
            "max_confuser_per_source": args.max_confuser_per_source,
            "max_other_per_source": args.max_other_per_source,
            "export_csv": args.export_csv,
            "export_overlays": args.export_overlays,
            "dry_run": args.dry_run,
        },
    }

    if not args.dry_run:
        candidates_root = args.output_dir / "candidates"
        images_root = candidates_root / "images"
        overlays_root = candidates_root / "overlays"

        images_root.mkdir(parents=True, exist_ok=True)

        manifest_path = candidates_root / "manifest.jsonl"
        csv_path = candidates_root / "review.csv"

        csv_writer: csv.DictWriter[str] | None = None
        csv_file = None

        manifest_file = None
        try:
            manifest_file = manifest_path.open("w", encoding="utf-8")
            if args.export_csv:
                csv_file = csv_path.open("w", encoding="utf-8", newline="")
                csv_writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "candidate_id",
                        "source_image",
                        "source_product_id",
                        "crop_path",
                        "ocr_prefill",
                        "ocr_confidence",
                        "priority_score",
                        "suggested_class",
                        "heuristic_tags",
                        "reviewer_decision",
                        "final_text",
                        "accepted",
                        "final_tag",
                    ],
                )
                csv_writer.writeheader()

            write_progress = ProgressTracker("Writing candidate crops/manifest", len(selected_candidates))
            for index, candidate in enumerate(selected_candidates, start=1):
                crop_name = f"{candidate.candidate_id}.jpg"
                crop_rel_path = f"candidates/images/{crop_name}"
                crop_abs_path = images_root / crop_name
                candidate.crop_image.save(crop_abs_path, format="JPEG")

                record = build_manifest_record(candidate, crop_rel_path)
                manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "candidate_id": record["candidate_id"],
                            "source_image": record["source_image"],
                            "source_product_id": record["source_product_id"],
                            "crop_path": record["crop_path"],
                            "ocr_prefill": record["ocr_prefill"],
                            "ocr_confidence": record["ocr_confidence"],
                            "priority_score": record["priority_score"],
                            "suggested_class": record["suggested_class"],
                            "heuristic_tags": "|".join(record["heuristic_tags"]),
                            "reviewer_decision": "",
                            "final_text": "",
                            "accepted": "",
                            "final_tag": "",
                        }
                    )
                write_progress.tick(index)

            if args.export_overlays:
                render_overlays(selected_candidates, overlays_root)

            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        finally:
            for candidate in selected_candidates:
                candidate.crop_image.close()
            if manifest_file is not None:
                manifest_file.close()
            if csv_file is not None:
                csv_file.close()
    else:
        for candidate in selected_candidates:
            candidate.crop_image.close()

    print("PP-OCRv5 candidate generation summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
