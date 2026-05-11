from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.ai.crop_normalization import (
    NormalizedTextLineCrop,
    TextLineCropConfig,
    TextLineGeometry,
    normalize_textline_crops,
)
from app.ai.parser import ExpiryDateParser
from app.ai.parseq import ParSeqRecognizer
from app.ai.preprocess import ImageVariant, ROIImagePreprocessor
from app.ai.types import ParsedDateData
from app.ai.ocr import PPOCRV5MobileRecognizer, RecognitionData
from app.infra.settings import get_settings
from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH, PROJECT_ROOT


EXPECTED_VARIANT_NAMES = (
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "clahe_gray",
    "unsharp_gray",
    "adaptive_binary",
    "adaptive_binary_inverted",
    "stamp_blackhat",
)
GENERIC_TEXT_TOKENS = (
    "COMPANY",
    "COMPAGNE",
    "COMPAGNER",
    "COMPAGNESS",
    "PROGRAPHY",
    "PHOTOGRAPHY",
    "COCA",
    "COLA",
)


@dataclass(slots=True)
class TruthCropItem:
    filename: str
    expected_date: date | None
    expected_day: int | None
    expected_month: int | None
    expected_year: int | None
    true_bbox_xyxy: list[float]
    true_polygon_xy: list[list[float]] | None = None
    image_url: str | None = None


@dataclass(slots=True)
class VariantRunResult:
    filename: str
    expected_date: str | None
    true_bbox_xyxy: list[float]
    variant_name: str
    crop_path: str
    parseq_raw_output: str
    parseq_normalized_output: str
    parseq_confidence: float | None
    parseq_reason: str | None
    parseq_runtime_ms: int
    parser_parsed_date: str | None
    parser_confidence: float
    parser_reason: str
    parser_candidates: list[str]
    parser_date_precision: str | None
    parser_parsed_day: int | None
    parser_parsed_month: int | None
    parser_parsed_year: int | None
    exact_match: bool
    malformed_but_potentially_recoverable: bool
    generic_text_output: bool
    compact_digit_analysis: dict[str, Any]
    true_polygon_xy: list[list[float]] | None = None
    crop_transform_used: str | None = None
    selected_orientation: str | None = None
    orientation_candidates_tried: list[str] | None = None
    original_crop_shape: list[int] | None = None
    normalized_crop_shape: list[int] | None = None
    selected_transform_reason: str | None = None


class BenchmarkRecognizer:
    def recognize(self, image: np.ndarray) -> RecognitionData:
        raise NotImplementedError

    def runtime_info(self) -> dict[str, Any]:
        return {}


class ParSeqBenchmarkRecognizer(BenchmarkRecognizer):
    def __init__(self, recognizer: ParSeqRecognizer) -> None:
        self.recognizer = recognizer

    def recognize(self, image: np.ndarray) -> RecognitionData:
        output = self.recognizer.recognize(image)
        return RecognitionData(
            raw_text=output.text,
            normalized_text=_normalize_text(output.text),
            confidence=output.confidence,
            reason=output.reason,
        )

    def runtime_info(self) -> dict[str, Any]:
        return self.recognizer.runtime_info()


class PPOCRBenchmarkRecognizer(BenchmarkRecognizer):
    def __init__(self, recognizer: PPOCRV5MobileRecognizer, *, model_name: str, model_dir: Path | None) -> None:
        self.recognizer = recognizer
        self.model_name = model_name
        self.model_dir = model_dir

    def recognize(self, image: np.ndarray) -> RecognitionData:
        return self.recognizer.recognize(image)

    def runtime_info(self) -> dict[str, Any]:
        info = self.recognizer.runtime_info()
        return {
            "model_name": self.model_name,
            "model_dir": str(self.model_dir) if self.model_dir is not None else None,
            "runtime_device": self.recognizer.runtime_device,
            "backend": "paddleocr.TextRecognition",
            "model_loaded": info.get("model_loaded"),
            "load_error": info.get("load_error"),
        }


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Type not JSON serializable: {type(value)}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_png(path: Path, image: np.ndarray) -> str:
    _ensure_dir(path.parent)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode png: {path}")
    path.write_bytes(encoded.tobytes())
    return str(path)


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").strip().upper().split())


def expected_label(day: int | None, month: int | None, year: int | None) -> str | None:
    if month is None or year is None:
        return None
    if day is None:
        return f"{int(month):02d}/{int(year):04d}"
    return date(int(year), int(month), int(day)).isoformat()


def parsed_label(parsed: ParsedDateData) -> str | None:
    parsed_month = parsed.parsed_month if parsed.parsed_month is not None else (parsed.parsed_date.month if parsed.parsed_date else None)
    parsed_year = parsed.parsed_year if parsed.parsed_year is not None else (parsed.parsed_date.year if parsed.parsed_date else None)
    parsed_day = (
        None
        if parsed.date_precision == "month"
        else parsed.parsed_day if parsed.parsed_day is not None else (parsed.parsed_date.day if parsed.parsed_date else None)
    )
    if parsed_month is None or parsed_year is None:
        return None
    if parsed.date_precision == "month" or parsed_day is None:
        return f"{int(parsed_month):02d}/{int(parsed_year):04d}"
    return date(int(parsed_year), int(parsed_month), int(parsed_day)).isoformat()


def parser_matches_expected(
    parsed: ParsedDateData,
    expected_day: int | None,
    expected_month: int | None,
    expected_year: int | None,
) -> bool:
    if expected_month is None or expected_year is None:
        return False
    parsed_month = parsed.parsed_month if parsed.parsed_month is not None else (parsed.parsed_date.month if parsed.parsed_date else None)
    parsed_year = parsed.parsed_year if parsed.parsed_year is not None else (parsed.parsed_date.year if parsed.parsed_date else None)
    parsed_day = (
        None
        if parsed.date_precision == "month"
        else parsed.parsed_day if parsed.parsed_day is not None else (parsed.parsed_date.day if parsed.parsed_date else None)
    )
    parsed_precision = parsed.date_precision or ("day" if parsed.parsed_date is not None else None)
    if parsed_month != int(expected_month) or parsed_year != int(expected_year):
        return False
    if expected_day is None:
        return True
    return parsed_precision == "day" and parsed_day == int(expected_day)


def _resolve_test64_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        if explicit.exists() and explicit.is_dir():
            return explicit
        raise SystemExit(f"test64 directory not found: {explicit}")
    candidates = [
        PROJECT_ROOT / "test64",
        Path.cwd() / "test64",
        Path("/app/test64"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise SystemExit(f"test64 directory not found. Tried: {', '.join(str(path) for path in candidates)}")


def _clamp_bbox(bbox: list[Any] | tuple[Any, Any, Any, Any], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(x1, max(0, w - 1)))
    y1 = max(0, min(y1, max(0, h - 1)))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]


def _clean_polygon(raw: Any, shape: tuple[int, ...]) -> list[list[float]] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    h, w = shape[:2]
    points: list[list[float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        points.append([max(0.0, min(float(w), x)), max(0.0, min(float(h), y))])
    return points


def _crop_polygon(image: np.ndarray, polygon: list[list[float]]) -> np.ndarray:
    points = [np.array(point, dtype=np.float32) for point in polygon]
    width = max(1, int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))))
    height = max(1, int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))))
    src = np.array(polygon, dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def load_annotated_truth_items(path: Path) -> list[TruthCropItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError("truth manifest must contain an items object")

    out: list[TruthCropItem] = []
    for filename in sorted(items):
        raw = items[filename]
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("true_bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            cleaned_bbox = [float(value) for value in bbox]
            day = raw.get("expected_day")
            month = raw.get("expected_month")
            year = raw.get("expected_year")
            expected = date(int(year), int(month), int(day)) if day and month and year else None
            expected_day = int(day) if day is not None else None
            expected_month = int(month) if month is not None else None
            expected_year = int(year) if year is not None else None
        except (TypeError, ValueError):
            continue
        if cleaned_bbox[2] <= cleaned_bbox[0] or cleaned_bbox[3] <= cleaned_bbox[1]:
            continue
        out.append(
            TruthCropItem(
                filename=str(raw.get("filename") or filename),
                expected_date=expected,
                expected_day=expected_day,
                expected_month=expected_month,
                expected_year=expected_year,
                true_bbox_xyxy=cleaned_bbox,
                true_polygon_xy=raw.get("true_polygon_xy") if isinstance(raw.get("true_polygon_xy"), list) else None,
                image_url=raw.get("image_url") if isinstance(raw.get("image_url"), str) else None,
            )
        )
    return out


def generate_benchmark_variants(crop: np.ndarray, preprocessor: ROIImagePreprocessor | None = None) -> list[ImageVariant]:
    preprocessor = preprocessor or ROIImagePreprocessor()
    variants = [ImageVariant("original_color_tight", crop.copy(), purpose="recognition")]
    generated = preprocessor.recognition_variants(crop, allowed_names=EXPECTED_VARIANT_NAMES[1:])
    seen = {"original_color_tight"}
    for variant in generated:
        if variant.name in EXPECTED_VARIANT_NAMES and variant.name not in seen:
            variants.append(variant)
            seen.add(variant.name)
    return variants


def resolve_recognizer_config(
    *,
    recognizer_name: str,
    settings: Any,
    ppocr_rec_model_dir: Path | None,
    ppocr_rec_model_name: str | None,
) -> dict[str, Any]:
    normalized = recognizer_name.strip().lower()
    if normalized == "parseq":
        return {
            "recognizer": "parseq",
            "model_name": "parseq",
            "model_dir": settings.parseq_model_dir,
            "device_mode": settings.parseq_device_mode,
        }
    if normalized in {"ppocrv5_server_rec", "ppocr_server_rec", "ppocr"}:
        return {
            "recognizer": "ppocrv5_server_rec",
            "model_name": ppocr_rec_model_name or "PP-OCRv5_server_rec",
            "model_dir": ppocr_rec_model_dir or settings.ocr_ppocrv5_main_model_dir,
            "device_mode": settings.ocr_device_mode,
        }
    if normalized == "svtrv2":
        return {
            "recognizer": "svtrv2",
            "model_name": ppocr_rec_model_name or getattr(settings, "svtrv2_rec_model_name", "ch_SVTRv2_rec"),
            "model_dir": ppocr_rec_model_dir or getattr(settings, "svtrv2_rec_model_dir", Path("models/svtrv2/ch_SVTRv2_rec")),
            "device_mode": getattr(settings, "svtrv2_device_mode", "cpu"),
        }
    raise ValueError("recognizer must be one of: parseq, ppocrv5_server_rec, svtrv2")


def build_benchmark_recognizer(config: dict[str, Any]) -> BenchmarkRecognizer:
    if config["recognizer"] == "parseq":
        return ParSeqBenchmarkRecognizer(
            ParSeqRecognizer(
                model_dir=Path(config["model_dir"]),
                device_mode=str(config["device_mode"]),
            )
        )
    if config["recognizer"] in {"ppocrv5_server_rec", "svtrv2"}:
        model_dir_raw = config.get("model_dir")
        model_dir = Path(model_dir_raw) if model_dir_raw is not None else None
        model_name = str(config.get("model_name") or ("ch_SVTRv2_rec" if config["recognizer"] == "svtrv2" else "PP-OCRv5_server_rec"))
        return PPOCRBenchmarkRecognizer(
            PPOCRV5MobileRecognizer(
                model_name=model_name,
                model_dir=model_dir,
                runtime_device=str(config.get("device_mode") or "cpu"),
                engine_name=str(config["recognizer"]),
            ),
            model_name=model_name,
            model_dir=model_dir,
        )
    raise ValueError(f"unsupported recognizer: {config['recognizer']}")


def is_generic_text_output(text: str | None) -> bool:
    normalized = _normalize_text(text)
    compact = re.sub(r"[^A-Z0-9]", "", normalized)
    if not compact:
        return False
    return any(token in normalized.split() or token in compact for token in GENERIC_TEXT_TOKENS)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _compact_candidate_dates(digits: str) -> list[date]:
    out: list[date] = []
    if len(digits) == 6:
        dd, mm, yy = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
        for year in (2000 + yy, 1900 + yy):
            parsed = _safe_date(year, mm, dd)
            if parsed:
                out.append(parsed)
    elif len(digits) == 8:
        shapes = (
            (int(digits[:2]), int(digits[2:4]), int(digits[4:8])),  # DDMMYYYY
            (int(digits[6:8]), int(digits[4:6]), int(digits[:4])),  # YYYYMMDD
        )
        for dd, mm, yyyy in shapes:
            parsed = _safe_date(yyyy, mm, dd)
            if parsed:
                out.append(parsed)
    return list(dict.fromkeys(out))


def analyze_compact_digits(text: str | None, expected_date: date | None = None) -> dict[str, Any]:
    value = text or ""
    digit_runs = re.findall(r"\d+", value)
    serial_runs = [run for run in digit_runs if len(run) > 8]
    candidate_dates: list[date] = []
    candidate_lengths: list[int] = []
    for run in digit_runs:
        if len(run) not in {6, 8}:
            continue
        dates = _compact_candidate_dates(run)
        if not dates:
            continue
        candidate_lengths.append(len(run))
        candidate_dates.extend(dates)
    candidate_dates = list(dict.fromkeys(candidate_dates))
    return {
        "digit_runs": digit_runs,
        "candidate_lengths": candidate_lengths,
        "candidate_dates": [item.isoformat() for item in candidate_dates],
        "has_compact_digit_candidate": bool(candidate_dates),
        "matches_expected": bool(expected_date is not None and expected_date in candidate_dates),
        "rejected_as_serial": bool(serial_runs and not candidate_dates),
        "serial_digit_runs": serial_runs,
    }


def _looks_malformed_date_like(text: str | None) -> bool:
    value = text or ""
    digits = sum(ch.isdigit() for ch in value)
    separators = sum(ch in "/.-:" for ch in value)
    digit_only = re.sub(r"\D", "", value)
    if len(digit_only) >= 8 and len(set(digit_only)) <= 2:
        return False
    return bool(digits >= 4 and (separators >= 1 or digits >= 6))


def _result_priority(result: VariantRunResult) -> tuple[int, float, float]:
    if result.exact_match:
        return (4, float(result.parser_confidence), float(result.parseq_confidence or 0.0))
    if result.parser_parsed_date:
        return (3, float(result.parser_confidence), float(result.parseq_confidence or 0.0))
    if result.malformed_but_potentially_recoverable:
        return (2, 0.0, float(result.parseq_confidence or 0.0))
    return (1, 0.0, float(result.parseq_confidence or 0.0))


def select_image_winner(results: list[VariantRunResult]) -> VariantRunResult | None:
    if not results:
        return None
    return max(results, key=_result_priority)


def classify_image_result(results: list[VariantRunResult], expected_date: date | None) -> str:
    if not results:
        return "image_unreadable"
    exact = [item for item in results if item.exact_match]
    original_exact = any(
        item.exact_match and item.variant_name.split("/")[-1] == "original_color_tight" for item in results
    )
    if exact and not original_exact:
        return "preprocessing_variant_needed"
    if exact:
        return "success_without_preprocessing_requirement"

    parseable = [item for item in results if item.parser_parsed_date]
    if parseable:
        parsed_dates = {item.parser_parsed_date for item in parseable}
        if len(parsed_dates) > 1:
            return "ambiguous_multi_date_crop"
        return "wrong_date_from_crop"

    recoverable = [item for item in results if item.malformed_but_potentially_recoverable]
    if recoverable and expected_date is not None and any(item.compact_digit_analysis.get("matches_expected") for item in recoverable):
        return "parser_too_strict"
    if recoverable:
        return "malformed_but_recoverable"

    if all(item.generic_text_output or _is_unreadable_text(item.parseq_raw_output) for item in results):
        return "recognizer_unreadable_crop"
    if any(item.generic_text_output for item in results):
        return "crop_label_or_bbox_issue"
    return "recognizer_unreadable_crop"


def _is_unreadable_text(text: str | None) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    digit_only = re.sub(r"\D", "", value)
    if len(digit_only) >= 8 and len(set(digit_only)) <= 2:
        return True
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    return bool(compact and len(compact) >= 6 and len(set(compact)) <= 2)


def _run_variant(
    *,
    recognizer: BenchmarkRecognizer,
    parser: ExpiryDateParser,
    filename: str,
    expected_date: date | None,
    expected_day: int | None,
    expected_month: int | None,
    expected_year: int | None,
    true_bbox: list[float],
    true_polygon: list[list[float]] | None,
    variant: ImageVariant,
    crop_path: str,
    today: date,
    crop_transform_used: str | None = None,
    selected_orientation: str | None = None,
    orientation_candidates_tried: list[str] | None = None,
    original_crop_shape: list[int] | None = None,
    normalized_crop_shape: list[int] | None = None,
    selected_transform_reason: str | None = None,
) -> VariantRunResult:
    start = perf_counter()
    rec = recognizer.recognize(variant.image)
    runtime_ms = int((perf_counter() - start) * 1000)
    normalized = _normalize_text(rec.normalized_text or rec.raw_text)
    parsed = parser.parse(normalized, reference_date=today)
    compact = analyze_compact_digits(rec.raw_text, expected_date)
    malformed = bool(parsed.parsed_date is None and (_looks_malformed_date_like(rec.raw_text) or compact["has_compact_digit_candidate"]))
    parsed_display = parsed_label(parsed)
    return VariantRunResult(
        filename=filename,
        expected_date=expected_label(expected_day, expected_month, expected_year),
        true_bbox_xyxy=true_bbox,
        true_polygon_xy=true_polygon,
        variant_name=variant.name,
        crop_path=crop_path,
        parseq_raw_output=rec.raw_text,
        parseq_normalized_output=normalized,
        parseq_confidence=rec.confidence,
        parseq_reason=rec.reason,
        parseq_runtime_ms=runtime_ms,
        parser_parsed_date=parsed_display,
        parser_confidence=parsed.confidence,
        parser_reason=parsed.reason,
        parser_candidates=parsed.candidates,
        parser_date_precision=parsed.date_precision,
        parser_parsed_day=parsed.parsed_day,
        parser_parsed_month=parsed.parsed_month,
        parser_parsed_year=parsed.parsed_year,
        exact_match=parser_matches_expected(parsed, expected_day, expected_month, expected_year),
        malformed_but_potentially_recoverable=malformed,
        generic_text_output=is_generic_text_output(rec.raw_text),
        compact_digit_analysis=compact,
        crop_transform_used=crop_transform_used,
        selected_orientation=selected_orientation,
        orientation_candidates_tried=orientation_candidates_tried,
        original_crop_shape=original_crop_shape,
        normalized_crop_shape=normalized_crop_shape,
        selected_transform_reason=selected_transform_reason,
    )


def _audit_one(
    *,
    item: TruthCropItem,
    test64_dir: Path,
    report_root: Path,
    recognizer: BenchmarkRecognizer,
    parser: ExpiryDateParser,
    preprocessor: ROIImagePreprocessor,
    today: date,
) -> dict[str, Any]:
    image_path = test64_dir / item.filename
    scan_dir = report_root / "images" / Path(item.filename).stem
    _ensure_dir(scan_dir / "variants")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "filename": item.filename,
            "expected_date": expected_label(item.expected_day, item.expected_month, item.expected_year),
            "outcome": "image_unreadable",
            "image_path": str(image_path),
            "variants": [],
            "winner": None,
        }

    bbox = _clamp_bbox(item.true_bbox_xyxy, image.shape)
    polygon = _clean_polygon(item.true_polygon_xy, image.shape)
    crop = _crop_polygon(image, polygon) if polygon else _crop(image, bbox)
    manual_crop_path = _write_png(scan_dir / "manual_true_crop.png", crop)
    normalized_crops = normalize_textline_crops(
        image,
        TextLineGeometry(
            bbox_xyxy=tuple(int(v) for v in bbox),
            polygon_xy=tuple((float(x), float(y)) for x, y in polygon) if polygon else None,
        ),
        TextLineCropConfig(padding_px=0, vertical_aspect_threshold=1.5),
    )

    results: list[VariantRunResult] = []
    for normalized_crop in normalized_crops:
        results.extend(
            _run_normalized_crop_variants(
                normalized_crop=normalized_crop,
                scan_dir=scan_dir,
                recognizer=recognizer,
                parser=parser,
                filename=item.filename,
                expected_date=item.expected_date,
                expected_day=item.expected_day,
                expected_month=item.expected_month,
                expected_year=item.expected_year,
                true_bbox=item.true_bbox_xyxy,
                true_polygon=polygon,
                preprocessor=preprocessor,
                today=today,
            )
        )
    if (
        len(normalized_crops) == 1
        and normalized_crops[0].selected_orientation == "original"
        and not any(result.exact_match for result in results)
    ):
        results.extend(
            _run_normalized_crop_variants(
                normalized_crop=_rotate_180_fallback_crop(normalized_crops[0]),
                scan_dir=scan_dir,
                recognizer=recognizer,
                parser=parser,
                filename=item.filename,
                expected_date=item.expected_date,
                expected_day=item.expected_day,
                expected_month=item.expected_month,
                expected_year=item.expected_year,
                true_bbox=item.true_bbox_xyxy,
                true_polygon=polygon,
                preprocessor=preprocessor,
                today=today,
            )
        )

    winner = select_image_winner(results)
    outcome = classify_image_result(results, item.expected_date)
    return {
        "filename": item.filename,
        "image_path": str(image_path),
        "expected_date": expected_label(item.expected_day, item.expected_month, item.expected_year),
        "expected_day": item.expected_day,
        "expected_month": item.expected_month,
        "expected_year": item.expected_year,
        "true_bbox_xyxy": item.true_bbox_xyxy,
        "true_polygon_xy": polygon,
        "manual_true_crop_source": "true_polygon_xy" if polygon else "true_bbox_xyxy",
        "manual_true_crop_path": manual_crop_path,
        "outcome": outcome,
        "winner": asdict(winner) if winner else None,
        "variants": [asdict(result) for result in results],
        "special_checks": _special_checks(item.filename, results, item.expected_date),
    }


def _run_normalized_crop_variants(
    *,
    normalized_crop: NormalizedTextLineCrop,
    scan_dir: Path,
    recognizer: BenchmarkRecognizer,
    parser: ExpiryDateParser,
    filename: str,
    expected_date: date | None,
    expected_day: int | None,
    expected_month: int | None,
    expected_year: int | None,
    true_bbox: list[float],
    true_polygon: list[list[float]] | None,
    preprocessor: ROIImagePreprocessor,
    today: date,
) -> list[VariantRunResult]:
    results: list[VariantRunResult] = []
    transform_dir = scan_dir / "variants" / normalized_crop.selected_orientation
    _write_png(transform_dir / "normalized_crop.png", normalized_crop.image)
    variants = generate_benchmark_variants(normalized_crop.image, preprocessor)
    for variant in variants:
        display_variant = ImageVariant(
            name=f"{normalized_crop.selected_orientation}/{variant.name}",
            image=variant.image,
            purpose=variant.purpose,
        )
        crop_path = _write_png(transform_dir / f"{variant.name}.png", variant.image)
        results.append(
            _run_variant(
                recognizer=recognizer,
                parser=parser,
                filename=filename,
                expected_date=expected_date,
                expected_day=expected_day,
                expected_month=expected_month,
                expected_year=expected_year,
                true_bbox=true_bbox,
                true_polygon=true_polygon,
                variant=display_variant,
                crop_path=crop_path,
                today=today,
                crop_transform_used=normalized_crop.crop_transform_used,
                selected_orientation=normalized_crop.selected_orientation,
                orientation_candidates_tried=normalized_crop.orientation_candidates_tried,
                original_crop_shape=normalized_crop.original_crop_shape,
                normalized_crop_shape=normalized_crop.normalized_crop_shape,
                selected_transform_reason=normalized_crop.selected_transform_reason,
            )
        )
    return results


def _rotate_180_fallback_crop(crop: NormalizedTextLineCrop) -> NormalizedTextLineCrop:
    image = cv2.rotate(crop.image, cv2.ROTATE_180) if crop.image.size else crop.image
    return NormalizedTextLineCrop(
        image=image,
        bbox_xyxy=crop.bbox_xyxy,
        polygon_used=crop.polygon_used,
        crop_transform_used="rotate_180",
        selected_orientation="rotate_180",
        original_crop_shape=crop.original_crop_shape,
        normalized_crop_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2]) if image.ndim == 3 else 1],
        orientation_candidates_tried=["original", "rotate_180"],
        selected_transform_reason="original_failed_expected_date",
    )


def _special_checks(filename: str, results: list[VariantRunResult], expected_date: date | None) -> dict[str, Any]:
    compact_filename = filename.lower()
    parsed_dates = sorted({item.parser_parsed_date for item in results if item.parser_parsed_date})
    raw_outputs = [item.parseq_raw_output for item in results]
    checks: dict[str, Any] = {}
    if "65940c3a" in compact_filename:
        checks["parses_expected_2027_12_12"] = "2027-12-12" in parsed_dates
        checks["parses_production_2025_12_12"] = "2025-12-12" in parsed_dates
    if filename == "IMG_0886.JPG":
        checks["contains_060528"] = any("060528" in output for output in raw_outputs)
        checks["compact_digit_outputs"] = [
            asdict(item) for item in results if item.compact_digit_analysis.get("has_compact_digit_candidate")
        ]
        checks["expected_date"] = expected_date.isoformat() if expected_date else None
        checks["060528_interpretation"] = (
            "manual true-crop variants did not reproduce 060528; current evidence points to recognition/crop-context "
            "instability rather than a parser-only compact-date bug"
            if not checks["contains_060528"]
            else "manual true-crop variants reproduced 060528; inspect compact-date parser constraints before accepting"
        )
    return checks


def _write_variant_table(report_root: Path, results: list[dict[str, Any]]) -> Path:
    lines = [
        "# Manual Crop Recognition Variant Comparison",
        "",
        "| filename | expected | outcome | variant | raw output | parsed date | exact | confidence | runtime ms |",
        "|---|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for item in results:
        for variant in item.get("variants", []):
            lines.append(
                "| {filename} | {expected} | {outcome} | {variant} | {raw} | {parsed} | {exact} | {conf} | {ms} |".format(
                    filename=item.get("filename"),
                    expected=item.get("expected_date"),
                    outcome=item.get("outcome"),
                    variant=variant.get("variant_name"),
                    raw=str(variant.get("parseq_raw_output") or "").replace("|", "\\|"),
                    parsed=variant.get("parser_parsed_date"),
                    exact=variant.get("exact_match"),
                    conf=variant.get("parseq_confidence"),
                    ms=variant.get("parseq_runtime_ms"),
                )
            )
    path = report_root / "variant_comparison_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_summary(report_root: Path, report: dict[str, Any]) -> Path:
    summary = report["summary"]
    lines = [
        "# Manual Crop Recognition Benchmark Summary",
        "",
        f"- total_crops: `{summary['total_crops']}`",
        f"- exact_match_crops: `{summary['exact_match_crops']}`",
        f"- generated_at: `{report['generated_at']}`",
        "",
        "## Outcomes",
    ]
    for key, count in sorted(summary["outcome_counts"].items()):
        lines.append(f"- {key}: `{count}`")
    lines.extend(["", "## Per Crop"])
    for item in report["items"]:
        winner = item.get("winner") or {}
        lines.extend(
            [
                f"### {item['filename']}",
                f"- expected: `{item.get('expected_date')}`",
                f"- outcome: `{item.get('outcome')}`",
                f"- winner_variant: `{winner.get('variant_name')}`",
                f"- winner_text: `{winner.get('parseq_raw_output')}`",
                f"- winner_parsed_date: `{winner.get('parser_parsed_date')}`",
                f"- manual_true_crop: `{item.get('manual_true_crop_path')}`",
                "",
            ]
        )
    path = report_root / "manual_crop_recognition_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual true-crop recognition benchmark")
    parser.add_argument("--truth-manifest", type=Path, default=CROP_TRUTH_ANNOTATIONS_PATH)
    parser.add_argument("--test64-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics"))
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--all-annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recognizer", choices=("parseq", "ppocrv5_server_rec", "svtrv2"), default="parseq")
    parser.add_argument("--ppocr-rec-model-dir", type=Path, default=None)
    parser.add_argument("--ppocr-rec-model-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_annotated_truth_items(args.truth_manifest)
    if not args.all_annotated:
        items = items[:10]
    if not items:
        raise SystemExit(f"no annotated truth bboxes found in {args.truth_manifest}")

    settings = get_settings()
    today = date.fromisoformat(str(args.today))
    test64_dir = _resolve_test64_dir(args.test64_dir)
    recognizer_config = resolve_recognizer_config(
        recognizer_name=args.recognizer,
        settings=settings,
        ppocr_rec_model_dir=args.ppocr_rec_model_dir,
        ppocr_rec_model_name=args.ppocr_rec_model_name,
    )
    report_root = args.output_dir.resolve() / f"manual_crop_recognition_{recognizer_config['recognizer']}_{_utc_stamp()}"
    _ensure_dir(report_root)
    recognizer = build_benchmark_recognizer(recognizer_config)
    preprocessor = ROIImagePreprocessor()
    parser = ExpiryDateParser(min_candidate_confidence=settings.parser_min_candidate_confidence)

    results: list[dict[str, Any]] = []
    for item in items:
        print(f"[manual-crop-recognition] benchmarking {item.filename}", flush=True)
        results.append(
            _audit_one(
                item=item,
                test64_dir=test64_dir,
                report_root=report_root,
                recognizer=recognizer,
                parser=parser,
                preprocessor=preprocessor,
                today=today,
            )
        )

    outcome_counts: dict[str, int] = {}
    exact_match_crops = 0
    for item in results:
        outcome = str(item.get("outcome") or "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        winner = item.get("winner") or {}
        if winner.get("exact_match"):
            exact_match_crops += 1

    summary = {
        "total_crops": len(results),
        "exact_match_crops": exact_match_crops,
        "outcome_counts": outcome_counts,
    }
    report = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "truth_manifest": str(args.truth_manifest),
        "test64_dir": str(test64_dir),
        "report_root": str(report_root),
        "variant_names": list(EXPECTED_VARIANT_NAMES),
        "recognizer": recognizer_config["recognizer"],
        "recognizer_config": {key: str(value) if isinstance(value, Path) else value for key, value in recognizer_config.items()},
        "recognizer_runtime_info": recognizer.runtime_info(),
        "summary": summary,
        "items": results,
    }
    report_path = report_root / "manual_crop_recognition_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    summary_path = _write_summary(report_root, report)
    table_path = _write_variant_table(report_root, results)
    print(f"manual crop recognition report: {report_path}")
    print(f"manual crop recognition summary: {summary_path}")
    print(f"variant comparison table: {table_path}")


if __name__ == "__main__":
    main()
