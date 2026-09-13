from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.ai.crop_normalization import TextLineCropConfig, TextLineGeometry, normalize_textline_crops
from app.ai.mobile_expiry_pipeline import MOBILE_RECOGNITION_VARIANTS, MobileExpiryPipeline
from app.infra.settings import get_settings
from app.scripts.mobile_test64_upload_benchmark import (
    IMAGE_SUFFIXES,
    _detected_matches_expected,
    _expected_label,
    _json_default,
    _load_expected_labels,
)
from app.workers.scan_jobs import build_pipeline


TARGET_FILENAMES = (
    "995e3d53-6e35-4cbb-ad8a-f673a5025887.jpg",
    "IMG_0898.JPG",
    "be79d962-d296-460c-8e5a-0f7e354a7072.jpg",
    "e774e8d5-c526-40d5-8fc3-10a82820fb63.jpg",
    "f4f5b08f-6fc8-41bb-8d03-80e1b2ce40a2.jpg",
    "IMG_0885.JPG",
    "cae792a3-f2cf-4a0d-b15f-4707c2eb3415.jpg",
)

DATE_LIKE_PATTERN = re.compile(r"(?<!\d)\d{1,4}\s*[./:-]\s*\d{1,2}(?:\s*[./:-]\s*\d{1,5})?(?!\d)")


@dataclass(frozen=True, slots=True)
class LineExtendVariant:
    name: str
    bbox_xyxy: tuple[int, int, int, int]
    metadata: dict[str, object]


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _load_expected_labels_from_report(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"report does not contain rows: {path}")
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        labels[filename] = {
            "filename": filename,
            "expected_day": row.get("expected_day"),
            "expected_month": row.get("expected_month"),
            "expected_year": row.get("expected_year"),
        }
    return labels


def _bbox_from_json(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    clipped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def _scaled_center_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width_scale: float,
    height_scale: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * (width_scale - 1.0) / 2.0))
    pad_y = int(round(height * (height_scale - 1.0) / 2.0))
    return _clip_bbox((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_shape)


def _left_right_extend_bbox(
    bbox: tuple[int, int, int, int],
    *,
    left_expansion: float,
    right_expansion: float,
    height_scale: float,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_y = int(round(height * (height_scale - 1.0) / 2.0))
    return _clip_bbox(
        (
            x1 - int(round(width * left_expansion)),
            y1 - pad_y,
            x2 + int(round(width * right_expansion)),
            y2 + pad_y,
        ),
        image_shape,
    )


def _line_extend_variants(
    bbox: tuple[int, int, int, int],
    *,
    image_shape: tuple[int, ...],
) -> list[LineExtendVariant]:
    specs: list[tuple[str, tuple[int, int, int, int] | None, dict[str, object]]] = [
        (
            "line_extend_x_1_6",
            _scaled_center_bbox(bbox, width_scale=1.6, height_scale=1.25, image_shape=image_shape),
            {"tight_bbox": list(bbox), "width_scale": 1.6, "height_scale": 1.25},
        ),
        (
            "line_extend_x_2_2",
            _scaled_center_bbox(bbox, width_scale=2.2, height_scale=1.35, image_shape=image_shape),
            {"tight_bbox": list(bbox), "width_scale": 2.2, "height_scale": 1.35},
        ),
        (
            "line_extend_left_heavy",
            _left_right_extend_bbox(
                bbox,
                left_expansion=1.5,
                right_expansion=0.6,
                height_scale=1.3,
                image_shape=image_shape,
            ),
            {"tight_bbox": list(bbox), "left_expansion": 1.5, "right_expansion": 0.6, "height_scale": 1.3},
        ),
        (
            "line_extend_right_heavy",
            _left_right_extend_bbox(
                bbox,
                left_expansion=0.6,
                right_expansion=1.5,
                height_scale=1.3,
                image_shape=image_shape,
            ),
            {"tight_bbox": list(bbox), "left_expansion": 0.6, "right_expansion": 1.5, "height_scale": 1.3},
        ),
    ]
    variants: list[LineExtendVariant] = []
    seen: set[tuple[int, int, int, int]] = set()
    for name, variant_bbox, metadata in specs:
        if variant_bbox is None or variant_bbox == bbox or variant_bbox in seen:
            continue
        seen.add(variant_bbox)
        variants.append(LineExtendVariant(name=name, bbox_xyxy=variant_bbox, metadata=metadata))
    return variants


def _write_png(path: Path, image: np.ndarray) -> str | None:
    if image.size == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writable = image
    if writable.ndim == 3:
        writable = cv2.cvtColor(writable, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", writable)
    if not ok:
        return None
    path.write_bytes(encoded.tobytes())
    return str(path)


def _clean_date_like_match_count(text: str) -> int:
    return len(DATE_LIKE_PATTERN.findall(text or ""))


def _line_extend_accepts(pipeline: MobileExpiryPipeline, text: str, parsed: Any) -> tuple[bool, str | None]:
    if parsed.parsed_date is None:
        return False, "no_parsed_date"
    if pipeline._date_precision_score(parsed) < 2:
        return False, f"precision_{parsed.date_precision or 'none'}"
    digit_count = len(re.findall(r"\d", text or ""))
    if digit_count > 12:
        return False, "too_many_digits"
    if _clean_date_like_match_count(text) != 1:
        return False, "not_one_clean_date_like_pattern"
    return True, None


def _evaluate_line_extend_variant(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    variant: LineExtendVariant,
    *,
    today: date,
    crop_path: Path,
) -> dict[str, Any]:
    crops = normalize_textline_crops(
        image,
        TextLineGeometry(bbox_xyxy=variant.bbox_xyxy, polygon_xy=None),
        TextLineCropConfig(padding_px=0),
    )
    attempts: list[dict[str, Any]] = []
    best_accept: dict[str, Any] | None = None
    saved_crop_path: str | None = None
    for crop_index, crop in enumerate(crops):
        if crop.image.size == 0:
            continue
        if saved_crop_path is None:
            saved_crop_path = _write_png(crop_path, crop.image)
        for image_variant in pipeline._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS):
            rec = pipeline._recognize_variant(
                image_variant.image,
                variant_name=image_variant.name,
                orientation=crop.selected_orientation,
            )
            parse_inputs = pipeline._build_parse_inputs(rec)
            parsed = pipeline._best_parse_for_inputs(parse_inputs, today=today)
            ocr_text = rec.raw_text or rec.normalized_text or ""
            accepted, rejection_reason = _line_extend_accepts(pipeline, ocr_text, parsed)
            row = {
                "crop_index": crop_index,
                "crop_policy": variant.name,
                "bbox_xyxy": list(variant.bbox_xyxy),
                "normalized_crop_bbox_xyxy": list(crop.bbox_xyxy),
                "recognition_variant": image_variant.name,
                "orientation": crop.selected_orientation,
                "raw_text": rec.raw_text,
                "normalized_text": rec.normalized_text,
                "ocr_confidence": rec.confidence,
                "parse_inputs": parse_inputs,
                "parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date is not None else None,
                "date_precision": parsed.date_precision,
                "parser_confidence": parsed.confidence,
                "date_like_pattern_count": _clean_date_like_match_count(ocr_text),
                "digit_count": len(re.findall(r"\d", ocr_text)),
                "accepted_as_late_date_evidence": accepted,
                "rejection_reason": rejection_reason,
            }
            attempts.append(row)
            if accepted and (
                best_accept is None
                or (
                    pipeline._date_precision_score(parsed),
                    float(parsed.confidence or 0.0),
                    float(rec.confidence or 0.0),
                )
                > (
                    3 if best_accept.get("date_precision") == "day" else 2,
                    float(best_accept.get("parser_confidence") or 0.0),
                    float(best_accept.get("ocr_confidence") or 0.0),
                )
            ):
                best_accept = row
    return {
        "crop_policy": variant.name,
        "bbox_xyxy": list(variant.bbox_xyxy),
        "metadata": {
            "source": "crop_policy_single_tight_line_extend",
            "extended_bbox": list(variant.bbox_xyxy),
            **variant.metadata,
        },
        "crop_path": saved_crop_path,
        "attempts": attempts,
        "accepted": [attempt for attempt in attempts if attempt["accepted_as_late_date_evidence"]],
        "best_accept": best_accept,
    }


def _selected_line_extend_accept(variant_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    accepted = [
        attempt
        for variant in variant_reports
        for attempt in variant["accepted"]
    ]
    if not accepted:
        return None
    return max(
        accepted,
        key=lambda attempt: (
            3 if attempt.get("date_precision") == "day" else 2,
            str(attempt.get("parsed_date") or ""),
            float(attempt.get("parser_confidence") or 0.0),
            float(attempt.get("ocr_confidence") or 0.0),
        ),
    )


async def _labels_for_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if args.expected_from_report:
        return _load_expected_labels_from_report(Path(args.expected_from_report))
    try:
        return await _load_expected_labels()
    except Exception as exc:
        print(f"Expected-label DB unavailable; continuing without expected labels: {exc}", flush=True)
        return {}


def _image_paths(args: argparse.Namespace) -> list[Path]:
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"test64 images directory not found: {images_dir}")
    selected_names = set(args.filename or ())
    if not selected_names and not args.all:
        selected_names = set(TARGET_FILENAMES)
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if selected_names:
        paths = [path for path in paths if path.name in selected_names]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("no images selected")
    return paths


async def main_async(args: argparse.Namespace) -> int:
    labels = await _labels_for_args(args)
    pipeline = build_pipeline(get_settings())
    report_root = Path(args.output_dir).resolve() / f"single_tight_line_extend_crop_{_utc_stamp()}"
    crops_root = report_root / "generated_crops"
    report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(_image_paths(args), start=1):
        started = perf_counter()
        output = pipeline.run(path.read_bytes(), today=date.today())
        image = pipeline.decode_image(path.read_bytes())
        expected = _expected_label(labels.get(path.name))
        baseline_detected = output.detected_expiry_date.isoformat() if output.detected_expiry_date else None
        baseline_exact = _detected_matches_expected(baseline_detected, expected)
        base_bbox = _bbox_from_json(output.final_recognition_bbox_xyxy)
        should_try = bool(
            image is not None
            and output.final_crop_policy == "single_tight"
            and output.detected_expiry_date is None
            and base_bbox is not None
        )
        variant_reports: list[dict[str, Any]] = []
        if should_try and image is not None and base_bbox is not None:
            for variant in _line_extend_variants(base_bbox, image_shape=image.shape):
                variant_reports.append(
                    _evaluate_line_extend_variant(
                        pipeline,
                        image,
                        variant,
                        today=date.today(),
                        crop_path=crops_root / path.stem / f"{variant.name}.png",
                    )
                )
        selected_accept = _selected_line_extend_accept(variant_reports)
        selected_detected = selected_accept.get("parsed_date") if selected_accept else None
        line_extend_exact = _detected_matches_expected(selected_detected, expected)
        latency_ms = (perf_counter() - started) * 1000.0
        row = {
            "filename": path.name,
            "expected_date": expected["label"],
            "expected_precision": expected["precision"],
            "baseline_status": output.status,
            "baseline_detected_expiry_date": baseline_detected,
            "baseline_exact_match": baseline_exact,
            "baseline_raw_text": output.raw_text,
            "baseline_normalized_text": output.normalized_text,
            "baseline_reason": output.reason,
            "final_crop_policy": output.final_crop_policy,
            "final_recognition_bbox_xyxy": output.final_recognition_bbox_xyxy,
            "tried_line_extend": should_try,
            "line_extend_accepted_count": sum(len(variant["accepted"]) for variant in variant_reports),
            "line_extend_selected": selected_accept,
            "line_extend_detected_expiry_date": selected_detected,
            "line_extend_exact_match": line_extend_exact,
            "line_extend_variants": variant_reports,
            "latency_ms": round(latency_ms, 3),
        }
        rows.append(row)
        print(
            f"[{index}] {path.name} baseline={output.status}/{output.final_crop_policy} "
            f"raw={output.raw_text!r} tried={should_try} accepted={row['line_extend_accepted_count']} "
            f"selected={selected_detected} expected={expected['label']} exact={line_extend_exact} "
            f"latency_ms={latency_ms:.1f}",
            flush=True,
        )

    tried = [row for row in rows if row["tried_line_extend"]]
    accepted_rows = [row for row in tried if row["line_extend_selected"]]
    fixed_rows = [row for row in accepted_rows if row["line_extend_exact_match"]]
    wrong_rows = [
        row
        for row in accepted_rows
        if row["expected_date"] and row["line_extend_detected_expiry_date"] and not row["line_extend_exact_match"]
    ]
    summary = {
        "total": len(rows),
        "eligible_single_tight_manual_review": len(tried),
        "rows_with_accepted_line_extend_evidence": len(accepted_rows),
        "expected_matches_from_line_extend": len(fixed_rows),
        "wrong_line_extend_dates": len(wrong_rows),
        "target_filenames": list(TARGET_FILENAMES),
    }
    payload = {
        "mode": "single_tight_line_extend_crop_experiment",
        "images_dir": str(Path(args.images_dir).resolve()),
        "created_at": datetime.utcnow(),
        "acceptance_rule": {
            "parsed_date_exists": True,
            "date_precision": ["day", "month"],
            "clean_date_like_pattern_count": 1,
            "max_total_digit_count": 12,
        },
        "summary": summary,
        "rows": rows,
        "fixed_rows": fixed_rows,
        "wrong_rows": wrong_rows,
    }
    report_json = report_root / "single_tight_line_extend_crop_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    summary_md = report_root / "single_tight_line_extend_crop_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Single Tight Line Extend Crop Experiment",
                "",
                f"- total: `{summary['total']}`",
                f"- eligible_single_tight_manual_review: `{summary['eligible_single_tight_manual_review']}`",
                f"- rows_with_accepted_line_extend_evidence: `{summary['rows_with_accepted_line_extend_evidence']}`",
                f"- expected_matches_from_line_extend: `{summary['expected_matches_from_line_extend']}`",
                f"- wrong_line_extend_dates: `{summary['wrong_line_extend_dates']}`",
                "",
                "## Accepted Rows",
                "",
                *[
                    (
                        f"- `{row['filename']}` selected `{row['line_extend_detected_expiry_date']}` "
                        f"expected `{row['expected_date']}` exact `{row['line_extend_exact_match']}` "
                        f"raw `{(row['line_extend_selected'] or {}).get('raw_text')}` "
                        f"policy `{(row['line_extend_selected'] or {}).get('crop_policy')}`"
                    )
                    for row in accepted_rows
                ],
            ]
        ),
        encoding="utf-8",
    )
    print("Report root:", report_root, flush=True)
    print(report_json, flush=True)
    print(summary_md, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment with late line-extended crops for single_tight manual-review fragments")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--expected-from-report", default=None)
    parser.add_argument("--filename", action="append", default=None, help="Specific image filename to test; repeatable")
    parser.add_argument("--all", action="store_true", help="Run every image in --images-dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
