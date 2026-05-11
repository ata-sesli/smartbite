from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import text

from app.ai.types import OCRResultData
from app.infra.db import get_session_factory
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage
from app.workers.scan_jobs import build_pipeline


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Type not JSON serializable: {type(value)}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _encode_png(image: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return encoded.tobytes()


def _write_png(path: Path, image: np.ndarray) -> str | None:
    data = _encode_png(image)
    if data is None:
        return None
    path.write_bytes(data)
    return str(path)


def _pad_border(image: np.ndarray, ratio: float = 0.12) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = max(1, int(round(w * ratio)))
    pad_y = max(1, int(round(h * ratio)))
    return cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)


async def _load_latest_batch_rows(source: str, max_rows: int, batch_gap_seconds: int) -> list[dict[str, Any]]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    id,
                    image_path,
                    roi_path,
                    status,
                    final_result_status,
                    failure_reason,
                    created_at
                FROM scans
                WHERE metadata_json->>'source' = :source
                ORDER BY created_at DESC
                LIMIT :max_rows
                """
            ),
            {"source": source, "max_rows": max_rows},
        )
        rows = [dict(row) for row in result.mappings().all()]

    if not rows:
        return []

    latest_batch: list[dict[str, Any]] = [rows[0]]
    prev = rows[0]["created_at"]
    for row in rows[1:]:
        current = row["created_at"]
        gap = abs((prev - current).total_seconds())
        if gap > batch_gap_seconds:
            break
        latest_batch.append(row)
        prev = current
    return latest_batch


async def _load_known_good_scans(limit: int = 50) -> list[dict[str, Any]]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    s.id,
                    s.image_path,
                    s.roi_path,
                    s.created_at,
                    p.parsed_date,
                    p.parse_confidence
                FROM scans s
                JOIN parsed_date_results p ON p.scan_id = s.id
                WHERE s.final_result_status = 'SUCCESS'
                  AND p.parsed_date IS NOT NULL
                ORDER BY p.parse_confidence DESC NULLS LAST, s.created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        return [dict(row) for row in result.mappings().all()]


def _variant_key(source: str, variant: str) -> str:
    return f"{source}:{variant}"


def _detect_variant_reports(
    *,
    pipeline,
    roi_source: str,
    roi_image: np.ndarray,
    detector_confidence: float | None,
    scan_dir: Path,
    today: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    variants = pipeline._variants_for_roi(roi_image)
    variant_reports: list[dict[str, Any]] = []
    evaluated_flat: list[dict[str, Any]] = []
    parseable_count = 0

    for variant_idx, (variant_name, variant_image) in enumerate(variants, start=1):
        key = _variant_key(roi_source, variant_name)
        detected_boxes, det_reason = pipeline.ocr_router.detect_text_boxes(variant_image)

        ranked = pipeline._build_ranked_candidates(variant_image, detected_boxes)
        ranked = pipeline._apply_geometry_shortlist(ranked)

        for candidate in ranked:
            if not candidate.selected_geometry:
                continue
            crop = candidate.crop(variant_image)
            if crop.size == 0:
                candidate.probe_reason = "candidate crop empty"
                continue

            probe = pipeline.ocr_router.recognize_probe(crop)
            probe_scores = pipeline._probe_signal_scores(probe)
            candidate.probe_text = probe.raw_text
            candidate.probe_normalized_text = probe.normalized_text
            candidate.probe_confidence = probe.confidence
            candidate.probe_reason = probe.reason
            candidate.score_breakdown.update(probe_scores)
            candidate.total_score = candidate.geometry_score + sum(probe_scores.values())

        final_candidates = pipeline._select_final_candidates(ranked)
        if not final_candidates and ranked:
            ranked[0].selected_final = True
            ranked[0].total_score = ranked[0].geometry_score
            final_candidates = [ranked[0]]

        final_candidate_reports: list[dict[str, Any]] = []
        for idx, candidate in enumerate(final_candidates, start=1):
            crop = candidate.crop(variant_image)
            if crop.size == 0:
                continue

            crop_path = scan_dir / f"{roi_source}_{variant_name}_final_{idx}_{candidate.candidate_id}.png"
            saved_crop_path = _write_png(crop_path, crop)

            parseq = pipeline.ocr_router.recognize_parseq(crop)
            ocr_data = OCRResultData(
                raw_text=parseq.raw_text,
                normalized_text=parseq.normalized_text,
                confidence=parseq.confidence,
                engine_name=pipeline.ocr_router.active_engine_name,
                runtime_device=pipeline.ocr_router.parseq_runtime_device,
                reason=parseq.reason,
            )
            parse_inputs = pipeline._build_parse_inputs(ocr_data)
            parsed = pipeline._best_parse_for_inputs(parse_inputs, today=today)
            if parsed.parsed_date is not None:
                parseable_count += 1

            candidate.parseq_text = parseq.raw_text
            candidate.parseq_confidence = parseq.confidence
            candidate.parseq_reason = parseq.reason
            candidate.parsed_date = parsed.parsed_date.isoformat() if parsed.parsed_date else None
            candidate.parse_confidence = parsed.confidence

            item = {
                "candidate_id": candidate.candidate_id,
                "bbox_xyxy": list(candidate.bbox),
                "total_score": candidate.total_score,
                "crop_path": saved_crop_path,
                "parseq_raw_text": parseq.raw_text,
                "parseq_confidence": parseq.confidence,
                "parseq_reason": parseq.reason,
                "parse_inputs": parse_inputs,
                "parser_parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date else None,
                "parser_date_format": parsed.date_format_detected,
                "parser_confidence": parsed.confidence,
                "parser_reason": parsed.reason,
            }
            final_candidate_reports.append(item)

            evaluated_flat.append(
                {
                    "variant_key": key,
                    "source": roi_source,
                    "variant": variant_name,
                    "detector_confidence": detector_confidence,
                    "candidate_id": candidate.candidate_id,
                    "candidate_total_score": candidate.total_score,
                    "candidate_bbox": list(candidate.bbox),
                    "ocr_raw_text": parseq.raw_text,
                    "ocr_confidence": parseq.confidence,
                    "ocr_reason": parseq.reason,
                    "parser_parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date else None,
                    "parser_confidence": parsed.confidence,
                    "parser_reason": parsed.reason,
                }
            )

        selected_id = final_candidates[0].candidate_id if final_candidates else None
        overlay = pipeline._render_debug_overlay(
            image=variant_image,
            det_boxes=detected_boxes,
            ranked=ranked,
            selected_candidate_id=selected_id,
        )
        overlay_path = scan_dir / f"{roi_source}_{variant_name}_overlay.png"
        saved_overlay_path = _write_png(overlay_path, overlay)

        variant_reports.append(
            {
                "source": roi_source,
                "variant": variant_name,
                "variant_key": key,
                "detector_confidence": detector_confidence,
                "text_detector_reason": det_reason,
                "text_detection_boxes": [asdict(box) for box in detected_boxes],
                "geometry_shortlist_top_n": pipeline.expiry_filter_geometry_top_n,
                "final_top_k": pipeline.expiry_filter_final_top_k,
                "geometry_shortlist_ids": [c.candidate_id for c in ranked if c.selected_geometry],
                "final_selected_ids": [c.candidate_id for c in ranked if c.selected_final],
                "mobile_probe_on_shortlist": [
                    {
                        "candidate_id": c.candidate_id,
                        "probe_text": c.probe_text,
                        "probe_normalized_text": c.probe_normalized_text,
                        "probe_confidence": c.probe_confidence,
                        "probe_reason": c.probe_reason,
                        "score_breakdown": c.score_breakdown,
                        "total_score": c.total_score,
                    }
                    for c in ranked
                    if c.selected_geometry
                ],
                "ranked_candidates": [c.to_debug_dict() for c in ranked],
                "final_candidates": final_candidate_reports,
                "overlay_path": saved_overlay_path,
            }
        )

    return variant_reports, evaluated_flat, parseable_count


def _build_scan_forensic_report(
    *,
    pipeline,
    storage: LocalStorage,
    scan_row: dict[str, Any],
    report_root: Path,
    today: date,
) -> dict[str, Any]:
    scan_id = str(scan_row["id"])
    scan_dir = report_root / "failed_scans" / scan_id
    _ensure_dir(scan_dir)

    image_abs = storage.absolute_path(scan_row["image_path"])
    image_bytes = image_abs.read_bytes()
    image = pipeline._decode_image(image_bytes)
    if image is None:
        return {
            "scan_id": scan_id,
            "image_path": str(image_abs),
            "error": "image unreadable",
        }

    original_copy_path = scan_dir / "original_image.png"
    _write_png(original_copy_path, image)

    detection = pipeline.detector.detect(image)
    roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
    fallback_roi_used = any(source == "full_image" for source, _, _ in roi_candidates)

    variant_reports_all: list[dict[str, Any]] = []
    evaluated_flat_all: list[dict[str, Any]] = []
    parseable_total = 0
    roi_sources: list[str] = []
    for source, roi_image, detector_conf in roi_candidates:
        roi_sources.append(source)
        variant_reports, evaluated_flat, parseable_count = _detect_variant_reports(
            pipeline=pipeline,
            roi_source=source,
            roi_image=roi_image,
            detector_confidence=detector_conf,
            scan_dir=scan_dir,
            today=today,
        )
        variant_reports_all.extend(variant_reports)
        evaluated_flat_all.extend(evaluated_flat)
        parseable_total += parseable_count

    evaluated_flat_all.sort(
        key=lambda item: (
            item["parser_parsed_date"] is not None,
            item["parser_confidence"] or 0.0,
            item["candidate_total_score"],
            item["ocr_confidence"] or 0.0,
        ),
        reverse=True,
    )

    if detection.reason:
        bottleneck = "yolo_detector_and_fallback_failed"
    elif not variant_reports_all or all(len(v["text_detection_boxes"]) == 0 for v in variant_reports_all):
        bottleneck = "text_detection_missing_regions"
    elif evaluated_flat_all and all((item["ocr_raw_text"] or "").strip() == "" for item in evaluated_flat_all):
        bottleneck = "parseq_empty_on_selected_candidates"
    elif evaluated_flat_all and all(item["parser_parsed_date"] is None for item in evaluated_flat_all):
        bottleneck = "parser_rejected_nonempty_ocr"
    else:
        bottleneck = "candidate_ranking_or_mixed"

    return {
        "scan_id": scan_id,
        "created_at": scan_row["created_at"],
        "status": scan_row["status"],
        "final_result_status": scan_row["final_result_status"],
        "failure_reason": scan_row["failure_reason"],
        "original_image_path": str(image_abs),
        "roi_saved_path_from_db": scan_row["roi_path"],
        "yolo": {
            "detected": detection.detected,
            "reason": detection.reason,
            "best_confidence": detection.confidence,
            "boxes": [asdict(box) for box in detection.boxes],
            "fallback_roi_used": fallback_roi_used,
            "roi_sources": roi_sources,
        },
        "variants": variant_reports_all,
        "evaluated_candidates_flat_sorted": evaluated_flat_all,
        "parseable_candidates_count": parseable_total,
        "suspected_bottleneck": bottleneck,
    }


def _run_parseq_sanity(
    *,
    pipeline,
    storage: LocalStorage,
    success_rows: list[dict[str, Any]],
    report_root: Path,
    samples: int = 3,
) -> list[dict[str, Any]]:
    recognizer = pipeline.ocr_router._runner.recognizer
    recognizer._ensure_loaded()
    parser = pipeline.parser
    today = date.today()

    model_file = recognizer._resolve_model_file()
    model_loaded = recognizer._model is not None
    tokenizer_present = bool(model_loaded and hasattr(recognizer._model, "tokenizer"))

    sanity_dir = report_root / "parseq_sanity"
    _ensure_dir(sanity_dir)

    out: list[dict[str, Any]] = []
    for row in success_rows:
        if len(out) >= samples:
            break
        image_path = row.get("roi_path") or row.get("image_path")
        if not image_path:
            continue
        abs_path = storage.absolute_path(str(image_path))
        if not abs_path.exists():
            continue

        image_bytes = abs_path.read_bytes()
        image = pipeline._decode_image(image_bytes)
        if image is None:
            continue

        boxes, _ = pipeline.ocr_router.detect_text_boxes(image)
        ranked = pipeline._build_ranked_candidates(image, boxes)
        ranked = pipeline._apply_geometry_shortlist(ranked)
        for candidate in ranked:
            if not candidate.selected_geometry:
                continue
            crop = candidate.crop(image)
            if crop.size == 0:
                continue

            base = recognizer.recognize(crop)
            if not base.text:
                continue

            rgb_swapped = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            padded = _pad_border(crop, 0.12)

            out_base = recognizer.recognize(crop)
            out_rgb = recognizer.recognize(rgb_swapped)
            out_gray = recognizer.recognize(gray)
            out_padded = recognizer.recognize(padded)

            tensor = recognizer._prepare_tensor(crop)
            crop_name = f"{row['id']}_{candidate.candidate_id}"
            crop_path = sanity_dir / f"{crop_name}.png"
            _write_png(crop_path, crop)

            parse_base = parser.parse(out_base.text, reference_date=today) if out_base.text else parser.parse("", reference_date=today)
            parse_rgb = parser.parse(out_rgb.text, reference_date=today) if out_rgb.text else parser.parse("", reference_date=today)
            parse_gray = parser.parse(out_gray.text, reference_date=today) if out_gray.text else parser.parse("", reference_date=today)
            parse_padded = parser.parse(out_padded.text, reference_date=today) if out_padded.text else parser.parse("", reference_date=today)

            out.append(
                {
                    "scan_id": str(row["id"]),
                    "source_image_path": str(abs_path),
                    "selected_candidate_id": candidate.candidate_id,
                    "selected_candidate_bbox": list(candidate.bbox),
                    "crop_path": str(crop_path),
                    "visual_crop_shape_hwc": list(crop.shape),
                    "parseq_model_dir": str(recognizer.model_dir),
                    "parseq_model_file": str(model_file) if model_file else None,
                    "parseq_model_loaded": model_loaded,
                    "parseq_has_tokenizer": tokenizer_present,
                    "parseq_charset_length": len(recognizer.charset),
                    "parseq_runtime_device": recognizer.runtime_device,
                    "preprocessing": {
                        "color_conversion": "BGR->RGB inside recognizer._prepare_tensor",
                        "target_shape_hwc": [32, 128, 3],
                        "normalization": "(x/255 - 0.5) / 0.5",
                        "layout": "CHW",
                        "prepared_tensor_shape": list(tensor.shape),
                        "prepared_tensor_min": float(np.min(tensor)),
                        "prepared_tensor_max": float(np.max(tensor)),
                    },
                    "runs": {
                        "bgr_input": {
                            "text": out_base.text,
                            "confidence": out_base.confidence,
                            "reason": out_base.reason,
                            "parser_parsed_date": parse_base.parsed_date.isoformat() if parse_base.parsed_date else None,
                            "parser_reason": parse_base.reason,
                        },
                        "rgb_swapped_input": {
                            "text": out_rgb.text,
                            "confidence": out_rgb.confidence,
                            "reason": out_rgb.reason,
                            "parser_parsed_date": parse_rgb.parsed_date.isoformat() if parse_rgb.parsed_date else None,
                            "parser_reason": parse_rgb.reason,
                        },
                        "grayscale_input": {
                            "text": out_gray.text,
                            "confidence": out_gray.confidence,
                            "reason": out_gray.reason,
                            "parser_parsed_date": parse_gray.parsed_date.isoformat() if parse_gray.parsed_date else None,
                            "parser_reason": parse_gray.reason,
                        },
                        "padded_crop_input": {
                            "text": out_padded.text,
                            "confidence": out_padded.confidence,
                            "reason": out_padded.reason,
                            "parser_parsed_date": parse_padded.parsed_date.isoformat() if parse_padded.parsed_date else None,
                            "parser_reason": parse_padded.reason,
                        },
                    },
                }
            )
            break

    return out


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    pipeline = build_pipeline(settings)

    latest_batch = await _load_latest_batch_rows(
        source=args.source,
        max_rows=args.max_rows,
        batch_gap_seconds=args.batch_gap_seconds,
    )
    if not latest_batch:
        print("No scans found for source:", args.source)
        return 1

    failed = [
        row
        for row in latest_batch
        if row.get("final_result_status") in {"OCR_FAILED", "DETECTOR_FAILED"}
    ][: args.samples]
    if not failed:
        print("No failed rows in latest batch for source:", args.source)
        return 1

    report_root = Path(args.output_dir).resolve() / f"test64_forensic_{_utc_stamp()}"
    _ensure_dir(report_root)
    _ensure_dir(report_root / "failed_scans")

    failed_reports = []
    for row in failed:
        report = _build_scan_forensic_report(
            pipeline=pipeline,
            storage=storage,
            scan_row=row,
            report_root=report_root,
            today=date.today(),
        )
        failed_reports.append(report)

    success_rows = await _load_known_good_scans(limit=args.good_scan_pool)
    parseq_sanity = _run_parseq_sanity(
        pipeline=pipeline,
        storage=storage,
        success_rows=success_rows,
        report_root=report_root,
        samples=args.parseq_sanity_samples,
    )

    bottleneck_counts: dict[str, int] = {}
    for item in failed_reports:
        key = item.get("suspected_bottleneck", "unknown")
        bottleneck_counts[key] = bottleneck_counts.get(key, 0) + 1

    payload = {
        "source": args.source,
        "latest_batch_size": len(latest_batch),
        "failed_sample_size": len(failed_reports),
        "bottleneck_counts": bottleneck_counts,
        "scan_reports": failed_reports,
        "parseq_sanity": parseq_sanity,
    }

    report_json = report_root / "forensic_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    summary_md = report_root / "summary.md"
    lines = [
        "# Forensic Expiry Debug Report",
        "",
        f"- Source: `{args.source}`",
        f"- Latest batch size: `{len(latest_batch)}`",
        f"- Failed sample size: `{len(failed_reports)}`",
        f"- Report JSON: `{report_json}`",
        "",
        "## Suspected Bottleneck Counts",
    ]
    for key, value in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## PARSeq Sanity Samples",
            f"- Requested: {args.parseq_sanity_samples}",
            f"- Collected: {len(parseq_sanity)}",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("Forensic report generated:")
    print(report_json)
    print(summary_md)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forensic debug pass on failed expiry scans")
    parser.add_argument("--source", default="test64-bulk-queue")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--good-scan-pool", type=int, default=80)
    parser.add_argument("--parseq-sanity-samples", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
