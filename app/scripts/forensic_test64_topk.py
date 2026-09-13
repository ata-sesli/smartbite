from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import text

from app.ai.pipeline import RankedCandidate
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


def _write_png(path: Path, image: np.ndarray) -> str | None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    path.write_bytes(encoded.tobytes())
    return str(path)


def _expected_to_strings(day: int | None, month: int | None, year: int | None) -> dict[str, str | None]:
    if month is None or year is None:
        return {
            "label": None,
            "iso": None,
            "dmy_slash": None,
            "dm_slash": None,
            "dmy_dot": None,
            "ymd_dash": None,
            "my_slash": None,
        }
    if day is None:
        return {
            "label": f"{month:02d}/{year:04d}",
            "iso": None,
            "dmy_slash": None,
            "dm_slash": None,
            "dmy_dot": None,
            "ymd_dash": None,
            "my_slash": f"{month:02d}/{year:04d}",
        }
    return {
        "label": f"{day:02d}/{month:02d}/{year:04d}",
        "iso": f"{year:04d}-{month:02d}-{day:02d}",
        "dmy_slash": f"{day:02d}/{month:02d}/{year:04d}",
        "dm_slash": f"{day}/{month}/{year}",
        "dmy_dot": f"{day:02d}.{month:02d}.{year:04d}",
        "ymd_dash": f"{year:04d}-{month:02d}-{day:02d}",
        "my_slash": f"{month:02d}/{year:04d}",
    }


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().upper())


def _date_like_tokens(day: int | None, month: int | None, year: int | None) -> list[str]:
    out: list[str] = []
    if month is None or year is None:
        return out
    y2 = year % 100
    if day is not None:
        out.extend(
            [
                f"{day:02d}/{month:02d}/{year:04d}",
                f"{day}/{month}/{year}",
                f"{day:02d}.{month:02d}.{year:04d}",
                f"{year:04d}-{month:02d}-{day:02d}",
                f"{day:02d}/{month:02d}/{y2:02d}",
                f"{day}/{month}/{y2}",
            ]
        )
    out.extend(
        [
            f"{month:02d}/{year:04d}",
            f"{month}/{year}",
            f"{month:02d}-{year:04d}",
        ]
    )
    return [token.upper() for token in out]


def _text_matches_expected(text_value: str | None, day: int | None, month: int | None, year: int | None) -> bool:
    text = _norm_text(text_value)
    if not text:
        return False
    for token in _date_like_tokens(day, month, year):
        if token in text:
            return True

    digits = re.sub(r"\D", "", text)
    if year is not None and str(year) in digits:
        if month is not None and f"{month:02d}" in digits:
            if day is None:
                return True
            if f"{day:02d}" in digits:
                return True
    return False


def _looks_lot_or_other(text_value: str | None) -> bool:
    text = _norm_text(text_value)
    return any(token in text for token in ("LOT", "BATCH", "PARTI", "SERI", "NO:"))


def _looks_production(text_value: str | None) -> bool:
    text = _norm_text(text_value)
    return any(token in text for token in ("URT", "U.T", "PROD", "PKD", "URETIM", "MFG"))


async def _load_latest_batch(source: str, max_rows: int, batch_gap_seconds: int) -> list[dict[str, Any]]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    s.id,
                    s.created_at,
                    s.image_path,
                    s.roi_path,
                    s.status,
                    s.final_result_status,
                    s.failure_reason,
                    s.metadata_json->>'source' AS source,
                    s.metadata_json->>'original_filename' AS filename,
                    p.parsed_date,
                    p.parse_confidence,
                    p.parser_reason
                FROM scans s
                LEFT JOIN parsed_date_results p ON p.scan_id = s.id
                WHERE s.metadata_json->>'source' = :source
                ORDER BY s.created_at DESC
                LIMIT :max_rows
                """
            ),
            {"source": source, "max_rows": max_rows},
        )
        rows = [dict(row) for row in result.mappings().all()]

    if not rows:
        return []

    latest_batch = [rows[0]]
    prev = rows[0]["created_at"]
    for row in rows[1:]:
        current = row["created_at"]
        gap = abs((prev - current).total_seconds())
        if gap > batch_gap_seconds:
            break
        latest_batch.append(row)
        prev = current
    return latest_batch


async def _load_expected_labels() -> dict[str, dict[str, Any]]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT filename, expected_day, expected_month, expected_year, updated_at
                FROM test64_expected_dates
                """
            )
        )
        rows = [dict(row) for row in result.mappings().all()]
    return {row["filename"]: row for row in rows}


def _verdict_for_row(row: dict[str, Any], label: dict[str, Any] | None) -> str:
    final_status = str(row.get("final_result_status") or "").lower()
    if final_status == "detector_failed":
        return "detection_fail"

    parsed = row.get("parsed_date")
    if parsed is None:
        return "recognition_fail"

    if label is None:
        return "wrong_guess"

    day = label.get("expected_day")
    month = label.get("expected_month")
    year = label.get("expected_year")
    if month is None or year is None:
        return "wrong_guess"

    if day is None:
        return "correct_guess" if (parsed.month == int(month) and parsed.year == int(year)) else "wrong_guess"

    return "correct_guess" if parsed == date(int(year), int(month), int(day)) else "wrong_guess"


def _pick_forensic_samples(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]], must_include: str) -> list[dict[str, Any]]:
    by_verdict: dict[str, list[dict[str, Any]]] = {"correct_guess": [], "wrong_guess": [], "recognition_fail": []}
    for row in rows:
        verdict = _verdict_for_row(row, labels.get(row.get("filename")))
        if verdict in by_verdict:
            by_verdict[verdict].append(row)

    selected: list[dict[str, Any]] = []
    selected.extend(by_verdict["correct_guess"][:3])
    selected.extend(by_verdict["wrong_guess"][:3])

    recog = by_verdict["recognition_fail"][:4]
    if not any(item.get("filename") == must_include for item in recog):
        required = next((item for item in by_verdict["recognition_fail"] if item.get("filename") == must_include), None)
        if required is not None:
            if len(recog) >= 4:
                recog[-1] = required
            else:
                recog.append(required)
    selected.extend(recog[:4])
    return selected


def _candidate_box(candidate: RankedCandidate) -> tuple[int, int, int, int]:
    return (int(candidate.bbox[0]), int(candidate.bbox[1]), int(candidate.bbox[2]), int(candidate.bbox[3]))


def _render_overlay(
    image: np.ndarray,
    det_boxes: list[Any],
    ranked: list[RankedCandidate],
    *,
    mode: str,
    true_candidate_ids: set[str] | None = None,
) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    for box in det_boxes:
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), (150, 150, 150), 1)

    for cand in ranked:
        show = True
        if mode == "shortlist":
            show = cand.selected_geometry
        elif mode == "final":
            show = cand.selected_final
        elif mode == "true":
            show = true_candidate_ids is not None and cand.candidate_id in true_candidate_ids

        if not show:
            continue
        x1, y1, x2, y2 = _candidate_box(cand)
        if mode == "all":
            color = (255, 220, 120)
        elif mode == "shortlist":
            color = (0, 215, 255)
        elif mode == "final":
            color = (255, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            cand.candidate_id,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _parse_reason_classification(
    *,
    verdict: str,
    expected_day: int | None,
    expected_month: int | None,
    expected_year: int | None,
    true_in_detected: bool,
    true_in_shortlist: bool,
    true_in_final: bool,
    final_candidates: list[dict[str, Any]],
    predicted_text: str | None,
) -> str:
    if verdict == "correct_guess":
        return "n/a_success"

    if verdict == "detection_fail":
        return "text_detector_missed_date"

    if not true_in_detected:
        return "text_detector_missed_date"
    if not true_in_shortlist:
        return "ranker_missed_date"
    if not true_in_final:
        return "top_k_too_restrictive"

    if verdict == "wrong_guess":
        if _looks_lot_or_other(predicted_text):
            return "picked_lot_or_other_code"
        if _looks_production(predicted_text):
            return "picked_production_date"
        return "parseq_misread_crop"

    has_empty_parseq = any(not (item.get("parseq_raw_text") or "").strip() for item in final_candidates)
    if has_empty_parseq:
        return "parseq_empty_on_readable_crop"

    has_parser_reject = any(
        (item.get("parseq_raw_text") or "").strip() and item.get("parser_parsed_date") is None for item in final_candidates
    )
    if has_parser_reject:
        return "parser_rejected_valid_text"

    if expected_month is None or expected_year is None:
        return "unknown"
    return "parseq_misread_crop"


def _run_forensic_for_scan(
    *,
    pipeline,
    storage: LocalStorage,
    scan_row: dict[str, Any],
    expected: dict[str, Any] | None,
    report_dir: Path,
) -> dict[str, Any]:
    scan_id = str(scan_row["id"])
    filename = str(scan_row.get("filename") or "")
    verdict = _verdict_for_row(scan_row, expected)
    expected_day = expected.get("expected_day") if expected else None
    expected_month = expected.get("expected_month") if expected else None
    expected_year = expected.get("expected_year") if expected else None
    expected_fmt = _expected_to_strings(expected_day, expected_month, expected_year)

    image_rel = str(scan_row["image_path"])
    image_abs = storage.absolute_path(image_rel)
    image_bytes = image_abs.read_bytes()
    image = pipeline._decode_image(image_bytes)
    if image is None:
        return {
            "filename": filename,
            "scan_id": scan_id,
            "error": "image unreadable",
        }

    detection = pipeline.detector.detect(image)
    roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
    fallback_used = any(source == "full_image" for source, _, _ in roi_candidates)

    scan_out_dir = report_dir / "overlays" / scan_id
    _ensure_dir(scan_out_dir)

    variant_reports: list[dict[str, Any]] = []
    all_candidate_matches: list[dict[str, Any]] = []
    final_candidates_global: list[dict[str, Any]] = []

    for source, roi_image, det_conf in roi_candidates:
        variants = pipeline._variants_for_roi(roi_image)
        for variant_name, variant_image in variants:
            detected_boxes, det_reason = pipeline.ocr_router.detect_text_boxes(variant_image)
            ranked = pipeline._build_ranked_candidates(variant_image, detected_boxes)
            ranked = pipeline._apply_geometry_shortlist(ranked)

            for cand in ranked:
                crop = cand.crop(variant_image)
                if crop.size == 0:
                    continue
                probe = pipeline.ocr_router.recognize_probe(crop)
                probe_scores = pipeline._probe_signal_scores(probe)
                cand.probe_text = probe.raw_text
                cand.probe_normalized_text = probe.normalized_text
                cand.probe_confidence = probe.confidence
                cand.probe_reason = probe.reason
                cand.score_breakdown.update(probe_scores)
                if cand.selected_geometry:
                    cand.total_score = cand.geometry_score + sum(probe_scores.values())

                match = _text_matches_expected(probe.raw_text, expected_day, expected_month, expected_year)
                all_candidate_matches.append(
                    {
                        "variant_key": f"{source}:{variant_name}",
                        "candidate_id": cand.candidate_id,
                        "selected_geometry": cand.selected_geometry,
                        "selected_final": cand.selected_final,
                        "probe_text": probe.raw_text,
                        "probe_confidence": probe.confidence,
                        "expected_match": match,
                    }
                )

            final_candidates = pipeline._select_final_candidates(ranked)
            if not final_candidates and ranked:
                ranked[0].selected_final = True
                ranked[0].total_score = ranked[0].geometry_score
                final_candidates = [ranked[0]]

            final_reports: list[dict[str, Any]] = []
            for idx, cand in enumerate(final_candidates, start=1):
                crop = cand.crop(variant_image)
                if crop.size == 0:
                    continue
                crop_path = scan_out_dir / f"{source}_{variant_name}_final_{idx}_{cand.candidate_id}.png"
                crop_saved = _write_png(crop_path, crop)

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
                parsed = pipeline._best_parse_for_inputs(parse_inputs, today=date.today())

                record = {
                    "candidate_id": cand.candidate_id,
                    "bbox_xyxy": [int(cand.bbox[0]), int(cand.bbox[1]), int(cand.bbox[2]), int(cand.bbox[3])],
                    "total_score": cand.total_score,
                    "crop_path": crop_saved,
                    "parseq_raw_text": parseq.raw_text,
                    "parseq_confidence": parseq.confidence,
                    "parseq_reason": parseq.reason,
                    "parse_inputs": parse_inputs,
                    "parser_parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date else None,
                    "parser_confidence": parsed.confidence,
                    "parser_reason": parsed.reason,
                    "expected_match_from_parseq": _text_matches_expected(parseq.raw_text, expected_day, expected_month, expected_year),
                }
                final_reports.append(record)
                final_candidates_global.append(
                    {
                        "variant_key": f"{source}:{variant_name}",
                        "source": source,
                        "variant": variant_name,
                        **record,
                    }
                )

            variant_reports.append(
                {
                    "variant_key": f"{source}:{variant_name}",
                    "source": source,
                    "variant": variant_name,
                    "detector_confidence": det_conf,
                    "text_detector_reason": det_reason,
                    "detected_box_count": len(detected_boxes),
                    "geometry_shortlist_ids": [c.candidate_id for c in ranked if c.selected_geometry],
                    "final_selected_ids": [c.candidate_id for c in ranked if c.selected_final],
                    "ranked_candidates": [c.to_debug_dict() for c in ranked],
                    "final_candidates": final_reports,
                    "detected_boxes": [asdict(b) for b in detected_boxes],
                    "_overlay_ref": {
                        "image": variant_image,
                        "boxes": detected_boxes,
                        "ranked": ranked,
                    },
                }
            )

    true_in_detected = any(item["expected_match"] for item in all_candidate_matches)
    true_in_shortlist = any(item["expected_match"] and item["selected_geometry"] for item in all_candidate_matches)
    true_in_final = any(item.get("expected_match_from_parseq") for item in final_candidates_global)

    focus_variant = None
    for variant in variant_reports:
        has_true_final = any(item.get("expected_match_from_parseq") for item in variant["final_candidates"])
        if has_true_final:
            focus_variant = variant
            break
    if focus_variant is None and variant_reports:
        focus_variant = max(
            variant_reports,
            key=lambda item: (
                len(item.get("final_candidates", [])),
                max((cand.get("total_score") or 0.0 for cand in item.get("final_candidates", [])), default=0.0),
            ),
        )

    overlays: dict[str, str | None] = {"all_boxes": None, "shortlist": None, "final_topk": None, "true_region": None}
    if focus_variant is not None:
        variant_image = focus_variant["_overlay_ref"]["image"]
        det_boxes = focus_variant["_overlay_ref"]["boxes"]
        ranked = focus_variant["_overlay_ref"]["ranked"]
        true_ids = {
            item["candidate_id"]
            for item in all_candidate_matches
            if item["variant_key"] == focus_variant["variant_key"] and item["expected_match"]
        }
        overlays["all_boxes"] = _write_png(scan_out_dir / "overlay_all_boxes.png", _render_overlay(variant_image, det_boxes, ranked, mode="all"))
        overlays["shortlist"] = _write_png(scan_out_dir / "overlay_shortlist.png", _render_overlay(variant_image, det_boxes, ranked, mode="shortlist"))
        overlays["final_topk"] = _write_png(scan_out_dir / "overlay_final_topk.png", _render_overlay(variant_image, det_boxes, ranked, mode="final"))
        overlays["true_region"] = _write_png(
            scan_out_dir / "overlay_true_region.png",
            _render_overlay(variant_image, det_boxes, ranked, mode="true", true_candidate_ids=true_ids),
        )

    parsed_date = scan_row.get("parsed_date")
    predicted_date = parsed_date.isoformat() if parsed_date is not None else None
    predicted_text = None
    if final_candidates_global:
        best_final = max(final_candidates_global, key=lambda item: item.get("parser_confidence") or 0.0)
        predicted_text = best_final.get("parseq_raw_text")

    failure_class = _parse_reason_classification(
        verdict=verdict,
        expected_day=expected_day,
        expected_month=expected_month,
        expected_year=expected_year,
        true_in_detected=true_in_detected,
        true_in_shortlist=true_in_shortlist,
        true_in_final=true_in_final,
        final_candidates=final_candidates_global,
        predicted_text=predicted_text,
    )

    for variant in variant_reports:
        variant.pop("_overlay_ref", None)

    return {
        "scan_id": scan_id,
        "filename": filename,
        "expected_expiry_label": expected_fmt["label"],
        "expected_expiry_iso": expected_fmt["iso"],
        "predicted_date": predicted_date,
        "predicted_parse_confidence": scan_row.get("parse_confidence"),
        "verdict": verdict,
        "yolo_product_roi": {
            "detected": detection.detected,
            "reason": detection.reason,
            "best_confidence": detection.confidence,
            "boxes": [asdict(box) for box in detection.boxes],
            "roi_sources": [src for src, _, _ in roi_candidates],
            "full_image_fallback_used": fallback_used,
        },
        "ppocr_text_detection_box_count_total": sum(int(v["detected_box_count"]) for v in variant_reports),
        "true_expiry_region_detected": true_in_detected,
        "true_expiry_region_in_geometry_shortlist": true_in_shortlist,
        "true_expiry_region_in_final_topk": true_in_final,
        "final_candidates_sent_to_parseq": final_candidates_global,
        "variants": variant_reports,
        "classification": failure_class,
        "final_reason": scan_row.get("failure_reason"),
        "overlays": overlays,
    }


def _summarize_verdicts(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, int]:
    summary = {"total": 0, "detection_fail": 0, "recognition_fail": 0, "wrong_guess": 0, "correct_guess": 0}
    for row in rows:
        verdict = _verdict_for_row(row, labels.get(row.get("filename")))
        summary["total"] += 1
        summary[verdict] = summary.get(verdict, 0) + 1
    return summary


def _evaluate_run_outputs(
    run_outputs: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = {"total": 0, "detection_fail": 0, "recognition_fail": 0, "wrong_guess": 0, "correct_guess": 0}
    parseable_candidates = 0
    selected_candidates_total = 0
    total_runtime_ms = 0

    per_scan: list[dict[str, Any]] = []
    for item in run_outputs:
        label = labels.get(item["filename"])
        output = item["output"]
        debug_summary = output.debug_summary or {}
        parseable_candidates += int(debug_summary.get("parseable_candidates") or 0)
        selected_candidates_total += int(debug_summary.get("selected_candidates_count") or 0)
        total_runtime_ms += int(sum(output.stage_timings_ms.values()))

        if str(output.final_status).lower() == "detector_failed":
            verdict = "detection_fail"
        elif output.parsed_date is None:
            verdict = "recognition_fail"
        else:
            if label is None:
                verdict = "wrong_guess"
            else:
                day = label.get("expected_day")
                month = label.get("expected_month")
                year = label.get("expected_year")
                if month is None or year is None:
                    verdict = "wrong_guess"
                elif day is None:
                    verdict = "correct_guess" if (output.parsed_date.month == int(month) and output.parsed_date.year == int(year)) else "wrong_guess"
                else:
                    verdict = "correct_guess" if output.parsed_date == date(int(year), int(month), int(day)) else "wrong_guess"

        summary["total"] += 1
        summary[verdict] = summary.get(verdict, 0) + 1

        per_scan.append(
            {
                "filename": item["filename"],
                "verdict": verdict,
                "predicted_date": output.parsed_date.isoformat() if output.parsed_date else None,
                "final_status": str(output.final_status),
                "reason": output.reason,
                "parseable_candidates": int(debug_summary.get("parseable_candidates") or 0),
                "selected_candidates": int(debug_summary.get("selected_candidates_count") or 0),
                "runtime_ms": int(sum(output.stage_timings_ms.values())),
            }
        )

    avg_candidates = (selected_candidates_total / summary["total"]) if summary["total"] else 0.0
    avg_runtime_ms = (total_runtime_ms / summary["total"]) if summary["total"] else 0.0
    return {
        "summary": summary,
        "parseable_candidates_total": parseable_candidates,
        "average_parseq_candidates_per_scan": round(avg_candidates, 3),
        "average_runtime_ms_per_scan": round(avg_runtime_ms, 3),
        "per_scan": per_scan,
    }


async def _run_topk_ablation(
    *,
    pipeline,
    storage: LocalStorage,
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    topk_values: list[int],
) -> dict[str, Any]:
    image_rows = [row for row in rows if row.get("filename")]
    out: dict[str, Any] = {}
    for topk in topk_values:
        pipeline.expiry_filter_final_top_k = int(topk)
        run_outputs: list[dict[str, Any]] = []
        for row in image_rows:
            image_bytes = storage.read_bytes(str(row["image_path"]))
            output = await asyncio.to_thread(pipeline.run, image_bytes, today=date.today())
            run_outputs.append({"filename": row["filename"], "output": output})
        out[str(topk)] = _evaluate_run_outputs(run_outputs, labels)
    return out


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    pipeline = build_pipeline(settings)

    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    if not rows:
        print("No rows found for source:", args.source)
        return 1
    labels = await _load_expected_labels()

    report_root = Path(args.output_dir).resolve() / f"test64_forensic_topk_{_utc_stamp()}"
    _ensure_dir(report_root)
    _ensure_dir(report_root / "overlays")

    baseline_summary = _summarize_verdicts(rows, labels)

    selected_scans = _pick_forensic_samples(rows, labels, args.must_include_filename)
    forensic_reports: list[dict[str, Any]] = []
    for row in selected_scans:
        forensic_reports.append(
            _run_forensic_for_scan(
                pipeline=pipeline,
                storage=storage,
                scan_row=row,
                expected=labels.get(row.get("filename")),
                report_dir=report_root,
            )
        )

    forensic_payload = {
        "source": args.source,
        "latest_batch_size": len(rows),
        "baseline_summary": baseline_summary,
        "selected_forensic_scans": [str(item.get("filename")) for item in selected_scans],
        "records": forensic_reports,
    }
    forensic_json = report_root / "forensic_10_report.json"
    forensic_json.write_text(json.dumps(forensic_payload, indent=2, default=_json_default), encoding="utf-8")

    summary_lines = [
        "# Forensic 10 Summary",
        "",
        f"- Source: `{args.source}`",
        f"- Latest batch size: `{len(rows)}`",
        f"- Baseline: `{baseline_summary}`",
        "",
        "## Selected Files",
    ]
    for row in selected_scans:
        summary_lines.append(f"- {row.get('filename')}")
    summary_lines.append("")
    summary_lines.append("## Failure Class Counts")
    class_counts: dict[str, int] = {}
    for item in forensic_reports:
        key = str(item.get("classification", "unknown"))
        class_counts[key] = class_counts.get(key, 0) + 1
    for key, val in sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True):
        summary_lines.append(f"- {key}: {val}")
    forensic_summary_md = report_root / "forensic_10_summary.md"
    forensic_summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    topk_values = [4, 8, 12]
    if args.include_topk20:
        topk_values.append(20)
    ablation_payload = await _run_topk_ablation(
        pipeline=pipeline,
        storage=storage,
        rows=rows,
        labels=labels,
        topk_values=topk_values,
    )
    topk_json = report_root / "topk_ablation_report.json"
    topk_json.write_text(
        json.dumps(
            {
                "source": args.source,
                "latest_batch_size": len(rows),
                "baseline_summary_from_db": baseline_summary,
                "topk_runs": ablation_payload,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    topk_lines = [
        "# Top-k Ablation Summary",
        "",
        f"- Source: `{args.source}`",
        f"- Latest batch size: `{len(rows)}`",
        "",
    ]
    for key in [str(k) for k in topk_values]:
        run = ablation_payload[key]
        topk_lines.append(f"## top_k={key}")
        topk_lines.append(f"- summary: `{run['summary']}`")
        topk_lines.append(f"- parseable_candidates_total: `{run['parseable_candidates_total']}`")
        topk_lines.append(f"- avg_candidates_per_scan: `{run['average_parseq_candidates_per_scan']}`")
        topk_lines.append(f"- avg_runtime_ms_per_scan: `{run['average_runtime_ms_per_scan']}`")
        topk_lines.append("")
    topk_summary_md = report_root / "topk_ablation_summary.md"
    topk_summary_md.write_text("\n".join(topk_lines), encoding="utf-8")

    target = next((item for item in forensic_reports if item.get("filename") == args.must_include_filename), None)
    special_path = report_root / "65940c3a_explanation.md"
    special_lines = ["# 65940c3a Specific Inspection", ""]
    if target is None:
        special_lines.append("- target image not part of selected forensic sample")
    else:
        special_lines.extend(
            [
                f"- filename: `{target['filename']}`",
                f"- expected: `{target.get('expected_expiry_label')}`",
                f"- predicted: `{target.get('predicted_date')}`",
                f"- verdict: `{target.get('verdict')}`",
                f"- true_expiry_region_detected: `{target.get('true_expiry_region_detected')}`",
                f"- in_geometry_shortlist: `{target.get('true_expiry_region_in_geometry_shortlist')}`",
                f"- in_final_topk: `{target.get('true_expiry_region_in_final_topk')}`",
                f"- classification: `{target.get('classification')}`",
                "",
                "## Final PARSeq Candidates",
            ]
        )
        for cand in target.get("final_candidates_sent_to_parseq", []):
            special_lines.append(
                f"- [{cand.get('source')}:{cand.get('variant')}] {cand.get('candidate_id')} "
                f"text=`{cand.get('parseq_raw_text')}` conf={cand.get('parseq_confidence')} "
                f"parsed={cand.get('parser_parsed_date')} reason={cand.get('parser_reason')}"
            )
    special_path.write_text("\n".join(special_lines), encoding="utf-8")

    print("Report root:", report_root)
    print("Generated:")
    print(report_root / "forensic_10_report.json")
    print(report_root / "forensic_10_summary.md")
    print(report_root / "topk_ablation_report.json")
    print(report_root / "topk_ablation_summary.md")
    print(report_root / "65940c3a_explanation.md")
    print(report_root / "overlays")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forensic audit + top-k ablation for SmartBite test64 expiry pipeline")
    parser.add_argument("--source", default="test64-bulk-queue")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--must-include-filename", default="65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg")
    parser.add_argument("--include-topk20", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
