from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re
from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import text

from app.ai.pipeline import EvaluationCandidate
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
    out.extend([f"{month:02d}/{year:04d}", f"{month}/{year}", f"{month:02d}-{year:04d}"])
    return [x.upper() for x in out]


def _text_matches_expected(text_value: str | None, day: int | None, month: int | None, year: int | None) -> bool:
    text = _norm_text(text_value)
    if not text:
        return False
    for token in _date_like_tokens(day, month, year):
        if token in text:
            return True

    digits = re.sub(r"\D", "", text)
    if year is not None and str(year) in digits and month is not None and f"{month:02d}" in digits:
        if day is None:
            return True
        if f"{day:02d}" in digits:
            return True
    return False


def _looks_date_like(text_value: str | None) -> bool:
    text = _norm_text(text_value)
    if not text:
        return False
    return bool(re.search(r"\d{1,4}[./-]\d{1,2}([./-]\d{1,4})?", text))


def _is_empty_text(text_value: str | None) -> bool:
    return not (text_value or "").strip()


def _expected_match_predicted(pred_date: date | None, expected: dict[str, Any] | None) -> bool:
    if pred_date is None or expected is None:
        return False
    day = expected.get("expected_day")
    month = expected.get("expected_month")
    year = expected.get("expected_year")
    if month is None or year is None:
        return False
    if day is None:
        return pred_date.month == int(month) and pred_date.year == int(year)
    return pred_date == date(int(year), int(month), int(day))


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


@dataclass(slots=True)
class CandidateEval:
    source: str
    variant: str
    variant_key: str
    candidate_id: str
    bbox: tuple[int, int, int, int]
    geometry_score: float
    total_score: float
    selected_geometry: bool
    selected_final: bool
    probe_text: str
    probe_confidence: float | None
    parseq_text: str | None
    parseq_confidence: float | None
    parser_date: date | None
    parser_confidence: float
    parser_reason: str
    expected_match_probe: bool
    expected_match_parseq: bool
    score_breakdown: dict[str, float]
    geometry_features: dict[str, float]


def _score_eval_candidate(pipeline, eval_item: CandidateEval, ocr: OCRResultData) -> tuple[float, float, float, float, float]:
    parsed_conf = eval_item.parser_confidence
    return (
        parsed_conf,
        eval_item.total_score,
        pipeline._score_ocr_candidate(ocr),
        pipeline._score_date_likeness(ocr.raw_text),
        pipeline._prefix_bonus(ocr.normalized_text),
    )


def _score_eval_unparseable(pipeline, eval_item: CandidateEval, ocr: OCRResultData) -> tuple[float, float, float, float]:
    return (
        eval_item.total_score,
        pipeline._score_ocr_candidate(ocr),
        pipeline._score_date_likeness(ocr.raw_text),
        pipeline._prefix_bonus(ocr.normalized_text),
    )


def _evaluate_single_scan(
    *,
    pipeline,
    image: np.ndarray,
    expected: dict[str, Any] | None,
    geometry_top_n: int | None,
    final_top_k: int,
) -> dict[str, Any]:
    expected_day = expected.get("expected_day") if expected else None
    expected_month = expected.get("expected_month") if expected else None
    expected_year = expected.get("expected_year") if expected else None

    detection = pipeline.detector.detect(image)
    roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
    fallback_used = any(src == "full_image" for src, _, _ in roi_candidates)

    parseq_cache: dict[tuple[str, str], tuple[str, float | None, str | None, date | None, float, str]] = {}
    all_candidates: list[CandidateEval] = []
    parseable_candidates_count = 0
    evaluated_count = 0

    if geometry_top_n is None:
        geometry_limit = 10**9
    else:
        geometry_limit = int(max(1, geometry_top_n))

    for source, roi_image, _det_conf in roi_candidates:
        variants = pipeline._variants_for_roi(roi_image)
        for variant_name, variant_image in variants:
            variant_key = f"{source}:{variant_name}"
            det_boxes, _det_reason = pipeline.ocr_router.detect_text_boxes(variant_image)
            ranked = pipeline._build_ranked_candidates(variant_image, det_boxes)
            if not ranked:
                continue

            top_geom_ids = {id(c) for c in ranked[:geometry_limit]}
            for candidate in ranked:
                candidate.selected_geometry = id(candidate) in top_geom_ids
                crop = candidate.crop(variant_image)
                if crop.size == 0:
                    candidate.probe_text = ""
                    candidate.probe_confidence = None
                    candidate.probe_reason = "candidate crop empty"
                    candidate.total_score = candidate.geometry_score
                    continue
                probe = pipeline.ocr_router.recognize_probe(crop)
                probe_scores = pipeline._probe_signal_scores(probe)
                candidate.probe_text = probe.raw_text
                candidate.probe_normalized_text = probe.normalized_text
                candidate.probe_confidence = probe.confidence
                candidate.probe_reason = probe.reason
                candidate.score_breakdown.update(probe_scores)
                if candidate.selected_geometry:
                    candidate.total_score = candidate.geometry_score + sum(probe_scores.values())
                else:
                    candidate.total_score = candidate.geometry_score

            shortlisted = [c for c in ranked if c.selected_geometry]
            shortlisted.sort(key=lambda c: c.total_score, reverse=True)
            final = shortlisted[: max(1, final_top_k)]
            final_ids = {id(c) for c in final}
            for c in ranked:
                c.selected_final = id(c) in final_ids

            for c in final:
                crop = c.crop(variant_image)
                if crop.size == 0:
                    continue
                cache_key = (variant_key, c.candidate_id)
                if cache_key not in parseq_cache:
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
                    parseq_cache[cache_key] = (
                        parseq.raw_text,
                        parseq.confidence,
                        parseq.reason,
                        parsed.parsed_date,
                        parsed.confidence,
                        parsed.reason,
                    )

                parseq_text, parseq_conf, _parseq_reason, parser_date, parser_conf, parser_reason = parseq_cache[cache_key]
                if parser_date is not None:
                    parseable_candidates_count += 1
                evaluated_count += 1

                all_candidates.append(
                    CandidateEval(
                        source=source,
                        variant=variant_name,
                        variant_key=variant_key,
                        candidate_id=c.candidate_id,
                        bbox=(int(c.bbox[0]), int(c.bbox[1]), int(c.bbox[2]), int(c.bbox[3])),
                        geometry_score=float(c.geometry_score),
                        total_score=float(c.total_score),
                        selected_geometry=True,
                        selected_final=True,
                        probe_text=c.probe_text,
                        probe_confidence=c.probe_confidence,
                        parseq_text=parseq_text,
                        parseq_confidence=parseq_conf,
                        parser_date=parser_date,
                        parser_confidence=float(parser_conf),
                        parser_reason=parser_reason,
                        expected_match_probe=_text_matches_expected(c.probe_text, expected_day, expected_month, expected_year),
                        expected_match_parseq=_text_matches_expected(parseq_text, expected_day, expected_month, expected_year),
                        score_breakdown={k: float(v) for k, v in c.score_breakdown.items()},
                        geometry_features={k: float(v) for k, v in c.geometry_features.items()},
                    )
                )

    parseable_eval = [item for item in all_candidates if item.parser_date is not None]

    selected: CandidateEval | None = None
    selected_date: date | None = None
    selected_reason = "no candidates evaluated"
    if parseable_eval:
        ranked_parseable: list[tuple[tuple[float, float, float, float, float], CandidateEval]] = []
        for item in parseable_eval:
            ocr = OCRResultData(
                raw_text=item.parseq_text or "",
                normalized_text=_norm_text(item.parseq_text),
                confidence=item.parseq_confidence,
                engine_name=pipeline.ocr_router.active_engine_name,
                runtime_device=pipeline.ocr_router.parseq_runtime_device,
                reason=None,
            )
            ranked_parseable.append((_score_eval_candidate(pipeline, item, ocr), item))
        ranked_parseable.sort(key=lambda x: x[0], reverse=True)
        selected = ranked_parseable[0][1]
        selected_date = selected.parser_date
        selected_reason = f"selected parseable candidate {selected.variant_key}:{selected.candidate_id}"
    else:
        with_text = [item for item in all_candidates if not _is_empty_text(item.parseq_text)]
        if with_text:
            ranked_text: list[tuple[tuple[float, float, float, float], CandidateEval]] = []
            for item in with_text:
                ocr = OCRResultData(
                    raw_text=item.parseq_text or "",
                    normalized_text=_norm_text(item.parseq_text),
                    confidence=item.parseq_confidence,
                    engine_name=pipeline.ocr_router.active_engine_name,
                    runtime_device=pipeline.ocr_router.parseq_runtime_device,
                    reason=None,
                )
                ranked_text.append((_score_eval_unparseable(pipeline, item, ocr), item))
            ranked_text.sort(key=lambda x: x[0], reverse=True)
            selected = ranked_text[0][1]
            selected_reason = f"selected non-parseable text candidate {selected.variant_key}:{selected.candidate_id}"
        elif detection.reason:
            selected_reason = f"detector failed: {detection.reason}"
        else:
            selected_reason = "ocr returned empty text"

    true_detected = any(item.expected_match_probe for item in all_candidates)
    true_in_final = any(item.expected_match_probe for item in all_candidates if item.selected_final)
    selected_parseq_text = selected.parseq_text if selected is not None else None

    return {
        "detection_detected": detection.detected,
        "detection_reason": detection.reason,
        "fallback_used": fallback_used,
        "final_selected_date": selected_date,
        "selected_parseq_text": selected_parseq_text,
        "selected_candidate": {
            "variant_key": selected.variant_key if selected else None,
            "candidate_id": selected.candidate_id if selected else None,
            "parseq_text": selected.parseq_text if selected else None,
            "parseq_confidence": selected.parseq_confidence if selected else None,
            "parser_date": selected.parser_date.isoformat() if (selected and selected.parser_date) else None,
            "parser_confidence": selected.parser_confidence if selected else None,
            "parser_reason": selected.parser_reason if selected else None,
        },
        "selected_reason": selected_reason,
        "evaluated_candidates": evaluated_count,
        "parseable_candidates": parseable_candidates_count,
        "true_expiry_region_detected": true_detected,
        "true_expiry_region_in_geometry_shortlist": true_detected,  # evaluated set is after shortlist stage
        "true_expiry_region_in_final_topk": true_in_final,
        "all_final_candidates": [
            {
                "variant_key": item.variant_key,
                "source": item.source,
                "variant": item.variant,
                "candidate_id": item.candidate_id,
                "bbox_xyxy": list(item.bbox),
                "geometry_score": item.geometry_score,
                "total_score": item.total_score,
                "probe_text": item.probe_text,
                "probe_confidence": item.probe_confidence,
                "parseq_text": item.parseq_text,
                "parseq_confidence": item.parseq_confidence,
                "parser_date": item.parser_date.isoformat() if item.parser_date else None,
                "parser_confidence": item.parser_confidence,
                "parser_reason": item.parser_reason,
                "expected_match_probe": item.expected_match_probe,
                "expected_match_parseq": item.expected_match_parseq,
                "score_breakdown": item.score_breakdown,
                "geometry_features": item.geometry_features,
            }
            for item in all_candidates
        ],
    }


def _verdict_from_eval(eval_result: dict[str, Any], expected: dict[str, Any] | None) -> str:
    if not eval_result.get("detection_detected", False):
        return "detection_fail"
    pred_date = eval_result.get("final_selected_date")
    if pred_date is None:
        return "recognition_fail"
    return "correct_guess" if _expected_match_predicted(pred_date, expected) else "wrong_guess"


def _classify_recognition_failure(
    *,
    eval_result: dict[str, Any],
    expected: dict[str, Any] | None,
) -> str:
    if not eval_result.get("detection_detected"):
        return "text_detector_missed_date"

    all_final = eval_result.get("all_final_candidates", [])
    true_detected = bool(eval_result.get("true_expiry_region_detected"))
    true_in_final = bool(eval_result.get("true_expiry_region_in_final_topk"))

    if not true_detected:
        return "text_detector_missed_date"
    if true_detected and not true_in_final:
        # If no date-like probe appears among final candidates, region was dropped by geometry shortlist;
        # otherwise it survived shortlist but ranking/probe phase chose other regions.
        any_date_like_final_probe = any(_looks_date_like(item.get("probe_text")) for item in all_final)
        if not any_date_like_final_probe:
            return "geometry_shortlist_missed_date"
        return "ranker_missed_date_after_probe"

    # True region reached final top-k; now inspect PARSeq/parser.
    true_final = [item for item in all_final if item.get("expected_match_probe")]
    if not true_final:
        return "unknown"

    any_parseq_empty = any(_is_empty_text(item.get("parseq_text")) for item in true_final)
    any_parseq_has_text = any(not _is_empty_text(item.get("parseq_text")) for item in true_final)
    any_parser_accept = any(item.get("parser_date") is not None for item in true_final)

    if any_parseq_empty and not any_parseq_has_text:
        return "bad_crop_to_parseq"
    if any_parseq_empty and any_parseq_has_text and not any_parser_accept:
        return "parseq_failed_on_selected_crop"
    if any_parseq_has_text and not any_parser_accept:
        # If parseq returns clear date-like text but parser still rejects.
        if any(_looks_date_like(item.get("parseq_text")) for item in true_final):
            return "parser_rejected_valid_text"
        return "parseq_failed_on_selected_crop"

    return "unknown"


async def _run_geometry_ablation(
    *,
    pipeline,
    storage: LocalStorage,
    rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    shortlist_values: list[int | None],
    final_top_k: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report: dict[str, Any] = {}
    per_run_eval: dict[str, dict[str, Any]] = {}
    for shortlist in shortlist_values:
        key = "all" if shortlist is None else str(shortlist)
        t0 = datetime.now()
        summary = {"total": 0, "correct_guess": 0, "wrong_guess": 0, "recognition_fail": 0, "detection_fail": 0}
        parseable_total = 0
        evaluated_total = 0
        true_detected = 0
        true_in_final = 0
        rows_eval: dict[str, Any] = {}

        for row in rows:
            filename = str(row.get("filename") or "")
            expected = labels.get(filename)
            image_bytes = storage.read_bytes(str(row["image_path"]))
            image = pipeline._decode_image(image_bytes)
            if image is None:
                eval_result = {
                    "detection_detected": False,
                    "detection_reason": "image unreadable",
                    "fallback_used": False,
                    "final_selected_date": None,
                    "selected_parseq_text": None,
                    "selected_reason": "image unreadable",
                    "evaluated_candidates": 0,
                    "parseable_candidates": 0,
                    "true_expiry_region_detected": False,
                    "true_expiry_region_in_geometry_shortlist": False,
                    "true_expiry_region_in_final_topk": False,
                    "all_final_candidates": [],
                }
            else:
                eval_result = _evaluate_single_scan(
                    pipeline=pipeline,
                    image=image,
                    expected=expected,
                    geometry_top_n=shortlist,
                    final_top_k=final_top_k,
                )

            verdict = _verdict_from_eval(eval_result, expected)
            summary["total"] += 1
            summary[verdict] += 1
            parseable_total += int(eval_result.get("parseable_candidates", 0))
            evaluated_total += int(eval_result.get("evaluated_candidates", 0))
            if eval_result.get("true_expiry_region_detected"):
                true_detected += 1
            if eval_result.get("true_expiry_region_in_final_topk"):
                true_in_final += 1

            rows_eval[filename] = {
                "verdict": verdict,
                "eval": eval_result,
            }

        elapsed = (datetime.now() - t0).total_seconds()
        report[key] = {
            "summary": summary,
            "parseable_candidates_total": parseable_total,
            "average_candidates_evaluated_per_scan": round((evaluated_total / summary["total"]) if summary["total"] else 0.0, 3),
            "true_expiry_region_detected_count": true_detected,
            "true_expiry_region_in_final_topk_count": true_in_final,
            "true_expiry_region_detected_rate": round((true_detected / summary["total"]) if summary["total"] else 0.0, 4),
            "true_expiry_region_in_final_topk_rate": round((true_in_final / summary["total"]) if summary["total"] else 0.0, 4),
            "elapsed_seconds": round(elapsed, 3),
        }
        per_run_eval[key] = rows_eval
    return report, per_run_eval


def _extract_65940c3a_details(
    run_eval: dict[str, Any],
    expected: dict[str, Any] | None,
) -> dict[str, Any]:
    target = run_eval.get("65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg")
    if target is None:
        return {"error": "target image not found in evaluation"}
    eval_result = target["eval"]
    finals = eval_result.get("all_final_candidates", [])

    # Focus on lower-right block heuristic: x_center > 55%, y_center > 55% of image area is
    # approximated from bbox in ROI space (for this report we keep bbox list only).
    lower_right = []
    for item in finals:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        lower_right.append(
            {
                "variant_key": item["variant_key"],
                "candidate_id": item["candidate_id"],
                "bbox_xyxy": item["bbox_xyxy"],
                "center_xy": [round(xc, 2), round(yc, 2)],
                "geometry_score": item["geometry_score"],
                "total_score": item["total_score"],
                "probe_text": item["probe_text"],
                "probe_confidence": item["probe_confidence"],
                "parseq_text": item["parseq_text"],
                "score_breakdown": item["score_breakdown"],
                "geometry_features": item["geometry_features"],
                "expected_match_probe": item["expected_match_probe"],
            }
        )

    lower_right.sort(key=lambda item: item["geometry_score"], reverse=True)
    matches = [item for item in lower_right if item.get("expected_match_probe")]
    grouped_like = [item for item in lower_right if "group_bonus" in (item.get("score_breakdown") or {}) and (item["score_breakdown"].get("group_bonus", 0) > 0)]

    explanation = {
        "expected": {
            "day": expected.get("expected_day") if expected else None,
            "month": expected.get("expected_month") if expected else None,
            "year": expected.get("expected_year") if expected else None,
        },
        "verdict": target["verdict"],
        "selected_reason": eval_result.get("selected_reason"),
        "true_expiry_region_detected": eval_result.get("true_expiry_region_detected"),
        "true_expiry_region_in_final_topk": eval_result.get("true_expiry_region_in_final_topk"),
        "lower_right_candidates_final_pool": lower_right,
        "expected_match_candidates": matches,
        "grouped_candidates_count": len(grouped_like),
        "grouped_candidates_preview": grouped_like[:10],
    }
    return explanation


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    pipeline = build_pipeline(settings)

    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    labels = await _load_expected_labels()
    if not rows:
        print("No rows found for source:", args.source)
        return 1

    report_root = Path(args.output_dir).resolve() / f"test64_geometry_shortlist_{_utc_stamp()}"
    _ensure_dir(report_root)

    shortlist_values: list[int | None] = [12, 30, 50, None]
    ablation_report, per_run_eval = await _run_geometry_ablation(
        pipeline=pipeline,
        storage=storage,
        rows=rows,
        labels=labels,
        shortlist_values=shortlist_values,
        final_top_k=args.final_top_k,
    )

    baseline_key = "12"
    baseline_eval = per_run_eval[baseline_key]
    recognition_fail_rows = [name for name, item in baseline_eval.items() if item["verdict"] == "recognition_fail"]
    breakdown_counts: dict[str, int] = {
        "text_detector_missed_date": 0,
        "geometry_shortlist_missed_date": 0,
        "ranker_missed_date_after_probe": 0,
        "bad_crop_to_parseq": 0,
        "parseq_failed_on_selected_crop": 0,
        "parser_rejected_valid_text": 0,
        "unknown": 0,
    }
    breakdown_rows: list[dict[str, Any]] = []
    for filename in recognition_fail_rows:
        expected = labels.get(filename)
        eval_result = baseline_eval[filename]["eval"]
        cls = _classify_recognition_failure(eval_result=eval_result, expected=expected)
        breakdown_counts[cls] = breakdown_counts.get(cls, 0) + 1
        breakdown_rows.append(
            {
                "filename": filename,
                "classification": cls,
                "selected_reason": eval_result.get("selected_reason"),
                "true_expiry_region_detected": eval_result.get("true_expiry_region_detected"),
                "true_expiry_region_in_final_topk": eval_result.get("true_expiry_region_in_final_topk"),
                "evaluated_candidates": eval_result.get("evaluated_candidates"),
                "parseable_candidates": eval_result.get("parseable_candidates"),
            }
        )

    special_expected = labels.get("65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg")
    special = _extract_65940c3a_details(baseline_eval, special_expected)

    payload = {
        "source": args.source,
        "latest_batch_size": len(rows),
        "final_top_k_fixed": args.final_top_k,
        "geometry_shortlist_ablation": ablation_report,
        "recognition_fail_count_baseline_12": len(recognition_fail_rows),
        "recognition_fail_breakdown_baseline_12": breakdown_counts,
        "recognition_fail_rows_baseline_12": breakdown_rows,
        "special_65940c3a": special,
    }
    report_json = report_root / "geometry_shortlist_ablation_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    summary_lines = [
        "# Geometry Shortlist Ablation Summary",
        "",
        f"- Source: `{args.source}`",
        f"- Latest batch size: `{len(rows)}`",
        f"- final_top_k fixed: `{args.final_top_k}`",
        "",
        "## Recognition-Fail Breakdown (baseline shortlist=12)",
    ]
    for key, val in breakdown_counts.items():
        summary_lines.append(f"- {key}: {val}")
    summary_lines.append("")
    summary_lines.append("## Ablation Results")
    for key in ("12", "30", "50", "all"):
        run = ablation_report[key]
        summary_lines.append(f"### geometry_shortlist={key}")
        summary_lines.append(f"- summary: `{run['summary']}`")
        summary_lines.append(f"- parseable_candidates_total: `{run['parseable_candidates_total']}`")
        summary_lines.append(f"- true_expiry_region_detected_count: `{run['true_expiry_region_detected_count']}`")
        summary_lines.append(f"- true_expiry_region_in_final_topk_count: `{run['true_expiry_region_in_final_topk_count']}`")
        summary_lines.append(f"- elapsed_seconds: `{run['elapsed_seconds']}`")
        summary_lines.append("")
    summary_md = report_root / "geometry_shortlist_ablation_summary.md"
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    special_md = report_root / "65940c3a_geometry_inspection.md"
    special_lines = [
        "# 65940c3a Geometry Inspection",
        "",
        f"- verdict: `{special.get('verdict')}`",
        f"- true_expiry_region_detected: `{special.get('true_expiry_region_detected')}`",
        f"- true_expiry_region_in_final_topk: `{special.get('true_expiry_region_in_final_topk')}`",
        f"- selected_reason: `{special.get('selected_reason')}`",
        "",
        "## Candidates (final pool, geometry sorted)",
    ]
    for item in special.get("lower_right_candidates_final_pool", [])[:40]:
        special_lines.append(
            f"- {item['variant_key']}::{item['candidate_id']} bbox={item['bbox_xyxy']} "
            f"geom={item['geometry_score']:.4f} total={item['total_score']:.4f} "
            f"probe=`{item['probe_text']}` parseq=`{item['parseq_text']}`"
        )
    special_lines.append("")
    special_lines.append("## Expected-Match Candidates")
    exp_matches = special.get("expected_match_candidates", [])
    if not exp_matches:
        special_lines.append("- none in final candidate pool")
    else:
        for item in exp_matches:
            special_lines.append(
                f"- {item['variant_key']}::{item['candidate_id']} geom={item['geometry_score']:.4f} total={item['total_score']:.4f} probe=`{item['probe_text']}`"
            )
    special_lines.append("")
    special_lines.append(f"- grouped_candidates_count: `{special.get('grouped_candidates_count')}`")
    special_md.write_text("\n".join(special_lines), encoding="utf-8")

    print("Report root:", report_root)
    print("Generated:")
    print(report_json)
    print(summary_md)
    print(special_md)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry shortlist forensic ablation for SmartBite test64")
    parser.add_argument("--source", default="test64-bulk-queue")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--final-top-k", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
