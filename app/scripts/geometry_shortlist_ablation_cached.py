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


def _looks_date_like(value: str | None) -> bool:
    text = _norm_text(value)
    return bool(re.search(r"\d{1,4}[./-]\d{1,2}([./-]\d{1,4})?", text))


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


def _looks_lot_or_other(text_value: str | None) -> bool:
    text = _norm_text(text_value)
    return any(token in text for token in ("LOT", "BATCH", "SERI", "PARTI", "NO"))


def _looks_production(text_value: str | None) -> bool:
    text = _norm_text(text_value)
    return any(token in text for token in ("URT", "U.T", "PROD", "PKD", "URETIM", "MFG"))


def _expected_match_predicted(pred_date: date | None, expected: dict[str, Any] | None) -> bool:
    if pred_date is None or expected is None:
        return False
    d = expected.get("expected_day")
    m = expected.get("expected_month")
    y = expected.get("expected_year")
    if m is None or y is None:
        return False
    if d is None:
        return pred_date.month == int(m) and pred_date.year == int(y)
    return pred_date == date(int(y), int(m), int(d))


@dataclass(slots=True)
class CandidateCache:
    variant_key: str
    source: str
    variant: str
    candidate_id: str
    geometry_rank: int
    geometry_score: float
    total_score: float
    bbox: tuple[int, int, int, int]
    probe_text: str
    probe_confidence: float | None
    expected_match_probe: bool
    variant_key_ref: str
    score_breakdown: dict[str, float]
    geometry_features: dict[str, float]
    parseq_text: str | None = None
    parseq_confidence: float | None = None
    parseq_reason: str | None = None
    parser_date: date | None = None
    parser_confidence: float | None = None
    parser_reason: str | None = None
    expected_match_parseq: bool | None = None


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
                    s.final_result_status,
                    s.failure_reason,
                    s.metadata_json->>'source' AS source,
                    s.metadata_json->>'original_filename' AS filename,
                    p.parsed_date,
                    p.parse_confidence
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


def _score_parseable_candidate(pipeline, candidate: CandidateCache) -> tuple[float, float, float, float, float]:
    ocr = OCRResultData(
        raw_text=candidate.parseq_text or "",
        normalized_text=_norm_text(candidate.parseq_text),
        confidence=candidate.parseq_confidence,
        engine_name=pipeline.ocr_router.active_engine_name,
        runtime_device=pipeline.ocr_router.parseq_runtime_device,
        reason=candidate.parseq_reason,
    )
    return (
        float(candidate.parser_confidence or 0.0),
        float(candidate.total_score),
        float(pipeline._score_ocr_candidate(ocr)),
        float(pipeline._score_date_likeness(ocr.raw_text)),
        float(pipeline._prefix_bonus(ocr.normalized_text)),
    )


def _score_unparseable_candidate(pipeline, candidate: CandidateCache) -> tuple[float, float, float, float]:
    ocr = OCRResultData(
        raw_text=candidate.parseq_text or "",
        normalized_text=_norm_text(candidate.parseq_text),
        confidence=candidate.parseq_confidence,
        engine_name=pipeline.ocr_router.active_engine_name,
        runtime_device=pipeline.ocr_router.parseq_runtime_device,
        reason=candidate.parseq_reason,
    )
    return (
        float(candidate.total_score),
        float(pipeline._score_ocr_candidate(ocr)),
        float(pipeline._score_date_likeness(ocr.raw_text)),
        float(pipeline._prefix_bonus(ocr.normalized_text)),
    )


def _run_parseq_and_parser(
    pipeline,
    candidate: CandidateCache,
    expected: dict[str, Any] | None,
    variant_images: dict[str, np.ndarray],
) -> None:
    if candidate.parseq_text is not None:
        return
    variant_image = variant_images.get(candidate.variant_key_ref)
    if variant_image is None:
        candidate.parseq_text = ""
        candidate.parseq_confidence = None
        candidate.parseq_reason = "missing variant image"
        candidate.parser_date = None
        candidate.parser_confidence = 0.0
        candidate.parser_reason = "missing variant image"
        candidate.expected_match_parseq = False
        return
    x1, y1, x2, y2 = candidate.bbox
    crop = variant_image[y1:y2, x1:x2]
    if crop.size == 0:
        candidate.parseq_text = ""
        candidate.parseq_confidence = None
        candidate.parseq_reason = "candidate crop empty"
        candidate.parser_date = None
        candidate.parser_confidence = 0.0
        candidate.parser_reason = "candidate crop empty"
        candidate.expected_match_parseq = False
        return
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
    candidate.parseq_text = parseq.raw_text
    candidate.parseq_confidence = parseq.confidence
    candidate.parseq_reason = parseq.reason
    candidate.parser_date = parsed.parsed_date
    candidate.parser_confidence = parsed.confidence
    candidate.parser_reason = parsed.reason
    day = expected.get("expected_day") if expected else None
    month = expected.get("expected_month") if expected else None
    year = expected.get("expected_year") if expected else None
    candidate.expected_match_parseq = _text_matches_expected(parseq.raw_text, day, month, year)


def _failure_classification_baseline(
    *,
    eval_summary: dict[str, Any],
    final_candidates: list[CandidateCache],
) -> str:
    if not eval_summary.get("detection_detected"):
        return "text_detector_missed_date"

    true_detected = bool(eval_summary.get("true_region_detected"))
    true_short = bool(eval_summary.get("true_region_in_shortlist"))
    true_final = bool(eval_summary.get("true_region_in_final_topk"))

    if not true_detected:
        return "text_detector_missed_date"
    if true_detected and not true_short:
        return "geometry_shortlist_missed_date"
    if true_short and not true_final:
        return "ranker_missed_date_after_probe"

    true_final_candidates = [c for c in final_candidates if c.expected_match_probe]
    if not true_final_candidates:
        return "unknown"

    any_empty = any(not (c.parseq_text or "").strip() for c in true_final_candidates)
    any_nonempty = any((c.parseq_text or "").strip() for c in true_final_candidates)
    any_parser_accept = any(c.parser_date is not None for c in true_final_candidates)
    any_date_like_text = any(_looks_date_like(c.parseq_text) for c in true_final_candidates)

    if any_empty and not any_nonempty:
        return "bad_crop_to_parseq"
    if any_nonempty and not any_parser_accept and any_date_like_text:
        return "parser_rejected_valid_text"
    if any_nonempty and not any_parser_accept:
        return "parseq_failed_on_selected_crop"
    return "unknown"


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    pipeline = build_pipeline(settings)

    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    labels = await _load_expected_labels()
    if not rows:
        print("No rows found for source:", args.source)
        return 1

    shortlist_values: list[int | None] = [12, 30, 50, None]
    shortlist_keys = ["12", "30", "50", "all"]
    metrics: dict[str, dict[str, Any]] = {}
    for k in shortlist_keys:
        metrics[k] = {
            "summary": {"total": 0, "correct_guess": 0, "wrong_guess": 0, "recognition_fail": 0, "detection_fail": 0},
            "parseable_candidates_total": 0,
            "candidates_evaluated_total": 0,
            "true_region_in_shortlist_count": 0,
            "true_region_in_final_topk_count": 0,
            "elapsed_seconds": 0.0,
        }

    baseline_recognition_fails: list[dict[str, Any]] = []
    special_65940: dict[str, Any] = {}
    start = datetime.now()

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        filename = str(row.get("filename") or "")
        expected = labels.get(filename)
        day = expected.get("expected_day") if expected else None
        month = expected.get("expected_month") if expected else None
        year = expected.get("expected_year") if expected else None

        image_bytes = storage.read_bytes(str(row["image_path"]))
        image = pipeline._decode_image(image_bytes)
        if image is None:
            print(f"[{idx}/{total}] {filename}: image unreadable")
            for key in shortlist_keys:
                metrics[key]["summary"]["total"] += 1
                metrics[key]["summary"]["detection_fail"] += 1
            continue

        detection = pipeline.detector.detect(image)
        roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
        fallback_used = any(src == "full_image" for src, _, _ in roi_candidates)

        variant_candidates: dict[str, list[CandidateCache]] = {}
        variant_images: dict[str, np.ndarray] = {}
        total_candidates = 0
        for source, roi_image, _det_conf in roi_candidates:
            variants = pipeline._variants_for_roi(roi_image)
            for variant_name, variant_image in variants:
                variant_key = f"{source}:{variant_name}"
                variant_images[variant_key] = variant_image
                det_boxes, _det_reason = pipeline.ocr_router.detect_text_boxes(variant_image)
                ranked = pipeline._build_ranked_candidates(variant_image, det_boxes)
                candidates: list[CandidateCache] = []
                for rank_i, cand in enumerate(ranked, start=1):
                    crop = cand.crop(variant_image)
                    if crop.size == 0:
                        continue
                    probe = pipeline.ocr_router.recognize_probe(crop)
                    probe_scores = pipeline._probe_signal_scores(probe)
                    total_score = float(cand.geometry_score + sum(probe_scores.values()))
                    candidates.append(
                        CandidateCache(
                            variant_key=variant_key,
                            source=source,
                            variant=variant_name,
                            candidate_id=cand.candidate_id,
                            geometry_rank=rank_i,
                            geometry_score=float(cand.geometry_score),
                            total_score=total_score,
                            bbox=(int(cand.bbox[0]), int(cand.bbox[1]), int(cand.bbox[2]), int(cand.bbox[3])),
                            probe_text=probe.raw_text,
                            probe_confidence=probe.confidence,
                            expected_match_probe=_text_matches_expected(probe.raw_text, day, month, year),
                            variant_key_ref=variant_key,
                            score_breakdown={**{k: float(v) for k, v in cand.score_breakdown.items()}, **{k: float(v) for k, v in probe_scores.items()}},
                            geometry_features={k: float(v) for k, v in cand.geometry_features.items()},
                        )
                    )
                if candidates:
                    variant_candidates[variant_key] = candidates
                    total_candidates += len(candidates)

        print(
            f"[{idx}/{total}] {filename}: variants={len(variant_candidates)} "
            f"candidates={total_candidates} fallback={fallback_used}"
        )

        for shortlist_value, key in zip(shortlist_values, shortlist_keys, strict=False):
            run_start = datetime.now()
            metrics[key]["summary"]["total"] += 1

            final_candidates: list[CandidateCache] = []
            any_true_short = False
            any_true_final = False
            any_true_detected = False

            for vkey, cand_list in variant_candidates.items():
                sorted_geom = sorted(cand_list, key=lambda c: c.geometry_score, reverse=True)
                if shortlist_value is None:
                    shortlist = sorted_geom
                else:
                    shortlist = sorted_geom[: int(max(1, shortlist_value))]
                if any(c.expected_match_probe for c in sorted_geom):
                    any_true_detected = True
                if any(c.expected_match_probe for c in shortlist):
                    any_true_short = True
                selected = sorted(shortlist, key=lambda c: c.total_score, reverse=True)[: args.final_top_k]
                if any(c.expected_match_probe for c in selected):
                    any_true_final = True
                for c in selected:
                    _run_parseq_and_parser(pipeline, c, expected, variant_images)
                final_candidates.extend(selected)

            metrics[key]["candidates_evaluated_total"] += len(final_candidates)
            metrics[key]["parseable_candidates_total"] += sum(1 for c in final_candidates if c.parser_date is not None)
            if any_true_short:
                metrics[key]["true_region_in_shortlist_count"] += 1
            if any_true_final:
                metrics[key]["true_region_in_final_topk_count"] += 1

            if not final_candidates and detection.reason and not detection.detected:
                verdict = "detection_fail"
                selected_date = None
                selected_text = None
            else:
                parseable = [c for c in final_candidates if c.parser_date is not None]
                if parseable:
                    selected = max(parseable, key=lambda c: _score_parseable_candidate(pipeline, c))
                    selected_date = selected.parser_date
                    selected_text = selected.parseq_text
                else:
                    with_text = [c for c in final_candidates if (c.parseq_text or "").strip()]
                    if with_text:
                        selected = max(with_text, key=lambda c: _score_unparseable_candidate(pipeline, c))
                        selected_date = None
                        selected_text = selected.parseq_text
                    else:
                        selected_date = None
                        selected_text = None

                if selected_date is None:
                    verdict = "recognition_fail"
                elif _expected_match_predicted(selected_date, expected):
                    verdict = "correct_guess"
                else:
                    verdict = "wrong_guess"

            metrics[key]["summary"][verdict] += 1
            metrics[key]["elapsed_seconds"] += (datetime.now() - run_start).total_seconds()

            if key == "12" and verdict == "recognition_fail":
                eval_summary = {
                    "detection_detected": detection.detected,
                    "detection_reason": detection.reason,
                    "fallback_used": fallback_used,
                    "true_region_detected": any_true_detected,
                    "true_region_in_shortlist": any_true_short,
                    "true_region_in_final_topk": any_true_final,
                    "selected_text": selected_text,
                }
                classification = _failure_classification_baseline(eval_summary=eval_summary, final_candidates=final_candidates)
                baseline_recognition_fails.append(
                    {
                        "filename": filename,
                        "classification": classification,
                        "detection_reason": detection.reason,
                        "fallback_used": fallback_used,
                        "true_region_detected": any_true_detected,
                        "true_region_in_shortlist": any_true_short,
                        "true_region_in_final_topk": any_true_final,
                        "selected_text": selected_text,
                        "selected_looks_lot_or_other": _looks_lot_or_other(selected_text),
                        "selected_looks_production": _looks_production(selected_text),
                        "final_candidates_count": len(final_candidates),
                    }
                )

                if filename == "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg":
                    # lower-right diagnostic: full_image:raw candidates sorted by geometry and position
                    lower_candidates = []
                    for c in variant_candidates.get("full_image:raw", []):
                        x1, y1, x2, y2 = c.bbox
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        cx = x1 + w / 2.0
                        cy = y1 + h / 2.0
                        # normalize against full_image:raw extent derived from bbox maxes
                        lower_candidates.append(
                            {
                                "candidate_id": c.candidate_id,
                                "bbox_xyxy": list(c.bbox),
                                "center_xy": [round(cx, 2), round(cy, 2)],
                                "geometry_rank": c.geometry_rank,
                                "geometry_score": c.geometry_score,
                                "total_score": c.total_score,
                                "probe_text": c.probe_text,
                                "probe_confidence": c.probe_confidence,
                                "expected_match_probe": c.expected_match_probe,
                                "group_bonus": c.score_breakdown.get("group_bonus", 0.0),
                                "geometry_features": c.geometry_features,
                                "score_breakdown": c.score_breakdown,
                            }
                        )
                    lower_candidates.sort(key=lambda x: x["geometry_score"], reverse=True)
                    shortlist_ids = {c.candidate_id for c in sorted(variant_candidates.get("full_image:raw", []), key=lambda c: c.geometry_score, reverse=True)[:12]}
                    special_65940 = {
                        "filename": filename,
                        "expected_label": "12/12/2027",
                        "detection_detected": detection.detected,
                        "detection_reason": detection.reason,
                        "fallback_used": fallback_used,
                        "true_region_detected": any_true_detected,
                        "true_region_in_shortlist_12": any_true_short,
                        "true_region_in_final_topk_12": any_true_final,
                        "shortlist_12_ids_full_image_raw": sorted(shortlist_ids),
                        "full_image_raw_candidates_geometry_sorted": lower_candidates,
                        "explanation": (
                            "Candidates containing expiry signal were present in detected set "
                            "but dropped before final selection if geometry_rank > shortlist threshold "
                            "or probe-based total_score stayed below competing packaging text candidates."
                        ),
                    }

        print(f"  -> done {filename}")

    elapsed_total = (datetime.now() - start).total_seconds()

    breakdown_counts = {
        "text_detector_missed_date": 0,
        "geometry_shortlist_missed_date": 0,
        "ranker_missed_date_after_probe": 0,
        "bad_crop_to_parseq": 0,
        "parseq_failed_on_selected_crop": 0,
        "parser_rejected_valid_text": 0,
        "unknown": 0,
    }
    for item in baseline_recognition_fails:
        cls = item["classification"]
        breakdown_counts[cls] = breakdown_counts.get(cls, 0) + 1

    for k in shortlist_keys:
        total_scans = metrics[k]["summary"]["total"] or 1
        metrics[k]["average_candidates_evaluated_per_scan"] = round(metrics[k]["candidates_evaluated_total"] / total_scans, 3)
        metrics[k]["elapsed_seconds"] = round(metrics[k]["elapsed_seconds"], 3)
        metrics[k]["true_region_in_shortlist_rate"] = round(metrics[k]["true_region_in_shortlist_count"] / total_scans, 4)
        metrics[k]["true_region_in_final_topk_rate"] = round(metrics[k]["true_region_in_final_topk_count"] / total_scans, 4)

    report_root = Path(args.output_dir).resolve() / f"test64_geometry_shortlist_cached_{_utc_stamp()}"
    _ensure_dir(report_root)

    payload = {
        "source": args.source,
        "latest_batch_size": len(rows),
        "final_top_k_fixed": args.final_top_k,
        "geometry_shortlist_ablation": metrics,
        "recognition_fail_count_baseline_12": len(baseline_recognition_fails),
        "recognition_fail_breakdown_baseline_12": breakdown_counts,
        "recognition_fail_rows_baseline_12": baseline_recognition_fails,
        "special_65940c3a": special_65940,
        "elapsed_total_seconds": round(elapsed_total, 3),
    }

    report_json = report_root / "geometry_shortlist_ablation_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    summary_lines = [
        "# Geometry Shortlist Ablation (Cached)",
        "",
        f"- source: `{args.source}`",
        f"- final_top_k fixed: `{args.final_top_k}`",
        f"- elapsed_total_seconds: `{round(elapsed_total, 3)}`",
        "",
        "## Recognition Fail Breakdown (shortlist=12)",
    ]
    for key, val in breakdown_counts.items():
        summary_lines.append(f"- {key}: {val}")
    summary_lines.append("")
    summary_lines.append("## Ablation")
    for key in ("12", "30", "50", "all"):
        run = metrics[key]
        summary_lines.append(f"### shortlist={key}")
        summary_lines.append(f"- summary: `{run['summary']}`")
        summary_lines.append(f"- parseable_candidates_total: `{run['parseable_candidates_total']}`")
        summary_lines.append(f"- true_region_in_shortlist_count: `{run['true_region_in_shortlist_count']}`")
        summary_lines.append(f"- true_region_in_final_topk_count: `{run['true_region_in_final_topk_count']}`")
        summary_lines.append(f"- avg_candidates_per_scan: `{run['average_candidates_evaluated_per_scan']}`")
        summary_lines.append(f"- elapsed_seconds: `{run['elapsed_seconds']}`")
        summary_lines.append("")
    summary_md = report_root / "geometry_shortlist_ablation_summary.md"
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    special_md = report_root / "65940c3a_geometry_inspection.md"
    special_lines = [
        "# 65940c3a Geometry Inspection",
        "",
        f"- expected: `12/12/2027`",
        f"- detection_detected: `{special_65940.get('detection_detected')}`",
        f"- true_region_detected: `{special_65940.get('true_region_detected')}`",
        f"- true_region_in_shortlist_12: `{special_65940.get('true_region_in_shortlist_12')}`",
        f"- true_region_in_final_topk_12: `{special_65940.get('true_region_in_final_topk_12')}`",
        "",
        "## full_image:raw candidates (geometry-sorted)",
    ]
    for item in special_65940.get("full_image_raw_candidates_geometry_sorted", [])[:80]:
        special_lines.append(
            f"- {item['candidate_id']} rank={item['geometry_rank']} geom={item['geometry_score']:.4f} "
            f"total={item['total_score']:.4f} probe=`{item['probe_text']}` "
            f"group_bonus={item['group_bonus']:.3f} bbox={item['bbox_xyxy']}"
        )
    special_md.write_text("\n".join(special_lines), encoding="utf-8")

    print("\nCompleted.")
    print("Report root:", report_root)
    print("Generated:")
    print(report_json)
    print(summary_md)
    print(special_md)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast cached geometry shortlist ablation for SmartBite test64")
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
