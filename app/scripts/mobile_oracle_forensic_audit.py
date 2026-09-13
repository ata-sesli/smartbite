from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.ai.mobile_expiry_pipeline import MOBILE_RECOGNITION_VARIANTS
from app.ai.mobile_expiry_pipeline import MobileExpiryPipeline
from app.ai.mobile_expiry_pipeline import _DateEvidence
from app.ai.mobile_expiry_pipeline import _EvaluatedCandidate
from app.ai.mobile_expiry_pipeline import _MobileGlobalCandidate
from app.ai.mobile_expiry_pipeline import _MobileRankedCandidate
from app.ai.mobile_expiry_pipeline import _RecognitionOutput
from app.ai.mobile_expiry_pipeline import _YoloCandidate
from app.ai.preprocess import ImageVariant
from app.ai.types import ParsedDateData
from app.infra.settings import Settings, get_settings
from app.scripts.mobile_test64_upload_benchmark import IMAGE_SUFFIXES
from app.scripts.mobile_test64_upload_benchmark import _expected_label
from app.scripts.mobile_test64_upload_benchmark import _json_default
from app.workers.scan_jobs import build_pipeline


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _clean_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _clean_polygon(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return points


def _bbox_overlap_metrics(
    candidate: tuple[int, int, int, int] | None,
    truth: tuple[int, int, int, int] | None,
) -> dict[str, float]:
    if candidate is None or truth is None:
        return {
            "iou": 0.0,
            "truth_coverage": 0.0,
            "candidate_coverage": 0.0,
            "containment": 0.0,
        }
    ix1 = max(candidate[0], truth[0])
    iy1 = max(candidate[1], truth[1])
    ix2 = min(candidate[2], truth[2])
    iy2 = min(candidate[3], truth[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
    truth_area = max(1, (truth[2] - truth[0]) * (truth[3] - truth[1]))
    union = max(1, candidate_area + truth_area - inter)
    return {
        "iou": inter / float(union),
        "truth_coverage": inter / float(truth_area),
        "candidate_coverage": inter / float(candidate_area),
        "containment": inter / float(max(1, min(candidate_area, truth_area))),
    }


def _classify_truth_overlap(metrics: dict[str, float]) -> str:
    if float(metrics.get("truth_coverage") or 0.0) >= 0.75 or float(metrics.get("iou") or 0.0) >= 0.50:
        return "covered"
    if float(metrics.get("truth_coverage") or 0.0) >= 0.45 and float(metrics.get("candidate_coverage") or 0.0) >= 0.35:
        return "tight"
    if float(metrics.get("truth_coverage") or 0.0) >= 0.20 or float(metrics.get("iou") or 0.0) >= 0.08:
        return "partial"
    return "missed"


def _expected_date_matches(parsed_date: date | None, expected: dict[str, Any]) -> bool:
    if parsed_date is None:
        return False
    year = expected.get("year")
    month = expected.get("month")
    day = expected.get("day")
    if parsed_date.year != year or parsed_date.month != month:
        return False
    if expected.get("precision") == "month" or day is None:
        return True
    return parsed_date.day == day


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _classify_row(row: dict[str, Any]) -> str:
    expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
    truth = row.get("truth") if isinstance(row.get("truth"), dict) else {}
    truth_overlap = row.get("truth_overlap") if isinstance(row.get("truth_overlap"), dict) else {}
    oracle = row.get("truth_crop_oracle") if isinstance(row.get("truth_crop_oracle"), dict) else {}
    if not truth.get("valid") or oracle.get("reliable") is False:
        return "F_truth_unreliable"
    if expected.get("precision") == "month":
        return "E_ambiguous_month_only"
    if row.get("expected_in_accepted"):
        return "A_selector"
    if row.get("expected_before_filter"):
        return "B_filter"
    if truth_overlap.get("best_verdict") in {"covered", "tight"}:
        return "C_crop_recognition"
    return "D_proposal_recall"


def _settings_for_stable_audit() -> Settings:
    settings = get_settings()
    return settings.model_copy(
        update={
            "mobile_product_cropper_rescue_enabled": False,
            "mobile_legacy_wide_group_rescue_enabled": False,
            "mobile_strict_evidence_acceptance_enabled": False,
            "mobile_rapidocr_role_constraint_enabled": False,
        }
    )


def _load_stable_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"stable report does not contain rows: {path}")
    return [row for row in rows if isinstance(row, dict) and row.get("exact_match") is not True]


def _load_truth_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError(f"truth manifest does not contain items: {path}")
    return {str(name): item for name, item in items.items() if isinstance(item, dict)}


def _truth_for_row(filename: str, truth_items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = truth_items.get(filename, {})
    bbox = _clean_bbox(item.get("true_bbox_xyxy")) if item else None
    polygon = _clean_polygon(item.get("true_polygon_xy")) if item else None
    return {
        "valid": bbox is not None,
        "bbox_xyxy": list(bbox) if bbox else None,
        "polygon_xy": polygon,
        "expected_day": item.get("expected_day") if item else None,
        "expected_month": item.get("expected_month") if item else None,
        "expected_year": item.get("expected_year") if item else None,
        "annotation_rotation_degrees": item.get("annotation_rotation_degrees") if item else None,
        "updated_at": item.get("updated_at") if item else None,
    }


def _scan_bbox(bbox: tuple[int, int, int, int] | None, *, offset_x: int = 0, offset_y: int = 0) -> list[int] | None:
    if bbox is None:
        return None
    return [int(bbox[0] + offset_x), int(bbox[1] + offset_y), int(bbox[2] + offset_x), int(bbox[3] + offset_y)]


def _score_overlap(
    bbox: tuple[int, int, int, int] | list[int] | None,
    truth_bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    cleaned = _clean_bbox(bbox) if bbox is not None else None
    metrics = _bbox_overlap_metrics(cleaned, truth_bbox)
    return {**metrics, "verdict": _classify_truth_overlap(metrics)}


def _parsed_json(parsed: ParsedDateData) -> dict[str, Any]:
    return {
        "parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date is not None else None,
        "date_format_detected": parsed.date_format_detected,
        "confidence": parsed.confidence,
        "candidates": parsed.candidates,
        "reason": parsed.reason,
        "date_precision": parsed.date_precision,
        "parsed_day": parsed.parsed_day,
        "parsed_month": parsed.parsed_month,
        "parsed_year": parsed.parsed_year,
    }


def _recognition_json(recognition: _RecognitionOutput) -> dict[str, Any]:
    return {
        "raw_text": recognition.raw_text,
        "normalized_text": recognition.normalized_text,
        "confidence": recognition.confidence,
        "reason": recognition.reason,
        "rotation": recognition.rotation,
    }


def _candidate_json(
    candidate: _MobileRankedCandidate,
    *,
    truth_bbox: tuple[int, int, int, int] | None,
    include_probe: bool = True,
) -> dict[str, Any]:
    scan_bbox = _scan_bbox(candidate.bbox_xyxy, offset_x=candidate.scan_offset_x, offset_y=candidate.scan_offset_y)
    final_bbox = _scan_bbox(candidate.final_crop_bbox, offset_x=candidate.scan_offset_x, offset_y=candidate.scan_offset_y)
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_type": candidate.candidate_type,
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "scan_bbox_xyxy": scan_bbox,
        "polygon_xy": [[float(x), float(y)] for x, y in candidate.polygon_xy] if candidate.polygon_xy else None,
        "detector_confidence": candidate.detector_confidence,
        "detector_sources": list(candidate.detector_sources),
        "detector_variant": candidate.detector_variant,
        "member_indices": candidate.member_indices,
        "member_bboxes": [list(bbox) for bbox in candidate.member_bboxes],
        "geometry_features": candidate.geometry_features,
        "geometry_score": candidate.geometry_score,
        "score_breakdown": candidate.score_breakdown,
        "total_score": candidate.total_score,
        "selected_geometry": candidate.selected_geometry,
        "selected_final": candidate.selected_final,
        "final_rank_before_force_include": candidate.final_rank_before_force_include,
        "final_rank_after_force_include": candidate.final_rank_after_force_include,
        "force_included_reason": candidate.force_included_reason,
        "recognition_bbox": list(candidate.recognition_bbox) if candidate.recognition_bbox else None,
        "evidence_bbox": list(candidate.evidence_bbox) if candidate.evidence_bbox else None,
        "final_crop_bbox": list(candidate.final_crop_bbox) if candidate.final_crop_bbox else None,
        "final_crop_bbox_scan": final_bbox,
        "final_crop_policy": candidate.final_crop_policy,
        "final_crop_padding_px": candidate.final_crop_padding_px,
        "context_probe_bboxes": [list(bbox) for bbox in candidate.context_probe_bboxes or []],
        "context_probe_texts": candidate.context_probe_texts or [],
        "context_probe_confidences": candidate.context_probe_confidences or [],
        "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
        "final_crop_truth_overlap": _score_overlap(final_bbox, truth_bbox),
        "probe": {
            "raw_text": candidate.probe_text,
            "normalized_text": candidate.probe_normalized_text,
            "confidence": candidate.probe_confidence,
            "reason": candidate.probe_reason,
            "selected_variant": candidate.selected_recognition_variant,
        }
        if include_probe
        else None,
    }


def _yolo_json(candidate: _YoloCandidate, *, truth_bbox: tuple[int, int, int, int] | None) -> dict[str, Any]:
    return {
        "bbox_xyxy": list(candidate.bbox_xyxy),
        "polygon_xy": candidate.polygon_xy,
        "confidence": candidate.confidence,
        "source": candidate.source,
        "sources": list(candidate.sources),
        "variant_name": candidate.variant_name,
        "truth_overlap": _score_overlap(candidate.bbox_xyxy, truth_bbox),
    }


def _proposal_json(
    proposal: Any,
    *,
    truth_bbox: tuple[int, int, int, int] | None,
    offset_x: int,
    offset_y: int,
) -> dict[str, Any]:
    scan_bbox = _scan_bbox(proposal.bbox_xyxy, offset_x=offset_x, offset_y=offset_y)
    polygon = None
    if proposal.polygon_xy is not None:
        polygon = [[float(x) + offset_x, float(y) + offset_y] for x, y in proposal.polygon_xy]
    return {
        "bbox_xyxy": list(proposal.bbox_xyxy),
        "scan_bbox_xyxy": scan_bbox,
        "polygon_xy": polygon,
        "confidence": proposal.confidence,
        "source": proposal.source,
        "sources": list(proposal.sources),
        "variant_name": proposal.variant_name,
        "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
    }


def _parse_inputs_report(pipeline: MobileExpiryPipeline, recognition: _RecognitionOutput, *, today: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in pipeline._build_parse_inputs(recognition):
        parsed = pipeline.parser.parse(text, reference_date=today)
        rows.append({"text": text, "parsed": _parsed_json(parsed)})
    return rows


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


def _trace_recognition_variants(
    pipeline: MobileExpiryPipeline,
    crop: Any,
    *,
    today: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in pipeline._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS):
        rec = pipeline._recognize_variant(variant.image, variant_name=variant.name, orientation=crop.selected_orientation)
        parse_inputs = _parse_inputs_report(pipeline, rec, today=today)
        best = pipeline._best_parse_for_inputs([item["text"] for item in parse_inputs], today=today)
        rows.append(
            {
                "variant": variant.name,
                "orientation": crop.selected_orientation,
                "recognition": _recognition_json(rec),
                "parse_inputs": parse_inputs,
                "best_parse": _parsed_json(best),
            }
        )
        if pipeline._recognition_result_needs_180_fallback(rec, best):
            rotated = pipeline._rotate_180_normalized_crop(crop)
            rotated_rec = pipeline._recognize_variant(rotated.image, variant_name=f"{variant.name}:rotate_180", orientation="rotate_180")
            rotated_inputs = _parse_inputs_report(pipeline, rotated_rec, today=today)
            rotated_best = pipeline._best_parse_for_inputs([item["text"] for item in rotated_inputs], today=today)
            rows.append(
                {
                    "variant": variant.name,
                    "orientation": "rotate_180",
                    "recognition": _recognition_json(rotated_rec),
                    "parse_inputs": rotated_inputs,
                    "best_parse": _parsed_json(rotated_best),
                }
            )
    return rows


def _date_evidence_json(
    pipeline: MobileExpiryPipeline,
    evidence: _DateEvidence,
    *,
    expected: dict[str, Any],
) -> dict[str, Any]:
    bbox = MobileExpiryPipeline._bbox_json_with_candidate_offset(
        evidence.normalized_crop.bbox_xyxy if evidence.normalized_crop is not None else evidence.candidate.final_crop_bbox,
        evidence.candidate,
    )
    return {
        "parsed": _parsed_json(evidence.parsed),
        "matches_expected": _expected_date_matches(evidence.parsed.parsed_date, expected),
        "recognition": _recognition_json(evidence.recognition),
        "parse_inputs_count": evidence.parse_inputs_count,
        "recognition_variant": evidence.recognition_variant,
        "crop_policy": evidence.crop_policy,
        "context_texts": list(evidence.context_texts),
        "candidate": _candidate_json(evidence.candidate, truth_bbox=None),
        "final_recognition_bbox_xyxy": bbox,
        "score": pipeline._score_date_evidence(evidence),
        "suppressed": pipeline._should_suppress_date_evidence(evidence),
    }


def _cluster_date_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[tuple[str | None, str | None], list[dict[str, Any]]] = {}
    for item in items:
        parsed = item.get("parsed") if isinstance(item.get("parsed"), dict) else {}
        key = (parsed.get("parsed_date"), parsed.get("date_precision"))
        clusters.setdefault(key, []).append(item)
    return [
        {
            "parsed_date": parsed_date,
            "date_precision": precision,
            "support": len(rows),
            "matches_expected": any(row.get("matches_expected") for row in rows),
            "evidence_indices": [row["evidence_index"] for row in rows],
            "variants": sorted({str(row.get("recognition_variant")) for row in rows}),
            "candidate_ids": sorted({str(row.get("candidate", {}).get("candidate_id")) for row in rows}),
        }
        for (parsed_date, precision), rows in sorted(clusters.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
    ]


def _save_final_candidate_crops(
    pipeline: MobileExpiryPipeline,
    global_candidate: _MobileGlobalCandidate,
    *,
    row_dir: Path,
    index: int,
    today: date,
    truth_bbox: tuple[int, int, int, int] | None,
) -> list[dict[str, Any]]:
    crop_rows: list[dict[str, Any]] = []
    crops = pipeline._normalized_crops_for_candidate(global_candidate.image, global_candidate.ranked)
    for crop_index, crop in enumerate(crops, start=1):
        crop_path = _write_png(row_dir / f"final_{index:02d}_crop_{crop_index:02d}.png", crop.image)
        scan_bbox = _scan_bbox(crop.bbox_xyxy, offset_x=global_candidate.ranked.scan_offset_x, offset_y=global_candidate.ranked.scan_offset_y)
        crop_rows.append(
            {
                "crop_index": crop_index,
                "crop_path": crop_path,
                "bbox_xyxy": list(crop.bbox_xyxy),
                "scan_bbox_xyxy": scan_bbox,
                "crop_transform_used": crop.crop_transform_used,
                "selected_orientation": crop.selected_orientation,
                "original_crop_shape": crop.original_crop_shape,
                "normalized_crop_shape": crop.normalized_crop_shape,
                "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
                "svtr_variants": _trace_recognition_variants(pipeline, crop, today=today),
            }
        )
    return crop_rows


def _update_best_overlap(best: dict[str, Any], item: dict[str, Any], *, stage: str, bbox_key: str = "scan_bbox_xyxy") -> dict[str, Any]:
    overlap = item.get("truth_overlap") if isinstance(item.get("truth_overlap"), dict) else {}
    current = float(overlap.get("truth_coverage") or 0.0), float(overlap.get("iou") or 0.0)
    previous = float(best.get("truth_coverage") or 0.0), float(best.get("iou") or 0.0)
    if current > previous:
        return {
            "stage": stage,
            "bbox_xyxy": item.get(bbox_key),
            "verdict": overlap.get("verdict"),
            "truth_coverage": overlap.get("truth_coverage"),
            "candidate_coverage": overlap.get("candidate_coverage"),
            "iou": overlap.get("iou"),
        }
    return best


def _expected_from_report_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": row.get("expected_date"),
        "precision": row.get("expected_precision"),
        "day": row.get("expected_day"),
        "month": row.get("expected_month"),
        "year": row.get("expected_year"),
    }


def _truth_crop_oracle(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    truth: dict[str, Any],
    *,
    expected: dict[str, Any],
    today: date,
    row_dir: Path,
) -> dict[str, Any]:
    bbox = _clean_bbox(truth.get("bbox_xyxy"))
    if bbox is None:
        return {"reliable": False, "reason": "missing_truth_bbox", "variants": []}
    x1, y1, x2, y2 = bbox
    crop_image = image[y1:y2, x1:x2]
    crop_path = _write_png(row_dir / "oracle_truth_crop.png", crop_image)
    if crop_image.size == 0:
        return {"reliable": False, "reason": "empty_truth_crop", "crop_path": crop_path, "variants": []}
    variant = ImageVariant("truth_bbox", crop_image, purpose="recognition")
    rows: list[dict[str, Any]] = []
    any_expected = False
    for rec_variant in pipeline._recognition_variants_for_crop(variant.image, allowed_names=MOBILE_RECOGNITION_VARIANTS):
        rec = pipeline._recognize_variant(rec_variant.image, variant_name=rec_variant.name, orientation="original")
        parse_inputs = _parse_inputs_report(pipeline, rec, today=today)
        best = pipeline._best_parse_for_inputs([item["text"] for item in parse_inputs], today=today)
        matches = _expected_date_matches(best.parsed_date, expected)
        any_expected = any_expected or matches
        rows.append(
            {
                "variant": rec_variant.name,
                "recognition": _recognition_json(rec),
                "parse_inputs": parse_inputs,
                "best_parse": _parsed_json(best),
                "matches_expected": matches,
            }
        )
    return {
        "reliable": bool(any_expected),
        "reason": "oracle_crop_matches_expected" if any_expected else "oracle_crop_did_not_parse_expected",
        "crop_path": crop_path,
        "variants": rows,
    }


def _trace_stable_pipeline(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    *,
    row_dir: Path,
    truth_bbox: tuple[int, int, int, int] | None,
    expected: dict[str, Any],
    today: date,
) -> dict[str, Any]:
    yolo_candidates, detection_reason = pipeline._detect_candidates(image)
    best_overlap: dict[str, Any] = {"stage": None, "verdict": "missed", "truth_coverage": 0.0, "iou": 0.0}
    yolo_rows = [_yolo_json(candidate, truth_bbox=truth_bbox) for candidate in yolo_candidates]
    for item in yolo_rows:
        best_overlap = _update_best_overlap(best_overlap, item, stage="yolo_obb", bbox_key="bbox_xyxy")

    stages: list[dict[str, Any]] = []
    global_candidates: list[_MobileGlobalCandidate] = []

    if yolo_candidates:
        direct_ranked = pipeline._build_ranked_candidates(image, yolo_candidates)
        direct_built = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in direct_ranked]
        pipeline._apply_geometry_shortlist(direct_ranked)
        direct_after_geometry = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in direct_ranked]
        for item in direct_after_geometry:
            best_overlap = _update_best_overlap(best_overlap, item, stage="direct_yolo_candidates")
        globals_for_stage = pipeline._global_candidates_from_ranked(
            image,
            direct_ranked,
            source="mobile_yolo",
            variant="yolo26s_obb",
            offset_x=0,
            offset_y=0,
        )
        global_candidates.extend(globals_for_stage)
        stages.append(
            {
                "stage": "direct_yolo_candidates",
                "ranked_before_geometry": direct_built,
                "ranked_after_geometry": direct_after_geometry,
                "global_candidate_count": len(globals_for_stage),
            }
        )

    rapidocr_reasons: list[str] = []
    for roi_index, roi in enumerate(pipeline._collect_advanced_roi_candidates(image, yolo_candidates), start=1):
        for variant in pipeline._detector_variants_for_roi(roi.image):
            proposals, reason = pipeline._combined_primary_proposals_for_roi(
                roi.image,
                variant=variant,
                roi_confidence=roi.confidence,
            )
            if reason:
                rapidocr_reasons.append(f"{roi.source}:{variant.name}:{reason}")
            proposal_rows = [
                _proposal_json(proposal, truth_bbox=truth_bbox, offset_x=roi.offset_x, offset_y=roi.offset_y)
                for proposal in proposals
            ]
            for item in proposal_rows:
                best_overlap = _update_best_overlap(best_overlap, item, stage="rapidocr_primary_proposals")
            ranked = pipeline._build_ranked_candidates_from_proposals(roi.image, proposals)
            built = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in ranked]
            pipeline._apply_geometry_shortlist(ranked)
            globals_for_stage = pipeline._global_candidates_from_ranked(
                roi.image,
                ranked,
                source=roi.source,
                variant=variant.name,
                offset_x=roi.offset_x,
                offset_y=roi.offset_y,
            )
            after_geometry = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in ranked]
            for item in after_geometry:
                best_overlap = _update_best_overlap(best_overlap, item, stage="rapidocr_ranked_candidates")
            global_candidates.extend(globals_for_stage)
            stages.append(
                {
                    "stage": "rapidocr_primary_roi",
                    "roi_index": roi_index,
                    "roi_source": roi.source,
                    "roi_offset": [roi.offset_x, roi.offset_y],
                    "roi_shape": list(roi.image.shape),
                    "roi_confidence": roi.confidence,
                    "detector_variant": variant.name,
                    "reason": reason,
                    "proposals": proposal_rows,
                    "ranked_before_geometry": built,
                    "ranked_after_geometry": after_geometry,
                    "global_candidate_count": len(globals_for_stage),
                }
            )

    if not global_candidates:
        evaluated: list[_EvaluatedCandidate] = []
        probe_candidates: list[_MobileGlobalCandidate] = []
        final_candidates: list[_MobileGlobalCandidate] = []
    else:
        probe_candidates = pipeline._select_global_probe_candidates(global_candidates)
        pipeline._probe_global_candidates(probe_candidates, today=today)
        final_candidates = pipeline._select_global_final_candidates(probe_candidates)
        evaluated = [pipeline._evaluate_candidate(candidate.image, candidate.ranked, today=today) for candidate in final_candidates]

    if pipeline._should_run_rapidocr_rescue(list(evaluated)):
        proposal_candidates, proposal_reason = pipeline._detect_rapidocr_rescue_candidates(image, yolo_candidates)
        proposal_rows = [_yolo_json(candidate, truth_bbox=truth_bbox) for candidate in proposal_candidates]
        for item in proposal_rows:
            best_overlap = _update_best_overlap(best_overlap, item, stage="rapidocr_rescue_yolo_candidates", bbox_key="bbox_xyxy")
        if proposal_candidates:
            ranked = pipeline._build_ranked_candidates(image, proposal_candidates)
            built = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in ranked]
            rescue_evaluated = pipeline._evaluate_architecture_candidates(image, ranked, today=today)
            accepted_rescue = pipeline._accepted_rapidocr_rescue_evaluations(list(evaluated), rescue_evaluated)
            evaluated.extend(accepted_rescue)
            stages.append(
                {
                    "stage": "rapidocr_rescue",
                    "reason": proposal_reason,
                    "proposals": proposal_rows,
                    "ranked_before_evaluation": built,
                    "evaluated_count": len(rescue_evaluated),
                    "accepted_count": len(accepted_rescue),
                }
            )
        else:
            stages.append({"stage": "rapidocr_rescue", "reason": proposal_reason, "proposals": []})

    if not any(item.parsed.parsed_date is not None for item in evaluated):
        rescue_candidates, rescue_reason = pipeline._detect_variant_rescue_candidates(image)
        rescue_rows = [_yolo_json(candidate, truth_bbox=truth_bbox) for candidate in rescue_candidates]
        for item in rescue_rows:
            best_overlap = _update_best_overlap(best_overlap, item, stage="variant_rescue_yolo_candidates", bbox_key="bbox_xyxy")
        if rescue_candidates:
            rescue_ranked = pipeline._build_ranked_candidates(image, rescue_candidates)
            built = [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in rescue_ranked]
            rescue_evaluated = pipeline._evaluate_architecture_candidates(image, rescue_ranked, today=today)
            evaluated.extend(rescue_evaluated)
            stages.append(
                {
                    "stage": "variant_rescue",
                    "reason": rescue_reason,
                    "proposals": rescue_rows,
                    "ranked_before_evaluation": built,
                    "evaluated_count": len(rescue_evaluated),
                }
            )
        else:
            stages.append({"stage": "variant_rescue", "reason": rescue_reason, "proposals": []})

    if yolo_candidates and not any(item.parsed.parsed_date is not None for item in evaluated):
        fallback_ranked = pipeline._build_ranked_candidates(image, [])
        fallback_evaluated = pipeline._evaluate_architecture_candidates(image, fallback_ranked, today=today)
        evaluated.extend(fallback_evaluated)
        stages.append(
            {
                "stage": "fallback_full",
                "ranked_before_evaluation": [_candidate_json(candidate, truth_bbox=truth_bbox, include_probe=False) for candidate in fallback_ranked],
                "evaluated_count": len(fallback_evaluated),
            }
        )

    final_candidate_rows: list[dict[str, Any]] = []
    for index, global_candidate in enumerate(final_candidates, start=1):
        item = _candidate_json(global_candidate.ranked, truth_bbox=truth_bbox)
        item["source"] = global_candidate.source
        item["variant"] = global_candidate.variant
        item["variant_key"] = global_candidate.variant_key
        item["final_crops"] = _save_final_candidate_crops(
            pipeline,
            global_candidate,
            row_dir=row_dir,
            index=index,
            today=today,
            truth_bbox=truth_bbox,
        )
        final_candidate_rows.append(item)
        for crop_item in item["final_crops"]:
            best_overlap = _update_best_overlap(best_overlap, crop_item, stage="final_candidate_crops")

    all_evidence = [
        evidence
        for item in evaluated
        for evidence in item.date_evidence
        if evidence.parsed.parsed_date is not None
    ]
    accepted_parseable, accepted_evidence = pipeline._accepted_parseable_items(evaluated)
    accepted_ids = {id(evidence) for evidence in accepted_evidence}
    evidence_rows: list[dict[str, Any]] = []
    for index, evidence in enumerate(all_evidence, start=1):
        row = _date_evidence_json(pipeline, evidence, expected=expected)
        row["evidence_index"] = index
        row["accepted"] = id(evidence) in accepted_ids
        evidence_rows.append(row)

    if accepted_evidence:
        selected_evidence = pipeline._select_date_evidence(accepted_evidence)
        selected = pipeline._evaluated_from_date_evidence(selected_evidence, date_evidence=accepted_evidence)
    elif accepted_parseable:
        selected = max(accepted_parseable, key=pipeline._score_parseable_candidate)
        selected_evidence = selected.selected_date_evidence
    elif evaluated:
        selected = max(evaluated, key=pipeline._score_unparseable_candidate)
        selected_evidence = None
    else:
        selected = None
        selected_evidence = None

    selected_json = None
    if selected is not None:
        accepted_parseable_ids = {id(item) for item in accepted_parseable}
        selected_json = {
            "status": "parsed_success" if selected.parsed.parsed_date is not None and id(selected) in accepted_parseable_ids else "manual_review_required",
            "parsed": _parsed_json(selected.parsed),
            "recognition": _recognition_json(selected.recognition),
            "recognition_variant": selected.recognition_variant,
            "candidate": _candidate_json(selected.candidate, truth_bbox=truth_bbox),
            "final_recognition_bbox_xyxy": MobileExpiryPipeline._final_recognition_bbox_json(selected),
            "reason": MobileExpiryPipeline._result_reason(selected),
            "selected_evidence_index": (
                next((row["evidence_index"] for row in evidence_rows if selected_evidence is not None and row["parsed"]["parsed_date"] == selected_evidence.parsed.parsed_date.isoformat()), None)
                if selected_evidence is not None and selected_evidence.parsed.parsed_date is not None
                else None
            ),
        }

    any_variant_expected = any(
        _expected_date_matches(_parse_iso_date(variant["best_parse"]["parsed_date"]), expected)
        for candidate in final_candidate_rows
        for crop in candidate.get("final_crops", [])
        for variant in crop.get("svtr_variants", [])
    )
    return {
        "detection_reason": detection_reason,
        "rapidocr_reasons": rapidocr_reasons,
        "yolo_obb_proposals": yolo_rows,
        "stages": stages,
        "probe_candidates": [_candidate_json(candidate.ranked, truth_bbox=truth_bbox) for candidate in probe_candidates],
        "final_candidates": final_candidate_rows,
        "evaluated_candidates": [
            {
                "candidate": _candidate_json(item.candidate, truth_bbox=truth_bbox),
                "recognition": _recognition_json(item.recognition),
                "parsed": _parsed_json(item.parsed),
                "recognition_variant": item.recognition_variant,
                "parse_inputs_count": item.parse_inputs_count,
                "date_evidence_count": len(item.date_evidence),
            }
            for item in evaluated
        ],
        "date_evidence": evidence_rows,
        "date_evidence_clusters": _cluster_date_evidence(evidence_rows),
        "accepted_date_evidence": [row for row in evidence_rows if row.get("accepted")],
        "selected": selected_json,
        "truth_overlap": {**best_overlap, "best_verdict": best_overlap.get("verdict")},
        "expected_in_accepted": any(row.get("accepted") and row.get("matches_expected") for row in evidence_rows),
        "expected_before_filter": any(row.get("matches_expected") for row in evidence_rows) or any_variant_expected,
    }


def _row_report(
    *,
    pipeline: MobileExpiryPipeline,
    image_path: Path,
    stable_row: dict[str, Any],
    truth: dict[str, Any],
    row_dir: Path,
    today: date,
) -> dict[str, Any]:
    image = pipeline.decode_image(image_path.read_bytes())
    expected = _expected_from_report_row(stable_row)
    if image is None:
        row = {
            "filename": image_path.name,
            "expected": expected,
            "detected": stable_row.get("detected_expiry_date"),
            "truth": truth,
            "truth_crop_oracle": {"reliable": False, "reason": "image_unreadable"},
            "truth_overlap": {"best_verdict": "missed"},
            "expected_in_accepted": False,
            "expected_before_filter": False,
        }
        row["class"] = _classify_row(row)
        return row
    truth_bbox = _clean_bbox(truth.get("bbox_xyxy"))
    started = perf_counter()
    oracle = _truth_crop_oracle(pipeline, image, truth, expected=expected, today=today, row_dir=row_dir)
    trace = _trace_stable_pipeline(
        pipeline,
        image,
        row_dir=row_dir,
        truth_bbox=truth_bbox,
        expected=expected,
        today=today,
    )
    row = {
        "filename": image_path.name,
        "runtime_ms": round((perf_counter() - started) * 1000.0, 3),
        "expected": expected,
        "detected": stable_row.get("detected_expiry_date"),
        "stable_status": stable_row.get("response_status"),
        "stable_reason": stable_row.get("reason"),
        "truth": truth,
        "truth_crop_oracle": oracle,
        **trace,
    }
    row["class"] = _classify_row(row)
    row["blocker"] = _blocker_for_class(row["class"])
    return row


def _blocker_for_class(class_name: str) -> str:
    return {
        "A_selector": "selector chose another date even though expected evidence survived",
        "B_filter": "expected evidence was generated but filtered before final selection",
        "C_crop_recognition": "truth region was proposed/cropped, but SVTR/parser did not recover expected date",
        "D_proposal_recall": "no candidate sufficiently covered the manual truth region",
        "E_ambiguous_month_only": "expected label is month-only or otherwise ambiguous",
        "F_truth_unreliable": "manual truth crop is missing or oracle recognition cannot verify it",
    }.get(class_name, "unknown")


def _report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row.get("class")) for row in rows)
    return {
        "total": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "exact_improvement_possible_without_training": [
            row["filename"]
            for row in rows
            if row.get("class") in {"A_selector", "B_filter"}
        ],
        "selector_can_be_fixed_safely": [
            row["filename"]
            for row in rows
            if row.get("class") == "A_selector"
        ],
        "proposal_recall_blocker": [
            row["filename"]
            for row in rows
            if row.get("class") == "D_proposal_recall"
        ],
        "recognizer_or_data_blocker": [
            row["filename"]
            for row in rows
            if row.get("class") in {"C_crop_recognition", "F_truth_unreliable"}
        ],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Mobile Oracle Forensic Audit",
        "",
        f"- total: `{summary['total']}`",
        f"- class_counts: `{summary['class_counts']}`",
        "",
        "## Improvement Buckets",
        "",
        f"- exact improvement possible without training: `{len(summary['exact_improvement_possible_without_training'])}`",
        f"- selector can be fixed safely: `{len(summary['selector_can_be_fixed_safely'])}`",
        f"- proposal recall blocker: `{len(summary['proposal_recall_blocker'])}`",
        f"- recognizer/data blocker: `{len(summary['recognizer_or_data_blocker'])}`",
        "",
        "## Rows",
        "",
        "| filename | expected | detected | class | selected bbox/source/variant | expected evidence stage | blocker |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        selected = row.get("selected") if isinstance(row.get("selected"), dict) else {}
        candidate = selected.get("candidate") if isinstance(selected.get("candidate"), dict) else {}
        selected_bits = [
            str(selected.get("final_recognition_bbox_xyxy") or ""),
            "/".join(candidate.get("detector_sources") or []),
            str(candidate.get("detector_variant") or ""),
            str(selected.get("recognition_variant") or ""),
        ]
        if row.get("expected_in_accepted"):
            expected_stage = "accepted_date_evidence"
        elif row.get("expected_before_filter"):
            expected_stage = "pre_filter_or_variant_trace"
        else:
            expected_stage = str(row.get("truth_overlap", {}).get("best_verdict") or "none")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("filename")),
                    str(row.get("expected", {}).get("label")),
                    str(row.get("detected")),
                    str(row.get("class")),
                    "<br>".join(bit for bit in selected_bits if bit),
                    expected_stage,
                    str(row.get("blocker")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    images_dir = Path(args.images_dir)
    stable_report = Path(args.stable_report)
    truth_manifest = Path(args.truth_manifest)
    output_root = Path(args.output_dir).resolve() / f"mobile_oracle_forensic_audit_{_utc_stamp()}"
    crops_root = output_root / "crops"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_stable_rows(stable_report)
    if args.limit is not None:
        rows = rows[: args.limit]
    truth_items = _load_truth_items(truth_manifest)
    pipeline = build_pipeline(_settings_for_stable_audit())

    audited: list[dict[str, Any]] = []
    today = date.today()
    for index, row in enumerate(rows, start=1):
        filename = str(row.get("filename"))
        image_path = images_dir / filename
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.exists():
            audited.append(
                {
                    "filename": filename,
                    "expected": _expected_label(row),
                    "detected": row.get("detected_expiry_date"),
                    "truth": {"valid": False, "bbox_xyxy": None, "reason": "image_missing"},
                    "truth_crop_oracle": {"reliable": False, "reason": "image_missing"},
                    "truth_overlap": {"best_verdict": "missed"},
                    "expected_in_accepted": False,
                    "expected_before_filter": False,
                    "class": "F_truth_unreliable",
                    "blocker": _blocker_for_class("F_truth_unreliable"),
                }
            )
            continue
        row_dir = crops_root / Path(filename).stem
        report = _row_report(
            pipeline=pipeline,
            image_path=image_path,
            stable_row=row,
            truth=_truth_for_row(filename, truth_items),
            row_dir=row_dir,
            today=today,
        )
        audited.append(report)
        print(
            f"[{index}/{len(rows)}] {filename} class={report['class']} "
            f"expected={report['expected']['label']} detected={report.get('detected')}",
            flush=True,
        )

    payload = {
        "mode": "mobile_oracle_forensic_audit",
        "created_at": datetime.utcnow(),
        "stable_report": str(stable_report),
        "truth_manifest": str(truth_manifest),
        "images_dir": str(images_dir),
        "summary": _report_summary(audited),
        "rows": audited,
    }
    json_path = output_root / "mobile_oracle_forensic_audit_report.json"
    md_path = output_root / "mobile_oracle_forensic_audit_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    print("Report root:", output_root)
    print(json_path)
    print(md_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle forensic audit for non-correct mobile test64 rows")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--stable-report", required=True)
    parser.add_argument("--truth-manifest", default="artifacts/forensics/crop_truth_annotations/test64_truth_bboxes.json")
    parser.add_argument("--output-dir", default="artifacts/onnx_parity")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    import asyncio

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
