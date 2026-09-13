from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ai.date_role import has_expiry_keyword, has_production_keyword
from app.ai.mobile_expiry_pipeline import MOBILE_RECOGNITION_VARIANTS
from app.ai.mobile_expiry_pipeline import MobileExpiryPipeline
from app.ai.mobile_expiry_pipeline import _DateEvidence
from app.ai.mobile_expiry_pipeline import _EvaluatedCandidate
from app.ai.mobile_expiry_pipeline import _MobileGlobalCandidate
from app.ai.mobile_expiry_pipeline import _MobileRankedCandidate
from app.ai.mobile_expiry_pipeline import _YoloCandidate
from app.ai.crop_normalization import TextLineCropConfig, TextLineGeometry, normalize_textline_crops
from app.infra.settings import get_settings
from app.scripts.mobile_oracle_forensic_audit import _bbox_overlap_metrics
from app.scripts.mobile_oracle_forensic_audit import _clean_bbox
from app.scripts.mobile_oracle_forensic_audit import _classify_truth_overlap
from app.scripts.mobile_oracle_forensic_audit import _expected_date_matches
from app.scripts.mobile_oracle_forensic_audit import _expected_from_report_row
from app.scripts.mobile_oracle_forensic_audit import _parse_inputs_report
from app.scripts.mobile_oracle_forensic_audit import _parsed_json
from app.scripts.mobile_oracle_forensic_audit import _recognition_json
from app.scripts.mobile_oracle_forensic_audit import _scan_bbox
from app.scripts.mobile_oracle_forensic_audit import _score_overlap
from app.scripts.mobile_oracle_forensic_audit import _truth_for_row
from app.scripts.mobile_test64_upload_benchmark import _json_default
from app.workers.scan_jobs import build_pipeline


TARGET_FILENAMES = (
    "418ca37c-5d00-4450-b556-bf0529464b88.jpg",
    "64046bed-36b9-4695-802d-dbd776a63782.jpg",
    "IMG_0903.JPG",
)

IMG_0903_OLD_HELPFUL_BBOX = (499, 1842, 1556, 2165)


@dataclass(slots=True)
class CropAttempt:
    index: int
    crop_policy: str
    bbox_xyxy: tuple[int, int, int, int]
    context_texts: tuple[str, ...]
    metadata: dict[str, object]
    crop_path: str | None
    truth_overlap: dict[str, Any]
    variants: list[dict[str, Any]]
    date_evidence: list[dict[str, Any]]


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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


def _draw_overlay(
    image: np.ndarray,
    boxes: list[dict[str, Any]],
    *,
    path: Path,
) -> str | None:
    canvas = image.copy()
    for item in boxes:
        bbox = _clean_bbox(item.get("bbox_xyxy"))
        if bbox is None:
            continue
        color = tuple(int(v) for v in item.get("color", (255, 220, 0)))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(2, int(round(max(image.shape[:2]) / 900))))
        label = str(item.get("label") or "")
        if label:
            cv2.putText(
                canvas,
                label[:48],
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return _write_png(path, canvas)


def _load_report_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"report has no rows: {path}")
    return {str(row.get("filename")): row for row in rows if isinstance(row, dict) and row.get("filename")}


def _load_truth_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError(f"truth manifest has no items: {path}")
    return {str(name): item for name, item in items.items() if isinstance(item, dict)}


def _settings_for_experiment() -> Any:
    settings = get_settings()
    return settings.model_copy(
        update={
            "mobile_product_cropper_rescue_enabled": False,
            "mobile_legacy_wide_group_rescue_enabled": False,
            "mobile_strict_evidence_acceptance_enabled": False,
            "mobile_rapidocr_role_constraint_enabled": False,
            "mobile_production_anchor_sibling_enabled": True,
            "mobile_paired_crop_evidence_enabled": True,
            "mobile_local_group_wide_crop_enabled": True,
        }
    )


def _trace_crop_variants(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    today: date,
) -> tuple[str | None, list[dict[str, Any]]]:
    crops = normalize_textline_crops(
        image,
        TextLineGeometry(bbox_xyxy=bbox, polygon_xy=None),
        TextLineCropConfig(padding_px=0),
    )
    if not crops:
        return None, []
    crop = crops[0]
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
    return None, rows


def _date_evidence_item(
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
        "recognition_variant": evidence.recognition_variant,
        "crop_policy": evidence.crop_policy,
        "context_texts": list(evidence.context_texts),
        "bbox_xyxy": bbox,
        "score": pipeline._score_date_evidence(evidence),
        "suppressed": pipeline._should_suppress_date_evidence(evidence),
        "metadata": evidence.metadata,
    }


def _build_primary_flow_trace(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    *,
    truth_bbox: tuple[int, int, int, int] | None,
    row_dir: Path,
) -> tuple[list[_YoloCandidate], list[_MobileGlobalCandidate], dict[str, Any]]:
    yolo_candidates, detection_reason = pipeline._detect_candidates(image)
    yolo_overlay = _draw_overlay(
        image,
        [
            {
                "bbox_xyxy": candidate.bbox_xyxy,
                "label": f"yolo {index} {candidate.confidence:.2f}",
                "color": (255, 220, 0),
            }
            for index, candidate in enumerate(yolo_candidates, start=1)
        ],
        path=row_dir / "02_yolo_proposals.png",
    )

    global_candidates: list[_MobileGlobalCandidate] = []
    rapidocr_boxes: list[dict[str, Any]] = []
    roi_reports: list[dict[str, Any]] = []
    if yolo_candidates:
        direct_ranked = pipeline._apply_geometry_shortlist(pipeline._build_ranked_candidates(image, yolo_candidates))
        global_candidates.extend(
            pipeline._global_candidates_from_ranked(
                image,
                direct_ranked,
                source="mobile_yolo",
                variant="yolo26s_obb",
                offset_x=0,
                offset_y=0,
            )
        )

    for roi_index, roi in enumerate(pipeline._collect_advanced_roi_candidates(image, yolo_candidates), start=1):
        for variant in pipeline._detector_variants_for_roi(roi.image):
            proposals, reason = pipeline._combined_primary_proposals_for_roi(
                roi.image,
                variant=variant,
                roi_confidence=roi.confidence,
            )
            proposal_rows: list[dict[str, Any]] = []
            for proposal in proposals:
                scan_bbox = _scan_bbox(proposal.bbox_xyxy, offset_x=roi.offset_x, offset_y=roi.offset_y)
                row = {
                    "bbox_xyxy": list(proposal.bbox_xyxy),
                    "scan_bbox_xyxy": scan_bbox,
                    "confidence": proposal.confidence,
                    "source": proposal.source,
                    "sources": list(proposal.sources),
                    "variant_name": proposal.variant_name,
                    "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
                }
                proposal_rows.append(row)
                if proposal.source == "rapidocr_ppocrv5":
                    rapidocr_boxes.append(row)
            ranked = pipeline._apply_geometry_shortlist(pipeline._build_ranked_candidates_from_proposals(roi.image, proposals))
            global_candidates.extend(
                pipeline._global_candidates_from_ranked(
                    roi.image,
                    ranked,
                    source=roi.source,
                    variant=variant.name,
                    offset_x=roi.offset_x,
                    offset_y=roi.offset_y,
                )
            )
            roi_reports.append(
                {
                    "roi_index": roi_index,
                    "source": roi.source,
                    "offset": [roi.offset_x, roi.offset_y],
                    "shape": list(roi.image.shape),
                    "variant": variant.name,
                    "rapidocr_reason": reason,
                    "proposals": proposal_rows,
                    "ranked_candidate_count": len(ranked),
                }
            )

    rapid_overlay = _draw_overlay(
        image,
        [
            {
                "bbox_xyxy": row["scan_bbox_xyxy"],
                "label": f"rapid {index}",
                "color": (0, 180, 255),
            }
            for index, row in enumerate(rapidocr_boxes, start=1)
            if row.get("scan_bbox_xyxy")
        ],
        path=row_dir / "03_rapidocr_boxes.png",
    )
    return yolo_candidates, global_candidates, {
        "detection_reason": detection_reason,
        "yolo_overlay": yolo_overlay,
        "rapidocr_overlay": rapid_overlay,
        "yolo": [
            {
                "bbox_xyxy": list(candidate.bbox_xyxy),
                "polygon_xy": candidate.polygon_xy,
                "confidence": candidate.confidence,
                "truth_overlap": _score_overlap(candidate.bbox_xyxy, truth_bbox),
            }
            for candidate in yolo_candidates
        ],
        "rapidocr_boxes": rapidocr_boxes,
        "roi_reports": roi_reports,
    }


def _instrument_generated_crops(
    pipeline: MobileExpiryPipeline,
    row_dir: Path,
    *,
    truth_bbox: tuple[int, int, int, int] | None,
    expected: dict[str, Any],
) -> tuple[list[CropAttempt], Any]:
    original = pipeline._date_evidence_from_bbox
    attempts: list[CropAttempt] = []

    def wrapped(
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        bbox: tuple[int, int, int, int],
        *,
        today: date,
        crop_policy: str,
        context_texts: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
        allowed_names: tuple[str, ...] = MOBILE_RECOGNITION_VARIANTS,
    ) -> list[_DateEvidence]:
        index = len(attempts) + 1
        clipped = pipeline._clip_bbox_to_image(bbox, image.shape)
        crop_path = None
        variants: list[dict[str, Any]] = []
        scan_bbox: tuple[int, int, int, int] | None = None
        if clipped is not None:
            scan_bbox = (
                clipped[0] + candidate.scan_offset_x,
                clipped[1] + candidate.scan_offset_y,
                clipped[2] + candidate.scan_offset_x,
                clipped[3] + candidate.scan_offset_y,
            )
            crop_image = image[clipped[1] : clipped[3], clipped[0] : clipped[2]]
            crop_path = _write_png(row_dir / "generated_crops" / f"{index:02d}_{crop_policy}.png", crop_image)
            _unused, variants = _trace_crop_variants(pipeline, image, clipped, today=today)
        evidence = original(
            image,
            candidate,
            bbox,
            today=today,
            crop_policy=crop_policy,
            context_texts=context_texts,
            metadata=metadata,
            allowed_names=allowed_names,
        )
        attempts.append(
            CropAttempt(
                index=index,
                crop_policy=crop_policy,
                bbox_xyxy=scan_bbox or bbox,
                context_texts=context_texts,
                metadata=dict(metadata or {}),
                crop_path=crop_path,
                truth_overlap=_score_overlap(scan_bbox or bbox, truth_bbox),
                variants=variants,
                date_evidence=[_date_evidence_item(pipeline, item, expected=expected) for item in evidence],
            )
        )
        return evidence

    pipeline._date_evidence_from_bbox = wrapped  # type: ignore[method-assign]
    return attempts, original


def _restore_generated_crop_instrumentation(pipeline: MobileExpiryPipeline, original: Any) -> None:
    pipeline._date_evidence_from_bbox = original  # type: ignore[method-assign]


def _evaluate_trace(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    global_candidates: list[_MobileGlobalCandidate],
    *,
    today: date,
    expected: dict[str, Any],
    truth_bbox: tuple[int, int, int, int] | None,
    row_dir: Path,
) -> dict[str, Any]:
    for candidate in global_candidates:
        candidate.ranked.selected_geometry = False
        candidate.ranked.selected_final = False
    probe_candidates = pipeline._select_global_probe_candidates(global_candidates)
    pipeline._probe_global_candidates(probe_candidates, today=today)
    final_candidates = pipeline._select_global_final_candidates(probe_candidates)

    anchor_windows: list[dict[str, Any]] = []
    paired_windows: list[dict[str, Any]] = []
    evaluated: list[_EvaluatedCandidate] = []
    attempts, original = _instrument_generated_crops(pipeline, row_dir, truth_bbox=truth_bbox, expected=expected)
    try:
        for global_candidate in final_candidates:
            item = pipeline._evaluate_candidate(global_candidate.image, global_candidate.ranked, today=today)
            evaluated.append(item)
            base_evidence = [
                evidence
                for evidence in item.date_evidence
                if evidence.crop_policy != "production_anchor_sibling" and not str(evidence.crop_policy or "").startswith("paired_crop")
            ]
            for anchor_bbox, anchor_context in pipeline._production_anchor_contexts(
                global_candidate.image,
                global_candidate.ranked,
                base_evidence,
                today=today,
            ):
                for relation, bbox in pipeline._production_anchor_sibling_bboxes(anchor_bbox, image_shape=global_candidate.image.shape):
                    scan_bbox = (
                        bbox[0] + global_candidate.ranked.scan_offset_x,
                        bbox[1] + global_candidate.ranked.scan_offset_y,
                        bbox[2] + global_candidate.ranked.scan_offset_x,
                        bbox[3] + global_candidate.ranked.scan_offset_y,
                    )
                    anchor_windows.append(
                        {
                            "candidate_id": global_candidate.ranked.candidate_id,
                            "anchor_bbox": _scan_bbox(anchor_bbox, offset_x=global_candidate.ranked.scan_offset_x, offset_y=global_candidate.ranked.scan_offset_y),
                            "anchor_context": list(anchor_context),
                            "relation": relation,
                            "bbox_xyxy": list(scan_bbox),
                            "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
                        }
                    )
            if pipeline._should_run_paired_crop_evidence(global_candidate.ranked, item, base_evidence):
                for policy, bbox in pipeline._paired_crop_bboxes(global_candidate.ranked):
                    clipped = pipeline._clip_bbox_to_image(bbox, global_candidate.image.shape)
                    if clipped is None:
                        continue
                    scan_bbox = (
                        clipped[0] + global_candidate.ranked.scan_offset_x,
                        clipped[1] + global_candidate.ranked.scan_offset_y,
                        clipped[2] + global_candidate.ranked.scan_offset_x,
                        clipped[3] + global_candidate.ranked.scan_offset_y,
                    )
                    paired_windows.append(
                        {
                            "candidate_id": global_candidate.ranked.candidate_id,
                            "crop_policy": policy,
                            "bbox_xyxy": list(scan_bbox),
                            "truth_overlap": _score_overlap(scan_bbox, truth_bbox),
                        }
                    )
    finally:
        _restore_generated_crop_instrumentation(pipeline, original)

    all_evidence = [
        evidence
        for item in evaluated
        for evidence in item.date_evidence
        if evidence.parsed.parsed_date is not None
    ]
    parseable, accepted_evidence = pipeline._accepted_parseable_items(evaluated)
    selected_evidence = pipeline._select_date_evidence(accepted_evidence) if accepted_evidence else None
    selected = pipeline._evaluated_from_date_evidence(selected_evidence, date_evidence=accepted_evidence) if selected_evidence else None

    selected_bbox = None
    selected_crop_path = None
    if selected is not None:
        selected_bbox = MobileExpiryPipeline._bbox_json_with_candidate_offset(
            selected.normalized_crop.bbox_xyxy if selected.normalized_crop is not None else selected.candidate.final_crop_bbox,
            selected.candidate,
        )
        if selected_bbox:
            bbox = _clean_bbox(selected_bbox)
            if bbox is not None:
                selected_crop_path = _write_png(row_dir / "selected_final_crop.png", image[bbox[1] : bbox[3], bbox[0] : bbox[2]])

    attempt_boxes = [
        {
            "bbox_xyxy": list(attempt.bbox_xyxy),
            "label": f"{attempt.index}:{attempt.crop_policy}",
            "color": (255, 128, 0) if attempt.crop_policy == "production_anchor_sibling" else (0, 255, 128),
        }
        for attempt in attempts
    ]
    anchor_overlay = _draw_overlay(
        image,
        [
            {"bbox_xyxy": item["anchor_bbox"], "label": f"anchor {index}", "color": (255, 0, 255)}
            for index, item in enumerate(anchor_windows, start=1)
            if item.get("anchor_bbox")
        ]
        + [
            {"bbox_xyxy": item["bbox_xyxy"], "label": f"{index}:{item['relation']}", "color": (255, 128, 0)}
            for index, item in enumerate(anchor_windows, start=1)
        ],
        path=row_dir / "04_role_anchors_and_sibling_windows.png",
    )
    generated_overlay = _draw_overlay(image, attempt_boxes, path=row_dir / "05_generated_sibling_paired_crops.png")

    return {
        "probe_candidate_count": len(probe_candidates),
        "final_candidate_count": len(final_candidates),
        "final_candidates": [
            {
                "candidate_id": candidate.ranked.candidate_id,
                "candidate_type": candidate.ranked.candidate_type,
                "source": candidate.source,
                "variant": candidate.variant,
                "scan_bbox": list(candidate.scan_bbox),
                "probe_text": candidate.ranked.probe_text,
                "probe_normalized_text": candidate.ranked.probe_normalized_text,
                "probe_confidence": candidate.ranked.probe_confidence,
                "selected_final": candidate.ranked.selected_final,
            }
            for candidate in final_candidates
        ],
        "role_anchor_windows": anchor_windows,
        "paired_windows": paired_windows,
        "role_anchor_overlay": anchor_overlay,
        "generated_overlay": generated_overlay,
        "generated_crop_attempts": [
            {
                "index": attempt.index,
                "crop_policy": attempt.crop_policy,
                "bbox_xyxy": list(attempt.bbox_xyxy),
                "context_texts": list(attempt.context_texts),
                "metadata": attempt.metadata,
                "crop_path": attempt.crop_path,
                "truth_overlap": attempt.truth_overlap,
                "variants": attempt.variants,
                "became_date_evidence": bool(attempt.date_evidence),
                "date_evidence": attempt.date_evidence,
            }
            for attempt in attempts
        ],
        "date_evidence": [_date_evidence_item(pipeline, evidence, expected=expected) for evidence in all_evidence],
        "accepted_date_evidence": [_date_evidence_item(pipeline, evidence, expected=expected) for evidence in accepted_evidence],
        "selected": _date_evidence_item(pipeline, selected_evidence, expected=expected) if selected_evidence is not None else None,
        "selected_crop_path": selected_crop_path,
        "selected_bbox": selected_bbox,
        "parseable_count": len(parseable),
    }


def _classify_disappearance(row: dict[str, Any], expected: dict[str, Any]) -> str:
    attempts = row.get("generated_crop_attempts") if isinstance(row.get("generated_crop_attempts"), list) else []
    accepted = row.get("accepted_date_evidence") if isinstance(row.get("accepted_date_evidence"), list) else []
    selected = row.get("selected") if isinstance(row.get("selected"), dict) else None
    expected_generated = [
        evidence
        for attempt in attempts
        for evidence in attempt.get("date_evidence", [])
        if evidence.get("matches_expected")
    ]
    expected_accepted = [evidence for evidence in accepted if evidence.get("matches_expected")]
    if selected and selected.get("matches_expected"):
        return "selected_expected"
    if expected_accepted:
        return "evidence_created_but_selector_rejected"
    if expected_generated:
        return "evidence_created_but_filtered"
    if not attempts:
        return "anchor_not_detected"
    best_attempt = max(
        attempts,
        key=lambda item: (
            float(item.get("truth_overlap", {}).get("truth_coverage") or 0.0),
            float(item.get("truth_overlap", {}).get("iou") or 0.0),
        ),
    )
    coverage = float(best_attempt.get("truth_overlap", {}).get("truth_coverage") or 0.0)
    if coverage < 0.20:
        return "search_window_missed_truth"
    if coverage < 0.75:
        return "crop_generated_but_wrong_geometry"
    any_parsed = any(
        variant.get("best_parse", {}).get("parsed_date")
        for attempt in attempts
        for variant in attempt.get("variants", [])
    )
    if any_parsed:
        return "parser_failed"
    return "svtr_failed_on_generated_crop"


def _save_reference_crop(
    pipeline: MobileExpiryPipeline,
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    today: date,
    row_dir: Path,
    name: str,
    truth_bbox: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    crop_path = _write_png(row_dir / f"{name}.png", crop)
    _unused, variants = _trace_crop_variants(pipeline, image, bbox, today=today)
    return {
        "bbox_xyxy": list(bbox),
        "crop_path": crop_path,
        "truth_overlap": _score_overlap(bbox, truth_bbox),
        "variants": variants,
    }


def _audit_row(
    pipeline: MobileExpiryPipeline,
    *,
    image_path: Path,
    report_row: dict[str, Any],
    truth_item: dict[str, Any],
    output_root: Path,
    today: date,
) -> dict[str, Any]:
    row_dir = output_root / image_path.stem
    row_dir.mkdir(parents=True, exist_ok=True)
    image = pipeline.decode_image(image_path.read_bytes())
    if image is None:
        raise ValueError(f"failed to decode {image_path}")

    expected = _expected_from_report_row(report_row)
    truth = _truth_for_row(image_path.name, {image_path.name: truth_item})
    truth_bbox = _clean_bbox(truth.get("bbox_xyxy"))
    original_overlay = _draw_overlay(
        image,
        [{"bbox_xyxy": truth.get("bbox_xyxy"), "label": f"expected {expected.get('label')}", "color": (0, 255, 0)}],
        path=row_dir / "01_original_expected_bbox.png",
    )
    yolo_candidates, global_candidates, proposal_trace = _build_primary_flow_trace(
        pipeline,
        image,
        truth_bbox=truth_bbox,
        row_dir=row_dir,
    )
    eval_trace = _evaluate_trace(
        pipeline,
        image,
        global_candidates,
        today=today,
        expected=expected,
        truth_bbox=truth_bbox,
        row_dir=row_dir,
    )

    img_0903_comparison = None
    if image_path.name == "IMG_0903.JPG":
        selected_bbox = _clean_bbox(eval_trace.get("selected_bbox"))
        img_0903_comparison = {
            "old_helpful_crop": _save_reference_crop(
                pipeline,
                image,
                IMG_0903_OLD_HELPFUL_BBOX,
                today=today,
                row_dir=row_dir,
                name="IMG_0903_old_helpful_bbox_crop",
                truth_bbox=truth_bbox,
            ),
            "current_selected_crop": _save_reference_crop(
                pipeline,
                image,
                selected_bbox,
                today=today,
                row_dir=row_dir,
                name="IMG_0903_current_selected_bbox_crop",
                truth_bbox=truth_bbox,
            )
            if selected_bbox is not None
            else None,
        }

    row = {
        "filename": image_path.name,
        "expected": expected,
        "baseline": {
            "detected_expiry_date": report_row.get("detected_expiry_date"),
            "status": report_row.get("response_status"),
            "final_crop_policy": report_row.get("final_crop_policy"),
            "reason": report_row.get("reason"),
        },
        "truth": truth,
        "original_overlay": original_overlay,
        **proposal_trace,
        **eval_trace,
        "img_0903_comparison": img_0903_comparison,
    }
    row["failure_stage"] = _classify_disappearance(row, expected)
    return row


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
            *["| " + " | ".join(row) + " |" for row in body],
        ]
    )


def _write_summary(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    lines = ["# Targeted Sibling/Paired Crop Forensic", ""]
    lines.extend(
        [
            "Forensic-only run. Pipeline defaults were not changed; the production-anchor sibling and paired-crop hooks were enabled only inside this script.",
            "",
            "## Overall",
            "",
            _md_table(
                [
                    ["filename", "expected", "baseline detected", "selected", "failure stage", "generated crops", "new evidence"],
                    *[
                        [
                            row["filename"],
                            str(row["expected"].get("label")),
                            str(row["baseline"].get("detected_expiry_date")),
                            str((row.get("selected") or {}).get("parsed", {}).get("parsed_date")),
                            str(row.get("failure_stage")),
                            str(len(row.get("generated_crop_attempts") or [])),
                            str(sum(1 for item in row.get("generated_crop_attempts") or [] if item.get("became_date_evidence"))),
                        ]
                        for row in rows
                    ],
                ]
            ),
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"## {row['filename']}",
                "",
                f"- expected: `{row['expected'].get('label')}`",
                f"- baseline detected: `{row['baseline'].get('detected_expiry_date')}`",
                f"- selected in forensic run: `{(row.get('selected') or {}).get('parsed', {}).get('parsed_date')}`",
                f"- failure stage: `{row.get('failure_stage')}`",
                f"- original/truth overlay: `{row.get('original_overlay')}`",
                f"- YOLO overlay: `{row.get('yolo_overlay')}`",
                f"- RapidOCR overlay: `{row.get('rapidocr_overlay')}`",
                f"- role/sibling overlay: `{row.get('role_anchor_overlay')}`",
                f"- generated crop overlay: `{row.get('generated_overlay')}`",
                "",
                "### Stage Counts",
                "",
                f"- YOLO proposals: `{len(row.get('yolo') or [])}`",
                f"- RapidOCR boxes: `{len(row.get('rapidocr_boxes') or [])}`",
                f"- final candidates: `{row.get('final_candidate_count')}`",
                f"- role-anchor sibling windows: `{len(row.get('role_anchor_windows') or [])}`",
                f"- paired windows: `{len(row.get('paired_windows') or [])}`",
                f"- generated sibling/paired crops: `{len(row.get('generated_crop_attempts') or [])}`",
                f"- accepted DateEvidence: `{len(row.get('accepted_date_evidence') or [])}`",
                "",
                "### Generated Crops",
                "",
            ]
        )
        crop_rows = [["#", "policy", "bbox", "truth overlap", "DateEvidence", "best parsed outputs", "crop"]]
        for attempt in row.get("generated_crop_attempts") or []:
            best_parsed = []
            for variant in attempt.get("variants") or []:
                parsed = variant.get("best_parse", {}).get("parsed_date")
                raw = variant.get("recognition", {}).get("raw_text")
                if parsed or raw:
                    best_parsed.append(f"{variant.get('variant')}:{raw}->{parsed}")
            crop_rows.append(
                [
                    str(attempt.get("index")),
                    str(attempt.get("crop_policy")),
                    str(attempt.get("bbox_xyxy")),
                    str(attempt.get("truth_overlap", {}).get("verdict")),
                    str(attempt.get("became_date_evidence")),
                    "<br>".join(best_parsed[:4]) if best_parsed else "none",
                    str(attempt.get("crop_path")),
                ]
            )
        lines.append(_md_table(crop_rows) if len(crop_rows) > 1 else "No sibling/paired crops were generated.")
        lines.append("")
        if row.get("img_0903_comparison"):
            comparison = row["img_0903_comparison"]
            lines.extend(
                [
                    "### IMG_0903 Crop Comparison",
                    "",
                    f"- old helpful bbox crop: `{comparison['old_helpful_crop'].get('crop_path')}`",
                    f"- current selected crop: `{(comparison.get('current_selected_crop') or {}).get('crop_path')}`",
                    "",
                ]
            )
    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic debug for production-anchor sibling and paired-crop experiment")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--baseline-report", default="artifacts/onnx_parity/test64_mobile_direct_20260519T120149Z/mobile_test64_direct_report.json")
    parser.add_argument("--truth-manifest", default="artifacts/forensics/crop_truth_annotations/test64_truth_bboxes.json")
    parser.add_argument("--output-dir", default="artifacts/onnx_parity")
    parser.add_argument("--today", default=None)
    args = parser.parse_args()

    today = date.fromisoformat(args.today) if args.today else date.today()
    output_root = Path(args.output_dir).resolve() / f"targeted_sibling_paired_forensic_{_utc_stamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    report_rows = _load_report_rows(Path(args.baseline_report))
    truth_items = _load_truth_items(Path(args.truth_manifest))
    pipeline = build_pipeline(_settings_for_experiment())

    rows: list[dict[str, Any]] = []
    for index, filename in enumerate(TARGET_FILENAMES, start=1):
        print(f"[{index}/{len(TARGET_FILENAMES)}] forensic {filename}", flush=True)
        image_path = Path(args.images_dir) / filename
        rows.append(
            _audit_row(
                pipeline,
                image_path=image_path,
                report_row=report_rows[filename],
                truth_item=truth_items.get(filename, {}),
                output_root=output_root,
                today=today,
            )
        )

    json_path = output_root / "targeted_sibling_paired_forensic.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, default=_json_default), encoding="utf-8")
    summary_path = _write_summary(output_root, rows)
    print("Report root:", output_root)
    print(summary_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
