from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ai.crop_normalization import TextLineCropConfig, TextLineGeometry, normalize_textline_crops
from app.ai.types import OCRResultData
from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH, CROP_TRUTH_AUDIT_FILENAMES, PROJECT_ROOT
from app.infra.settings import get_settings
from app.workers.scan_jobs import build_pipeline


class CropTruthManifestError(ValueError):
    pass


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Type not JSON serializable: {type(value)}")


def load_truth_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CropTruthManifestError(f"truth manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CropTruthManifestError(f"truth manifest is invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CropTruthManifestError("truth manifest must be a JSON object")
    return payload


def validate_truth_manifest(payload: dict[str, Any]) -> list[str]:
    items = payload.get("items")
    if not isinstance(items, dict):
        raise CropTruthManifestError("truth manifest must contain an items object")

    missing: list[str] = []
    for filename in CROP_TRUTH_AUDIT_FILENAMES:
        raw = items.get(filename)
        if not isinstance(raw, dict):
            missing.append(filename)
            continue
        bbox = raw.get("true_bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            missing.append(filename)
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            missing.append(filename)
            continue
        if x2 <= x1 or y2 <= y1:
            missing.append(filename)
    return missing


def _resolve_test64_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "test64",
        Path.cwd() / "test64",
        Path("/app/test64"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise SystemExit(f"test64 directory not found. Tried: {', '.join(str(path) for path in candidates)}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_png(path: Path, image: np.ndarray) -> str:
    _ensure_dir(path.parent)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode png: {path}")
    path.write_bytes(encoded.tobytes())
    return str(path)


def _clamp_bbox(bbox: list[Any] | tuple[Any, Any, Any, Any], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2


def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]


def _clean_polygon(raw: Any, shape: tuple[int, ...]) -> list[tuple[float, float]] | None:
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    h, w = shape[:2]
    points: list[tuple[float, float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        points.append((max(0.0, min(float(w), x)), max(0.0, min(float(h), y))))
    return points


def _crop_polygon(image: np.ndarray, polygon: list[tuple[float, float]]) -> np.ndarray:
    widths = [
        np.linalg.norm(np.array(polygon[1]) - np.array(polygon[0])),
        np.linalg.norm(np.array(polygon[2]) - np.array(polygon[3])),
    ]
    heights = [
        np.linalg.norm(np.array(polygon[3]) - np.array(polygon[0])),
        np.linalg.norm(np.array(polygon[2]) - np.array(polygon[1])),
    ]
    width = max(1, int(round(max(widths))))
    height = max(1, int(round(max(heights))))
    src = np.array(polygon, dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> dict[str, float]:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = max(1, area_a + area_b - inter)
    return {
        "iou": inter / float(union),
        "truth_coverage": inter / float(area_b),
        "candidate_coverage": inter / float(area_a),
        "containment": inter / float(max(1, min(area_a, area_b))),
    }


def _crop_purity_metrics(
    crop_bbox: tuple[int, int, int, int],
    truth_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    overlap = _bbox_overlap(crop_bbox, truth_bbox)
    crop_area = max(1, (crop_bbox[2] - crop_bbox[0]) * (crop_bbox[3] - crop_bbox[1]))
    truth_area = max(1, (truth_bbox[2] - truth_bbox[0]) * (truth_bbox[3] - truth_bbox[1]))
    ratio = crop_area / float(truth_area)
    contains_large_extra = bool(
        overlap["truth_coverage"] >= 0.45
        and (ratio >= 3.0 or overlap["candidate_coverage"] <= 0.35)
    )
    return {
        **overlap,
        "crop_area": crop_area,
        "truth_area": truth_area,
        "crop_area_to_truth_area_ratio": ratio,
        "crop_contains_large_extra_text_area": contains_large_extra,
    }


def _is_overlap_hit(box: tuple[int, int, int, int], truth: tuple[int, int, int, int]) -> bool:
    overlap = _bbox_overlap(box, truth)
    return bool(overlap["iou"] >= 0.20 or overlap["truth_coverage"] >= 0.60 or overlap["containment"] >= 0.70)


def _looks_date_like_text(value: str | None) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    digits = sum(ch.isdigit() for ch in text)
    separators = sum(ch in "/.-:" for ch in text)
    return digits >= 4 and (separators >= 1 or digits >= 6)


def _scan_box(box: Any, *, offset_x: int, offset_y: int) -> tuple[int, int, int, int]:
    return (box.x1 + offset_x, box.y1 + offset_y, box.x2 + offset_x, box.y2 + offset_y)


def _draw_boxes(
    image: np.ndarray,
    *,
    truth: tuple[int, int, int, int],
    ppocr_boxes: list[tuple[int, int, int, int]],
    craft_boxes: list[tuple[int, int, int, int]],
    merged_boxes: list[tuple[int, int, int, int]],
    selected_boxes: list[tuple[int, int, int, int]],
    truth_polygon: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    for box in ppocr_boxes:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (255, 150, 40), 2)
    for box in craft_boxes:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (0, 180, 255), 2)
    for box in merged_boxes:
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (190, 80, 255), 1)
    for idx, box in enumerate(selected_boxes, start=1):
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 4)
        cv2.putText(canvas, f"S{idx}", (box[0], max(18, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    if truth_polygon:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in truth_polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=5)
    else:
        cv2.rectangle(canvas, (truth[0], truth[1]), (truth[2], truth[3]), (0, 255, 0), 5)
    cv2.putText(canvas, "TRUE", (truth[0], max(24, truth[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def _fit_panel(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if image.size == 0:
        cv2.putText(canvas, "empty", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        return canvas
    src = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / float(max(1, src.shape[1])), height / float(max(1, src.shape[0])))
    resized = cv2.resize(src, (max(1, int(src.shape[1] * scale)), max(1, int(src.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _put_label(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    labeled = panel.copy()
    y = 26
    for line in lines[:5]:
        cv2.putText(labeled, line[:70], (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
        y += 24
    return labeled


def _write_side_by_side_panel(
    path: Path,
    *,
    original_overlay: np.ndarray,
    manual_crop: np.ndarray,
    manual_parseq: dict[str, Any],
    selected_reports: list[dict[str, Any]],
) -> str:
    panel_w = 420
    panel_h = 320
    original_panel = _put_label(
        _fit_panel(original_overlay, width=panel_w, height=panel_h),
        ["original + truth/detectors/selected"],
    )
    manual_panel = _put_label(
        _fit_panel(manual_crop, width=panel_w, height=panel_h),
        [
            "manual true crop",
            f"text: {manual_parseq.get('raw_text')}",
            f"date: {manual_parseq.get('parser_parsed_date')}",
        ],
    )
    selected = _primary_selected_report(selected_reports) or {}
    selected_crop = cv2.imread(str(selected.get("crop_path") or ""), cv2.IMREAD_COLOR) if selected else None
    if selected_crop is None:
        selected_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    selected_panel = _put_label(
        _fit_panel(selected_crop, width=panel_w, height=panel_h),
        [
            "pipeline selected crop",
            f"text: {(selected.get('direct_parseq') or {}).get('raw_text')}",
            f"date: {(selected.get('direct_parseq') or {}).get('parser_parsed_date')}",
            f"bbox: {selected.get('final_crop_bbox_xyxy') or selected.get('scan_bbox_xyxy')}",
            f"ratio: {round(float((selected.get('crop_purity') or {}).get('crop_area_to_truth_area_ratio') or 0.0), 2)}",
        ],
    )
    return _write_png(path, np.concatenate([original_panel, selected_panel, manual_panel], axis=1))


def _primary_selected_report(selected_reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not selected_reports:
        return None
    return max(
        selected_reports,
        key=lambda report: (
            1 if (report.get("direct_parseq") or {}).get("parser_parsed_date") else 0,
            float((report.get("crop_purity") or {}).get("truth_coverage") or 0.0),
            float((report.get("direct_parseq") or {}).get("confidence") or 0.0),
            -float((report.get("crop_purity") or {}).get("crop_area_to_truth_area_ratio") or 999.0),
        ),
    )


def _parseq_and_parse(pipeline: Any, crop: np.ndarray, *, today: date) -> dict[str, Any]:
    if crop.size == 0:
        return {
            "raw_text": "",
            "confidence": None,
            "reason": "empty crop",
            "parser_parsed_date": None,
            "parser_confidence": 0.0,
            "parser_reason": "empty crop",
        }
    parseq = pipeline.ocr_router.recognize_parseq(crop)
    ocr = OCRResultData(
        raw_text=parseq.raw_text,
        normalized_text=parseq.normalized_text,
        confidence=parseq.confidence,
        engine_name=pipeline.ocr_router.active_engine_name,
        runtime_device=pipeline.ocr_router.parseq_runtime_device,
        reason=parseq.reason,
    )
    parse_inputs = pipeline._build_parse_inputs(ocr)
    parsed = pipeline._best_parse_for_inputs(parse_inputs, today=today)
    return {
        "raw_text": parseq.raw_text,
        "normalized_text": parseq.normalized_text,
        "confidence": parseq.confidence,
        "reason": parseq.reason,
        "parse_inputs": parse_inputs,
        "parser_parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date else None,
        "parser_confidence": parsed.confidence,
        "parser_reason": parsed.reason,
    }


def _best_parseq_for_normalized_crops(
    pipeline: Any,
    crops: list[Any],
    *,
    today: date,
) -> tuple[dict[str, Any], np.ndarray, Any | None]:
    if not crops:
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        return _parseq_and_parse(pipeline, empty, today=today), empty, None
    best: tuple[tuple[float, float], dict[str, Any], np.ndarray, Any] | None = None
    for crop in crops:
        report = _parseq_and_parse(pipeline, crop.image, today=today)
        score = (
            float(report.get("parser_confidence") or 0.0) if report.get("parser_parsed_date") else 0.0,
            float(report.get("confidence") or 0.0),
        )
        item = (score, report, crop.image, crop)
        if best is None or score > best[0]:
            best = item
        if score[0] >= 0.9 and score[1] >= 0.85:
            break
    assert best is not None
    return best[1], best[2], best[3]


def _best_recognition_variant_for_candidate(pipeline: Any, crop: np.ndarray, variant_name: str | None) -> tuple[str, np.ndarray]:
    if crop.size == 0:
        return variant_name or "empty", crop
    variants = pipeline._recognition_variants_for_crop(crop, allowed_names=(variant_name,) if variant_name else None)
    if not variants:
        return "original_padded", crop
    selected = variants[0]
    return selected.name, selected.image


def _selected_candidates_from_debug(debug_payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in debug_payload.get("variants", []):
        if not isinstance(variant, dict):
            continue
        variant_key = str(variant.get("variant_key") or "")
        for candidate in variant.get("candidates", []):
            if not isinstance(candidate, dict) or not candidate.get("selected_final"):
                continue
            item = dict(candidate)
            item["variant_key"] = variant_key
            item["source"] = variant.get("source")
            item["variant"] = variant.get("variant")
            out.append(item)
    return out


def _detector_truth_audit(pipeline: Any, image: np.ndarray, truth_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    detection = pipeline.detector.detect(image)
    roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
    ppocr_scan_boxes: list[tuple[int, int, int, int]] = []
    craft_scan_boxes: list[tuple[int, int, int, int]] = []
    merged_scan_boxes: list[tuple[int, int, int, int]] = []
    detector_variants: list[dict[str, Any]] = []

    for roi_candidate in roi_candidates:
        variants = pipeline._variants_for_roi(roi_candidate.image)
        for variant in variants:
            if variant.name != pipeline.craft_rescue_variant_name:
                continue
            ppocr_raw, ppocr_reason = pipeline._detect_text_boxes(variant.image, variant_name=variant.name)
            ppocr_mapped = pipeline._map_text_boxes_to_original(
                boxes=ppocr_raw,
                variant=variant,
                original_shape=roi_candidate.image.shape,
            )
            craft_raw, craft_reason, craft_ms, craft_timed_out = pipeline._run_craft_rescue(
                roi_candidate.image,
                variant_name=pipeline.craft_rescue_variant_name,
            )
            craft_mapped = pipeline._cap_craft_boxes(craft_raw)
            merged = pipeline._merge_text_boxes(ppocr_mapped, craft_mapped)

            ppocr_boxes = [_scan_box(box, offset_x=roi_candidate.offset_x, offset_y=roi_candidate.offset_y) for box in ppocr_mapped]
            craft_boxes = [_scan_box(box, offset_x=roi_candidate.offset_x, offset_y=roi_candidate.offset_y) for box in craft_mapped]
            merged_boxes = [_scan_box(box, offset_x=roi_candidate.offset_x, offset_y=roi_candidate.offset_y) for box in merged]
            ppocr_scan_boxes.extend(ppocr_boxes)
            craft_scan_boxes.extend(craft_boxes)
            merged_scan_boxes.extend(merged_boxes)
            detector_variants.append(
                {
                    "roi_source": roi_candidate.source,
                    "variant": variant.name,
                    "ppocr_count": len(ppocr_boxes),
                    "craft_count": len(craft_boxes),
                    "merged_count": len(merged_boxes),
                    "ppocr_reason": ppocr_reason,
                    "craft_reason": craft_reason,
                    "craft_runtime_ms": craft_ms,
                    "craft_timed_out": craft_timed_out,
                    "ppocr_truth_hits": [box for box in ppocr_boxes if _is_overlap_hit(box, truth_bbox)],
                    "craft_truth_hits": [box for box in craft_boxes if _is_overlap_hit(box, truth_bbox)],
                    "merged_truth_hits": [box for box in merged_boxes if _is_overlap_hit(box, truth_bbox)],
                }
            )

    return {
        "yolo_detected": detection.detected,
        "yolo_reason": detection.reason,
        "roi_count": len(roi_candidates),
        "variants": detector_variants,
        "ppocr_boxes": ppocr_scan_boxes,
        "craft_boxes": craft_scan_boxes,
        "merged_boxes": merged_scan_boxes,
        "ppocr_has_true_overlap": any(_is_overlap_hit(box, truth_bbox) for box in ppocr_scan_boxes),
        "craft_has_true_overlap": any(_is_overlap_hit(box, truth_bbox) for box in craft_scan_boxes),
        "merged_has_true_overlap": any(_is_overlap_hit(box, truth_bbox) for box in merged_scan_boxes),
    }


def _classify_failure(
    *,
    image_unreadable: bool,
    detector: dict[str, Any],
    selected_has_true_overlap: bool,
    selected_parseq: dict[str, Any] | None,
    selected_purity: dict[str, Any] | None,
    true_parseq: dict[str, Any],
) -> str:
    if image_unreadable:
        return "image_unreadable"
    if not detector["ppocr_has_true_overlap"] and not detector["craft_has_true_overlap"]:
        return "detector_missed_true_region"
    if (detector["ppocr_has_true_overlap"] or detector["craft_has_true_overlap"]) and not detector["merged_has_true_overlap"]:
        return "craft_mapping_or_merge_bug"
    if detector["merged_has_true_overlap"] and not selected_has_true_overlap:
        return "ranker_dropped_true_region"
    if selected_has_true_overlap and selected_purity and selected_purity.get("crop_contains_large_extra_text_area"):
        return "bad_crop_extraction"
    if not true_parseq.get("raw_text"):
        return "parseq_fails_on_true_crop"
    if true_parseq.get("raw_text") and not true_parseq.get("parser_parsed_date"):
        if not _looks_date_like_text(str(true_parseq.get("raw_text") or "")):
            return "parseq_fails_on_true_crop"
        return "parser_rejects_valid_parseq_text"
    if selected_parseq and true_parseq.get("parser_parsed_date") and not selected_parseq.get("parser_parsed_date"):
        return "bad_crop_extraction"
    return "n/a_success_or_needs_manual_review"


def _audit_one(
    *,
    pipeline: Any,
    filename: str,
    item: dict[str, Any],
    test64_dir: Path,
    report_root: Path,
    today: date,
) -> dict[str, Any]:
    image_path = test64_dir / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    scan_dir = report_root / "images" / Path(filename).stem
    _ensure_dir(scan_dir)

    if image is None:
        return {"filename": filename, "classification": "image_unreadable", "image_path": str(image_path)}

    truth_bbox = _clamp_bbox(item["true_bbox_xyxy"], image.shape)
    truth_polygon = _clean_polygon(item.get("true_polygon_xy"), image.shape)
    true_crop = _crop_polygon(image, truth_polygon) if truth_polygon else _crop(image, truth_bbox)
    true_crop_path = _write_png(scan_dir / "manual_true_crop.png", true_crop)
    true_normalized_crops = normalize_textline_crops(
        image,
        TextLineGeometry(
            bbox_xyxy=truth_bbox,
            polygon_xy=tuple((float(x), float(y)) for x, y in truth_polygon) if truth_polygon else None,
        ),
        TextLineCropConfig(padding_px=0, vertical_aspect_threshold=1.5),
    )
    true_parseq, true_normalized_crop, true_selected_transform = _best_parseq_for_normalized_crops(
        pipeline,
        true_normalized_crops,
        today=today,
    )
    true_normalized_crop_path = _write_png(scan_dir / "manual_true_crop_normalized.png", true_normalized_crop)

    detector_audit = _detector_truth_audit(pipeline, image, truth_bbox)

    image_bytes = image_path.read_bytes()
    output = pipeline.run(image_bytes, today=today)
    debug_payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8")) if output.debug_candidates_json_bytes else {}
    selected_candidates = _selected_candidates_from_debug(debug_payload)
    selected_boxes: list[tuple[int, int, int, int]] = []
    selected_reports: list[dict[str, Any]] = []

    for idx, candidate in enumerate(selected_candidates, start=1):
        raw_bbox = candidate.get("scan_bbox_xyxy") or candidate.get("bbox_xyxy")
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            continue
        scan_bbox = _clamp_bbox(raw_bbox, image.shape)
        final_roi_bbox = candidate.get("final_crop_bbox_xyxy")
        final_scan_bbox = scan_bbox
        if isinstance(final_roi_bbox, list) and len(final_roi_bbox) == 4 and isinstance(candidate.get("bbox_xyxy"), list):
            candidate_roi_bbox = candidate["bbox_xyxy"]
            offset_x = scan_bbox[0] - int(round(float(candidate_roi_bbox[0])))
            offset_y = scan_bbox[1] - int(round(float(candidate_roi_bbox[1])))
            final_scan_bbox = _clamp_bbox(
                [
                    float(final_roi_bbox[0]) + offset_x,
                    float(final_roi_bbox[1]) + offset_y,
                    float(final_roi_bbox[2]) + offset_x,
                    float(final_roi_bbox[3]) + offset_y,
                ],
                image.shape,
            )
        selected_boxes.append(final_scan_bbox)
        raw_crop = _crop(image, final_scan_bbox)
        recognition_variant_name, parseq_crop = _best_recognition_variant_for_candidate(
            pipeline,
            raw_crop,
            candidate.get("recognition_variant") if isinstance(candidate.get("recognition_variant"), str) else None,
        )
        original_color_crop_path = _write_png(scan_dir / f"pipeline_selected_crop_{idx}_original_color.png", raw_crop)
        selected_crop_path = _write_png(scan_dir / f"pipeline_selected_crop_{idx}_{recognition_variant_name}.png", parseq_crop)
        direct_parseq = _parseq_and_parse(pipeline, parseq_crop, today=today)
        original_color_parseq = _parseq_and_parse(pipeline, raw_crop, today=today)
        purity = _crop_purity_metrics(final_scan_bbox, truth_bbox)
        selected_reports.append(
            {
                "index": idx,
                "variant_key": candidate.get("variant_key"),
                "candidate_id": candidate.get("candidate_id"),
                "scan_bbox_xyxy": list(scan_bbox),
                "final_crop_bbox_xyxy": list(final_scan_bbox),
                "truth_overlap": _bbox_overlap(final_scan_bbox, truth_bbox),
                "crop_purity": purity,
                "selected_recognition_variant": recognition_variant_name,
                "crop_path": selected_crop_path,
                "original_color_crop_path": original_color_crop_path,
                "original_color_parseq": original_color_parseq,
                "pipeline_debug_parseq_text": candidate.get("parseq_text"),
                "pipeline_debug_parser_date": candidate.get("parsed_date"),
                "pipeline_debug_final_crop_policy": candidate.get("final_crop_policy"),
                "pipeline_debug_final_crop_padding_px": candidate.get("final_crop_padding_px"),
                "pipeline_debug_original_color_fallback_used": candidate.get("original_color_fallback_used"),
                "direct_parseq": direct_parseq,
            }
        )

    selected_has_true_overlap = any(_is_overlap_hit(box, truth_bbox) for box in selected_boxes)
    primary_selected = _primary_selected_report(selected_reports)
    selected_parseq = primary_selected["direct_parseq"] if primary_selected else None
    selected_purity = primary_selected["crop_purity"] if primary_selected else None

    overlay = _draw_boxes(
        image,
        truth=truth_bbox,
        truth_polygon=truth_polygon,
        ppocr_boxes=detector_audit["ppocr_boxes"],
        craft_boxes=detector_audit["craft_boxes"],
        merged_boxes=detector_audit["merged_boxes"],
        selected_boxes=selected_boxes,
    )
    overlay_path = _write_png(scan_dir / "overlay_truth_detectors_selected.png", overlay)
    side_by_side_path = _write_side_by_side_panel(
        scan_dir / "crop_truth_side_by_side.png",
        original_overlay=overlay,
        manual_crop=true_crop,
        manual_parseq=true_parseq,
        selected_reports=selected_reports,
    )

    classification = _classify_failure(
        image_unreadable=False,
        detector=detector_audit,
        selected_has_true_overlap=selected_has_true_overlap,
        selected_parseq=selected_parseq,
        selected_purity=selected_purity,
        true_parseq=true_parseq,
    )

    return {
        "filename": filename,
        "image_path": str(image_path),
        "expected": {
            "day": item.get("expected_day"),
            "month": item.get("expected_month"),
            "year": item.get("expected_year"),
        },
        "true_bbox_xyxy": list(truth_bbox),
        "true_polygon_xy": [list(point) for point in truth_polygon] if truth_polygon else None,
        "manual_true_crop_source": "true_polygon_xy" if truth_polygon else "true_bbox_xyxy",
        "manual_true_crop_path": true_crop_path,
        "manual_true_crop_normalized_path": true_normalized_crop_path,
        "manual_true_crop_orientation_candidates_tried": (
            true_normalized_crops[0].orientation_candidates_tried if true_normalized_crops else ["original"]
        ),
        "manual_true_crop_selected_orientation": (
            true_selected_transform.selected_orientation if true_selected_transform is not None else "original"
        ),
        "overlay_path": overlay_path,
        "side_by_side_path": side_by_side_path,
        "classification": classification,
        "pipeline": {
            "final_status": output.final_status,
            "reason": output.reason,
            "parsed_date": output.parsed_date.isoformat() if output.parsed_date else None,
            "raw_text": output.raw_text,
            "debug_summary": output.debug_summary,
        },
        "detector_audit": {
            key: value
            for key, value in detector_audit.items()
            if key not in {"ppocr_boxes", "craft_boxes", "merged_boxes"}
        },
        "detector_counts": {
            "ppocr_boxes": len(detector_audit["ppocr_boxes"]),
            "craft_boxes": len(detector_audit["craft_boxes"]),
            "merged_boxes": len(detector_audit["merged_boxes"]),
        },
        "ranker_selected_true_candidate": selected_has_true_overlap,
        "pipeline_selected_crops": selected_reports,
        "manual_true_crop_parseq": true_parseq,
    }


def _write_summary(report_root: Path, report: dict[str, Any]) -> Path:
    summary = report["summary"]
    lines = [
        "# Crop Truth Audit Summary",
        "",
        f"- total_images: `{summary['total_images']}`",
        f"- generated_at: `{report['generated_at']}`",
        f"- selected_true_candidate_count: `{summary.get('selected_true_candidate_count')}`",
        f"- correct_parsed_dates: `{summary.get('correct_parsed_dates')}`",
        "",
        "## Classifications",
    ]
    for key, count in sorted(summary["classifications"].items()):
        lines.append(f"- {key}: `{count}`")
    lines.extend(["", "## Per Image"])
    for item in report["items"]:
        lines.extend(
            [
                f"### {item['filename']}",
                f"- classification: `{item['classification']}`",
                f"- pipeline_status: `{item['pipeline']['final_status']}`",
                f"- pipeline_text: `{item['pipeline'].get('raw_text')}`",
                f"- pipeline_date: `{item['pipeline'].get('parsed_date')}`",
                f"- ppocr_true_overlap: `{item['detector_audit']['ppocr_has_true_overlap']}`",
                f"- craft_true_overlap: `{item['detector_audit']['craft_has_true_overlap']}`",
                f"- merged_true_overlap: `{item['detector_audit']['merged_has_true_overlap']}`",
                f"- selected_true_candidate: `{item['ranker_selected_true_candidate']}`",
                f"- true_crop_parseq: `{item['manual_true_crop_parseq'].get('raw_text')}`",
                f"- true_crop_parser_date: `{item['manual_true_crop_parseq'].get('parser_parsed_date')}`",
                f"- overlay: `{item['overlay_path']}`",
                f"- side_by_side: `{item.get('side_by_side_path')}`",
                "",
            ]
        )
        selected = item.get("pipeline_selected_crops") or []
        if selected:
            purity = selected[0].get("crop_purity") or {}
            lines.extend(
                [
                    f"- selected_crop_ratio: `{round(float(purity.get('crop_area_to_truth_area_ratio') or 0.0), 3)}`",
                    f"- selected_crop_large_extra_text: `{purity.get('crop_contains_large_extra_text_area')}`",
                    f"- selected_original_fallback_used: `{selected[0].get('pipeline_debug_original_color_fallback_used')}`",
                    "",
                ]
            )
    path = report_root / "crop_truth_audit_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop-level truth audit for SmartBite test64")
    parser.add_argument(
        "--truth-manifest",
        type=Path,
        default=CROP_TRUTH_ANNOTATIONS_PATH,
        help="Path to website-generated crop truth bbox manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics"))
    parser.add_argument("--today", default=date.today().isoformat())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_truth_manifest(args.truth_manifest)
    missing = validate_truth_manifest(payload)
    if missing:
        lines = "\n".join(f"- {filename}" for filename in missing)
        raise SystemExit(f"crop truth audit requires all 10 true_bbox_xyxy values. Missing:\n{lines}")

    test64_dir = _resolve_test64_dir()
    report_root = args.output_dir.resolve() / f"crop_truth_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    today = date.fromisoformat(str(args.today))
    pipeline = build_pipeline(get_settings())

    items_payload = payload["items"]
    results: list[dict[str, Any]] = []
    for filename in CROP_TRUTH_AUDIT_FILENAMES:
        print(f"[crop-truth] auditing {filename}", flush=True)
        results.append(
            _audit_one(
                pipeline=pipeline,
                filename=filename,
                item=items_payload[filename],
                test64_dir=test64_dir,
                report_root=report_root,
                today=today,
            )
        )

    classifications: dict[str, int] = {}
    for item in results:
        key = str(item.get("classification") or "unknown")
        classifications[key] = classifications.get(key, 0) + 1
    selected_true_candidate_count = sum(1 for item in results if item.get("ranker_selected_true_candidate"))
    correct_parsed_dates = 0
    for item in results:
        expected = item.get("expected") or {}
        expected_date = None
        if expected.get("year") and expected.get("month") and expected.get("day"):
            expected_date = f"{int(expected['year']):04d}-{int(expected['month']):02d}-{int(expected['day']):02d}"
        if expected_date and (item.get("pipeline") or {}).get("parsed_date") == expected_date:
            correct_parsed_dates += 1

    report = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "truth_manifest": str(args.truth_manifest),
        "report_root": str(report_root),
        "summary": {
            "total_images": len(results),
            "classifications": classifications,
            "selected_true_candidate_count": selected_true_candidate_count,
            "correct_parsed_dates": correct_parsed_dates,
        },
        "items": results,
    }
    report_path = report_root / "crop_truth_audit_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=_json_default), encoding="utf-8")
    summary_path = _write_summary(report_root, report)
    print(f"crop truth audit report: {report_path}")
    print(f"crop truth audit summary: {summary_path}")


if __name__ == "__main__":
    main()
