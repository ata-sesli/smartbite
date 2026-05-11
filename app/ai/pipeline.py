from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
import json
import re
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ai.crop_normalization import (
    NormalizedTextLineCrop,
    TextLineCropConfig,
    TextLineGeometry,
    normalize_textline_crops,
)
from app.ai.date_role import (
    compact_evidence_text,
    has_expiry_keyword,
    has_production_keyword,
    has_weak_expiry_keyword,
    normalize_evidence_text,
    role_selection_key,
    score_date_role,
)
from app.ai.decision import ExpiryDecisionEngine
from app.ai.detector import ExpiryRegionDetector
from app.ai.ocr import OCRRouter, RecognitionData, TextDetectionBox
from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ImageVariant, ROIImagePreprocessor
from app.ai.types import DecisionResult, DetectionResult, OCRResultData, ParsedDateData
from app.domain.enums import ExpiryClassification, FinalResultStatus

PREFIX_HINTS = ("EXP", "USE BY", "BEST BEFORE", "SKT", "TETT", "BBE", "SON TUKETIM", "SON KULLANMA")
EXPIRY_RESCUE_KEYWORDS = (
    "EXP",
    "EXPIRE",
    "EXPIRY",
    "BBE",
    "BEST BEFORE",
    "USE BY",
    "TETT",
    "SKT",
    "SON TUKETIM",
    "SON KULLANMA",
)
PRODUCTION_KEYWORDS = (
    "URT",
    "MFG",
    "MFD",
    "PROD",
    "PRODUCTION",
    "MANUFACTURE",
    "MANUFACTURING",
)
GENERIC_BRAND_TOKENS = ("COMPANY", "COMPAGNE", "COMPAGNER", "COCA", "COLA", "COCACOLA", "COCACOLATE", "STATES")
NEGATIVE_HINTS = (
    "INGREDIENT",
    "NUTRITION",
    "CARBO",
    "PROTEIN",
    "ENERGY",
    "KCAL",
    "NET",
    "GRAM",
    "KG",
    "ML",
    "LITER",
    "LOT",
    "BATCH",
    "ADDRESS",
    "PRODUCER",
    "DISTRIBUTOR",
    "IMPORTER",
)

DATE_PATTERN = re.compile(r"\b\d{1,4}[./:-]\d{1,2}(?:[./:-]\d{1,4})?\b")
LOT_ONLY_PATTERN = re.compile(r"\b(?:LOT|BATCH)\b")
WEIGHT_VOLUME_ONLY_PATTERN = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*(?:G|GR|GRAM|KG|ML|L|LT|LITER|LITRE)\s*$")


@dataclass(slots=True)
class PipelineRunOutput:
    detected: bool
    detector_confidence: float | None
    raw_text: str | None
    normalized_text: str | None
    ocr_confidence: float | None
    ocr_engine: str | None
    ocr_runtime_device: str | None
    parsed_date: date | None
    date_format_detected: str | None
    parse_confidence: float | None
    expiry_classification: str
    days_remaining: int | None
    alert_required: bool
    needs_review: bool
    final_status: str
    reason: str
    stage_timings_ms: dict[str, int]
    roi_png_bytes: bytes | None
    debug_candidates_json_bytes: bytes | None = None
    debug_overlay_png_bytes: bytes | None = None
    debug_summary: dict[str, object] | None = None


@dataclass(slots=True)
class ScanROICandidate:
    source: str
    image: np.ndarray
    confidence: float | None
    offset_x: int
    offset_y: int


@dataclass(slots=True)
class GlobalCandidate:
    source: str
    variant: str
    variant_key: str
    roi: np.ndarray
    detector_confidence: float | None
    ranked: RankedCandidate
    scan_bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class EvaluationCandidate:
    source: str
    variant: str
    variant_key: str
    candidate_id: str
    roi: np.ndarray
    detector_confidence: float | None
    candidate_bbox: tuple[int, int, int, int]
    candidate_total_score: float
    ocr: OCRResultData
    parsed: ParsedDateData
    parse_inputs_count: int
    recognition_variant: str
    scan_bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class FinalParseqCrop:
    image: np.ndarray
    bbox: tuple[int, int, int, int]
    padding_px: int
    policy: str
    polygon_xy: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class RankedCandidate:
    candidate_id: str
    candidate_type: str
    bbox: tuple[int, int, int, int]
    member_indices: list[int]
    member_bboxes: list[tuple[int, int, int, int]]
    detector_sources: tuple[str, ...]
    detector_variant: str | None
    geometry_features: dict[str, float]
    geometry_score: float
    score_breakdown: dict[str, float]
    recognition_bbox: tuple[int, int, int, int] | None = None
    evidence_bbox: tuple[int, int, int, int] | None = None
    polygon_xy: tuple[tuple[float, float], ...] | None = None
    probe_text: str = ""
    probe_normalized_text: str = ""
    probe_confidence: float | None = None
    probe_reason: str | None = None
    total_score: float = 0.0
    selected_geometry: bool = False
    selected_final: bool = False
    parseq_text: str = ""
    parseq_confidence: float | None = None
    parseq_reason: str | None = None
    parseq_input_shape: list[int] | None = None
    recognizer_engine: str = ""
    recognizer_model_name: str = ""
    recognition_variant: str = ""
    selected_recognition_variant: str = ""
    recognition_variant_image: np.ndarray | None = None
    final_crop_bbox: tuple[int, int, int, int] | None = None
    final_crop_padding_px: int | None = None
    final_crop_policy: str | None = None
    crop_transform_used: str | None = None
    orientation_candidates_tried: list[str] | None = None
    selected_orientation: str | None = None
    original_crop_shape: list[int] | None = None
    normalized_crop_shape: list[int] | None = None
    selected_transform_reason: str | None = None
    original_color_fallback_used: bool = False
    parsed_date: str | None = None
    parser_date_precision: str | None = None
    parser_parsed_day: int | None = None
    parser_parsed_month: int | None = None
    parser_parsed_year: int | None = None
    parse_confidence: float | None = None
    scan_bbox: tuple[int, int, int, int] | None = None
    multiline_split_source_id: str | None = None
    line_index_in_group: int | None = None
    force_included_reason: str | None = None
    final_rank_before_force_include: int | None = None
    final_rank_after_force_include: int | None = None
    context_probe_bboxes: list[tuple[int, int, int, int]] | None = None
    context_probe_indices: list[int] | None = None
    context_probe_texts: list[str] | None = None
    context_probe_confidences: list[float | None] | None = None
    adjacent_expiry_keyword: bool = False
    adjacent_production_keyword: bool = False
    keyword_relation: str | None = None
    context_recognizer_engine: str = ""
    context_recognizer_model_name: str = ""
    final_recognizer_engine: str = ""
    final_recognizer_model_name: str = ""

    def crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return image[y1:y2, x1:x2]

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_type": self.candidate_type,
            "bbox_xyxy": [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]],
            "recognition_bbox_xyxy": (
                [self.recognition_bbox[0], self.recognition_bbox[1], self.recognition_bbox[2], self.recognition_bbox[3]]
                if self.recognition_bbox is not None
                else None
            ),
            "evidence_bbox_xyxy": (
                [self.evidence_bbox[0], self.evidence_bbox[1], self.evidence_bbox[2], self.evidence_bbox[3]]
                if self.evidence_bbox is not None
                else None
            ),
            "polygon_xy": (
                [[float(x), float(y)] for x, y in self.polygon_xy]
                if self.polygon_xy is not None
                else None
            ),
            "scan_bbox_xyxy": (
                [self.scan_bbox[0], self.scan_bbox[1], self.scan_bbox[2], self.scan_bbox[3]]
                if self.scan_bbox is not None
                else None
            ),
            "member_indices": self.member_indices,
            "detector_sources": list(self.detector_sources),
            "detector_variant": self.detector_variant,
            "geometry_features": self.geometry_features,
            "geometry_score": self.geometry_score,
            "score_breakdown": self.score_breakdown,
            "probe_text": self.probe_text,
            "probe_normalized_text": self.probe_normalized_text,
            "probe_confidence": self.probe_confidence,
            "probe_reason": self.probe_reason,
            "total_score": self.total_score,
            "selected_geometry": self.selected_geometry,
            "selected_final": self.selected_final,
            "parseq_text": self.parseq_text,
            "parseq_confidence": self.parseq_confidence,
            "parseq_reason": self.parseq_reason,
            "parseq_input_shape": self.parseq_input_shape,
            "recognizer_engine": self.recognizer_engine,
            "recognizer_model_name": self.recognizer_model_name,
            "recognition_variant": self.recognition_variant,
            "selected_recognition_variant": self.selected_recognition_variant or self.recognition_variant,
            "final_crop_bbox_xyxy": (
                [self.final_crop_bbox[0], self.final_crop_bbox[1], self.final_crop_bbox[2], self.final_crop_bbox[3]]
                if self.final_crop_bbox is not None
                else None
            ),
            "final_crop_padding_px": self.final_crop_padding_px,
            "final_crop_policy": self.final_crop_policy,
            "crop_transform_used": self.crop_transform_used,
            "orientation_candidates_tried": self.orientation_candidates_tried or [],
            "selected_orientation": self.selected_orientation,
            "original_crop_shape": self.original_crop_shape,
            "normalized_crop_shape": self.normalized_crop_shape,
            "selected_transform_reason": self.selected_transform_reason,
            "original_color_fallback_used": self.original_color_fallback_used,
            "parsed_date": self.parsed_date,
            "parser_date_precision": self.parser_date_precision,
            "parser_parsed_day": self.parser_parsed_day,
            "parser_parsed_month": self.parser_parsed_month,
            "parser_parsed_year": self.parser_parsed_year,
            "parse_confidence": self.parse_confidence,
            "multiline_split_source_id": self.multiline_split_source_id,
            "line_index_in_group": self.line_index_in_group,
            "force_included_reason": self.force_included_reason,
            "final_rank_before_force_include": self.final_rank_before_force_include,
            "final_rank_after_force_include": self.final_rank_after_force_include,
            "date_like_score": self.score_breakdown.get("date_like_score", 0.0),
            "expiry_keyword_score": self.score_breakdown.get("expiry_keyword_score", 0.0),
            "production_keyword_penalty": self.score_breakdown.get("production_keyword_penalty", 0.0),
            "brand_text_penalty": self.score_breakdown.get("brand_text_penalty", 0.0),
            "parser_probe_bonus": self.score_breakdown.get("parser_probe_bonus", 0.0),
            "keyword_date_group_bonus": self.score_breakdown.get("keyword_date_group_bonus", 0.0),
            "context_probe_bboxes": [list(bbox) for bbox in (self.context_probe_bboxes or [])],
            "context_probe_texts": self.context_probe_texts or [],
            "context_probe_confidences": self.context_probe_confidences or [],
            "adjacent_expiry_keyword": self.adjacent_expiry_keyword,
            "adjacent_production_keyword": self.adjacent_production_keyword,
            "keyword_relation": self.keyword_relation,
            "context_recognizer_engine": self.context_recognizer_engine,
            "context_recognizer_model_name": self.context_recognizer_model_name,
            "final_recognizer_engine": self.final_recognizer_engine or self.recognizer_engine,
            "final_recognizer_model_name": self.final_recognizer_model_name or self.recognizer_model_name,
        }


class ExpiryPipeline:
    def __init__(
        self,
        detector: ExpiryRegionDetector,
        preprocessor: ROIImagePreprocessor,
        ocr_router: OCRRouter,
        parser: ExpiryDateParser,
        decision: ExpiryDecisionEngine,
        *,
        top_k: int = 4,
        roi_padding_ratio: float = 0.12,
        high_recall_mode: bool = True,
        expiry_filter_geometry_top_n: int = 12,
        expiry_filter_final_top_k: int = 4,
        save_debug_candidates: bool = True,
        save_debug_overlays: bool = True,
        craft_rescue_enabled: bool = True,
        craft_rescue_variant_name: str = "raw",
        craft_rescue_canvas_size: int = 640,
        craft_rescue_mag_ratio: float = 1.0,
        craft_max_rois_per_scan: int = 1,
        craft_max_boxes_accepted: int = 40,
        craft_trigger_min_ppocr_boxes: int = 3,
        craft_trigger_min_best_score: float = 1.6,
        craft_trigger_large_box_area_ratio: float = 0.15,
        craft_timeout_seconds: float = 12.0,
        expiry_global_final_top_k: int = 6,
        expiry_global_debug_final_top_k: int = 12,
        expiry_global_max_per_roi: int = 2,
        expiry_global_max_per_variant: int = 3,
        expiry_global_probe_top_k: int = 20,
        expiry_max_detector_variants: int = 1,
        expiry_detector_variants: tuple[str, ...] = ("raw",),
        expiry_max_rois_per_scan: int = 1,
        expiry_probe_variants: tuple[str, ...] = ("original_padded", "clahe_gray", "adaptive_binary"),
        expiry_parseq_variant_policy: str = "best_probe",
        expiry_parseq_max_variants_per_candidate: int = 1,
        expiry_debug_all_recognition_variants: bool = False,
        expiry_final_crop_padding_ratio: float = 0.04,
        expiry_final_crop_min_padding_px: int = 3,
        expiry_final_crop_max_padding_px: int = 18,
        expiry_debug_parseq_whole_groups: bool = False,
        max_group_candidates_per_roi: int = 100,
        context_probe_enabled: bool = True,
        context_max_boxes_per_candidate: int = 2,
        context_max_candidates_per_scan: int = 20,
    ) -> None:
        self.detector = detector
        self.preprocessor = preprocessor
        self.ocr_router = ocr_router
        self.parser = parser
        self.decision = decision
        self.top_k = max(1, int(top_k))
        self.roi_padding_ratio = max(0.0, min(float(roi_padding_ratio), 0.5))
        self.high_recall_mode = bool(high_recall_mode)
        self.expiry_filter_geometry_top_n = max(1, int(expiry_filter_geometry_top_n))
        self.expiry_filter_final_top_k = max(1, int(expiry_filter_final_top_k))
        self.save_debug_candidates = bool(save_debug_candidates)
        self.save_debug_overlays = bool(save_debug_overlays)
        self.craft_rescue_enabled = bool(craft_rescue_enabled)
        self.craft_rescue_variant_name = craft_rescue_variant_name.strip() or "raw"
        self.craft_rescue_canvas_size = max(1, int(craft_rescue_canvas_size))
        self.craft_rescue_mag_ratio = max(0.1, float(craft_rescue_mag_ratio))
        self.craft_max_rois_per_scan = max(0, int(craft_max_rois_per_scan))
        self.craft_max_boxes_accepted = max(1, int(craft_max_boxes_accepted))
        self.craft_trigger_min_ppocr_boxes = max(0, int(craft_trigger_min_ppocr_boxes))
        self.craft_trigger_min_best_score = float(craft_trigger_min_best_score)
        self.craft_trigger_large_box_area_ratio = max(0.0, min(float(craft_trigger_large_box_area_ratio), 1.0))
        self.craft_timeout_seconds = max(0.1, float(craft_timeout_seconds))
        self.expiry_global_final_top_k = max(1, int(expiry_global_final_top_k))
        self.expiry_global_debug_final_top_k = max(self.expiry_global_final_top_k, int(expiry_global_debug_final_top_k))
        self.expiry_global_max_per_roi = max(1, int(expiry_global_max_per_roi))
        self.expiry_global_max_per_variant = max(1, int(expiry_global_max_per_variant))
        self.expiry_global_probe_top_k = max(1, int(expiry_global_probe_top_k))
        self.expiry_max_detector_variants = max(1, int(expiry_max_detector_variants))
        self.expiry_detector_variants = tuple(v.strip() for v in expiry_detector_variants if v.strip()) or ("raw",)
        self.expiry_max_rois_per_scan = max(1, int(expiry_max_rois_per_scan))
        self.expiry_probe_variants = tuple(v.strip() for v in expiry_probe_variants if v.strip()) or (
            "original_padded",
            "clahe_gray",
            "adaptive_binary",
        )
        self.expiry_parseq_variant_policy = expiry_parseq_variant_policy.strip().lower() or "best_probe"
        self.expiry_parseq_max_variants_per_candidate = max(1, int(expiry_parseq_max_variants_per_candidate))
        self.expiry_debug_all_recognition_variants = bool(expiry_debug_all_recognition_variants)
        self.expiry_final_crop_padding_ratio = max(0.0, min(float(expiry_final_crop_padding_ratio), 0.5))
        self.expiry_final_crop_min_padding_px = max(0, int(expiry_final_crop_min_padding_px))
        self.expiry_final_crop_max_padding_px = max(
            self.expiry_final_crop_min_padding_px,
            int(expiry_final_crop_max_padding_px),
        )
        self.expiry_debug_parseq_whole_groups = bool(expiry_debug_parseq_whole_groups)
        self.max_group_candidates_per_roi = max(0, int(max_group_candidates_per_roi))
        self.context_probe_enabled = bool(context_probe_enabled)
        self.context_max_boxes_per_candidate = max(0, int(context_max_boxes_per_candidate))
        self.context_max_candidates_per_scan = max(0, int(context_max_candidates_per_scan))

    def run(self, image_bytes: bytes, *, today: date) -> PipelineRunOutput:
        run_start = perf_counter()
        timings: dict[str, int] = {
            "preprocess": 0,
            "text_detect": 0,
            "probe": 0,
            "ocr": 0,
            "parse": 0,
        }
        performance = self._new_performance_stats()

        t0 = perf_counter()
        image = self._decode_image(image_bytes)
        timings["decode"] = int((perf_counter() - t0) * 1000)
        if image is None:
            return self._failed(
                final_status=FinalResultStatus.DETECTOR_FAILED,
                reason="image unreadable",
                timings=timings,
            )

        t1 = perf_counter()
        detection: DetectionResult = self.detector.detect(image)
        timings["detect"] = int((perf_counter() - t1) * 1000)
        performance["yolo_runtime_ms"] = timings["detect"]

        roi_candidates = self._collect_roi_candidates(image=image, detection=detection)
        performance["roi_count"] = len(roi_candidates)
        evaluated: list[EvaluationCandidate] = []
        global_candidates: list[GlobalCandidate] = []
        scan_box_bboxes: list[tuple[int, int, int, int]] = []
        variants_debug: list[dict[str, object]] = []
        variant_debug_records: dict[str, tuple[np.ndarray, list[TextDetectionBox], list[RankedCandidate]]] = {}
        craft_rois_used = 0

        for roi_candidate in roi_candidates:
            source = roi_candidate.source
            roi_image = roi_candidate.image
            confidence = roi_candidate.confidence
            t_pre = perf_counter()
            variants = self._variants_for_roi(roi_image)
            timings["preprocess"] += int((perf_counter() - t_pre) * 1000)
            performance["detector_variants_by_roi"][source] = [variant.name for variant in variants]

            for variant in variants:
                variant_name = variant.name
                variant_image = variant.image
                variant_key = f"{source}:{variant_name}"

                t_det = perf_counter()
                detected_boxes_raw, det_reason = self._detect_text_boxes(
                    variant_image,
                    variant_name=variant_name,
                )
                det_ms = int((perf_counter() - t_det) * 1000)
                timings["text_detect"] += det_ms
                if getattr(self.ocr_router, "text_detector_mode", "unknown") == "craft":
                    performance["craft_call_count"] += 1
                    performance["craft_total_ms"] += det_ms
                else:
                    performance["ppocr_detection_call_count"] += 1
                    performance["ppocr_detection_total_ms"] += det_ms

                detected_boxes = self._map_text_boxes_to_original(
                    boxes=detected_boxes_raw,
                    variant=variant,
                    original_shape=roi_image.shape,
                )
                performance["detected_boxes_before_dedupe"] += len(detected_boxes)
                scan_box_bboxes.extend(
                    self._scan_bbox_from_box(box, offset_x=roi_candidate.offset_x, offset_y=roi_candidate.offset_y)
                    for box in detected_boxes
                )
                ranked = self._build_ranked_candidates(roi_image, detected_boxes)
                performance["candidates_before_geometry_cap"] += len(ranked)
                ranked = self._apply_geometry_shortlist(ranked)

                craft_debug = self._default_craft_debug()
                should_rescue, rescue_reasons, skipped_reason = self._should_run_craft_rescue(
                    ranked_candidates=ranked,
                    ppocr_boxes=detected_boxes,
                    variant_name=variant_name,
                    craft_rois_used=craft_rois_used,
                )
                craft_debug["craft_trigger_reason"] = rescue_reasons
                craft_debug["craft_skipped_reason"] = skipped_reason

                if should_rescue:
                    craft_rois_used += 1
                    craft_debug["craft_triggered"] = True
                    craft_boxes_raw, craft_reason, craft_runtime_ms, craft_timed_out = self._run_craft_rescue(
                        roi_image,
                        variant_name=self.craft_rescue_variant_name,
                    )
                    timings["text_detect"] += craft_runtime_ms
                    craft_debug["craft_runtime_ms"] = craft_runtime_ms
                    craft_debug["craft_box_count"] = len(craft_boxes_raw)
                    performance["craft_call_count"] += 1
                    performance["craft_total_ms"] += craft_runtime_ms
                    performance["detected_boxes_before_dedupe"] += len(craft_boxes_raw)

                    craft_accepted = self._cap_craft_boxes(craft_boxes_raw)
                    if craft_timed_out:
                        craft_accepted = []
                        craft_debug["craft_skipped_reason"] = "craft_timeout"
                    elif craft_reason and self._reason_is_detector_unavailable(craft_reason):
                        craft_accepted = []
                        craft_debug["craft_skipped_reason"] = "craft_unavailable"
                    elif not craft_accepted:
                        craft_debug["craft_skipped_reason"] = "craft_returned_no_boxes"
                    else:
                        craft_debug["craft_skipped_reason"] = None

                    craft_debug["craft_accepted_box_count"] = len(craft_accepted)
                    if craft_reason:
                        det_reason = "; ".join(dict.fromkeys(part for part in (det_reason, craft_reason) if part))

                    if craft_accepted:
                        detected_boxes = self._merge_text_boxes(detected_boxes, craft_accepted)
                        scan_box_bboxes.extend(
                            self._scan_bbox_from_box(box, offset_x=roi_candidate.offset_x, offset_y=roi_candidate.offset_y)
                            for box in craft_accepted
                        )
                        ranked = self._build_ranked_candidates(roi_image, detected_boxes)
                        performance["candidates_before_geometry_cap"] += len(ranked)
                        ranked = self._apply_geometry_shortlist(ranked)

                for candidate in ranked:
                    candidate.selected_final = False
                    candidate.scan_bbox = self._scan_bbox_for_candidate(
                        candidate,
                        offset_x=roi_candidate.offset_x,
                        offset_y=roi_candidate.offset_y,
                    )
                    if not candidate.selected_geometry:
                        continue
                    global_candidates.append(
                        GlobalCandidate(
                            source=source,
                            variant=variant_name,
                            variant_key=variant_key,
                            roi=roi_image,
                            detector_confidence=confidence,
                            ranked=candidate,
                            scan_bbox=candidate.scan_bbox,
                        )
                    )

                variants_debug.append(
                    {
                        "source": source,
                        "variant": variant_name,
                        "variant_key": variant_key,
                        "detector_mode": getattr(self.ocr_router, "text_detector_mode", "unknown"),
                        "detector_box_count": len(detected_boxes),
                        "merged_box_count": len(detected_boxes),
                        "detector_counts": self._detector_counts(detected_boxes),
                        "detected_boxes": [
                            self._detector_box_to_debug_dict(
                                box,
                                offset_x=roi_candidate.offset_x,
                                offset_y=roi_candidate.offset_y,
                            )
                            for box in detected_boxes
                        ],
                        "detector_reason": det_reason,
                        "detector_unavailable_reasons": self._detector_unavailable_reasons(det_reason),
                        **craft_debug,
                        "candidate_count": len(ranked),
                        "geometry_selected_count": sum(1 for candidate in ranked if candidate.selected_geometry),
                        "final_selected_count": 0,
                        "geometry_top_n": self.expiry_filter_geometry_top_n,
                        "final_top_k": self.expiry_filter_final_top_k,
                        "global_final_top_k": self._effective_global_final_top_k(),
                        "candidates": [candidate.to_debug_dict() for candidate in ranked],
                    }
                )
                variant_debug_records[variant_key] = (roi_image, detected_boxes, ranked)

        performance["detected_boxes_after_dedupe"] = len(self._dedupe_bbox_list(scan_box_bboxes))
        global_probe_candidates, global_probe_summary = self._select_global_probe_candidates(global_candidates)
        performance["candidates_sent_to_mobile_probe"] = len(global_probe_candidates)
        self._probe_global_candidates(
            global_probe_candidates,
            today=today,
            timings=timings,
            performance=performance,
        )
        global_selected, global_selection_summary = self._select_global_final_candidates(global_probe_candidates)
        performance["candidates_before_global_final_cap"] = len(global_probe_candidates)
        performance["parseq_candidate_count"] = len(global_selected)
        global_selection_summary.update(global_probe_summary)
        self._refresh_variant_debug_records(variants_debug, variant_debug_records)

        for global_candidate in global_selected:
            candidate = global_candidate.ranked
            final_crop, normalized_crops = self._normalized_final_crops_for_candidate(candidate, global_candidate.roi)
            if not normalized_crops:
                continue
            candidate.final_crop_bbox = final_crop.bbox
            candidate.final_crop_padding_px = final_crop.padding_px
            candidate.final_crop_policy = final_crop.policy

            t_ocr = perf_counter()
            (
                parseq,
                parsed,
                parse_inputs_count,
                recognition_variant,
                input_shape,
                parse_ms,
                original_color_fallback_used,
                selected_normalized_crop,
            ) = self._best_parseq_for_crop(
                normalized_crops,
                today=today,
                selected_variant_name=candidate.recognition_variant or None,
                selected_variant_image=(
                    candidate.recognition_variant_image
                    if final_crop.bbox == candidate.bbox and normalized_crops[0].selected_orientation == "original"
                    else None
                ),
                performance=performance,
            )
            timings["ocr"] += int((perf_counter() - t_ocr) * 1000)
            timings["parse"] += parse_ms
            performance["parser_total_ms"] += parse_ms
            candidate.parseq_input_shape = input_shape
            candidate.recognition_variant = recognition_variant
            candidate.original_color_fallback_used = original_color_fallback_used
            candidate.crop_transform_used = selected_normalized_crop.crop_transform_used
            candidate.orientation_candidates_tried = selected_normalized_crop.orientation_candidates_tried
            candidate.selected_orientation = selected_normalized_crop.selected_orientation
            candidate.original_crop_shape = selected_normalized_crop.original_crop_shape
            candidate.normalized_crop_shape = selected_normalized_crop.normalized_crop_shape
            candidate.selected_transform_reason = selected_normalized_crop.selected_transform_reason

            ocr = OCRResultData(
                raw_text=parseq.raw_text,
                normalized_text=parseq.normalized_text,
                confidence=parseq.confidence,
                engine_name=self.ocr_router.active_engine_name,
                runtime_device=getattr(
                    self.ocr_router,
                    "final_recognition_runtime_device",
                    getattr(self.ocr_router, "parseq_runtime_device", None),
                ),
                reason=parseq.reason,
            )

            candidate.parseq_text = parseq.raw_text
            candidate.parseq_confidence = parseq.confidence
            candidate.parseq_reason = parseq.reason
            candidate.recognizer_engine = str(getattr(self.ocr_router, "active_engine_name", ""))
            candidate.recognizer_model_name = str(getattr(self.ocr_router, "final_recognizer_model_name", ""))
            candidate.final_recognizer_engine = candidate.recognizer_engine
            candidate.final_recognizer_model_name = candidate.recognizer_model_name
            candidate.parsed_date = parsed.parsed_date.isoformat() if parsed.parsed_date else None
            candidate.parser_date_precision = parsed.date_precision
            candidate.parser_parsed_day = parsed.parsed_day
            candidate.parser_parsed_month = parsed.parsed_month
            candidate.parser_parsed_year = parsed.parsed_year
            candidate.parse_confidence = parsed.confidence

            evaluated.append(
                EvaluationCandidate(
                    source=global_candidate.source,
                    variant=global_candidate.variant,
                    variant_key=global_candidate.variant_key,
                    candidate_id=candidate.candidate_id,
                    roi=global_candidate.roi,
                    detector_confidence=global_candidate.detector_confidence,
                    candidate_bbox=candidate.bbox,
                    candidate_total_score=candidate.total_score,
                    ocr=ocr,
                    parsed=parsed,
                    parse_inputs_count=parse_inputs_count,
                    recognition_variant=recognition_variant,
                    scan_bbox=global_candidate.scan_bbox,
                )
            )

        self._refresh_variant_debug_records(
            variants_debug,
            variant_debug_records,
            global_selection_summary=global_selection_summary,
        )
        performance_summary = self._finalize_performance_stats(
            performance,
            total_runtime_ms=int((perf_counter() - run_start) * 1000),
        )
        global_selection_summary["performance"] = performance_summary

        parseable = [candidate for candidate in evaluated if candidate.parsed.parsed_date is not None]
        if parseable:
            selected = max(parseable, key=self._score_parseable_candidate)
            t_decision = perf_counter()
            decision: DecisionResult = self.decision.decide(
                selected.parsed.parsed_date,
                today=today,
                parse_confidence=selected.parsed.confidence,
            )
            timings["decision"] = int((perf_counter() - t_decision) * 1000)
            reason = self._compose_reason(
                selected=selected,
                evaluated=evaluated,
                parseable_count=len(parseable),
                base_reason=f"{selected.parsed.reason};{decision.reason}",
            )

            debug_candidates_json, debug_overlay_png, debug_summary = self._build_debug_artifacts(
                variants_debug=variants_debug,
                variant_debug_records=variant_debug_records,
                selected_variant_key=selected.variant_key,
                selected_candidate_id=selected.candidate_id,
                evaluated_count=len(evaluated),
                parseable_count=len(parseable),
                global_selection_summary=global_selection_summary,
            )

            return PipelineRunOutput(
                detected=detection.detected,
                detector_confidence=selected.detector_confidence if selected.detector_confidence is not None else detection.confidence,
                raw_text=selected.ocr.raw_text,
                normalized_text=selected.ocr.normalized_text,
                ocr_confidence=selected.ocr.confidence,
                ocr_engine=selected.ocr.engine_name,
                ocr_runtime_device=selected.ocr.runtime_device,
                parsed_date=selected.parsed.parsed_date,
                date_format_detected=selected.parsed.date_format_detected,
                parse_confidence=selected.parsed.confidence,
                expiry_classification=decision.expiry_classification,
                days_remaining=decision.days_remaining,
                alert_required=decision.alert_required,
                needs_review=decision.needs_review,
                final_status=decision.final_status,
                reason=reason,
                stage_timings_ms=timings,
                roi_png_bytes=self._encode_png(selected.roi),
                debug_candidates_json_bytes=debug_candidates_json,
                debug_overlay_png_bytes=debug_overlay_png,
                debug_summary=debug_summary,
            )

        with_text = [candidate for candidate in evaluated if candidate.ocr.raw_text.strip()]
        if with_text:
            selected = max(with_text, key=self._score_unparseable_candidate)
            reason = self._compose_reason(
                selected=selected,
                evaluated=evaluated,
                parseable_count=0,
                base_reason=f"manual_review_no_parse;{selected.parsed.reason}",
            )

            debug_candidates_json, debug_overlay_png, debug_summary = self._build_debug_artifacts(
                variants_debug=variants_debug,
                variant_debug_records=variant_debug_records,
                selected_variant_key=selected.variant_key,
                selected_candidate_id=selected.candidate_id,
                evaluated_count=len(evaluated),
                parseable_count=0,
                global_selection_summary=global_selection_summary,
            )

            return self._failed(
                final_status=FinalResultStatus.MANUAL_REVIEW_REQUIRED,
                reason=reason,
                timings=timings,
                detected=detection.detected,
                detector_confidence=selected.detector_confidence if selected.detector_confidence is not None else detection.confidence,
                ocr=selected.ocr,
                parsed=selected.parsed,
                roi=selected.roi,
                debug_candidates_json_bytes=debug_candidates_json,
                debug_overlay_png_bytes=debug_overlay_png,
                debug_summary=debug_summary,
            )

        if detection.reason:
            debug_candidates_json, debug_overlay_png, debug_summary = self._build_debug_artifacts(
                variants_debug=variants_debug,
                variant_debug_records=variant_debug_records,
                selected_variant_key=None,
                selected_candidate_id=None,
                evaluated_count=len(evaluated),
                parseable_count=0,
                global_selection_summary=global_selection_summary,
            )
            return self._failed(
                final_status=FinalResultStatus.DETECTOR_FAILED,
                reason=self._compose_empty_reason(
                    detection_reason=detection.reason,
                    evaluated_count=len(evaluated),
                ),
                timings=timings,
                detected=False,
                detector_confidence=detection.confidence,
                debug_candidates_json_bytes=debug_candidates_json,
                debug_overlay_png_bytes=debug_overlay_png,
                debug_summary=debug_summary,
            )

        debug_candidates_json, debug_overlay_png, debug_summary = self._build_debug_artifacts(
            variants_debug=variants_debug,
            variant_debug_records=variant_debug_records,
            selected_variant_key=None,
            selected_candidate_id=None,
            evaluated_count=len(evaluated),
            parseable_count=0,
            global_selection_summary=global_selection_summary,
        )
        return self._failed(
            final_status=FinalResultStatus.OCR_FAILED,
            reason=self._compose_empty_reason(
                detection_reason="ocr returned empty text",
                evaluated_count=len(evaluated),
            ),
            timings=timings,
            detected=detection.detected,
            detector_confidence=detection.confidence,
            debug_candidates_json_bytes=debug_candidates_json,
            debug_overlay_png_bytes=debug_overlay_png,
            debug_summary=debug_summary,
        )

    def _collect_roi_candidates(
        self,
        *,
        image: np.ndarray,
        detection: DetectionResult,
    ) -> list[ScanROICandidate]:
        roi_candidates: list[ScanROICandidate] = []
        if detection.detected and detection.boxes:
            detector_boxes = detection.boxes[: min(self.top_k, self.expiry_max_rois_per_scan)]
            for index, box in enumerate(detector_boxes):
                crop = self._crop_box_with_padding(image=image, box=box)
                if crop is None:
                    continue
                roi, offset_x, offset_y = crop
                roi_candidates.append(
                    ScanROICandidate(
                        source=f"detector_box_{index + 1}",
                        image=roi,
                        confidence=box.confidence,
                        offset_x=offset_x,
                        offset_y=offset_y,
                    )
                )

        # Always include full-image candidate in high-recall mode.
        if (self.high_recall_mode and len(roi_candidates) < self.expiry_max_rois_per_scan) or not roi_candidates:
            roi_candidates.append(
                ScanROICandidate(
                    source="full_image",
                    image=image,
                    confidence=detection.confidence,
                    offset_x=0,
                    offset_y=0,
                )
            )

        return roi_candidates

    def _variants_for_roi(self, roi: np.ndarray) -> list[ImageVariant]:
        detector_variants = getattr(self.preprocessor, "detector_variants", None)
        if callable(detector_variants):
            variants = list(detector_variants(roi))
        else:
            variants = [ImageVariant(name="raw", image=roi, purpose="detector")]

        allowed = set(self.expiry_detector_variants)
        if allowed:
            variants = [variant for variant in variants if variant.name in allowed]
        if not variants:
            variants = [ImageVariant(name="raw", image=roi, purpose="detector")]
        return variants[: self.expiry_max_detector_variants]

    def _detect_text_boxes(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        try:
            return self.ocr_router.detect_text_boxes(image, variant_name=variant_name)
        except TypeError as exc:
            if "variant_name" not in str(exc):
                raise
            return self.ocr_router.detect_text_boxes(image)

    def _probe_geometry_candidates(
        self,
        *,
        roi_image: np.ndarray,
        ranked: list[RankedCandidate],
        today: date,
        timings: dict[str, int],
    ) -> None:
        for candidate in ranked:
            if not candidate.selected_geometry:
                continue
            crop = candidate.crop(roi_image)
            if crop.size == 0:
                candidate.probe_reason = "candidate crop empty"
                continue

            t_probe = perf_counter()
            probe, recognition_variant, recognition_variant_image = self._best_probe_for_crop(crop, today=today)
            timings["probe"] += int((perf_counter() - t_probe) * 1000)

            probe_scores = self._probe_signal_scores(probe, candidate=candidate, today=today)
            candidate.recognition_variant = recognition_variant
            candidate.probe_text = probe.raw_text
            candidate.probe_normalized_text = probe.normalized_text
            candidate.probe_confidence = probe.confidence
            candidate.probe_reason = probe.reason
            candidate.score_breakdown.update(probe_scores)
            candidate.total_score = candidate.geometry_score + sum(probe_scores.values())
            candidate.recognition_variant_image = recognition_variant_image
            candidate.selected_recognition_variant = recognition_variant

    @staticmethod
    def _new_performance_stats() -> dict[str, object]:
        return {
            "total_runtime_ms": 0,
            "yolo_runtime_ms": 0,
            "roi_count": 0,
            "detector_variants_by_roi": {},
            "ppocr_detection_call_count": 0,
            "ppocr_detection_total_ms": 0,
            "ppocr_detection_ms_per_call": [],
            "craft_call_count": 0,
            "craft_total_ms": 0,
            "craft_ms_per_call": [],
            "detected_boxes_before_dedupe": 0,
            "detected_boxes_after_dedupe": 0,
            "candidates_before_geometry_cap": 0,
            "candidates_sent_to_mobile_probe": 0,
            "mobile_probe_call_count": 0,
            "mobile_probe_total_ms": 0,
            "mobile_probe_ms_per_candidate": 0.0,
            "context_probe_call_count": 0,
            "context_probe_total_ms": 0,
            "context_probe_ms_per_call": 0.0,
            "candidates_before_global_final_cap": 0,
            "parseq_call_count": 0,
            "parseq_candidate_count": 0,
            "parseq_total_ms": 0,
            "final_recognition_call_count": 0,
            "final_recognition_total_ms": 0,
            "parser_total_ms": 0,
            "recognition_variants_generated": 0,
            "probe_variants_per_candidate": 0.0,
            "parseq_variants_per_candidate": 0.0,
        }

    @staticmethod
    def _finalize_performance_stats(
        performance: dict[str, object],
        *,
        total_runtime_ms: int,
    ) -> dict[str, object]:
        finalized = dict(performance)
        finalized["total_runtime_ms"] = total_runtime_ms

        ppocr_calls = int(finalized.get("ppocr_detection_call_count") or 0)
        ppocr_total = int(finalized.get("ppocr_detection_total_ms") or 0)
        craft_calls = int(finalized.get("craft_call_count") or 0)
        craft_total = int(finalized.get("craft_total_ms") or 0)
        probe_candidates = int(finalized.get("candidates_sent_to_mobile_probe") or 0)
        probe_total = int(finalized.get("mobile_probe_total_ms") or 0)
        context_calls = int(finalized.get("context_probe_call_count") or 0)
        context_total = int(finalized.get("context_probe_total_ms") or 0)
        parseq_calls = int(finalized.get("parseq_call_count") or 0)
        parseq_total = int(finalized.get("parseq_total_ms") or 0)
        final_recognition_calls = int(finalized.get("final_recognition_call_count") or parseq_calls)
        final_recognition_total = int(finalized.get("final_recognition_total_ms") or parseq_total)
        final_candidates = int(finalized.get("parseq_candidate_count") or finalized.get("candidates_before_global_final_cap") or 0)

        finalized["ppocr_detection_ms_per_call"] = round(ppocr_total / ppocr_calls, 2) if ppocr_calls else 0.0
        finalized["craft_ms_per_call"] = round(craft_total / craft_calls, 2) if craft_calls else 0.0
        finalized["mobile_probe_ms_per_candidate"] = (
            round(probe_total / probe_candidates, 2) if probe_candidates else 0.0
        )
        finalized["context_probe_ms_per_call"] = round(context_total / context_calls, 2) if context_calls else 0.0
        finalized["parseq_ms_per_call"] = round(parseq_total / parseq_calls, 2) if parseq_calls else 0.0
        finalized["final_recognition_call_count"] = final_recognition_calls
        finalized["final_recognition_total_ms"] = final_recognition_total
        finalized["final_recognition_ms_per_call"] = (
            round(final_recognition_total / final_recognition_calls, 2) if final_recognition_calls else 0.0
        )
        finalized["probe_variants_per_candidate"] = (
            round(int(finalized.get("mobile_probe_call_count") or 0) / probe_candidates, 2) if probe_candidates else 0.0
        )
        finalized["parseq_variants_per_candidate"] = (
            round(parseq_calls / final_candidates, 2) if final_candidates else 0.0
        )
        return finalized

    def _select_global_probe_candidates(self, candidates: list[GlobalCandidate]) -> tuple[list[GlobalCandidate], dict[str, object]]:
        for candidate in candidates:
            candidate.ranked.selected_geometry = False
            candidate.ranked.selected_final = False

        deduped = self._dedupe_global_candidates(candidates)
        ordered = sorted(
            deduped,
            key=lambda item: (
                1 if item.ranked.candidate_type == "line" else 0,
                item.ranked.geometry_score,
                float(item.ranked.probe_confidence or 0.0),
                -item.ranked.geometry_features.get("area_ratio", 1.0),
            ),
            reverse=True,
        )
        selected = ordered[: self.expiry_global_probe_top_k]
        for candidate in selected:
            candidate.ranked.selected_geometry = True

        return selected, {
            "global_probe_top_k": self.expiry_global_probe_top_k,
            "global_probe_candidate_count": len(candidates),
            "global_probe_deduped_candidate_count": len(deduped),
            "global_probe_selected_candidate_count": len(selected),
        }

    def _probe_global_candidates(
        self,
        candidates: list[GlobalCandidate],
        *,
        today: date,
        timings: dict[str, int],
        performance: dict[str, object],
    ) -> None:
        context_cache: dict[tuple[int, int, int, int], RecognitionData] = {}
        context_candidates_seen = 0
        for global_candidate in candidates:
            candidate = global_candidate.ranked
            crop = candidate.crop(global_candidate.roi)
            if crop.size == 0:
                candidate.probe_reason = "candidate crop empty"
                continue

            t_probe = perf_counter()
            probe, recognition_variant, recognition_variant_image = self._best_probe_for_crop(
                crop,
                today=today,
                performance=performance,
            )
            timings["probe"] += int((perf_counter() - t_probe) * 1000)

            probe_scores = self._probe_signal_scores(probe, candidate=candidate, today=today)
            candidate.recognition_variant = recognition_variant
            candidate.probe_text = probe.raw_text
            candidate.probe_normalized_text = probe.normalized_text
            candidate.probe_confidence = probe.confidence
            candidate.probe_reason = probe.reason
            candidate.score_breakdown.update(probe_scores)
            candidate.total_score = candidate.geometry_score + sum(probe_scores.values())
            candidate.recognition_variant_image = recognition_variant_image
            candidate.selected_recognition_variant = recognition_variant
            if (
                self.context_probe_enabled
                and self.context_max_boxes_per_candidate > 0
                and context_candidates_seen < self.context_max_candidates_per_scan
                and candidate.context_probe_bboxes
            ):
                context_candidates_seen += 1
                t_context = perf_counter()
                context_scores = self._probe_candidate_context(
                    candidate,
                    roi=global_candidate.roi,
                    cache=context_cache,
                    performance=performance,
                )
                timings["probe"] += int((perf_counter() - t_context) * 1000)
                candidate.score_breakdown.update(context_scores)
                candidate.total_score = candidate.geometry_score + sum(candidate.score_breakdown.values())

    def _probe_candidate_context(
        self,
        candidate: RankedCandidate,
        *,
        roi: np.ndarray,
        cache: dict[tuple[int, int, int, int], RecognitionData],
        performance: dict[str, object],
    ) -> dict[str, float]:
        recognize_context = getattr(self.ocr_router, "recognize_context", None)
        if not callable(recognize_context):
            return {}

        texts: list[str] = []
        confidences: list[float | None] = []
        expiry_hits = 0
        production_hits = 0
        context_bboxes = candidate.context_probe_bboxes or []
        for bbox in context_bboxes[: self.context_max_boxes_per_candidate]:
            x1, y1, x2, y2 = bbox
            crop = roi[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if bbox in cache:
                rec = cache[bbox]
            else:
                t_call = perf_counter()
                rec = recognize_context(crop)
                call_ms = int((perf_counter() - t_call) * 1000)
                performance["context_probe_call_count"] = int(performance["context_probe_call_count"]) + 1
                performance["context_probe_total_ms"] = int(performance["context_probe_total_ms"]) + call_ms
                cache[bbox] = rec
            text = rec.normalized_text or rec.raw_text
            if text:
                texts.append(text)
                confidences.append(rec.confidence)
                if self._has_expiry_keyword(text) or has_weak_expiry_keyword(text):
                    expiry_hits += 1
                if self._has_production_keyword(text):
                    production_hits += 1

        if not texts:
            return {}

        candidate.context_probe_texts = texts
        candidate.context_probe_confidences = confidences
        candidate.context_recognizer_engine = str(getattr(self.ocr_router, "context_recognizer_name", "context"))
        candidate.context_recognizer_model_name = str(getattr(self.ocr_router, "context_recognizer_model_name", ""))
        candidate.evidence_bbox = self._union_many_bboxes([candidate.recognition_bbox or candidate.bbox, *context_bboxes[: len(texts)]])
        candidate.adjacent_expiry_keyword = expiry_hits > 0
        candidate.adjacent_production_keyword = production_hits > 0

        max_conf = max((float(conf or 0.0) for conf in confidences), default=0.0)
        scores: dict[str, float] = {
            "context_probe_confidence_score": min(0.25, max_conf * 0.25),
        }
        if expiry_hits:
            scores["adjacent_expiry_keyword_bonus"] = 4.0
            scores["expiry_keyword_score"] = max(float(candidate.score_breakdown.get("expiry_keyword_score", 0.0)), 2.0)
            scores["keyword_date_group_bonus"] = max(float(candidate.score_breakdown.get("keyword_date_group_bonus", 0.0)), 2.0)
        if production_hits and not expiry_hits:
            scores["adjacent_production_keyword_penalty"] = -2.5
            scores["production_keyword_penalty"] = min(float(candidate.score_breakdown.get("production_keyword_penalty", 0.0)), -1.5)
        return scores

    def _effective_global_final_top_k(self) -> int:
        if self.high_recall_mode:
            return self.expiry_global_debug_final_top_k
        return self.expiry_global_final_top_k

    @staticmethod
    def _scan_bbox_for_candidate(
        candidate: RankedCandidate,
        *,
        offset_x: int,
        offset_y: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = candidate.bbox
        return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)

    @staticmethod
    def _scan_bbox_from_box(
        box: TextDetectionBox,
        *,
        offset_x: int,
        offset_y: int,
    ) -> tuple[int, int, int, int]:
        return (box.x1 + offset_x, box.y1 + offset_y, box.x2 + offset_x, box.y2 + offset_y)

    def _select_global_final_candidates(
        self,
        candidates: list[GlobalCandidate],
    ) -> tuple[list[GlobalCandidate], dict[str, object]]:
        for candidate in candidates:
            candidate.ranked.selected_final = False
            candidate.ranked.force_included_reason = None
            candidate.ranked.final_rank_before_force_include = None
            candidate.ranked.final_rank_after_force_include = None

        deduped = self._dedupe_global_candidates(candidates)
        cap = self._effective_global_final_top_k()

        selected: list[GlobalCandidate] = []
        selected_ids: set[int] = set()
        per_roi_counts: dict[str, int] = {}
        per_variant_counts: dict[str, int] = {}

        def sort_key(item: GlobalCandidate) -> tuple[int, float, float, float, float]:
            ranked = item.ranked
            return (
                self._candidate_evidence_priority(ranked),
                ranked.total_score,
                float(ranked.probe_confidence or 0.0),
                ranked.geometry_score,
                -float(ranked.geometry_features.get("area_ratio", 1.0)),
            )

        ordered = sorted(deduped, key=sort_key, reverse=True)
        for rank, item in enumerate(ordered, start=1):
            item.ranked.final_rank_before_force_include = rank

        def add(item: GlobalCandidate, *, force: bool, reason: str | None = None) -> None:
            if len(selected) >= cap or id(item.ranked) in selected_ids:
                return
            if not force:
                if per_roi_counts.get(item.source, 0) >= self.expiry_global_max_per_roi:
                    return
                if per_variant_counts.get(item.variant, 0) >= self.expiry_global_max_per_variant:
                    return
            selected.append(item)
            selected_ids.add(id(item.ranked))
            per_roi_counts[item.source] = per_roi_counts.get(item.source, 0) + 1
            per_variant_counts[item.variant] = per_variant_counts.get(item.variant, 0) + 1
            item.ranked.selected_final = True
            item.ranked.final_rank_after_force_include = len(selected)
            if force:
                item.ranked.force_included_reason = reason or self._force_include_reason(item.ranked)

        for item in ordered:
            force_reason = self._force_include_reason(item.ranked)
            if force_reason:
                add(item, force=True, reason=force_reason)
        for item in ordered:
            add(item, force=False)

        return selected, {
            "global_candidate_count": len(candidates),
            "global_deduped_candidate_count": len(deduped),
            "global_selected_candidate_count": len(selected),
            "global_final_top_k": cap,
            "global_max_per_roi": self.expiry_global_max_per_roi,
            "global_max_per_detector_variant": self.expiry_global_max_per_variant,
        }

    def _candidate_evidence_priority(self, candidate: RankedCandidate) -> int:
        text = self._candidate_signal_text(candidate)
        priority = self._text_evidence_priority(
            text,
            parser_bonus=float(candidate.score_breakdown.get("parser_probe_bonus", 0.0) or 0.0),
            expiry_score=float(candidate.score_breakdown.get("expiry_keyword_score", 0.0) or 0.0),
        )
        if priority >= 3 and candidate.candidate_type == "line":
            return 4
        return priority

    def _text_evidence_priority(self, text: str, *, parser_bonus: float = 0.0, expiry_score: float = 0.0) -> int:
        has_date = self._has_date_like_text(text) or parser_bonus > 0
        has_expiry = self._has_expiry_keyword(text) or expiry_score > 0
        has_production = self._has_production_keyword(text)
        if has_expiry and has_date:
            return 3
        if has_date and not has_production:
            return 2
        if has_date and has_production:
            return 1
        return 0

    def _force_include_reason(self, candidate: RankedCandidate) -> str | None:
        text = self._candidate_signal_text(candidate)
        has_date = self._has_date_like_text(text) or candidate.score_breakdown.get("parser_probe_bonus", 0.0) > 0
        has_expiry = self._has_expiry_keyword(text) or candidate.score_breakdown.get("expiry_keyword_score", 0.0) > 0
        if candidate.candidate_type == "line" and candidate.multiline_split_source_id and has_expiry and has_date:
            return "line_expiry_keyword_date"
        if has_expiry and has_date:
            return "expiry_keyword_date"
        if candidate.score_breakdown.get("keyword_date_group_bonus", 0.0) > 0:
            return "keyword_date_group"
        if has_date and candidate.score_breakdown.get("parser_probe_bonus", 0.0) > 0:
            return "parseable_date_probe"
        if has_date:
            return "date_like_probe"
        return None

    def _dedupe_global_candidates(self, candidates: list[GlobalCandidate]) -> list[GlobalCandidate]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                1 if item.ranked.candidate_type == "line" else 0,
                self._candidate_evidence_priority(item.ranked),
                item.ranked.total_score,
                float(item.ranked.probe_confidence or 0.0),
                item.ranked.geometry_score,
                -float(item.ranked.geometry_features.get("area_ratio", 1.0)),
            ),
            reverse=True,
        )
        kept: list[GlobalCandidate] = []
        for candidate in ordered:
            if any(self._scan_bboxes_are_duplicates(candidate.scan_bbox, existing.scan_bbox) for existing in kept):
                continue
            kept.append(candidate)
        return kept

    def _dedupe_bbox_list(self, bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        kept: list[tuple[int, int, int, int]] = []
        for bbox in sorted(bboxes, key=lambda item: ((item[2] - item[0]) * (item[3] - item[1]), item[1], item[0])):
            if any(self._scan_bboxes_are_duplicates(bbox, existing) for existing in kept):
                continue
            kept.append(bbox)
        return kept

    @classmethod
    def _scan_bboxes_are_duplicates(
        cls,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> bool:
        iou, containment = cls._bbox_overlap(a, b)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        area_ratio = max(area_a, area_b) / float(max(1, min(area_a, area_b)))
        return iou >= 0.35 or (containment >= 0.80 and area_ratio <= 4.0)

    @staticmethod
    def _bbox_overlap(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0, 0.0
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = max(1, area_a + area_b - inter)
        containment = inter / float(max(1, min(area_a, area_b)))
        return inter / float(union), containment

    def _ranked_candidate_has_strong_expiry_signal(self, candidate: RankedCandidate) -> bool:
        text = self._candidate_signal_text(candidate)
        return bool(self._has_date_like_text(text) and self._has_expiry_keyword(text))

    @staticmethod
    def _refresh_variant_debug_records(
        variants_debug: list[dict[str, object]],
        variant_debug_records: dict[str, tuple[np.ndarray, list[TextDetectionBox], list[RankedCandidate]]],
        *,
        global_selection_summary: dict[str, object] | None = None,
    ) -> None:
        _ = global_selection_summary
        for variant in variants_debug:
            variant_key = str(variant.get("variant_key", ""))
            record = variant_debug_records.get(variant_key)
            if record is None:
                continue
            _roi_image, _detected_boxes, ranked = record
            variant["final_selected_count"] = sum(1 for candidate in ranked if candidate.selected_final)
            variant["candidates"] = [candidate.to_debug_dict() for candidate in ranked]

    @staticmethod
    def _default_craft_debug() -> dict[str, object]:
        return {
            "craft_triggered": False,
            "craft_trigger_reason": [],
            "craft_runtime_ms": None,
            "craft_box_count": 0,
            "craft_accepted_box_count": 0,
            "craft_skipped_reason": None,
        }

    def _should_run_craft_rescue(
        self,
        *,
        ranked_candidates: list[RankedCandidate],
        ppocr_boxes: list[TextDetectionBox],
        variant_name: str | None,
        craft_rois_used: int,
    ) -> tuple[bool, list[str], str | None]:
        detector_mode = getattr(self.ocr_router, "text_detector_mode", "unknown")
        if detector_mode != "ensemble":
            return False, [], "mode_not_ensemble"
        if not self.craft_rescue_enabled:
            return False, [], "craft_rescue_disabled"
        if (variant_name or "raw") != self.craft_rescue_variant_name:
            return False, [], "not_rescue_variant"
        if craft_rois_used >= self.craft_max_rois_per_scan:
            return False, [], "roi_budget_exhausted"

        reasons: list[str] = []
        has_date_like = self._ranked_candidates_have_date_like_signal(ranked_candidates)
        has_expiry_keyword = self._ranked_candidates_have_expiry_keyword(ranked_candidates)
        best_score = max((candidate.total_score for candidate in ranked_candidates if candidate.selected_geometry), default=0.0)
        ppocr_pool_is_strong = has_date_like and has_expiry_keyword and best_score >= self.craft_trigger_min_best_score

        if len(ppocr_boxes) < self.craft_trigger_min_ppocr_boxes and not ppocr_pool_is_strong:
            reasons.append("ppocr_box_count_below_threshold")
        has_probe_signal = any(
            bool(candidate.probe_text or candidate.probe_normalized_text or candidate.probe_reason)
            for candidate in ranked_candidates
            if candidate.selected_geometry
        )
        if has_probe_signal and not has_date_like:
            reasons.append("no_date_like_candidate")
        if has_probe_signal and not has_expiry_keyword:
            reasons.append("no_expiry_keyword_candidate")

        if best_score < self.craft_trigger_min_best_score:
            reasons.append("best_candidate_score_below_threshold")
        if self._ranked_candidates_are_mostly_large_brand_like(ranked_candidates):
            reasons.append("mostly_large_brand_like_candidates")
        if self.high_recall_mode:
            reasons.append("high_recall_mode")

        if reasons:
            return True, reasons, None
        return False, [], "ppocr_pool_strong"

    @staticmethod
    def _candidate_signal_text(candidate: RankedCandidate) -> str:
        return " ".join(
            part.strip().upper()
            for part in (
                candidate.probe_normalized_text,
                candidate.probe_text,
                *((candidate.context_probe_texts or [])),
            )
            if part and part.strip()
        )

    def _ranked_candidates_have_date_like_signal(self, candidates: list[RankedCandidate]) -> bool:
        for candidate in candidates:
            if not candidate.selected_geometry:
                continue
            text = self._candidate_signal_text(candidate)
            if self._has_date_like_text(text) or self._score_date_likeness(text) >= 0.65:
                return True
        return False

    def _ranked_candidates_have_expiry_keyword(self, candidates: list[RankedCandidate]) -> bool:
        for candidate in candidates:
            if not candidate.selected_geometry:
                continue
            text = self._candidate_signal_text(candidate)
            if self._has_expiry_keyword(text):
                return True
        return False

    def _ranked_candidates_are_mostly_large_brand_like(self, candidates: list[RankedCandidate]) -> bool:
        top = [candidate for candidate in candidates if candidate.selected_geometry][:3]
        if not top:
            return False
        if self._ranked_candidates_have_date_like_signal(top) or self._ranked_candidates_have_expiry_keyword(top):
            return False
        large_count = sum(
            1
            for candidate in top
            if candidate.geometry_features.get("area_ratio", 0.0) >= self.craft_trigger_large_box_area_ratio
        )
        needed = 2 if len(top) >= 3 else len(top)
        return large_count >= needed

    def _run_craft_rescue(
        self,
        image: np.ndarray,
        *,
        variant_name: str,
    ) -> tuple[list[TextDetectionBox], str | None, int, bool]:
        detect_craft = getattr(self.ocr_router, "detect_craft_text_boxes", None)
        if not callable(detect_craft):
            return [], "craft detector unavailable", 0, False

        start = perf_counter()
        try:
            try:
                boxes, reason = detect_craft(
                    image,
                    variant_name=variant_name,
                    canvas_size=self.craft_rescue_canvas_size,
                    mag_ratio=self.craft_rescue_mag_ratio,
                )
            except TypeError as exc:
                if "canvas_size" not in str(exc) and "mag_ratio" not in str(exc):
                    raise
                boxes, reason = detect_craft(image, variant_name=variant_name)
        except Exception as exc:
            runtime_ms = int((perf_counter() - start) * 1000)
            return [], f"craft detector unavailable: {exc}", runtime_ms, False

        runtime_ms = int((perf_counter() - start) * 1000)
        timed_out = runtime_ms > int(self.craft_timeout_seconds * 1000)
        normalized = [
            TextDetectionBox(
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                confidence=box.confidence,
                source=box.source,
                sources=box.sources,
                variant_name=box.variant_name or variant_name,
                polygon_xy=box.polygon_xy,
            )
            for box in boxes
        ]
        return normalized, reason, runtime_ms, timed_out

    def _cap_craft_boxes(self, boxes: list[TextDetectionBox]) -> list[TextDetectionBox]:
        valid = [box for box in boxes if box.x2 > box.x1 and box.y2 > box.y1]
        with_conf = [box for box in valid if box.confidence is not None]
        without_conf = [box for box in valid if box.confidence is None]
        with_conf.sort(key=lambda box: float(box.confidence or 0.0), reverse=True)
        without_conf.sort(key=lambda box: (max(1, box.x2 - box.x1) * max(1, box.y2 - box.y1), box.y1, box.x1))
        return [*with_conf, *without_conf][: self.craft_max_boxes_accepted]

    def _merge_text_boxes(
        self,
        ppocr_boxes: list[TextDetectionBox],
        craft_boxes: list[TextDetectionBox],
    ) -> list[TextDetectionBox]:
        merge_text_boxes = getattr(self.ocr_router, "merge_text_boxes", None)
        if callable(merge_text_boxes):
            return merge_text_boxes(ppocr_boxes, craft_boxes)
        return [*ppocr_boxes, *craft_boxes]

    @staticmethod
    def _reason_is_detector_unavailable(reason: str | None) -> bool:
        if not reason:
            return False
        reason_lower = reason.lower()
        return any(token in reason_lower for token in ("unavailable", "disabled", "not found", "import failed"))

    @staticmethod
    def _map_text_boxes_to_original(
        *,
        boxes: list[TextDetectionBox],
        variant: ImageVariant,
        original_shape: tuple[int, ...],
    ) -> list[TextDetectionBox]:
        if not boxes:
            return []
        h = int(original_shape[0])
        w = int(original_shape[1])
        scale_x = variant.scale_x if variant.scale_x > 0 else 1.0
        scale_y = variant.scale_y if variant.scale_y > 0 else 1.0
        mapped: list[TextDetectionBox] = []
        for box in boxes:
            x1 = max(0, min(int(np.floor(box.x1 / scale_x)), w - 1))
            y1 = max(0, min(int(np.floor(box.y1 / scale_y)), h - 1))
            x2 = max(0, min(int(np.ceil(box.x2 / scale_x)), w))
            y2 = max(0, min(int(np.ceil(box.y2 / scale_y)), h))
            if x2 <= x1 or y2 <= y1:
                continue
            polygon_xy = None
            if box.polygon_xy is not None:
                mapped_points: list[tuple[float, float]] = []
                for px, py in box.polygon_xy:
                    mapped_points.append(
                        (
                            max(0.0, min(float(px) / scale_x, float(w))),
                            max(0.0, min(float(py) / scale_y, float(h))),
                        )
                    )
                polygon_xy = tuple(mapped_points)
            mapped.append(
                TextDetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=box.confidence,
                    source=box.source,
                    sources=box.sources,
                    variant_name=box.variant_name or variant.name,
                    polygon_xy=polygon_xy,
                )
            )
        return mapped

    @staticmethod
    def _detector_counts(boxes: list[TextDetectionBox]) -> dict[str, int]:
        counts = {"ppocrv5_server": 0, "craft": 0, "ensemble": 0}
        for box in boxes:
            for source in box.sources:
                counts[source] = counts.get(source, 0) + 1
            if box.source == "ensemble":
                counts["ensemble"] = counts.get("ensemble", 0) + 1
        return counts

    @staticmethod
    def _detector_box_to_debug_dict(
        box: TextDetectionBox,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> dict[str, object]:
        roi_bbox = [float(value) for value in box.bbox_xyxy]
        scan_bbox = [float(box.x1 + offset_x), float(box.y1 + offset_y), float(box.x2 + offset_x), float(box.y2 + offset_y)]
        roi_polygon = (
            [[float(point[0]), float(point[1])] for point in box.polygon_xy]
            if box.polygon_xy
            else None
        )
        scan_polygon = (
            [[float(point[0]) + offset_x, float(point[1]) + offset_y] for point in box.polygon_xy]
            if box.polygon_xy
            else None
        )
        return {
            "bbox_xyxy": scan_bbox,
            "roi_bbox_xyxy": roi_bbox,
            "polygon_xy": scan_polygon,
            "roi_polygon_xy": roi_polygon,
            "confidence": float(box.confidence) if box.confidence is not None else None,
            "source": box.source,
            "sources": list(box.sources),
            "variant_name": box.variant_name,
        }

    @staticmethod
    def _detector_unavailable_reasons(reason: str | None) -> list[str]:
        if not reason:
            return []
        parts = [part.strip() for part in reason.split(";") if part.strip()]
        return [
            part
            for part in parts
            if any(token in part.lower() for token in ("unavailable", "disabled", "not found", "import failed"))
        ]

    def _recognition_variants_for_crop(
        self,
        crop: np.ndarray,
        *,
        allowed_names: tuple[str, ...] | None = None,
        performance: dict[str, object] | None = None,
    ) -> list[ImageVariant]:
        recognition_variants = getattr(self.preprocessor, "recognition_variants", None)
        if callable(recognition_variants):
            try:
                variants = list(recognition_variants(crop, allowed_names=allowed_names))
            except TypeError:
                variants = list(recognition_variants(crop))
            if variants:
                if performance is not None:
                    performance["recognition_variants_generated"] = int(performance["recognition_variants_generated"]) + len(variants)
                filtered = self._filter_recognition_variants(variants, allowed_names=allowed_names)
                if filtered:
                    return filtered
        fallback = [ImageVariant(name="original_padded", image=crop, purpose="recognition")]
        if performance is not None:
            performance["recognition_variants_generated"] = int(performance["recognition_variants_generated"]) + 1
        return self._filter_recognition_variants(fallback, allowed_names=allowed_names) or fallback

    @staticmethod
    def _filter_recognition_variants(
        variants: list[ImageVariant],
        *,
        allowed_names: tuple[str, ...] | None,
    ) -> list[ImageVariant]:
        if not allowed_names:
            return variants
        allowed = set(allowed_names)
        selected = [variant for variant in variants if variant.name in allowed]
        if selected:
            return selected
        original = [variant for variant in variants if variant.name == "original_padded"]
        return original or variants[:1]

    def _final_parseq_crop_for_candidate(self, candidate: RankedCandidate, image: np.ndarray) -> FinalParseqCrop:
        tight_bbox, policy = self._final_tight_bbox_for_candidate(candidate)
        padded_bbox, padding_px = self._pad_bbox_for_final_crop(tight_bbox, image.shape)
        x1, y1, x2, y2 = padded_bbox
        polygon_xy = candidate.polygon_xy if tight_bbox == candidate.bbox else None
        return FinalParseqCrop(
            image=image[y1:y2, x1:x2],
            bbox=padded_bbox,
            padding_px=padding_px,
            policy=policy,
            polygon_xy=polygon_xy,
        )

    def _normalized_final_crops_for_candidate(
        self,
        candidate: RankedCandidate,
        image: np.ndarray,
    ) -> tuple[FinalParseqCrop, list[NormalizedTextLineCrop]]:
        final_crop = self._final_parseq_crop_for_candidate(candidate, image)
        geometry = TextLineGeometry(
            bbox_xyxy=final_crop.bbox,
            polygon_xy=final_crop.polygon_xy,
        )
        config = TextLineCropConfig(
            padding_px=final_crop.padding_px if final_crop.polygon_xy is not None else 0,
            vertical_aspect_threshold=1.5,
        )
        return final_crop, normalize_textline_crops(image, geometry, config)

    def _final_tight_bbox_for_candidate(self, candidate: RankedCandidate) -> tuple[tuple[int, int, int, int], str]:
        if candidate.recognition_bbox is not None and candidate.recognition_bbox != candidate.bbox:
            return candidate.recognition_bbox, "recognition_bbox_tight"

        if candidate.candidate_type == "line":
            line_boxes = self._same_line_member_bboxes(candidate.member_bboxes)
            if line_boxes:
                return self._union_many_bboxes(line_boxes), "line_tight"
            return candidate.bbox, "line_tight"

        if (
            candidate.candidate_type in {"group", "fallback"}
            and candidate.member_bboxes
            and not self.expiry_debug_parseq_whole_groups
        ):
            lines = self._cluster_bboxes_into_lines(candidate.member_bboxes)
            if len(lines) > 1:
                selected = self._select_best_final_crop_line(candidate, lines)
                return self._union_many_bboxes(selected), "line_from_multiline_group"
            if len(lines) == 1 and self._bboxes_share_text_line(lines[0]):
                return self._union_many_bboxes(lines[0]), "same_line_group"

        return candidate.bbox, f"{candidate.candidate_type}_tight"

    def _pad_bbox_for_final_crop(
        self,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, ...],
    ) -> tuple[tuple[int, int, int, int], int]:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        if self.expiry_final_crop_max_padding_px <= 0:
            padding = 0
        else:
            padding = int(round(max(bw, bh) * self.expiry_final_crop_padding_ratio))
            padding = max(self.expiry_final_crop_min_padding_px, padding)
            padding = min(self.expiry_final_crop_max_padding_px, padding)
        return (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(w, x2 + padding),
            min(h, y2 + padding),
        ), padding

    @staticmethod
    def _same_line_member_bboxes(bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        if not bboxes:
            return []
        if ExpiryPipeline._bboxes_share_text_line(bboxes):
            return bboxes
        lines = ExpiryPipeline._cluster_bboxes_into_lines(bboxes)
        return max(lines, key=lambda line: (len(line), ExpiryPipeline._bbox_area(ExpiryPipeline._union_many_bboxes(line)))) if lines else []

    @staticmethod
    def _cluster_bboxes_into_lines(bboxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
        ordered = sorted(bboxes, key=lambda box: (((box[1] + box[3]) / 2.0), box[0]))
        lines: list[list[tuple[int, int, int, int]]] = []
        for box in ordered:
            cy = (box[1] + box[3]) / 2.0
            height = max(1, box[3] - box[1])
            placed = False
            for line in lines:
                line_cy = sum((item[1] + item[3]) / 2.0 for item in line) / len(line)
                line_height = sum(max(1, item[3] - item[1]) for item in line) / len(line)
                if abs(cy - line_cy) <= max(height, line_height) * 0.65:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])
        return [sorted(line, key=lambda box: box[0]) for line in lines]

    @staticmethod
    def _bboxes_share_text_line(bboxes: list[tuple[int, int, int, int]]) -> bool:
        if len(bboxes) <= 1:
            return True
        base = bboxes[0]
        for other in bboxes[1:]:
            base_h = max(1, base[3] - base[1])
            other_h = max(1, other[3] - other[1])
            y_overlap = max(0, min(base[3], other[3]) - max(base[1], other[1]))
            base_cy = (base[1] + base[3]) / 2.0
            other_cy = (other[1] + other[3]) / 2.0
            if y_overlap < min(base_h, other_h) * 0.35 and abs(base_cy - other_cy) > max(base_h, other_h) * 0.65:
                return False
        return True

    def _select_best_final_crop_line(
        self,
        candidate: RankedCandidate,
        lines: list[list[tuple[int, int, int, int]]],
    ) -> list[tuple[int, int, int, int]]:
        probe_lines = [line.strip() for line in re.split(r"[\r\n]+", candidate.probe_text or "") if line.strip()]

        def score(index: int, line: list[tuple[int, int, int, int]]) -> tuple[int, float, float]:
            text = probe_lines[index] if index < len(probe_lines) else ""
            evidence = self._text_evidence_priority(text)
            if self._has_production_keyword(text) and not self._has_expiry_keyword(text):
                evidence -= 1
            bbox = self._union_many_bboxes(line)
            aspect = (bbox[2] - bbox[0]) / float(max(1, bbox[3] - bbox[1]))
            return evidence, min(aspect, 12.0), -float(self._bbox_area(bbox))

        indexed = list(enumerate(lines))
        _, selected = max(indexed, key=lambda item: score(item[0], item[1]))
        return selected

    def _best_probe_for_crop(
        self,
        crop: np.ndarray,
        *,
        today: date,
        performance: dict[str, object] | None = None,
    ) -> tuple[RecognitionData, str, np.ndarray]:
        best: RecognitionData | None = None
        best_variant = "original_padded"
        best_variant_image = crop
        best_score: tuple[float, float, float, float] | None = None
        allowed_names = None if self.expiry_debug_all_recognition_variants else self.expiry_probe_variants
        for variant in self._recognition_variants_for_crop(crop, allowed_names=allowed_names, performance=performance):
            t_probe_call = perf_counter()
            probe = self.ocr_router.recognize_probe(variant.image)
            probe_call_ms = int((perf_counter() - t_probe_call) * 1000)
            if performance is not None:
                performance["mobile_probe_call_count"] = int(performance["mobile_probe_call_count"]) + 1
                performance["mobile_probe_total_ms"] = int(performance["mobile_probe_total_ms"]) + probe_call_ms
            scores = self._probe_signal_scores(probe, today=today)
            parseable_score = 0.0
            if probe.raw_text.strip():
                ocr = OCRResultData(
                    raw_text=probe.raw_text,
                    normalized_text=probe.normalized_text,
                    confidence=probe.confidence,
                    engine_name=getattr(self.ocr_router, "probe_engine_name", "probe"),
                    runtime_device=getattr(self.ocr_router, "parseq_runtime_device", None),
                    reason=probe.reason,
                )
                parse_inputs = self._build_parse_inputs(ocr)
                parsed = self._best_parse_for_inputs(parse_inputs, today=today)
                parseable_score = float(parsed.confidence if parsed.parsed_date is not None else 0.0)
            score = (
                float(probe.confidence or 0.0),
                float(self._score_date_likeness(probe.raw_text)),
                parseable_score,
                float(sum(scores.values())),
            )
            if best_score is None or score > best_score:
                best = probe
                best_variant = variant.name
                best_variant_image = variant.image
                best_score = score
        if best is None:
            return RecognitionData(raw_text="", normalized_text="", confidence=None, reason="no recognition variants"), best_variant, best_variant_image
        return best, best_variant, best_variant_image

    def _best_parseq_for_crop(
        self,
        crops: list[NormalizedTextLineCrop],
        *,
        today: date,
        selected_variant_name: str | None = None,
        selected_variant_image: np.ndarray | None = None,
        performance: dict[str, object] | None = None,
    ) -> tuple[RecognitionData, ParsedDateData, int, str, list[int] | None, int, bool, NormalizedTextLineCrop]:
        best: tuple[
            tuple[float, float, float],
            RecognitionData,
            ParsedDateData,
            int,
            str,
            list[int] | None,
            int,
            bool,
            NormalizedTextLineCrop,
        ] | None = None
        total_parse_ms = 0
        for idx, normalized_crop in enumerate(crops):
            if normalized_crop.image.size == 0:
                continue
            (
                parseq,
                parsed,
                parse_inputs_count,
                recognition_variant,
                input_shape,
                parse_ms,
                original_color_fallback_used,
            ) = self._best_parseq_for_single_crop(
                normalized_crop.image,
                today=today,
                selected_variant_name=selected_variant_name,
                selected_variant_image=selected_variant_image if idx == 0 else None,
                performance=performance,
            )
            total_parse_ms += parse_ms
            score = (
                float(parsed.confidence if parsed.parsed_date is not None else 0.0),
                float(self._score_date_likeness(parseq.raw_text)),
                float(parseq.confidence or 0.0),
            )
            item = (
                score,
                parseq,
                parsed,
                parse_inputs_count,
                recognition_variant,
                input_shape,
                total_parse_ms,
                original_color_fallback_used,
                normalized_crop,
            )
            if best is None or score > best[0]:
                best = item
            if parsed.parsed_date is not None and parsed.confidence >= 0.9 and (parseq.confidence or 0.0) >= 0.85:
                break

        if best is not None and self._should_try_rotate_180_fallback(crops, best[1], best[2]):
            rotated_crop = self._rotate_180_normalized_crop(crops[0])
            (
                parseq,
                parsed,
                parse_inputs_count,
                recognition_variant,
                input_shape,
                parse_ms,
                original_color_fallback_used,
            ) = self._best_parseq_for_single_crop(
                rotated_crop.image,
                today=today,
                selected_variant_name=selected_variant_name,
                selected_variant_image=None,
                performance=performance,
            )
            total_parse_ms += parse_ms
            score = (
                float(parsed.confidence if parsed.parsed_date is not None else 0.0),
                float(self._score_date_likeness(parseq.raw_text)),
                float(parseq.confidence or 0.0),
            )
            item = (
                score,
                parseq,
                parsed,
                parse_inputs_count,
                recognition_variant,
                input_shape,
                total_parse_ms,
                original_color_fallback_used,
                rotated_crop,
            )
            if score > best[0]:
                best = item

        if best is None:
            empty_crop = crops[0] if crops else NormalizedTextLineCrop(
                image=np.zeros((1, 1, 3), dtype=np.uint8),
                bbox_xyxy=(0, 0, 1, 1),
                polygon_used=False,
                crop_transform_used="bbox_raw",
                selected_orientation="original",
                original_crop_shape=[1, 1, 3],
                normalized_crop_shape=[1, 1, 3],
                orientation_candidates_tried=["original"],
                selected_transform_reason="empty",
            )
            empty = RecognitionData(raw_text="", normalized_text="", confidence=None, reason="no normalized crops")
            parsed = self.parser.parse("", reference_date=today)
            return empty, parsed, 1, "original_padded", None, 0, False, empty_crop

        _, parseq, parsed, parse_inputs_count, recognition_variant, input_shape, _parse_ms, fallback, normalized_crop = best
        return parseq, parsed, parse_inputs_count, recognition_variant, input_shape, total_parse_ms, fallback, normalized_crop

    def _should_try_rotate_180_fallback(
        self,
        crops: list[NormalizedTextLineCrop],
        parseq: RecognitionData,
        parsed: ParsedDateData,
    ) -> bool:
        if self.expiry_debug_all_recognition_variants:
            return False
        if len(crops) != 1:
            return False
        crop = crops[0]
        if crop.selected_orientation != "original" or "rotate_180" in crop.orientation_candidates_tried:
            return False
        return self._parseq_result_needs_original_fallback(parseq, parsed)

    def _rotate_180_normalized_crop(self, crop: NormalizedTextLineCrop) -> NormalizedTextLineCrop:
        image = cv2.rotate(crop.image, cv2.ROTATE_180) if crop.image.size else crop.image
        return NormalizedTextLineCrop(
            image=image,
            bbox_xyxy=crop.bbox_xyxy,
            polygon_used=crop.polygon_used,
            crop_transform_used="rotate_180",
            selected_orientation="rotate_180",
            original_crop_shape=crop.original_crop_shape,
            normalized_crop_shape=[
                int(image.shape[0]),
                int(image.shape[1]),
                int(image.shape[2]) if image.ndim == 3 else 1,
            ],
            orientation_candidates_tried=["original", "rotate_180"],
            selected_transform_reason="original_failed_final_recognition",
        )

    def _best_parseq_for_single_crop(
        self,
        crop: np.ndarray,
        *,
        today: date,
        selected_variant_name: str | None = None,
        selected_variant_image: np.ndarray | None = None,
        performance: dict[str, object] | None = None,
    ) -> tuple[RecognitionData, ParsedDateData, int, str, list[int] | None, int, bool]:
        best_parseable: tuple[tuple[float, float, float], RecognitionData, ParsedDateData, int, str, list[int]] | None = None
        best_with_text: tuple[tuple[float, float, float], RecognitionData, ParsedDateData, int, str, list[int]] | None = None
        best_empty: tuple[RecognitionData, ParsedDateData, int, str, list[int]] | None = None
        parse_ms = 0

        def run_variant(image: np.ndarray, name: str) -> tuple[RecognitionData, ParsedDateData, int, str, list[int]]:
            nonlocal parse_ms
            t_parseq_call = perf_counter()
            recognize_final = getattr(self.ocr_router, "recognize_final", None)
            parseq = recognize_final(image) if callable(recognize_final) else self.ocr_router.recognize_parseq(image)
            parseq_call_ms = int((perf_counter() - t_parseq_call) * 1000)
            if performance is not None:
                performance["parseq_call_count"] = int(performance["parseq_call_count"]) + 1
                performance["parseq_total_ms"] = int(performance["parseq_total_ms"]) + parseq_call_ms
                performance["final_recognition_call_count"] = int(performance["final_recognition_call_count"]) + 1
                performance["final_recognition_total_ms"] = int(performance["final_recognition_total_ms"]) + parseq_call_ms
            ocr = OCRResultData(
                raw_text=parseq.raw_text,
                normalized_text=parseq.normalized_text,
                confidence=parseq.confidence,
                engine_name=self.ocr_router.active_engine_name,
                runtime_device=getattr(
                    self.ocr_router,
                    "final_recognition_runtime_device",
                    getattr(self.ocr_router, "parseq_runtime_device", None),
                ),
                reason=parseq.reason,
            )
            t_parse = perf_counter()
            parse_inputs = self._build_parse_inputs(ocr)
            parsed = self._best_parse_for_inputs(parse_inputs, today=today)
            parse_ms += int((perf_counter() - t_parse) * 1000)
            input_shape = [
                int(image.shape[0]),
                int(image.shape[1]),
                int(image.shape[2]) if image.ndim == 3 else 1,
            ]
            return parseq, parsed, len(parse_inputs), name, input_shape

        def maybe_with_original_fallback(
            parseq: RecognitionData,
            parsed: ParsedDateData,
            parse_inputs_count: int,
            variant_name: str,
            input_shape: list[int],
        ) -> tuple[RecognitionData, ParsedDateData, int, str, list[int], bool]:
            if (
                self.expiry_debug_all_recognition_variants
                or variant_name in {"original_color", "original_padded"}
                or not self._parseq_result_needs_original_fallback(parseq, parsed)
            ):
                return parseq, parsed, parse_inputs_count, variant_name, input_shape, False

            original_parseq, original_parsed, original_inputs_count, original_name, original_shape = run_variant(
                crop,
                "original_color",
            )
            if original_parsed.parsed_date is not None or (
                not parseq.raw_text.strip() and original_parseq.raw_text.strip()
            ):
                return original_parseq, original_parsed, original_inputs_count, original_name, original_shape, True
            return parseq, parsed, parse_inputs_count, variant_name, input_shape, True

        allowed_names: tuple[str, ...] | None = None
        if not self.expiry_debug_all_recognition_variants and self.expiry_parseq_variant_policy == "best_probe":
            allowed_names = (selected_variant_name or "original_padded",)
        if (
            selected_variant_image is not None
            and not self.expiry_debug_all_recognition_variants
            and self.expiry_parseq_variant_policy == "best_probe"
        ):
            variants = [
                ImageVariant(
                    name=selected_variant_name or "original_padded",
                    image=selected_variant_image,
                    purpose="recognition",
                )
            ]
        elif (
            not self.expiry_debug_all_recognition_variants
            and self.expiry_parseq_variant_policy == "best_probe"
            and (selected_variant_name or "original_padded") == "original_padded"
        ):
            variants = [ImageVariant(name="original_padded", image=crop, purpose="recognition")]
        else:
            variants = self._recognition_variants_for_crop(crop, allowed_names=allowed_names, performance=performance)
        if not self.expiry_debug_all_recognition_variants:
            variants = variants[: self.expiry_parseq_max_variants_per_candidate]

        for variant in variants:
            parseq, parsed, parse_inputs_count, variant_name, input_shape = run_variant(variant.image, variant.name)

            if parsed.parsed_date is not None:
                score = (
                    float(parsed.confidence),
                    float(parseq.confidence or 0.0),
                    float(self._score_date_likeness(parseq.raw_text)),
                )
                item = (score, parseq, parsed, parse_inputs_count, variant_name, input_shape)
                if best_parseable is None or score > best_parseable[0]:
                    best_parseable = item
                if parsed.confidence >= 0.9 and (parseq.confidence or 0.0) >= 0.85:
                    final = maybe_with_original_fallback(parseq, parsed, parse_inputs_count, variant_name, input_shape)
                    return (*final[:5], parse_ms, final[5])
                continue

            if parseq.raw_text.strip():
                ocr = OCRResultData(
                    raw_text=parseq.raw_text,
                    normalized_text=parseq.normalized_text,
                    confidence=parseq.confidence,
                    engine_name=self.ocr_router.active_engine_name,
                    runtime_device=self.ocr_router.parseq_runtime_device,
                    reason=parseq.reason,
                )
                score = (
                    float(self._score_ocr_candidate(ocr)),
                    float(parseq.confidence or 0.0),
                    float(self._score_date_likeness(parseq.raw_text)),
                )
                item = (score, parseq, parsed, parse_inputs_count, variant_name, input_shape)
                if best_with_text is None or score > best_with_text[0]:
                    best_with_text = item
            elif best_empty is None:
                best_empty = (parseq, parsed, parse_inputs_count, variant_name, input_shape)

        if best_parseable is not None:
            _, parseq, parsed, parse_inputs_count, variant_name, input_shape = best_parseable
            final = maybe_with_original_fallback(parseq, parsed, parse_inputs_count, variant_name, input_shape)
            return (*final[:5], parse_ms, final[5])
        if best_with_text is not None:
            _, parseq, parsed, parse_inputs_count, variant_name, input_shape = best_with_text
            final = maybe_with_original_fallback(parseq, parsed, parse_inputs_count, variant_name, input_shape)
            return (*final[:5], parse_ms, final[5])
        if best_empty is not None:
            parseq, parsed, parse_inputs_count, variant_name, input_shape = best_empty
            final = maybe_with_original_fallback(parseq, parsed, parse_inputs_count, variant_name, input_shape)
            return (*final[:5], parse_ms, final[5])

        empty = RecognitionData(raw_text="", normalized_text="", confidence=None, reason="no recognition variants")
        parsed = self.parser.parse("", reference_date=today)
        return empty, parsed, 1, "original_padded", None, parse_ms, False

    def _parseq_result_needs_original_fallback(self, parseq: RecognitionData, parsed: ParsedDateData) -> bool:
        text = parseq.raw_text or parseq.normalized_text or ""
        if parsed.parsed_date is None:
            return True
        return self._is_generic_brand_text(text)

    def _build_ranked_candidates(self, image: np.ndarray, boxes: list[TextDetectionBox]) -> list[RankedCandidate]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []

        if not boxes:
            area_ratio = 1.0
            return [
                RankedCandidate(
                    candidate_id="fallback_full",
                    candidate_type="fallback",
                    bbox=(0, 0, w, h),
                    member_indices=[],
                    member_bboxes=[],
                    detector_sources=("fallback",),
                    detector_variant=None,
                    geometry_features={
                        "area_ratio": area_ratio,
                        "aspect_ratio": float(w) / float(max(h, 1)),
                        "line_likeness": 0.15,
                        "edge_proximity": 0.0,
                        "neighbor_count": 0.0,
                    },
                    geometry_score=0.05,
                    score_breakdown={"fallback_bonus": 0.05},
                    total_score=0.05,
                )
            ]

        neighbor_counts = self._neighbor_counts(boxes)

        candidates: list[RankedCandidate] = []
        seen: set[tuple[int, int, int, int, tuple[int, ...]]] = set()

        for idx, box in enumerate(boxes):
            bbox = (box.x1, box.y1, box.x2, box.y2)
            candidate = self._candidate_from_bbox(
                bbox=bbox,
                member_indices=[idx],
                candidate_type="single",
                image_shape=(h, w),
                boxes=boxes,
                neighbor_counts=neighbor_counts,
            )
            key = (*bbox, tuple(candidate.member_indices))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

        line_clusters = self._cluster_boxes_into_lines(boxes)
        if len(line_clusters) > 1:
            multiline_source_id = "multiline_block_1"
            for line_index, member_indices in enumerate(line_clusters):
                if len(member_indices) < 2:
                    continue
                bbox = self._union_many_bboxes(
                    [(boxes[idx].x1, boxes[idx].y1, boxes[idx].x2, boxes[idx].y2) for idx in member_indices]
                )
                candidate = self._candidate_from_bbox(
                    bbox=bbox,
                    member_indices=member_indices,
                    candidate_type="line",
                    image_shape=(h, w),
                    boxes=boxes,
                    neighbor_counts=neighbor_counts,
                )
                candidate.multiline_split_source_id = multiline_source_id
                candidate.line_index_in_group = line_index
                key = (*bbox, tuple(candidate.member_indices))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

        group_count = 0
        for i in range(len(boxes)):
            if self.max_group_candidates_per_roi <= 0 or group_count >= self.max_group_candidates_per_roi:
                break
            if not self._box_has_grouping_geometry(boxes[i], image_shape=(h, w)):
                continue
            neighbors = [
                (
                    self._box_center_distance(boxes[i], boxes[j]),
                    j,
                )
                for j in range(len(boxes))
                if j != i and self._box_has_grouping_geometry(boxes[j], image_shape=(h, w))
            ]
            neighbors.sort(key=lambda item: item[0])
            for _distance, j in neighbors[:2]:
                if i >= j:
                    continue
                if not self._boxes_are_groupable(boxes[i], boxes[j]):
                    continue
                bbox = self._union_bbox(
                    (boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2),
                    (boxes[j].x1, boxes[j].y1, boxes[j].x2, boxes[j].y2),
                )
                candidate = self._candidate_from_bbox(
                    bbox=bbox,
                    member_indices=[i, j],
                    candidate_type="group",
                    image_shape=(h, w),
                    boxes=boxes,
                    neighbor_counts=neighbor_counts,
                )
                key = (*bbox, tuple(candidate.member_indices))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                group_count += 1
                if group_count >= self.max_group_candidates_per_roi:
                    break

        candidates.sort(key=lambda c: c.geometry_score, reverse=True)
        for idx, candidate in enumerate(candidates, start=1):
            candidate.candidate_id = f"cand_{idx}"
        return candidates

    @staticmethod
    def _cluster_boxes_into_lines(boxes: list[TextDetectionBox]) -> list[list[int]]:
        if not boxes:
            return []
        ordered = sorted(range(len(boxes)), key=lambda idx: ((boxes[idx].y1 + boxes[idx].y2) / 2.0, boxes[idx].x1))
        lines: list[list[int]] = []
        for idx in ordered:
            box = boxes[idx]
            cy = (box.y1 + box.y2) / 2.0
            height = max(1, box.y2 - box.y1)
            placed = False
            for line in lines:
                line_boxes = [boxes[item] for item in line]
                line_cy = sum((item.y1 + item.y2) / 2.0 for item in line_boxes) / len(line_boxes)
                line_height = sum(max(1, item.y2 - item.y1) for item in line_boxes) / len(line_boxes)
                y_overlap = max(0, min(max(item.y2 for item in line_boxes), box.y2) - max(min(item.y1 for item in line_boxes), box.y1))
                if abs(cy - line_cy) <= max(height, line_height) * 0.65 or y_overlap >= min(height, line_height) * 0.35:
                    line.append(idx)
                    placed = True
                    break
            if not placed:
                lines.append([idx])
        return [sorted(line, key=lambda idx: boxes[idx].x1) for line in lines]

    @staticmethod
    def _box_center_distance(a: TextDetectionBox, b: TextDetectionBox) -> float:
        return float(np.hypot(((a.x1 + a.x2) / 2.0) - ((b.x1 + b.x2) / 2.0), ((a.y1 + a.y2) / 2.0) - ((b.y1 + b.y2) / 2.0)))

    @staticmethod
    def _box_has_grouping_geometry(box: TextDetectionBox, *, image_shape: tuple[int, int]) -> bool:
        h, w = image_shape
        bw = max(1, box.x2 - box.x1)
        bh = max(1, box.y2 - box.y1)
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        return bool(0.0003 <= area_ratio <= 0.18 and 0.35 <= aspect <= 14.0)

    def _apply_geometry_shortlist(self, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        if not candidates:
            return []
        primary = candidates[: self.expiry_filter_geometry_top_n]
        cutoff = primary[-1].geometry_score if primary else candidates[0].geometry_score
        secondary_limit = max(self.expiry_filter_geometry_top_n, self.expiry_global_probe_top_k * 4)
        top = set(id(candidate) for candidate in primary)
        for candidate in candidates[:secondary_limit]:
            if candidate.candidate_type == "line":
                top.add(id(candidate))
                continue
            if candidate.geometry_score >= cutoff - 0.30 and candidate.geometry_features.get("area_ratio", 1.0) <= 0.16:
                top.add(id(candidate))
        for candidate in candidates:
            candidate.selected_geometry = id(candidate) in top
            if not candidate.selected_geometry:
                candidate.total_score = candidate.geometry_score
        return candidates

    def _select_final_candidates(self, candidates: list[RankedCandidate]) -> list[RankedCandidate]:
        shortlisted = [candidate for candidate in candidates if candidate.selected_geometry]
        if not shortlisted:
            return []

        shortlisted.sort(key=lambda c: c.total_score, reverse=True)
        selected = shortlisted[: self.expiry_filter_final_top_k]
        selected_ids = {id(candidate) for candidate in selected}
        for candidate in shortlisted:
            candidate.selected_final = id(candidate) in selected_ids
        return selected

    @staticmethod
    def _neighbor_counts(boxes: list[TextDetectionBox]) -> dict[int, int]:
        counts = {idx: 0 for idx in range(len(boxes))}
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if ExpiryPipeline._boxes_are_groupable(boxes[i], boxes[j]):
                    counts[i] += 1
                    counts[j] += 1
        return counts

    @staticmethod
    def _boxes_are_groupable(a: TextDetectionBox, b: TextDetectionBox) -> bool:
        aw, ah = max(1, a.x2 - a.x1), max(1, a.y2 - a.y1)
        bw, bh = max(1, b.x2 - b.x1), max(1, b.y2 - b.y1)

        a_cx = (a.x1 + a.x2) / 2.0
        a_cy = (a.y1 + a.y2) / 2.0
        b_cx = (b.x1 + b.x2) / 2.0
        b_cy = (b.y1 + b.y2) / 2.0

        center_distance = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
        diag_ref = max(np.hypot(aw, ah), np.hypot(bw, bh))

        x_overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        y_overlap = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))

        x_gap = max(0, max(a.x1, b.x1) - min(a.x2, b.x2))
        y_gap = max(0, max(a.y1, b.y1) - min(a.y2, b.y2))

        close_centers = center_distance <= (diag_ref * 2.4)
        shared_line = y_overlap >= min(ah, bh) * 0.25
        stacked_line = x_overlap >= min(aw, bw) * 0.25
        close_gaps = x_gap <= max(aw, bw) * 1.5 and y_gap <= max(ah, bh) * 1.5
        return bool(close_centers and (shared_line or stacked_line or close_gaps))

    @staticmethod
    def _bbox_vertical_overlap_ratio(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return float(overlap) / float(max(1, min(a[3] - a[1], b[3] - b[1])))

    @staticmethod
    def _bbox_horizontal_overlap_ratio(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        return float(overlap) / float(max(1, min(a[2] - a[0], b[2] - b[0])))

    @classmethod
    def _context_relation(
        cls,
        candidate_bbox: tuple[int, int, int, int],
        context_bbox: tuple[int, int, int, int],
    ) -> str | None:
        cx1, cy1, cx2, cy2 = candidate_bbox
        bx1, by1, bx2, by2 = context_bbox
        candidate_h = max(1, cy2 - cy1)
        context_h = max(1, by2 - by1)
        ref_h = max(candidate_h, context_h)

        vertical_overlap = cls._bbox_vertical_overlap_ratio(candidate_bbox, context_bbox)
        if bx2 <= cx1 and vertical_overlap >= 0.45:
            gap = cx1 - bx2
            if gap <= 4 * ref_h:
                return "left_same_line"

        if by2 <= cy1:
            gap = cy1 - by2
            horizontal_overlap = cls._bbox_horizontal_overlap_ratio(candidate_bbox, context_bbox)
            candidate_cx = (cx1 + cx2) / 2.0
            context_cx = (bx1 + bx2) / 2.0
            center_aligned = abs(candidate_cx - context_cx) <= max(cx2 - cx1, bx2 - bx1)
            if gap <= 2 * ref_h and (horizontal_overlap >= 0.25 or center_aligned):
                return "above_aligned"

        return None

    def _context_probe_matches(
        self,
        *,
        candidate_bbox: tuple[int, int, int, int],
        member_indices: list[int],
        boxes: list[TextDetectionBox],
    ) -> list[tuple[int, tuple[int, int, int, int], str]]:
        if not self.context_probe_enabled or self.context_max_boxes_per_candidate <= 0:
            return []
        excluded = set(member_indices)
        matches: list[tuple[tuple[int, int, float], int, tuple[int, int, int, int], str]] = []
        cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
        cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
        for idx, box in enumerate(boxes):
            if idx in excluded:
                continue
            bbox = (box.x1, box.y1, box.x2, box.y2)
            relation = self._context_relation(candidate_bbox, bbox)
            if relation is None:
                continue
            bx = (box.x1 + box.x2) / 2.0
            by = (box.y1 + box.y2) / 2.0
            distance = float(np.hypot(cx - bx, cy - by))
            priority = 0 if relation == "left_same_line" else 1
            matches.append(((priority, int(distance), -float(box.confidence or 0.0)), idx, bbox, relation))
        matches.sort(key=lambda item: item[0])
        return [(idx, bbox, relation) for _key, idx, bbox, relation in matches[: self.context_max_boxes_per_candidate]]

    @staticmethod
    def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        )

    @staticmethod
    def _union_many_bboxes(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        return (
            min(item[0] for item in bboxes),
            min(item[1] for item in bboxes),
            max(item[2] for item in bboxes),
            max(item[3] for item in bboxes),
        )

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def _candidate_from_bbox(
        self,
        *,
        bbox: tuple[int, int, int, int],
        member_indices: list[int],
        candidate_type: str,
        image_shape: tuple[int, int],
        boxes: list[TextDetectionBox],
        neighbor_counts: dict[int, int],
    ) -> RankedCandidate:
        h, w = image_shape
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        area_ratio = float(area) / float(max(1, h * w))
        aspect = float(bw) / float(max(1, bh))

        if aspect >= 1.0:
            line_likeness = min(1.0, aspect / 6.0)
        else:
            line_likeness = max(0.0, aspect * 0.4)

        edge_dist = min(x1, y1, max(0, w - x2), max(0, h - y2))
        edge_proximity = float(edge_dist) / float(max(1, min(h, w)))

        member_neighbor_counts = [neighbor_counts.get(idx, 0) for idx in member_indices]
        neighbor_avg = float(sum(member_neighbor_counts) / len(member_neighbor_counts)) if member_neighbor_counts else 0.0

        det_scores = [boxes[idx].confidence for idx in member_indices if idx < len(boxes)]
        det_scores = [score for score in det_scores if score is not None]
        det_conf_avg = float(sum(det_scores) / len(det_scores)) if det_scores else 0.0
        detector_sources = tuple(
            dict.fromkeys(
                source
                for idx in member_indices
                if idx < len(boxes)
                for source in boxes[idx].sources
            )
        ) or ("unknown",)
        detector_variants = tuple(
            dict.fromkeys(
                boxes[idx].variant_name
                for idx in member_indices
                if idx < len(boxes) and boxes[idx].variant_name
            )
        )
        polygon_xy = boxes[member_indices[0]].polygon_xy if len(member_indices) == 1 and member_indices[0] < len(boxes) else None

        size_score = self._geometry_size_score(area_ratio)
        aspect_score = self._geometry_aspect_score(aspect)
        edge_score = self._geometry_edge_score(edge_proximity)
        neighbor_score = min(0.2, neighbor_avg * 0.05)
        det_score = (det_conf_avg - 0.5) * 0.25 if det_scores else 0.0
        group_bonus = 0.1 if candidate_type == "group" else 0.0
        multi_source_bonus = 0.25 if len(detector_sources) > 1 else 0.0

        geometry_score = size_score + aspect_score + edge_score + neighbor_score + det_score + group_bonus + multi_source_bonus
        score_breakdown = {
            "size_score": size_score,
            "aspect_score": aspect_score,
            "edge_score": edge_score,
            "neighbor_score": neighbor_score,
            "det_score": det_score,
            "group_bonus": group_bonus,
            "multi_source_bonus": multi_source_bonus,
        }
        context_matches = self._context_probe_matches(
            candidate_bbox=bbox,
            member_indices=member_indices,
            boxes=boxes,
        )
        context_indices = [idx for idx, _context_bbox, _relation in context_matches]
        context_bboxes = [context_bbox for _idx, context_bbox, _relation in context_matches]
        keyword_relation = context_matches[0][2] if context_matches else None

        return RankedCandidate(
            candidate_id="pending",
            candidate_type=candidate_type,
            bbox=bbox,
            member_indices=member_indices,
            member_bboxes=[
                (boxes[idx].x1, boxes[idx].y1, boxes[idx].x2, boxes[idx].y2)
                for idx in member_indices
                if idx < len(boxes)
            ],
            detector_sources=detector_sources,
            detector_variant=detector_variants[0] if detector_variants else None,
            geometry_features={
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect, 4),
                "line_likeness": round(line_likeness, 4),
                "edge_proximity": round(edge_proximity, 4),
                "neighbor_count": round(neighbor_avg, 2),
                "det_conf_avg": round(det_conf_avg, 4),
            },
            geometry_score=geometry_score,
            score_breakdown=score_breakdown,
            recognition_bbox=bbox,
            evidence_bbox=self._union_many_bboxes([bbox, *context_bboxes]) if context_bboxes else bbox,
            polygon_xy=polygon_xy,
            context_probe_bboxes=context_bboxes,
            context_probe_indices=context_indices,
            keyword_relation=keyword_relation,
            total_score=geometry_score,
        )

    @staticmethod
    def _geometry_size_score(area_ratio: float) -> float:
        if area_ratio < 0.0006:
            return -0.6
        if area_ratio < 0.002:
            return -0.2
        if area_ratio <= 0.12:
            return 0.55
        if area_ratio <= 0.32:
            return 0.2
        return -0.35

    @staticmethod
    def _geometry_aspect_score(aspect: float) -> float:
        if aspect < 0.5:
            return -0.2
        if aspect < 1.0:
            return 0.05
        if aspect <= 8.0:
            return 0.35
        if aspect <= 14.0:
            return 0.2
        return -0.1

    @staticmethod
    def _geometry_edge_score(edge_proximity: float) -> float:
        if edge_proximity < 0.01:
            return -0.18
        if edge_proximity < 0.03:
            return -0.07
        if edge_proximity < 0.12:
            return 0.08
        return 0.14

    @classmethod
    def _normalize_evidence_text(cls, text: str | None) -> str:
        return normalize_evidence_text(text)

    @classmethod
    def _compact_evidence_text(cls, text: str | None) -> str:
        return compact_evidence_text(text)

    @classmethod
    def _has_expiry_keyword(cls, text: str | None) -> bool:
        return has_expiry_keyword(text)

    @classmethod
    def _has_production_keyword(cls, text: str | None) -> bool:
        return has_production_keyword(text)

    @classmethod
    def _has_date_like_text(cls, text: str | None) -> bool:
        value = text or ""
        if DATE_PATTERN.search(value):
            return True
        digits = sum(ch.isdigit() for ch in value)
        separators = sum(ch in "/.-:" for ch in value)
        return digits >= 4 and separators >= 1

    @classmethod
    def _is_generic_brand_text(cls, text: str | None) -> bool:
        normalized = cls._normalize_evidence_text(text)
        compact = cls._compact_evidence_text(text)
        if not normalized:
            return False
        if any(token in normalized.split() or token in compact for token in GENERIC_BRAND_TOKENS):
            return True
        letters = sum(ch.isalpha() for ch in compact)
        digits = sum(ch.isdigit() for ch in compact)
        return bool(digits == 0 and letters >= 6)

    @classmethod
    def _is_weight_or_volume_only(cls, text: str | None) -> bool:
        return bool(WEIGHT_VOLUME_ONLY_PATTERN.match(cls._normalize_evidence_text(text)))

    @classmethod
    def _is_barcode_like_numeric(cls, text: str | None) -> bool:
        value = text or ""
        digits = re.sub(r"\D", "", value)
        separators = sum(ch in "/.-:" for ch in value)
        return bool(len(digits) >= 8 and separators == 0)

    def _probe_signal_scores(
        self,
        probe: RecognitionData,
        *,
        candidate: RankedCandidate | None = None,
        today: date | None = None,
    ) -> dict[str, float]:
        text = (probe.normalized_text or probe.raw_text or "").upper()
        if not text:
            return {"probe_empty_penalty": -0.25}

        digits = sum(ch.isdigit() for ch in text)
        separators = sum(ch in "/.-:" for ch in text)
        compact_len = max(1, len([ch for ch in text if not ch.isspace()]))
        date_likeness = (digits + separators) / float(compact_len)

        has_date_pattern = 1.0 if self._has_date_like_text(text) else 0.0
        has_expiry_keyword = self._has_expiry_keyword(text)
        has_production_keyword = self._has_production_keyword(text)
        negative_hits = sum(1 for token in NEGATIVE_HINTS if token in text)

        lot_only_penalty = 0.0
        if LOT_ONLY_PATTERN.search(text) and not self._has_date_like_text(text):
            lot_only_penalty = -2.0

        production_keyword_penalty = 0.0
        if has_production_keyword and self._has_date_like_text(text) and not has_expiry_keyword:
            production_keyword_penalty = -3.0

        brand_text_penalty = 0.0
        if self._is_generic_brand_text(text):
            brand_text_penalty -= 4.0
        if self._is_weight_or_volume_only(text):
            brand_text_penalty -= 2.5
        if self._is_barcode_like_numeric(text):
            brand_text_penalty -= 2.0

        parser_probe_bonus = 0.0
        if today is not None and text.strip():
            ocr = OCRResultData(
                raw_text=probe.raw_text,
                normalized_text=probe.normalized_text,
                confidence=probe.confidence,
                engine_name=getattr(self.ocr_router, "probe_engine_name", "probe"),
                runtime_device=getattr(self.ocr_router, "parseq_runtime_device", None),
                reason=probe.reason,
            )
            parsed = self._best_parse_for_inputs(self._build_parse_inputs(ocr), today=today)
            if parsed.parsed_date is not None:
                parser_probe_bonus = 2.0

        confidence_score = (probe.confidence or 0.0) * 0.75
        date_like_score = max(0.0, min(1.0, date_likeness)) * 1.2 + (2.0 * has_date_pattern)
        expiry_keyword_score = 0.0
        if has_expiry_keyword and self._has_date_like_text(text):
            expiry_keyword_score = 6.0
        elif has_expiry_keyword:
            expiry_keyword_score = 2.5
        keyword_date_group_bonus = 0.0
        if candidate is not None and has_expiry_keyword and self._has_date_like_text(text):
            if candidate.candidate_type in {"group", "line"}:
                keyword_date_group_bonus += 1.5
            if candidate.multiline_split_source_id and candidate.candidate_type == "line":
                keyword_date_group_bonus += 1.0
        negative_penalty = -min(2.4, negative_hits * 0.4)

        return {
            "probe_confidence_score": confidence_score,
            "probe_date_likeness_score": date_like_score,
            "probe_pattern_bonus": 0.0,
            "probe_keyword_bonus": expiry_keyword_score,
            "probe_negative_penalty": negative_penalty,
            "probe_lot_only_penalty": lot_only_penalty,
            "date_like_score": date_like_score,
            "expiry_keyword_score": expiry_keyword_score,
            "production_keyword_penalty": production_keyword_penalty,
            "brand_text_penalty": brand_text_penalty,
            "parser_probe_bonus": parser_probe_bonus,
            "keyword_date_group_bonus": keyword_date_group_bonus,
        }

    @staticmethod
    def _upscale_roi_variant(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image

        target_min_height = 192
        if h >= target_min_height:
            return image

        scale = target_min_height / float(h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray | None:
        try:
            with Image.open(BytesIO(image_bytes)) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
                return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        if array.size == 0:
            return None
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    @classmethod
    def _score_ocr_candidate(cls, ocr: OCRResultData) -> float:
        if not ocr.raw_text:
            return -1.0
        score = (ocr.confidence or 0.0) * 10.0
        text = ocr.raw_text
        if any(sep in text for sep in ("/", "-", ".")):
            score += 1.0
        if any(char.isdigit() for char in text):
            score += 0.5
        if cls._is_generic_brand_text(text):
            score -= 4.0
        if cls._has_production_keyword(text) and cls._has_date_like_text(text) and not cls._has_expiry_keyword(text):
            score -= 3.0
        if cls._has_expiry_keyword(text) and cls._has_date_like_text(text):
            score += 4.0
        return score

    @staticmethod
    def _score_date_likeness(text: str) -> float:
        compact = [ch for ch in text if not ch.isspace()]
        if not compact:
            return 0.0
        date_chars = sum(ch.isdigit() or ch in {"/", "-", ".", ":"} for ch in compact)
        ratio = date_chars / len(compact)
        if any(prefix in text.upper() for prefix in PREFIX_HINTS):
            ratio += 0.1
        return ratio

    @classmethod
    def _prefix_bonus(cls, text: str) -> float:
        return 1.0 if cls._has_expiry_keyword(text) else 0.0

    def _score_parseable_candidate(self, candidate: EvaluationCandidate) -> tuple[int, int, int, float, float, float, float, float]:
        text = " ".join(part for part in (candidate.ocr.normalized_text, candidate.ocr.raw_text) if part)
        role = score_date_role(text)
        return (
            role.priority,
            role.non_production,
            candidate.parsed.parsed_date.toordinal() if candidate.parsed.parsed_date is not None else 0,
            self._text_evidence_priority(text, parser_bonus=1.0),
            candidate.parsed.confidence,
            candidate.candidate_total_score,
            self._score_ocr_candidate(candidate.ocr),
            self._score_date_likeness(candidate.ocr.raw_text),
        )

    def _score_unparseable_candidate(self, candidate: EvaluationCandidate) -> tuple[int, float, float, float, float]:
        text = " ".join(part for part in (candidate.ocr.normalized_text, candidate.ocr.raw_text) if part)
        return (
            self._text_evidence_priority(text),
            candidate.candidate_total_score,
            self._score_ocr_candidate(candidate.ocr),
            self._score_date_likeness(candidate.ocr.raw_text),
            self._prefix_bonus(candidate.ocr.normalized_text),
        )

    @staticmethod
    def _normalize_ocr_confusions(text: str) -> str:
        mapping = str.maketrans(
            {
                "O": "0",
                "I": "1",
                "L": "1",
                "S": "5",
                "B": "8",
            }
        )

        def replace_token(match: "re.Match[str]") -> str:
            token = match.group(0)
            has_digit_or_separator = any(ch.isdigit() or ch in "/.-:" for ch in token)
            has_confusable_chars = any(ch in "OILSB" for ch in token)
            if has_digit_or_separator and has_confusable_chars:
                return token.translate(mapping)
            return token

        pattern = r"[A-Z0-9/.\-:]+"
        return re.sub(pattern, replace_token, text.upper())

    @classmethod
    def _build_parse_inputs(cls, ocr: OCRResultData) -> list[str]:
        base = cls._normalize_ocr_confusions(ocr.normalized_text or "")
        raw = cls._normalize_ocr_confusions(ocr.raw_text or "")
        inputs: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            normalized = " ".join(value.strip().upper().split())
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            inputs.append(normalized)

        add(base)
        for line in raw.splitlines():
            if any(ch.isdigit() for ch in line):
                add(line)
        tokens = [t for t in raw.split() if any(ch.isdigit() for ch in t)]
        for token in tokens:
            if any(sep in token for sep in ("/", "-", ".")):
                add(token)

        for index in range(len(tokens) - 1):
            add(f"{tokens[index]} {tokens[index + 1]}")
        for index in range(len(tokens) - 2):
            add(f"{tokens[index]} {tokens[index + 1]} {tokens[index + 2]}")

        return inputs or ([base] if base else [raw])

    def _best_parse_for_inputs(self, parse_inputs: list[str], *, today: date) -> ParsedDateData:
        parsed_inputs: list[tuple[str, ParsedDateData]] = []
        for text in parse_inputs:
            candidate = self.parser.parse(text, reference_date=today)
            if candidate.parsed_date is None:
                continue
            parsed_inputs.append((text, candidate))
        if len(parsed_inputs) == 1:
            return parsed_inputs[0][1]
        if parsed_inputs:
            _text, best = max(parsed_inputs, key=lambda item: self._parse_input_selection_key(item[0], item[1]))
            return best
        return self.parser.parse(parse_inputs[0], reference_date=today)

    @staticmethod
    def _parse_input_selection_key(text: str, parsed: ParsedDateData) -> tuple[int, int, int, int, float]:
        return role_selection_key(
            evidence=score_date_role(text),
            parsed_date=parsed.parsed_date,
            specificity=1,
            confidence=parsed.confidence,
        )

    def _build_debug_artifacts(
        self,
        *,
        variants_debug: list[dict[str, object]],
        variant_debug_records: dict[str, tuple[np.ndarray, list[TextDetectionBox], list[RankedCandidate]]],
        selected_variant_key: str | None,
        selected_candidate_id: str | None,
        evaluated_count: int,
        parseable_count: int,
        global_selection_summary: dict[str, object] | None = None,
    ) -> tuple[bytes | None, bytes | None, dict[str, object] | None]:
        final_info_fn = getattr(self.ocr_router, "final_recognition_runtime_info", None)
        parseq_info_fn = getattr(self.ocr_router, "parseq_runtime_info", None)
        final_info: dict[str, object] | None = None
        if callable(final_info_fn):
            final_info = final_info_fn()
        if callable(parseq_info_fn):
            parseq_info = parseq_info_fn()
        else:
            parseq_info = {
                "model_dir": None,
                "model_file": None,
                "requested_device_mode": None,
                "runtime_device": getattr(self.ocr_router, "parseq_runtime_device", None),
                "backend": None,
                "charset_length": None,
                "model_loaded": None,
                "mps_fallback_triggered": None,
            }
        selected_candidates_count = sum(int(variant.get("final_selected_count", 0)) for variant in variants_debug)
        selected_input_shapes: list[dict[str, object]] = []
        for variant in variants_debug:
            variant_key = str(variant.get("variant_key", ""))
            for candidate in variant.get("candidates", []):
                if not isinstance(candidate, dict):
                    continue
                if not candidate.get("selected_final"):
                    continue
                selected_input_shapes.append(
                    {
                        "variant_key": variant_key,
                        "candidate_id": candidate.get("candidate_id"),
                        "parseq_input_shape": candidate.get("parseq_input_shape"),
                        "recognition_variant": candidate.get("recognition_variant"),
                    }
                )
        summary = {
            "selected_variant_key": selected_variant_key,
            "selected_candidate_id": selected_candidate_id,
            "selected_candidates_count": selected_candidates_count,
            "selected_candidate_input_shapes": selected_input_shapes,
            "evaluated_candidates": evaluated_count,
            "parseable_candidates": parseable_count,
            "variants_count": len(variants_debug),
            "final_recognition_runtime": final_info or parseq_info,
            "parseq_runtime": parseq_info,
        }
        if global_selection_summary:
            summary.update(global_selection_summary)

        candidates_bytes: bytes | None = None
        if self.save_debug_candidates:
            payload = {
                "summary": summary,
                "variants": variants_debug,
            }
            candidates_bytes = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")

        overlay_bytes: bytes | None = None
        if self.save_debug_overlays and selected_variant_key and selected_variant_key in variant_debug_records:
            variant_image, det_boxes, ranked = variant_debug_records[selected_variant_key]
            overlay = self._render_debug_overlay(
                image=variant_image,
                det_boxes=det_boxes,
                ranked=ranked,
                selected_candidate_id=selected_candidate_id,
            )
            overlay_bytes = self._encode_png(overlay)

        return candidates_bytes, overlay_bytes, summary

    @staticmethod
    def _render_debug_overlay(
        *,
        image: np.ndarray,
        det_boxes: list[TextDetectionBox],
        ranked: list[RankedCandidate],
        selected_candidate_id: str | None,
    ) -> np.ndarray:
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        for box in det_boxes:
            cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), (255, 180, 80), 1)

        for candidate in ranked:
            x1, y1, x2, y2 = candidate.bbox
            color = (160, 160, 160)
            thickness = 1
            if candidate.selected_geometry:
                color = (0, 215, 255)
            if candidate.selected_final:
                color = (255, 0, 255)
                thickness = 2
            if selected_candidate_id and candidate.candidate_id == selected_candidate_id:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                canvas,
                candidate.candidate_id,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _compose_reason(
        self,
        *,
        selected: EvaluationCandidate,
        evaluated: list[EvaluationCandidate],
        parseable_count: int,
        base_reason: str,
    ) -> str:
        return (
            f"{base_reason};"
            f"selected_source={selected.source};"
            f"selected_variant={selected.variant};"
            f"selected_candidate={selected.candidate_id};"
            f"evaluated_candidates={len(evaluated)};"
            f"parseable_candidates={parseable_count};"
            f"parse_inputs={selected.parse_inputs_count}"
        )

    @staticmethod
    def _compose_empty_reason(*, detection_reason: str, evaluated_count: int) -> str:
        return f"{detection_reason};evaluated_candidates={evaluated_count};parseable_candidates=0"

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None
        return encoded.tobytes()

    def _crop_box_with_padding(self, *, image: np.ndarray, box) -> tuple[np.ndarray, int, int] | None:
        h, w = image.shape[:2]
        bw = max(1, box.x2 - box.x1)
        bh = max(1, box.y2 - box.y1)
        pad_x = int(round(bw * self.roi_padding_ratio))
        pad_y = int(round(bh * self.roi_padding_ratio))

        x1 = max(0, min(box.x1 - pad_x, w - 1))
        y1 = max(0, min(box.y1 - pad_y, h - 1))
        x2 = max(0, min(box.x2 + pad_x, w))
        y2 = max(0, min(box.y2 + pad_y, h))

        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2], x1, y1

    def _failed(
        self,
        *,
        final_status: str,
        reason: str,
        timings: dict[str, int],
        detected: bool = False,
        detector_confidence: float | None = None,
        ocr: OCRResultData | None = None,
        parsed: ParsedDateData | None = None,
        roi: np.ndarray | None = None,
        debug_candidates_json_bytes: bytes | None = None,
        debug_overlay_png_bytes: bytes | None = None,
        debug_summary: dict[str, object] | None = None,
    ) -> PipelineRunOutput:
        roi_png = self._encode_png(roi) if roi is not None else None

        return PipelineRunOutput(
            detected=detected,
            detector_confidence=detector_confidence,
            raw_text=ocr.raw_text if ocr else None,
            normalized_text=ocr.normalized_text if ocr else None,
            ocr_confidence=ocr.confidence if ocr else None,
            ocr_engine=ocr.engine_name if ocr else None,
            ocr_runtime_device=ocr.runtime_device if ocr else None,
            parsed_date=parsed.parsed_date if parsed else None,
            date_format_detected=parsed.date_format_detected if parsed else None,
            parse_confidence=parsed.confidence if parsed else None,
            expiry_classification=ExpiryClassification.MANUAL_REVIEW_REQUIRED,
            days_remaining=None,
            alert_required=True,
            needs_review=True,
            final_status=final_status,
            reason=reason,
            stage_timings_ms=timings,
            roi_png_bytes=roi_png,
            debug_candidates_json_bytes=debug_candidates_json_bytes,
            debug_overlay_png_bytes=debug_overlay_png_bytes,
            debug_summary=debug_summary,
        )
