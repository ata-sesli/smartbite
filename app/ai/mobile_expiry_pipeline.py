from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ai.crop_normalization import NormalizedTextLineCrop, TextLineCropConfig, TextLineGeometry, normalize_textline_crops
from app.ai.date_role import (
    compact_evidence_text,
    has_expiry_keyword,
    has_production_keyword,
    has_weak_expiry_keyword,
    normalize_evidence_text,
    role_selection_key,
    score_date_role,
)
from app.ai.expiry_candidate_engine import ExpiryCandidateEngine, ProposalBox
from app.ai.onnx_inference import SVTROnnxTextRecognizer, YoloObbOnnxDetector
from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ImageVariant, ROIImagePreprocessor
from app.ai.rapidocr_text_detector import RapidOCRTextProposalDetector
from app.ai.types import ParsedDateData
from app.infra.settings import PROJECT_ROOT


DATE_PATTERN = re.compile(r"\b\d{1,4}[./:-]\d{1,2}(?:[./:-]\d{1,4})?\b")
LOT_ONLY_PATTERN = re.compile(r"\b(?:LOT|BATCH)\b")
WEIGHT_VOLUME_ONLY_PATTERN = re.compile(r"^\s*\d+(?:[.,]\d+)?\s*(?:G|GR|GRAM|KG|ML|L|LT|LITER|LITRE)\s*$")
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
MOBILE_RECOGNITION_VARIANTS = (
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "clahe_gray",
    "unsharp_gray",
    "adaptive_binary",
    "adaptive_binary_inverted",
    "stamp_blackhat",
)
MOBILE_ENHANCED_RECOGNITION_VARIANTS = tuple(
    name for name in MOBILE_RECOGNITION_VARIANTS if name != "original_color_tight"
)
PRODUCTION_ANCHOR_SIBLING_RECOGNITION_VARIANTS = (
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "unsharp_gray",
)
ANCHOR_LOCAL_SIBLING_RECOGNITION_VARIANTS = (
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "unsharp_gray",
)
PARTIAL_DAY_MONTH_RIGHT_EXPANSION_VARIANTS = (
    "stamp_blackhat",
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "unsharp_gray",
)
DOT_MATRIX_RESCUE_RECOGNITION_VARIANTS = (
    "stamp_blackhat",
    "original_color_tight",
    "original_padded",
    "gray_upscaled",
    "unsharp_gray",
    "adaptive_binary",
)
RAPIDOCR_GATED_PROBE_RECOGNITION_VARIANTS = (
    "original_color_tight",
)
PRODUCT_CROPPER_RECOGNITION_VARIANTS = (
    "original_color_tight",
    "original_padded",
    "clahe_gray",
    "adaptive_binary",
)
NORMAL_SELECTED_VARIANT_PRIORITY = {
    "original_color_tight",
    "original_padded",
    "clahe_gray",
    "stamp_blackhat",
}
SVTR_RECOGNITION_BATCH_SIZE = 8
DEBUG_PROFILE_TIMING_KEYS = (
    "decode_ms",
    "yolo_ms",
    "rapidocr_primary_ms",
    "candidate_build_ms",
    "probe_svtr_ms",
    "context_probe_ms",
    "final_svtr_primary_ms",
    "hardcase_svtr_ms",
    "dot_matrix_ms",
    "pp_mobile_full_image_ms",
    "date_tail_crop_ms",
    "partial_expansion_ms",
    "production_sibling_ms",
    "anchor_local_sibling_ms",
    "detector_variant_rescue_ms",
    "rapidocr_rescue_ms",
    "total_ms",
)
DEBUG_PROFILE_COUNT_KEYS = (
    "num_yolo_boxes",
    "num_pp_boxes",
    "num_probe_candidates",
    "num_final_candidates",
    "num_primary_svtr_calls",
    "num_hardcase_svtr_calls",
    "num_dot_matrix_crops",
)


def normalize_recognition_text(raw_text: str) -> str:
    return " ".join(raw_text.strip().upper().split())


@dataclass(slots=True)
class MobileExpiryPipelineResult:
    status: str
    detected_expiry_date: date | None
    raw_text: str | None
    normalized_text: str | None
    recognition_confidence: float | None
    detector_confidence: float | None
    reason: str | None
    detection_polygon_json: list[list[float]] | None
    runtime_ms: int
    final_recognition_bbox_xyxy: list[int] | None = None
    final_recognition_polygon_json: list[list[float]] | None = None
    final_crop_policy: str | None = None
    final_crop_padding_px: int | None = None
    date_evidence_json: list[dict[str, object]] | None = None
    debug_profile: dict[str, float | int] | None = None
    debug_candidates_json_bytes: bytes | None = None


@dataclass(slots=True)
class _YoloCandidate:
    polygon_xy: list[list[float]]
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str = "yolo26s_obb"
    sources: tuple[str, ...] = ("yolo26s_obb",)
    variant_name: str | None = "yolo26s_obb"


@dataclass(slots=True)
class _RecognitionOutput:
    raw_text: str
    normalized_text: str
    confidence: float | None
    reason: str | None
    rotation: str


@dataclass(slots=True)
class _MobileRankedCandidate:
    candidate_id: str
    candidate_type: str
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: tuple[tuple[float, float], ...] | None
    detector_confidence: float | None
    detector_sources: tuple[str, ...]
    detector_variant: str | None
    member_indices: list[int]
    member_bboxes: list[tuple[int, int, int, int]]
    geometry_features: dict[str, float]
    geometry_score: float
    score_breakdown: dict[str, float]
    total_score: float
    recognition_bbox: tuple[int, int, int, int] | None = None
    evidence_bbox: tuple[int, int, int, int] | None = None
    probe_text: str = ""
    probe_normalized_text: str = ""
    probe_confidence: float | None = None
    probe_reason: str | None = None
    recognition_variant_image: np.ndarray | None = None
    selected_recognition_output: _RecognitionOutput | None = None
    selected_recognition_variant: str = ""
    selected_geometry: bool = False
    selected_final: bool = False
    final_rank_before_force_include: int | None = None
    final_rank_after_force_include: int | None = None
    force_included_reason: str | None = None
    final_crop_bbox: tuple[int, int, int, int] | None = None
    final_crop_padding_px: int | None = None
    final_crop_policy: str | None = None
    context_probe_bboxes: list[tuple[int, int, int, int]] | None = None
    context_probe_indices: list[int] | None = None
    context_probe_texts: list[str] | None = None
    context_probe_confidences: list[float | None] | None = None
    rapidocr_probe_text: str | None = None
    rapidocr_probe_normalized_text: str | None = None
    rapidocr_probe_confidence: float | None = None
    rapidocr_probe_score: float | None = None
    rapidocr_probe_orientation: str | None = None
    rapidocr_probe_polygon_used: bool = False
    adjacent_expiry_keyword: bool = False
    adjacent_production_keyword: bool = False
    keyword_relation: str | None = None
    scan_offset_x: int = 0
    scan_offset_y: int = 0

    def crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy
        return image[y1:y2, x1:x2]


@dataclass(slots=True)
class _MobileGlobalCandidate:
    source: str
    variant: str
    variant_key: str
    image: np.ndarray
    detector_confidence: float | None
    ranked: _MobileRankedCandidate
    scan_bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class _MobileROICandidate:
    source: str
    image: np.ndarray
    confidence: float | None
    offset_x: int
    offset_y: int
    polygon_xy: list[list[float]] | None = None


@dataclass(slots=True)
class _ProductCropperRoi:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str


@dataclass(slots=True)
class _DateEvidence:
    candidate: _MobileRankedCandidate
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int
    recognition_variant: str
    normalized_crop: NormalizedTextLineCrop | None
    crop_policy: str | None = None
    context_texts: tuple[str, ...] = ()
    original_color_fallback_used: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _EvaluatedCandidate:
    candidate: _MobileRankedCandidate
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int
    recognition_variant: str
    normalized_crop: NormalizedTextLineCrop | None
    original_color_fallback_used: bool = False
    date_evidence: list[_DateEvidence] = field(default_factory=list)
    selected_date_evidence: _DateEvidence | None = None


@dataclass(slots=True)
class _RecognitionEvidence:
    variant_name: str
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int


class SVTRTextRecognizer:
    cacheable = True

    def __init__(self, *, model_name: str, model_dir: Path | None, device: str) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = "gpu:0" if device == "cuda" else "cpu"
        self._model: Any | None = None
        self._load_error: str | None = None

    @staticmethod
    def _is_inference_dir(path: Path) -> bool:
        return (path / "inference.yml").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            from paddleocr import TextRecognition
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"SVTR recognizer import failed: {exc}"
            return

        kwargs: dict[str, object] = {"device": self.device, "model_name": self.model_name}
        if self.model_dir is not None:
            model_dir = self.model_dir
            if not model_dir.is_absolute():
                model_dir = PROJECT_ROOT / model_dir
            if not model_dir.exists():
                self._load_error = f"SVTR recognizer model directory not found: {model_dir}"
                return
            if not self._is_inference_dir(model_dir):
                self._load_error = f"SVTR recognizer model directory missing inference.yml: {model_dir}"
                return
            kwargs["model_dir"] = str(model_dir)

        try:
            self._model = TextRecognition(**kwargs)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"SVTR recognizer initialization failed: {exc}"

    @staticmethod
    def _prepare_input(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    @staticmethod
    def _output_from_entry(entry: object) -> _RecognitionOutput:
        if not isinstance(entry, dict):
            return _RecognitionOutput("", "", None, "SVTR recognizer returned malformed result", "original")
        raw = str(entry.get("rec_text", "") or "").strip()
        confidence: float | None = None
        try:
            confidence = float(entry["rec_score"]) if entry.get("rec_score") is not None else None
        except Exception:
            confidence = None
        if not raw:
            return _RecognitionOutput("", "", confidence, "SVTR recognizer returned empty text", "original")
        return _RecognitionOutput(raw, normalize_recognition_text(raw), confidence, None, "original")

    def recognize_many(self, images: list[np.ndarray], *, batch_size: int = SVTR_RECOGNITION_BATCH_SIZE) -> list[_RecognitionOutput]:
        self._ensure_loaded()
        if not images:
            return []
        if self._load_error:
            return [_RecognitionOutput("", "", None, self._load_error, "original") for _ in images]
        if self._model is None:
            return [_RecognitionOutput("", "", None, "SVTR recognizer unavailable", "original") for _ in images]

        recognizer_inputs = [self._prepare_input(image) for image in images]
        try:
            result = self._model.predict(input=recognizer_inputs, batch_size=max(1, int(batch_size)))
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return [_RecognitionOutput("", "", None, f"SVTR recognition failed: {exc}", "original") for _ in images]

        if not isinstance(result, list):
            return [_RecognitionOutput("", "", None, "SVTR recognizer returned no result", "original") for _ in images]
        outputs = [self._output_from_entry(entry) for entry in result[: len(images)]]
        if len(outputs) < len(images):
            outputs.extend(
                _RecognitionOutput("", "", None, "SVTR recognizer returned fewer results than inputs", "original")
                for _ in range(len(images) - len(outputs))
            )
        return outputs

    def recognize(self, image: np.ndarray) -> _RecognitionOutput:
        self._ensure_loaded()
        if self._load_error:
            return _RecognitionOutput("", "", None, self._load_error, "original")
        if self._model is None:
            return _RecognitionOutput("", "", None, "SVTR recognizer unavailable", "original")

        try:
            result = self._model.predict(self._prepare_input(image))
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return _RecognitionOutput("", "", None, f"SVTR recognition failed: {exc}", "original")

        if not isinstance(result, list) or not result:
            return _RecognitionOutput("", "", None, "SVTR recognizer returned no result", "original")
        return self._output_from_entry(result[0])


class MobileExpiryPipeline:
    def __init__(
        self,
        *,
        detector_model_path: Path,
        detector_confidence_threshold: float,
        detector_imgsz: int,
        max_candidates: int,
        crop_padding_px: int,
        svtr_model_name: str,
        svtr_model_dir: Path,
        svtr_device: str,
        parser_min_candidate_confidence: float,
        detector_backend: str = "ultralytics",
        detector_onnx_path: Path | None = None,
        full_image_rotation_detector_rescue_enabled: bool = True,
        full_image_rotation_detector_rescue_max_candidates: int = 1,
        svtr_backend: str = "paddle",
        svtr_onnx_model_path: Path | None = None,
        proposal_rescue_backend: str = "rapidocr_ppocrv5",
        rapidocr_primary_enabled: bool = True,
        rapidocr_primary_mode: str = "gated",
        rapidocr_rescue_enabled: bool = True,
        rapidocr_ocr_version: str = "PP-OCRv5",
        rapidocr_model_type: str = "mobile",
        rapidocr_lang_type: str = "ch",
        rapidocr_limit_side_len: int = 512,
        rapidocr_limit_type: str = "max",
        rapidocr_max_candidates: int = 64,
        rapidocr_primary_max_rois_per_scan: int = 12,
        rapidocr_primary_max_boxes_accepted: int = 40,
        rapidocr_primary_timeout_seconds: float = 12.0,
        rapidocr_gated_max_boxes_accepted: int = 36,
        rapidocr_gated_probe_top_k: int = 8,
        rapidocr_gated_final_top_k: int = 4,
        rapidocr_gated_prefilter_probe_top_k: int = 36,
        rapidocr_gated_min_probe_digits: int = 4,
        rapidocr_gated_min_date_likeness_score: float = 1.0,
        rapidocr_max_rois_per_scan: int = 1,
        rapidocr_max_boxes_accepted: int = 5,
        rapidocr_timeout_seconds: float = 0.75,
        rapidocr_min_confidence: float = 0.10,
        product_cropper_rescue_enabled: bool = False,
        product_cropper_model_path: Path | None = None,
        product_cropper_confidence_threshold: float = 0.15,
        product_cropper_imgsz: int = 1024,
        product_cropper_padding_ratio: float = 0.12,
        product_cropper_max_candidates: int = 3,
        legacy_wide_group_rescue_enabled: bool = False,
        strict_evidence_acceptance_enabled: bool = False,
        rapidocr_role_constraint_enabled: bool = False,
        production_anchor_sibling_enabled: bool = False,
        paired_crop_evidence_enabled: bool = False,
        local_group_wide_crop_enabled: bool = False,
        rotation_wide_crop_enabled: bool = False,
        hardcase_recognizer_rescue_enabled: bool = False,
        hardcase_svtr_model_name: str | None = None,
        hardcase_svtr_model_dir: Path | None = None,
        hardcase_svtr_backend: str | None = None,
        hardcase_svtr_onnx_model_path: Path | None = None,
        dot_matrix_rescue_enabled: bool = True,
        dot_matrix_rescue_max_candidates: int = 8,
    ) -> None:
        self.detector_model_path = detector_model_path
        self.detector_backend = detector_backend.lower().strip()
        self.detector_onnx_path = detector_onnx_path
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_imgsz = detector_imgsz
        self.max_candidates = max(1, int(max_candidates))
        self.crop_padding_px = max(0, int(crop_padding_px))
        self.full_image_rotation_detector_rescue_enabled = bool(full_image_rotation_detector_rescue_enabled)
        self.full_image_rotation_detector_rescue_max_candidates = max(1, int(full_image_rotation_detector_rescue_max_candidates))
        self.high_recall_mode = True
        self.expiry_filter_geometry_top_n = max(1, int(max_candidates))
        self.expiry_global_probe_top_k = 20
        self.expiry_global_final_top_k = 6
        self.expiry_global_debug_final_top_k = 12
        self.expiry_global_max_per_roi = 2
        self.expiry_global_max_per_variant = 3
        self.expiry_max_detector_variants = 1
        self.expiry_detector_variants = ("raw",)
        self.expiry_final_crop_padding_ratio = 0.04
        self.expiry_final_crop_min_padding_px = 3
        self.expiry_final_crop_max_padding_px = 18
        self.expiry_debug_parseq_whole_groups = False
        self.context_probe_enabled = True
        self.context_max_boxes_per_candidate = 2
        self.context_max_candidates_per_scan = 20
        self.max_group_candidates_per_roi = 100
        self.candidate_engine = ExpiryCandidateEngine(
            max_group_candidates_per_roi=self.max_group_candidates_per_roi,
            geometry_top_n=self.expiry_filter_geometry_top_n,
            context_probe_enabled=True,
            context_max_boxes_per_candidate=2,
            require_multiline_for_line_candidates=True,
        )
        self.proposal_rescue_backend = proposal_rescue_backend.strip().lower() or "rapidocr_ppocrv5"
        self.rapidocr_primary_enabled = bool(rapidocr_primary_enabled)
        rapidocr_primary_mode_normalized = rapidocr_primary_mode.strip().lower() or "gated"
        if rapidocr_primary_mode_normalized not in {"gated", "always", "off"}:
            raise ValueError("rapidocr_primary_mode must be one of: gated, always, off")
        self.rapidocr_primary_mode = rapidocr_primary_mode_normalized
        self.rapidocr_rescue_enabled = bool(rapidocr_rescue_enabled)
        self.rapidocr_ocr_version = rapidocr_ocr_version.strip() or "PP-OCRv5"
        self.rapidocr_model_type = rapidocr_model_type.strip().lower() or "server"
        self.rapidocr_lang_type = rapidocr_lang_type.strip().lower() or "ch"
        self.rapidocr_limit_side_len = max(128, int(rapidocr_limit_side_len))
        self.rapidocr_limit_type = rapidocr_limit_type.strip().lower() or "max"
        self.rapidocr_max_candidates = max(1, int(rapidocr_max_candidates))
        self.rapidocr_primary_max_rois_per_scan = max(0, int(rapidocr_primary_max_rois_per_scan))
        self.rapidocr_primary_max_boxes_accepted = max(1, int(rapidocr_primary_max_boxes_accepted))
        self.rapidocr_primary_timeout_seconds = max(0.1, float(rapidocr_primary_timeout_seconds))
        self.rapidocr_gated_max_boxes_accepted = max(1, int(rapidocr_gated_max_boxes_accepted))
        self.rapidocr_gated_probe_top_k = max(1, int(rapidocr_gated_probe_top_k))
        self.rapidocr_gated_final_top_k = max(1, int(rapidocr_gated_final_top_k))
        self.rapidocr_gated_prefilter_probe_top_k = max(1, int(rapidocr_gated_prefilter_probe_top_k))
        self.rapidocr_gated_min_probe_digits = max(1, int(rapidocr_gated_min_probe_digits))
        self.rapidocr_gated_min_date_likeness_score = max(0.0, float(rapidocr_gated_min_date_likeness_score))
        self.rapidocr_max_rois_per_scan = max(0, int(rapidocr_max_rois_per_scan))
        self.rapidocr_max_boxes_accepted = max(1, int(rapidocr_max_boxes_accepted))
        self.rapidocr_timeout_seconds = max(0.1, float(rapidocr_timeout_seconds))
        self.rapidocr_min_confidence = max(0.0, min(float(rapidocr_min_confidence), 1.0))
        self.product_cropper_rescue_enabled = bool(product_cropper_rescue_enabled)
        self.product_cropper_model_path = product_cropper_model_path or Path("models/yolo20n/yolo26s/yolo26s-best.pt")
        self.product_cropper_confidence_threshold = max(0.01, min(float(product_cropper_confidence_threshold), 0.99))
        self.product_cropper_imgsz = max(1, int(product_cropper_imgsz))
        self.product_cropper_padding_ratio = max(0.0, min(float(product_cropper_padding_ratio), 0.5))
        self.product_cropper_max_candidates = max(1, int(product_cropper_max_candidates))
        self.legacy_wide_group_rescue_enabled = bool(legacy_wide_group_rescue_enabled)
        self.strict_evidence_acceptance_enabled = bool(strict_evidence_acceptance_enabled)
        self.rapidocr_role_constraint_enabled = bool(rapidocr_role_constraint_enabled)
        self.production_anchor_sibling_enabled = bool(production_anchor_sibling_enabled)
        self.paired_crop_evidence_enabled = bool(paired_crop_evidence_enabled)
        self.local_group_wide_crop_enabled = bool(local_group_wide_crop_enabled)
        self.rotation_wide_crop_enabled = bool(rotation_wide_crop_enabled)
        self.hardcase_recognizer_rescue_enabled = bool(hardcase_recognizer_rescue_enabled)
        self.dot_matrix_rescue_enabled = bool(dot_matrix_rescue_enabled)
        self.dot_matrix_rescue_max_candidates = max(1, int(dot_matrix_rescue_max_candidates))
        self.svtr_batch_recognition_enabled = False
        self.parser = ExpiryDateParser(min_candidate_confidence=parser_min_candidate_confidence)
        self.preprocessor = ROIImagePreprocessor()
        svtr_backend_normalized = svtr_backend.lower().strip()
        if svtr_backend_normalized == "onnx":
            self.recognizer = SVTROnnxTextRecognizer(
                model_path=svtr_onnx_model_path or Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx"),
                paddle_model_dir=svtr_model_dir,
                runtime_device=svtr_device,
            )
        elif svtr_backend_normalized == "paddle":
            self.recognizer = SVTRTextRecognizer(
                model_name=svtr_model_name,
                model_dir=svtr_model_dir,
                device=svtr_device,
            )
        else:
            raise ValueError("svtr_backend must be one of: paddle, onnx")
        self.hardcase_recognizer: Any | None = None
        if self.hardcase_recognizer_rescue_enabled:
            hardcase_backend = (hardcase_svtr_backend or svtr_backend).lower().strip()
            hardcase_model_name = hardcase_svtr_model_name or svtr_model_name
            hardcase_model_dir = hardcase_svtr_model_dir or svtr_model_dir
            if hardcase_backend == "onnx":
                self.hardcase_recognizer = SVTROnnxTextRecognizer(
                    model_path=hardcase_svtr_onnx_model_path
                    or hardcase_model_dir.with_suffix(".onnx"),
                    paddle_model_dir=hardcase_model_dir,
                    runtime_device=svtr_device,
                    engine_name="hardcase_svtrv2_onnx",
                )
            elif hardcase_backend == "paddle":
                self.hardcase_recognizer = SVTRTextRecognizer(
                    model_name=hardcase_model_name,
                    model_dir=hardcase_model_dir,
                    device=svtr_device,
                )
            else:
                raise ValueError("hardcase_svtr_backend must be one of: paddle, onnx")
        self._detector_model: Any | None = None
        self._onnx_detector: YoloObbOnnxDetector | None = None
        self._detector_error: str | None = None
        self._rapidocr_detector: RapidOCRTextProposalDetector | None = None
        self._rapidocr_error: str | None = None
        self._product_cropper_model: Any | None = None
        self._product_cropper_error: str | None = None
        self._last_rapidocr_primary_debug: dict[str, object] = {}
        self._last_rapidocr_rescue_debug: dict[str, object] = {}
        self._debug_records: dict[str, list[dict[str, object]]] = {}
        self._debug_profile: dict[str, float | int] = {}
        self._active_svtr_profile_key: str | None = None
        self._svtr_recognition_cache: dict[tuple[int, str, str, str, tuple[int, ...], str], _RecognitionOutput] = {}
        self._rapidocr_gated_probe_metadata: dict[tuple[int, int, int, int], dict[str, object]] = {}

    def _reset_debug_records(self) -> None:
        self._svtr_recognition_cache = {}
        self._rapidocr_gated_probe_metadata = {}
        self._debug_profile = {
            **{key: 0.0 for key in DEBUG_PROFILE_TIMING_KEYS},
            **{key: 0 for key in DEBUG_PROFILE_COUNT_KEYS},
        }
        self._active_svtr_profile_key = None
        self._debug_records = {
            "yolo_obb": [],
            "rotation_detector_rescue": [],
            "rapidocr_text_boxes": [],
            "ppmobile_gated_prefilter": [],
            "candidate_engine_candidates": [],
            "geometry_shortlist": [],
            "probe_ocr": [],
            "final_candidate_selection": [],
            "svtr_final_attempts": [],
            "date_evidence": [],
            "final_selected_evidence": [],
            "hardcase_recognizer_rescue": [],
            "ppmobile_gated_trigger": [],
            "dot_matrix_rescue": [],
        }

    def _debug_append(self, key: str, row: dict[str, object]) -> None:
        self._debug_records.setdefault(key, []).append(row)

    def _debug_profile_snapshot(self) -> dict[str, float | int]:
        if not self._debug_profile:
            self._debug_profile = {
                **{key: 0.0 for key in DEBUG_PROFILE_TIMING_KEYS},
                **{key: 0 for key in DEBUG_PROFILE_COUNT_KEYS},
            }
        snapshot: dict[str, float | int] = {}
        for key in DEBUG_PROFILE_TIMING_KEYS:
            snapshot[key] = round(float(self._debug_profile.get(key, 0.0) or 0.0), 3)
        for key in DEBUG_PROFILE_COUNT_KEYS:
            snapshot[key] = int(self._debug_profile.get(key, 0) or 0)
        return snapshot

    def _profile_add(self, key: str, value_ms: float) -> None:
        if not self._debug_profile:
            self._debug_profile_snapshot()
        self._debug_profile[key] = float(self._debug_profile.get(key, 0.0) or 0.0) + float(value_ms)

    def _profile_inc(self, key: str, value: int = 1) -> None:
        if not self._debug_profile:
            self._debug_profile_snapshot()
        self._debug_profile[key] = int(self._debug_profile.get(key, 0) or 0) + int(value)

    def _profile_set(self, key: str, value: float | int) -> None:
        if not self._debug_profile:
            self._debug_profile_snapshot()
        self._debug_profile[key] = value

    @contextmanager
    def _profile_timer(self, key: str):
        started = perf_counter()
        try:
            yield
        finally:
            self._profile_add(key, (perf_counter() - started) * 1000.0)

    @contextmanager
    def _svtr_profile_stage(self, key: str):
        previous = self._active_svtr_profile_key
        self._active_svtr_profile_key = key
        try:
            yield
        finally:
            self._active_svtr_profile_key = previous

    def _svtr_profile_key(self, recognizer: Any) -> str:
        if self.hardcase_recognizer is not None and recognizer is self.hardcase_recognizer:
            return "hardcase_svtr_ms"
        return self._active_svtr_profile_key or "final_svtr_primary_ms"

    def _record_svtr_profile(self, recognizer: Any, *, elapsed_ms: float, call_count: int) -> None:
        timing_key = self._svtr_profile_key(recognizer)
        count_key = "num_hardcase_svtr_calls" if timing_key == "hardcase_svtr_ms" else "num_primary_svtr_calls"
        self._profile_add(timing_key, elapsed_ms)
        self._profile_inc(count_key, call_count)

    def _finish_debug_profile(self, started: float) -> dict[str, float | int]:
        self._profile_set("total_ms", (perf_counter() - started) * 1000.0)
        return self._debug_profile_snapshot()

    def _debug_bytes(self) -> bytes:
        payload: dict[str, object] = dict(self._debug_records)
        payload["profile"] = self._debug_profile_snapshot()
        return json.dumps(payload, default=str).encode("utf-8")

    @classmethod
    def _debug_yolo_candidate(cls, candidate: _YoloCandidate) -> dict[str, object]:
        return {
            "bbox_xyxy": cls._bbox_json(candidate.bbox_xyxy),
            "polygon_xy": candidate.polygon_xy,
            "confidence": candidate.confidence,
            "source": candidate.source,
            "sources": list(candidate.sources),
            "variant_name": candidate.variant_name,
        }

    @classmethod
    def _debug_proposal(cls, proposal: ProposalBox) -> dict[str, object]:
        return {
            "bbox_xyxy": cls._bbox_json(proposal.bbox_xyxy),
            "polygon_xy": [[float(x), float(y)] for x, y in proposal.polygon_xy] if proposal.polygon_xy else None,
            "confidence": proposal.confidence,
            "source": proposal.source,
            "sources": list(proposal.sources),
            "variant_name": proposal.variant_name,
        }

    @classmethod
    def _debug_ranked_candidate(cls, candidate: _MobileRankedCandidate) -> dict[str, object]:
        return {
            "candidate_id": candidate.candidate_id,
            "candidate_type": candidate.candidate_type,
            "bbox_xyxy": cls._bbox_json(candidate.bbox_xyxy),
            "polygon_xy": [[float(x), float(y)] for x, y in candidate.polygon_xy] if candidate.polygon_xy else None,
            "detector_confidence": candidate.detector_confidence,
            "detector_sources": list(candidate.detector_sources),
            "detector_variant": candidate.detector_variant,
            "member_indices": list(candidate.member_indices),
            "member_bboxes": [cls._bbox_json(box) for box in candidate.member_bboxes],
            "geometry_score": candidate.geometry_score,
            "geometry_features": dict(candidate.geometry_features),
            "score_breakdown": dict(candidate.score_breakdown),
            "total_score": candidate.total_score,
            "selected_geometry": candidate.selected_geometry,
            "selected_final": candidate.selected_final,
            "final_rank_before_force_include": candidate.final_rank_before_force_include,
            "final_rank_after_force_include": candidate.final_rank_after_force_include,
            "force_included_reason": candidate.force_included_reason,
            "final_crop_bbox": cls._bbox_json(candidate.final_crop_bbox),
            "final_crop_policy": candidate.final_crop_policy,
            "final_crop_padding_px": candidate.final_crop_padding_px,
            "probe_text": candidate.probe_text,
            "probe_normalized_text": candidate.probe_normalized_text,
            "probe_confidence": candidate.probe_confidence,
            "probe_reason": candidate.probe_reason,
            "rapidocr_probe_text": candidate.rapidocr_probe_text,
            "rapidocr_probe_normalized_text": candidate.rapidocr_probe_normalized_text,
            "rapidocr_probe_confidence": candidate.rapidocr_probe_confidence,
            "rapidocr_probe_score": candidate.rapidocr_probe_score,
            "rapidocr_probe_orientation": candidate.rapidocr_probe_orientation,
            "rapidocr_probe_polygon_used": candidate.rapidocr_probe_polygon_used,
            "selected_recognition_variant": candidate.selected_recognition_variant,
            "context_probe_bboxes": [cls._bbox_json(box) for box in candidate.context_probe_bboxes or []],
            "context_probe_texts": list(candidate.context_probe_texts or []),
            "scan_offset": [candidate.scan_offset_x, candidate.scan_offset_y],
        }

    @staticmethod
    def _debug_parsed_date(parsed: ParsedDateData) -> dict[str, object]:
        return {
            "parsed_date": parsed.parsed_date.isoformat() if parsed.parsed_date is not None else None,
            "date_format_detected": parsed.date_format_detected,
            "confidence": parsed.confidence,
            "reason": parsed.reason,
            "date_precision": parsed.date_precision,
            "parsed_day": parsed.parsed_day,
            "parsed_month": parsed.parsed_month,
            "parsed_year": parsed.parsed_year,
        }

    @classmethod
    def _debug_date_evidence(cls, evidence: _DateEvidence) -> dict[str, object]:
        orientation = evidence.normalized_crop.selected_orientation if evidence.normalized_crop is not None else evidence.recognition.rotation
        return {
            "parsed": cls._debug_parsed_date(evidence.parsed),
            "raw_text": evidence.recognition.raw_text,
            "normalized_text": evidence.recognition.normalized_text,
            "ocr_confidence": evidence.recognition.confidence,
            "recognition_variant": evidence.recognition_variant,
            "orientation": orientation,
            "parse_inputs_count": evidence.parse_inputs_count,
            "crop_policy": evidence.crop_policy,
            "crop_bbox_xyxy": cls._bbox_json(evidence.normalized_crop.bbox_xyxy if evidence.normalized_crop is not None else evidence.candidate.final_crop_bbox),
            "context_texts": list(evidence.context_texts),
            "candidate": cls._debug_ranked_candidate(evidence.candidate),
            "metadata": dict(evidence.metadata),
        }

    def _ensure_detector(self) -> Any | None:
        if self._detector_model is not None or self._detector_error is not None:
            return self._detector_model
        model_path = self.detector_model_path
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            self._detector_error = f"YOLO OBB detector model not found: {model_path}"
            return None
        try:
            from ultralytics import YOLO

            self._detector_model = YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._detector_error = f"YOLO OBB detector initialization failed: {exc}"
        return self._detector_model

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray | None:
        try:
            with Image.open(BytesIO(image_bytes)) as image_raw:
                image = ImageOps.exif_transpose(image_raw).convert("RGB")
                return np.asarray(image)
        except Exception:
            return None

    @staticmethod
    def _xyxy_from_polygon(polygon: list[list[float]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return (int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys))))

    def _detect_candidates_for_variant(
        self,
        image: np.ndarray,
        *,
        variant_name: str,
    ) -> tuple[list[_YoloCandidate], str | None]:
        if self.detector_backend == "onnx":
            if self._onnx_detector is None:
                self._onnx_detector = YoloObbOnnxDetector(
                    model_path=self.detector_onnx_path
                    or self.detector_model_path.with_suffix(".onnx"),
                    confidence_threshold=self.detector_confidence_threshold,
                    imgsz=self.detector_imgsz,
                    max_candidates=self.max_candidates,
                )
            detections, reason = self._onnx_detector.detect(image)
            candidates = [
                _YoloCandidate(
                    polygon_xy=item.polygon_xy,
                    bbox_xyxy=item.bbox_xyxy,
                    confidence=item.confidence,
                    variant_name=variant_name,
                )
                for item in detections
            ]
            self._debug_append(
                "yolo_obb",
                {
                    "stage": "onnx_after_threshold_cap",
                    "variant_name": variant_name,
                    "backend": "onnx",
                    "threshold": self.detector_confidence_threshold,
                    "max_candidates": self.max_candidates,
                    "reason": reason,
                    "boxes": [self._debug_yolo_candidate(candidate) for candidate in candidates],
                },
            )
            return candidates, reason
        if self.detector_backend != "ultralytics":
            return [], "detector_backend must be one of: ultralytics, onnx"

        model = self._ensure_detector()
        if self._detector_error:
            return [], self._detector_error
        if model is None:
            return [], "YOLO OBB detector unavailable"

        try:
            results = model.predict(
                image,
                conf=self.detector_confidence_threshold,
                imgsz=self.detector_imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return [], f"YOLO OBB detector inference failed: {exc}"
        if not results:
            return [], "YOLO OBB detector returned no result"

        obb = getattr(results[0], "obb", None)
        if obb is None or len(obb) == 0:
            return [], "no expiry region detected"
        polys_raw = getattr(obb, "xyxyxyxy", None)
        conf_raw = getattr(obb, "conf", None)
        if polys_raw is None or conf_raw is None:
            return [], "YOLO OBB output missing polygons or confidences"

        polygons = polys_raw.cpu().numpy().tolist()
        confidences = conf_raw.cpu().numpy().tolist()
        raw_candidates: list[_YoloCandidate] = []
        candidates: list[_YoloCandidate] = []
        height, width = image.shape[:2]
        for polygon, confidence in zip(polygons, confidences, strict=False):
            conf = float(confidence)
            cleaned = [
                [max(0.0, min(float(x), float(width))), max(0.0, min(float(y), float(height)))]
                for x, y in polygon
            ]
            bbox = self._xyxy_from_polygon(cleaned)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            raw_candidates.append(_YoloCandidate(polygon_xy=cleaned, bbox_xyxy=bbox, confidence=conf, variant_name=variant_name))
            if conf < self.detector_confidence_threshold:
                continue
            candidates.append(raw_candidates[-1])
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        capped = candidates[: self.max_candidates]
        self._debug_append(
            "yolo_obb",
            {
                "stage": "ultralytics_before_pipeline_cap",
                "variant_name": variant_name,
                "backend": "ultralytics",
                "model_call_conf": self.detector_confidence_threshold,
                "pipeline_threshold": self.detector_confidence_threshold,
                "instrumentation_limit": "Ultralytics outputs are captured before pipeline cap, but after the model call confidence threshold.",
                "max_candidates": self.max_candidates,
                "raw_count": len(raw_candidates),
                "after_threshold_count": len(candidates),
                "after_cap_count": len(capped),
                "raw_boxes": [self._debug_yolo_candidate(candidate) for candidate in raw_candidates],
                "after_threshold_boxes": [self._debug_yolo_candidate(candidate) for candidate in candidates],
                "after_cap_boxes": [self._debug_yolo_candidate(candidate) for candidate in capped],
            },
        )
        return capped, None if capped else "no candidate passed confidence threshold"

    def _detect_candidates(self, image: np.ndarray) -> tuple[list[_YoloCandidate], str | None]:
        candidates, reason = self._detect_candidates_for_variant(image, variant_name="yolo26s_obb")
        return candidates, reason

    @staticmethod
    def _rotate_full_image_for_detector(image: np.ndarray, orientation: str) -> np.ndarray:
        if orientation == "rot90_cw":
            return np.rot90(image, k=3)
        if orientation == "rot90_ccw":
            return np.rot90(image, k=1)
        raise ValueError(f"unsupported detector rotation rescue orientation: {orientation}")

    @staticmethod
    def _map_rotated_point_to_original(
        x: float,
        y: float,
        *,
        orientation: str,
        width: int,
        height: int,
    ) -> list[float]:
        if orientation == "rot90_cw":
            return [y, height - x]
        if orientation == "rot90_ccw":
            return [width - y, x]
        raise ValueError(f"unsupported detector rotation rescue orientation: {orientation}")

    @classmethod
    def _map_rotated_yolo_candidate_to_original(
        cls,
        candidate: _YoloCandidate,
        *,
        orientation: str,
        original_shape: tuple[int, ...],
    ) -> _YoloCandidate | None:
        height, width = original_shape[:2]
        polygon = [
            cls._map_rotated_point_to_original(
                float(x),
                float(y),
                orientation=orientation,
                width=width,
                height=height,
            )
            for x, y in candidate.polygon_xy
        ]
        cleaned = [
            [max(0.0, min(float(x), float(width))), max(0.0, min(float(y), float(height)))]
            for x, y in polygon
        ]
        bbox = cls._xyxy_from_polygon(cleaned)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return _YoloCandidate(
            polygon_xy=cleaned,
            bbox_xyxy=bbox,
            confidence=candidate.confidence,
            source="yolo26s_obb_rotation_rescue",
            sources=("yolo26s_obb", "rotation_rescue"),
            variant_name=orientation,
        )

    def _detect_full_image_rotation_rescue_candidates(
        self,
        image: np.ndarray,
    ) -> tuple[list[_YoloCandidate], str | None]:
        if not self.full_image_rotation_detector_rescue_enabled:
            return [], "full_image_rotation_detector_rescue_disabled"
        collected: list[_YoloCandidate] = []
        reasons: list[str] = []
        orientation_rows: list[dict[str, object]] = []
        for orientation in ("rot90_cw", "rot90_ccw"):
            rotated = self._rotate_full_image_for_detector(image, orientation)
            candidates, reason = self._detect_candidates_for_variant(rotated, variant_name=orientation)
            if reason:
                reasons.append(f"{orientation}:{reason}")
            mapped: list[_YoloCandidate] = []
            for candidate in candidates:
                mapped_candidate = self._map_rotated_yolo_candidate_to_original(
                    candidate,
                    orientation=orientation,
                    original_shape=image.shape,
                )
                if mapped_candidate is not None:
                    mapped.append(mapped_candidate)
            collected.extend(mapped)
            orientation_rows.append(
                {
                    "orientation": orientation,
                    "rotated_shape": list(rotated.shape[:2]),
                    "reason": reason,
                    "raw_count": len(candidates),
                    "mapped_count": len(mapped),
                    "mapped_boxes": [self._debug_yolo_candidate(candidate) for candidate in mapped],
                }
            )
        deduped = self._dedupe_yolo_candidates(collected)
        limit = max(self.max_candidates, self.expiry_global_probe_top_k, self.max_candidates * 2)
        capped = deduped[:limit]
        self._debug_append(
            "rotation_detector_rescue",
            {
                "trigger": "no_initial_yolo_candidates",
                "orientations": orientation_rows,
                "deduped_count": len(deduped),
                "after_cap_count": len(capped),
                "boxes": [self._debug_yolo_candidate(candidate) for candidate in capped],
            },
        )
        return capped, None if capped else "; ".join(reasons) or "rotation detector rescue found no boxes"

    def _detect_variant_rescue_candidates(self, image: np.ndarray) -> tuple[list[_YoloCandidate], str | None]:
        collected: list[_YoloCandidate] = []
        reasons: list[str] = []
        for variant in self.preprocessor.detector_variants(image):
            if variant.name == "raw":
                continue
            candidates, reason = self._detect_candidates_for_variant(variant.image, variant_name=variant.name)
            if reason:
                reasons.append(f"{variant.name}:{reason}")
            collected.extend(self._map_yolo_candidates_to_original(candidates, variant=variant, original_shape=image.shape))
        deduped = self._dedupe_yolo_candidates(collected)
        limit = max(self.max_candidates, self.expiry_global_probe_top_k, self.max_candidates * 2)
        return deduped[:limit], None if deduped else "; ".join(reasons) or "no expiry region detected"

    @staticmethod
    def _rect_integral_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    def _dot_matrix_detail_components(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        masks: list[np.ndarray] = []
        for kernel_size in (9, 15, 21):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _threshold, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(mask)
        detail_mask = np.maximum.reduce(masks)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(detail_mask, 8)
        components: list[tuple[float, float, float]] = []
        for idx in range(1, count):
            x, y, width, height, area = stats[idx]
            if area < 3 or area > 300:
                continue
            if width > 50 or height > 50:
                continue
            cx, cy = centers[idx]
            components.append((float(cx), float(cy), float(area)))
        if not components:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(components, dtype=np.float32)

    def _dot_matrix_fine_detail_components(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _threshold, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        count, _labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        components: list[tuple[float, float, float]] = []
        for idx in range(1, count):
            x, y, width, height, area = stats[idx]
            if area < 3 or area > 750:
                continue
            if width > 70 or height > 95:
                continue
            cx, cy = centers[idx]
            components.append((float(cx), float(cy), float(area)))
        if not components:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(components, dtype=np.float32)

    def _refined_dot_matrix_bbox(
        self,
        components: np.ndarray,
        window: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = window
        inside = components[
            (components[:, 0] >= x1)
            & (components[:, 0] <= x2)
            & (components[:, 1] >= y1)
            & (components[:, 1] <= y2)
        ]
        if len(inside) < 40:
            return None
        xs = inside[:, 0]
        ys = inside[:, 1]
        rx1 = int(np.floor(np.percentile(xs, 2.0)))
        rx2 = int(np.ceil(np.percentile(xs, 98.0)))
        ry1 = int(np.floor(np.percentile(ys, 2.0)))
        ry2 = int(np.ceil(np.percentile(ys, 98.0)))
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        width = max(1, rx2 - rx1)
        height = max(1, ry2 - ry1)
        pad_x = max(28, int(round(width * 0.18)))
        pad_y = max(18, int(round(height * 0.28)))
        return self._clip_bbox_to_image((rx1 - pad_x, ry1 - pad_y, rx2 + pad_x, ry2 + pad_y), image_shape)

    def _dot_matrix_rescue_sub_bboxes(
        self,
        components: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> list[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        if width < 900 and height < 420:
            return []
        sub_width = min(width, max(420, int(round(width * 0.58))))
        sub_height = min(height, max(180, int(round(height * 0.55))))
        x_positions = [x1]
        if width > sub_width:
            x_positions.extend([x1 + (width - sub_width) // 2, x2 - sub_width])
        y_positions = [y1]
        if height > sub_height:
            y_positions.extend([y1 + (height - sub_height) // 2, y2 - sub_height])
        out: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for sy in y_positions:
            for sx in x_positions:
                candidate = self._clip_bbox_to_image((sx, sy, sx + sub_width, sy + sub_height), image_shape)
                if candidate is None or candidate in seen:
                    continue
                cx1, cy1, cx2, cy2 = candidate
                inside = components[
                    (components[:, 0] >= cx1)
                    & (components[:, 0] <= cx2)
                    & (components[:, 1] >= cy1)
                    & (components[:, 1] <= cy2)
                ]
                if len(inside) < 45:
                    continue
                seen.add(candidate)
                out.append(candidate)
        return out

    def _dot_matrix_rescue_vertical_line_boxes(
        self,
        components: np.ndarray,
        *,
        image_shape: tuple[int, ...],
    ) -> list[tuple[float, tuple[int, int, int, int], dict[str, float], tuple[int, int, int, int]]]:
        height, width = image_shape[:2]
        if len(components) < 35 or height <= 0 or width <= 0:
            return []

        count_image = np.zeros((height, width), dtype=np.uint8)
        area_image = np.zeros((height, width), dtype=np.float32)
        for cx, cy, area in components:
            x = int(round(float(cx)))
            y = int(round(float(cy)))
            if 0 <= x < width and 0 <= y < height:
                count_image[y, x] = min(255, int(count_image[y, x]) + 1)
                area_image[y, x] += float(area)
        count_integral = cv2.integral(count_image, sdepth=cv2.CV_32S)
        area_integral = cv2.integral(area_image, sdepth=cv2.CV_64F)

        sizes: set[tuple[int, int]] = set()
        for width_ratio, height_ratio in ((0.045, 0.13), (0.06, 0.16), (0.075, 0.20), (0.09, 0.24)):
            window_width = max(110, min(width - 1, int(round(width * width_ratio))))
            window_height = max(360, min(height - 1, int(round(height * height_ratio))))
            if window_width > 0 and window_height > 0:
                sizes.add((window_width, window_height))

        scored: list[tuple[float, tuple[int, int, int, int], dict[str, float], tuple[int, int, int, int]]] = []
        for window_width, window_height in sorted(sizes):
            if window_width >= width or window_height >= height:
                continue
            step_x = max(40, window_width // 3)
            step_y = max(65, window_height // 5)
            for y1 in range(0, height - window_height + 1, step_y):
                for x1 in range(0, width - window_width + 1, step_x):
                    x2 = x1 + window_width
                    y2 = y1 + window_height
                    touches_edge = x1 <= 6 or y1 <= 6 or x2 >= width - 6 or y2 >= height - 6
                    if touches_edge:
                        continue
                    component_count = self._rect_integral_sum(count_integral, x1, y1, x2, y2)
                    if component_count < 35:
                        continue
                    component_area = self._rect_integral_sum(area_integral, x1, y1, x2, y2)
                    density = component_area / float(max(1, window_width * window_height))
                    if density < 0.006 or density > 0.28:
                        continue
                    inside = components[
                        (components[:, 0] >= x1)
                        & (components[:, 0] <= x2)
                        & (components[:, 1] >= y1)
                        & (components[:, 1] <= y2)
                    ]
                    if len(inside) < 35:
                        continue
                    span_x = float(np.percentile(inside[:, 0], 95.0) - np.percentile(inside[:, 0], 5.0))
                    span_y = float(np.percentile(inside[:, 1], 95.0) - np.percentile(inside[:, 1], 5.0))
                    verticality = span_y / max(span_x, 1.0)
                    if verticality < 1.45:
                        continue
                    center_x = float(np.percentile(inside[:, 0], 25.0))
                    center_y = float(np.percentile(inside[:, 1], 50.0))
                    mid_span_x = float(np.percentile(inside[:, 0], 60.0) - np.percentile(inside[:, 0], 2.0))
                    mid_span_y = float(np.percentile(inside[:, 1], 82.0) - np.percentile(inside[:, 1], 18.0))
                    line_width = min(165.0, max(115.0, mid_span_x + 50.0))
                    line_height = min(480.0, max(360.0, mid_span_y + 70.0))
                    line_bbox = self._clip_bbox_to_image(
                        (
                            int(round(center_x - line_width / 2.0)),
                            int(round(center_y - line_height / 2.0)),
                            int(round(center_x + line_width / 2.0)),
                            int(round(center_y + line_height / 2.0)),
                        ),
                        image_shape,
                    )
                    if line_bbox is None:
                        continue
                    score = float(component_count) * 0.025 + density * 24.0 + min(verticality, 6.0) * 0.8
                    scored.append(
                        (
                            score,
                            line_bbox,
                            {
                                "vertical_dot_line": 1.0,
                                "vertical_component_count": float(component_count),
                                "vertical_component_density": float(density),
                                "verticality": float(verticality),
                            },
                            (x1, y1, x2, y2),
                        )
                    )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, tuple[int, int, int, int], dict[str, float], tuple[int, int, int, int]]] = []
        for item in scored:
            bbox = item[1]
            if any(self._bbox_overlap(bbox, existing[1])[0] >= 0.35 for existing in selected):
                continue
            selected.append(item)
            if len(selected) >= 6:
                break
        return selected

    @staticmethod
    def _oriented_rect_polygon(
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        angle_degrees: float,
    ) -> tuple[tuple[float, float], ...]:
        theta = math.radians(angle_degrees)
        axis_x = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
        axis_y = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)
        center = np.asarray([center_x, center_y], dtype=np.float32)
        half_w = float(width) / 2.0
        half_h = float(height) / 2.0
        points = (
            center - axis_x * half_w - axis_y * half_h,
            center + axis_x * half_w - axis_y * half_h,
            center + axis_x * half_w + axis_y * half_h,
            center - axis_x * half_w + axis_y * half_h,
        )
        return tuple((float(point[0]), float(point[1])) for point in points)

    def _dot_matrix_rescue_oriented_line_proposals(
        self,
        bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> list[tuple[tuple[int, int, int, int], tuple[tuple[float, float], ...], dict[str, float]]]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        image_h, image_w = image_shape[:2]
        if width < 700 or height < 260:
            return []

        line_width = min(max(620.0, width * 0.68), width * 0.92, image_w * 0.75)
        line_height = min(240.0, max(120.0, height * 0.28))
        layouts = (
            (0.11, 0.26, 6.0),
            (0.11, 0.26, -6.0),
            (0.36, 0.26, 6.0),
            (0.11, 0.52, 6.0),
            (0.36, 0.52, 6.0),
            (0.11, 0.74, 6.0),
        )
        out: list[tuple[tuple[int, int, int, int], tuple[tuple[float, float], ...], dict[str, float]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for x_frac, y_frac, angle in layouts:
            center_x = x1 + width * x_frac
            center_y = y1 + height * y_frac
            polygon = self._oriented_rect_polygon(center_x, center_y, line_width, line_height, angle)
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            if min(xs) < 0 or min(ys) < 0 or max(xs) > image_w or max(ys) > image_h:
                continue
            candidate_bbox = self._clip_bbox_to_image(
                (
                    int(math.floor(min(xs))),
                    int(math.floor(min(ys))),
                    int(math.ceil(max(xs))),
                    int(math.ceil(max(ys))),
                ),
                image_shape,
            )
            if candidate_bbox is None or candidate_bbox in seen:
                continue
            seen.add(candidate_bbox)
            out.append(
                (
                    candidate_bbox,
                    polygon,
                    {
                        "oriented_line": 1.0,
                        "oriented_line_x_frac": float(x_frac),
                        "oriented_line_y_frac": float(y_frac),
                        "oriented_line_angle": float(angle),
                    },
                )
            )
        return out

    def _detect_dot_matrix_rescue_proposals(self, image: np.ndarray) -> tuple[list[ProposalBox], str | None]:
        if not self.dot_matrix_rescue_enabled:
            return [], "dot_matrix_rescue_disabled"
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return [], "empty_image"
        components = self._dot_matrix_detail_components(image)
        if len(components) < 40:
            self._debug_append(
                "dot_matrix_rescue",
                {"stage": "proposal_detection", "component_count": int(len(components)), "proposals": []},
            )
            return [], "not_enough_dot_matrix_detail"

        count_image = np.zeros((height, width), dtype=np.uint8)
        area_image = np.zeros((height, width), dtype=np.float32)
        for cx, cy, area in components:
            x = int(round(float(cx)))
            y = int(round(float(cy)))
            if 0 <= x < width and 0 <= y < height:
                count_image[y, x] = min(255, int(count_image[y, x]) + 1)
                area_image[y, x] += float(area)
        count_integral = cv2.integral(count_image, sdepth=cv2.CV_32S)
        area_integral = cv2.integral(area_image, sdepth=cv2.CV_64F)

        scored: list[tuple[float, tuple[int, int, int, int], dict[str, float]]] = []
        sizes: set[tuple[int, int]] = set()
        for width_ratio, height_ratio in ((0.22, 0.07), (0.32, 0.10), (0.45, 0.16), (0.38, 0.14)):
            window_width = max(220, min(width - 1, int(round(width * width_ratio))))
            window_height = max(90, min(height - 1, int(round(height * height_ratio))))
            if window_width <= 0 or window_height <= 0:
                continue
            sizes.add((window_width, window_height))

        for window_width, window_height in sorted(sizes):
            step_x = max(70, window_width // 4)
            step_y = max(45, window_height // 4)
            if window_width >= width or window_height >= height:
                continue
            for y1 in range(0, height - window_height + 1, step_y):
                for x1 in range(0, width - window_width + 1, step_x):
                    x2 = x1 + window_width
                    y2 = y1 + window_height
                    touches_edge = x1 <= 6 or y1 <= 6 or x2 >= width - 6 or y2 >= height - 6
                    if touches_edge:
                        continue
                    component_count = self._rect_integral_sum(count_integral, x1, y1, x2, y2)
                    if component_count < 55:
                        continue
                    component_area = self._rect_integral_sum(area_integral, x1, y1, x2, y2)
                    density = component_area / float(max(1, window_width * window_height))
                    if density < 0.002 or density > 0.18:
                        continue
                    inside = components[
                        (components[:, 0] >= x1)
                        & (components[:, 0] <= x2)
                        & (components[:, 1] >= y1)
                        & (components[:, 1] <= y2)
                    ]
                    if len(inside) < 55:
                        continue
                    try:
                        covariance = np.cov(inside[:, :2].T)
                        eigenvalues = np.linalg.eigvalsh(covariance)
                        elongation = float(eigenvalues[1] / max(float(eigenvalues[0]), 1e-3))
                    except Exception:
                        elongation = 1.0
                    lower_half_bonus = 1.4 if y1 >= height * 0.38 else 0.0
                    score = (
                        float(component_count) * 0.018
                        + density * 28.0
                        + min(elongation, 12.0) * 0.22
                        + lower_half_bonus
                    )
                    scored.append(
                        (
                            score,
                            (x1, y1, x2, y2),
                            {
                                "component_count": float(component_count),
                                "component_density": float(density),
                                "elongation": float(elongation),
                            },
                        )
                    )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[tuple[float, tuple[int, int, int, int], dict[str, float]]] = []

        def add_window(item: tuple[float, tuple[int, int, int, int], dict[str, float]]) -> None:
            if len(selected) >= self.dot_matrix_rescue_max_candidates:
                return
            bbox = item[1]
            for _score, existing, _features in selected:
                iou, containment = self._bbox_overlap(bbox, existing)
                if iou >= 0.40 or containment >= 0.80:
                    return
            selected.append(item)

        for item in scored[:4]:
            add_window(item)
        band_edges = [0, int(height * 0.34), int(height * 0.67), height]
        for band_start, band_end in zip(band_edges, band_edges[1:], strict=False):
            band_items = [
                item
                for item in scored
                if band_start <= ((item[1][1] + item[1][3]) / 2.0) < band_end
            ]
            for item in band_items[:2]:
                add_window(item)
        for item in scored:
            add_window(item)
            if len(selected) >= self.dot_matrix_rescue_max_candidates:
                break

        proposals: list[ProposalBox] = []
        proposal_debug: list[dict[str, object]] = []
        seen_refined: list[tuple[int, int, int, int]] = []
        scored_boxes: list[
            tuple[
                float,
                tuple[int, int, int, int],
                dict[str, float],
                tuple[int, int, int, int],
                tuple[tuple[float, float], ...] | None,
            ]
        ] = []
        for score, window, features in selected:
            refined = self._refined_dot_matrix_bbox(components, window, image_shape=image.shape)
            if refined is None:
                continue
            if refined[1] >= height - max(80, int(round(height * 0.08))):
                continue
            scored_boxes.append((score, refined, features, window, None))
            for line_bbox, line_polygon, line_features in self._dot_matrix_rescue_oriented_line_proposals(
                refined,
                image_shape=image.shape,
            ):
                scored_boxes.append((score * 1.04, line_bbox, {**features, **line_features}, window, line_polygon))
            for sub_bbox in self._dot_matrix_rescue_sub_bboxes(components, refined, image_shape=image.shape):
                scored_boxes.append((score * 0.97, sub_bbox, {**features, "sub_window": 1.0}, window, None))

        fine_components = self._dot_matrix_fine_detail_components(image)
        for score, line_bbox, line_features, window in self._dot_matrix_rescue_vertical_line_boxes(
            fine_components,
            image_shape=image.shape,
        ):
            if line_bbox[1] >= height - max(80, int(round(height * 0.08))):
                continue
            x1, y1, x2, y2 = line_bbox
            line_polygon = (
                (float(x1), float(y1)),
                (float(x2), float(y1)),
                (float(x2), float(y2)),
                (float(x1), float(y2)),
            )
            scored_boxes.append((score * 1.05, line_bbox, line_features, window, line_polygon))

        scored_boxes.sort(key=lambda item: item[0], reverse=True)
        for score, refined, features, window, polygon in scored_boxes:
            if any(self._bbox_overlap(refined, existing)[0] >= 0.45 for existing in seen_refined):
                continue
            seen_refined.append(refined)
            confidence = max(0.15, min(0.65, score / 35.0))
            proposal = ProposalBox(
                bbox_xyxy=refined,
                confidence=confidence,
                source="dot_matrix_rescue",
                sources=("dot_matrix_rescue",),
                variant_name="dot_matrix_rescue",
                polygon_xy=polygon,
            )
            proposals.append(proposal)
            proposal_debug.append(
                {
                    "window_bbox_xyxy": self._bbox_json(window),
                    "refined_bbox_xyxy": self._bbox_json(refined),
                    "score": round(float(score), 4),
                    "confidence": round(float(confidence), 4),
                    **{key: round(float(value), 4) for key, value in features.items()},
                }
            )
            if len(proposals) >= self.dot_matrix_rescue_max_candidates:
                break

        self._debug_append(
            "dot_matrix_rescue",
            {
                "stage": "proposal_detection",
                "component_count": int(len(components)),
                "fine_component_count": int(len(fine_components)),
                "scored_window_count": len(scored),
                "selected_window_count": len(selected),
                "proposal_count": len(proposals),
                "proposals": proposal_debug,
            },
        )
        self._profile_inc("num_dot_matrix_crops", len(proposals))
        return proposals, None if proposals else "dot_matrix_rescue_found_no_boxes"

    def _ensure_product_cropper_model(self) -> Any | None:
        if self._product_cropper_model is not None or self._product_cropper_error is not None:
            return self._product_cropper_model
        model_path = self.product_cropper_model_path
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not model_path.exists():
            self._product_cropper_error = f"product cropper model not found: {model_path}"
            return None
        try:
            from ultralytics import YOLO

            self._product_cropper_model = YOLO(str(model_path))
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._product_cropper_error = f"product cropper initialization failed: {exc}"
        return self._product_cropper_model

    @staticmethod
    def _float_value(value: Any) -> float:
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _list_value(value: Any) -> list[float]:
        if hasattr(value, "tolist"):
            return [float(item) for item in value.tolist()]
        return [float(item) for item in value]

    @staticmethod
    def _ratio_padded_bbox(
        bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
        padding_ratio: float,
    ) -> tuple[int, int, int, int] | None:
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(round(bw * padding_ratio))
        pad_y = int(round(bh * padding_ratio))
        padded = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )
        if padded[2] <= padded[0] or padded[3] <= padded[1]:
            return None
        return padded

    def _detect_product_cropper_rois(self, image: np.ndarray) -> tuple[list[_ProductCropperRoi], str | None]:
        if not self.product_cropper_rescue_enabled:
            return [], "product_cropper_rescue_disabled"
        model = self._ensure_product_cropper_model()
        if self._product_cropper_error:
            return [], self._product_cropper_error
        if model is None:
            return [], "product cropper unavailable"
        try:
            results = model.predict(
                image,
                conf=self.product_cropper_confidence_threshold,
                imgsz=self.product_cropper_imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return [], f"product cropper inference failed: {exc}"
        if not results:
            return [], "product cropper returned no result"

        rois: list[_ProductCropperRoi] = []
        height, width = image.shape[:2]
        obb = getattr(results[0], "obb", None)
        if obb is not None and len(obb) > 0:
            polys_raw = getattr(obb, "xyxyxyxy", None)
            conf_raw = getattr(obb, "conf", None)
            if polys_raw is not None and conf_raw is not None:
                polygons = polys_raw.cpu().numpy().tolist() if hasattr(polys_raw, "cpu") else polys_raw
                confidences = conf_raw.cpu().numpy().tolist() if hasattr(conf_raw, "cpu") else conf_raw
                for polygon, confidence in zip(polygons, confidences, strict=False):
                    conf = float(confidence)
                    if conf < self.product_cropper_confidence_threshold:
                        continue
                    clipped_polygon = [
                        [max(0.0, min(float(x), float(width))), max(0.0, min(float(y), float(height)))]
                        for x, y in polygon
                    ]
                    padded = self._ratio_padded_bbox(
                        self._xyxy_from_polygon(clipped_polygon),
                        image_shape=image.shape,
                        padding_ratio=self.product_cropper_padding_ratio,
                    )
                    if padded is not None:
                        rois.append(_ProductCropperRoi(padded, conf, "product_yolo_obb"))

        boxes = getattr(results[0], "boxes", None)
        if not rois and boxes is not None and len(boxes) > 0:
            for row in boxes:
                conf = self._float_value(getattr(row, "conf", 0.0))
                if conf < self.product_cropper_confidence_threshold:
                    continue
                raw_xyxy = getattr(row, "xyxy", None)
                if raw_xyxy is None:
                    continue
                xyxy_source = raw_xyxy[0] if isinstance(raw_xyxy, (list, tuple)) else raw_xyxy[0]
                values = self._list_value(xyxy_source)
                if len(values) < 4:
                    continue
                bbox = (
                    max(0, min(width, int(round(values[0])))),
                    max(0, min(height, int(round(values[1])))),
                    max(0, min(width, int(round(values[2])))),
                    max(0, min(height, int(round(values[3])))),
                )
                padded = self._ratio_padded_bbox(
                    bbox,
                    image_shape=image.shape,
                    padding_ratio=self.product_cropper_padding_ratio,
                )
                if padded is not None:
                    rois.append(_ProductCropperRoi(padded, conf, "product_yolo"))

        rois.sort(key=lambda roi: (roi.confidence, self._bbox_area(roi.bbox_xyxy)), reverse=True)
        return rois[:1], None if rois else "product cropper returned no product ROI"

    def _detect_product_cropper_rescue_candidates(self, image: np.ndarray) -> tuple[list[_YoloCandidate], str | None]:
        product_rois, roi_reason = self._detect_product_cropper_rois(image)
        if not product_rois:
            return [], roi_reason

        roi = product_rois[0]
        x1, y1, x2, y2 = roi.bbox_xyxy
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return [], "product cropper ROI crop empty"

        local_candidates, expiry_reason = self._detect_candidates_for_variant(
            crop,
            variant_name="product_cropper_rescue",
        )
        mapped: list[_YoloCandidate] = []
        height, width = image.shape[:2]
        for candidate in local_candidates[: self.product_cropper_max_candidates]:
            polygon = [
                [
                    max(0.0, min(float(px) + x1, float(width))),
                    max(0.0, min(float(py) + y1, float(height))),
                ]
                for px, py in candidate.polygon_xy
            ]
            bbox = self._xyxy_from_polygon(polygon)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            mapped.append(
                _YoloCandidate(
                    polygon_xy=polygon,
                    bbox_xyxy=bbox,
                    confidence=candidate.confidence,
                    source="product_cropper_rescue",
                    sources=("product_yolo", "yolo26s_obb"),
                    variant_name="product_cropper_rescue",
                )
            )
        deduped = self._dedupe_yolo_candidates(mapped)
        return (
            deduped[: self.product_cropper_max_candidates],
            None if deduped else expiry_reason or "no expiry region detected inside product ROI",
        )

    def _ranked_candidate_from_product_rescue(self, candidate: _YoloCandidate, *, index: int) -> _MobileRankedCandidate:
        return _MobileRankedCandidate(
            candidate_id=f"product_rescue_{index}",
            candidate_type="single",
            bbox_xyxy=candidate.bbox_xyxy,
            polygon_xy=tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
            detector_confidence=candidate.confidence,
            detector_sources=candidate.sources,
            detector_variant=candidate.variant_name,
            member_indices=[0],
            member_bboxes=[candidate.bbox_xyxy],
            geometry_features={"area_ratio": 0.0},
            geometry_score=0.0,
            score_breakdown={"product_cropper_direct": 1.0},
            total_score=1.0,
            recognition_bbox=candidate.bbox_xyxy,
            evidence_bbox=candidate.bbox_xyxy,
            final_crop_bbox=candidate.bbox_xyxy,
            final_crop_padding_px=0,
            final_crop_policy="product_cropper_rescue",
        )

    def _product_rescue_crops_for_candidate(
        self,
        image: np.ndarray,
        candidate: _YoloCandidate,
    ) -> list[NormalizedTextLineCrop]:
        polygon = tuple((float(x), float(y)) for x, y in candidate.polygon_xy) if candidate.polygon_xy else None
        return normalize_textline_crops(
            image,
            TextLineGeometry(
                bbox_xyxy=candidate.bbox_xyxy,
                polygon_xy=polygon,
            ),
            TextLineCropConfig(padding_px=0),
        )

    def _product_rescue_same_date_support(self, evidence: _DateEvidence, evidence_items: list[_DateEvidence]) -> int:
        if evidence.parsed.parsed_date is None:
            return 0
        return len(
            {
                item.recognition_variant
                for item in evidence_items
                if item.parsed.parsed_date == evidence.parsed.parsed_date
                and float(item.parsed.confidence or 0.0) >= 0.70
                and float(item.recognition.confidence or 0.0) >= 0.75
            }
        )

    def _is_strong_product_rescue_evidence(self, evidence: _DateEvidence, evidence_items: list[_DateEvidence]) -> bool:
        if evidence.crop_policy != "product_cropper_rescue" or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = self._date_evidence_text(evidence)
        if has_production_keyword(text) and not has_expiry_keyword(text):
            return False
        if self._unsupported_compact_future_penalty(evidence) < 0:
            return False
        parser_conf = float(evidence.parsed.confidence or 0.0)
        ocr_conf = float(evidence.recognition.confidence or 0.0)
        raw = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        if self._has_alpha_attached_leading_day_token(raw, evidence.parsed.parsed_date):
            return False
        if self._has_clean_full_date_pattern(raw) and parser_conf >= 0.90 and ocr_conf >= 0.90:
            return True
        return self._product_rescue_same_date_support(evidence, evidence_items) >= 2

    def _same_date_variant_support(
        self,
        evidence: _DateEvidence,
        evidence_items: list[_DateEvidence],
        *,
        min_confidence: float = 0.75,
    ) -> int:
        if evidence.parsed.parsed_date is None:
            return 0
        return len(
            {
                item.recognition_variant
                for item in evidence_items
                if item.parsed.parsed_date == evidence.parsed.parsed_date
                and float(item.parsed.confidence or 0.0) >= min_confidence
                and float(item.recognition.confidence or 0.0) >= min_confidence
            }
        )

    @staticmethod
    def _is_rapidocr_evidence(evidence: _DateEvidence) -> bool:
        return "rapidocr_ppocrv5" in evidence.candidate.detector_sources

    @staticmethod
    def _is_rapidocr_gated_evidence(evidence: _DateEvidence) -> bool:
        return (evidence.candidate.detector_variant or "").startswith("rapidocr_gated")

    @staticmethod
    def _is_product_rescue_evidence(evidence: _DateEvidence) -> bool:
        return evidence.crop_policy == "product_cropper_rescue" or evidence.candidate.detector_variant == "product_cropper_rescue"

    @staticmethod
    def _is_dot_matrix_rescue_evidence(evidence: _DateEvidence) -> bool:
        return evidence.crop_policy == "dot_matrix_rescue" or "dot_matrix_rescue" in evidence.candidate.detector_sources

    @staticmethod
    def _is_core_yolo_evidence(evidence: _DateEvidence) -> bool:
        return (
            "yolo26s_obb" in evidence.candidate.detector_sources
            and "rapidocr_ppocrv5" not in evidence.candidate.detector_sources
            and "rotation_rescue" not in evidence.candidate.detector_sources
            and "product_yolo" not in evidence.candidate.detector_sources
            and "dot_matrix_rescue" not in evidence.candidate.detector_sources
            and evidence.crop_policy
            not in {
                "legacy_wide_group",
                "local_group_wide_crop",
                "rotation_wide_crop",
                "product_cropper_rescue",
                "anchor_local_sibling",
                "dot_matrix_rescue",
            }
        )

    def _is_strong_dot_matrix_rescue_evidence(self, evidence: _DateEvidence, evidence_items: list[_DateEvidence]) -> bool:
        if not self._is_dot_matrix_rescue_evidence(evidence) or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = self._date_evidence_text(evidence)
        if has_production_keyword(text) and not (has_expiry_keyword(text) or has_weak_expiry_keyword(text)):
            return False
        if self._unsupported_compact_future_penalty(evidence) < 0:
            return False
        raw = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        parser_conf = float(evidence.parsed.confidence or 0.0)
        ocr_conf = float(evidence.recognition.confidence or 0.0)
        detector_conf = float(evidence.candidate.detector_confidence or 0.0)
        digit_count = len(re.findall(r"\d", raw))
        if (
            evidence.candidate.polygon_xy is not None
            and evidence.parsed.date_format_detected in {"TRAILING_DD/MM/YY", "TRAILING_DD/MM/YYYY"}
            and parser_conf >= 0.90
            and ocr_conf >= 0.74
            and detector_conf >= 0.45
            and 6 <= digit_count <= 10
        ):
            return True
        if self._has_clean_full_date_pattern(raw) and parser_conf >= 0.80 and ocr_conf >= 0.80:
            return True
        return self._same_date_variant_support(evidence, evidence_items, min_confidence=0.70) >= 2

    def _is_strong_rapidocr_gated_evidence(self, evidence: _DateEvidence, evidence_items: list[_DateEvidence]) -> bool:
        if not self._is_rapidocr_gated_evidence(evidence) or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = self._date_evidence_text(evidence)
        if has_production_keyword(text) and not (has_expiry_keyword(text) or has_weak_expiry_keyword(text)):
            return False
        if self._unsupported_compact_future_penalty(evidence) < 0:
            return False
        if self._is_strong_rapidocr_partial_completion_evidence(evidence):
            return True
        raw_parts = list(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        raw = " ".join(raw_parts)
        clean_supported_date = any(
            self._clean_supported_rapidocr_full_date_text(part, evidence.parsed.parsed_date)
            for part in raw_parts
        ) or self._clean_supported_rapidocr_full_date_text(raw, evidence.parsed.parsed_date)
        noisy_leading_supported_date = any(
            self._noisy_leading_day_supported_rapidocr_full_date_text(part, evidence.parsed.parsed_date)
            for part in raw_parts
        ) or self._noisy_leading_day_supported_rapidocr_full_date_text(raw, evidence.parsed.parsed_date)
        clean_single_date = any(self._clean_single_full_date_text(part) for part in raw_parts) or self._clean_single_full_date_text(raw)
        contains_supported_date_token = any(
            self._contains_supported_full_date_token(part, evidence.parsed.parsed_date) for part in raw_parts + [raw]
        )
        parser_conf = float(evidence.parsed.confidence or 0.0)
        ocr_conf = float(evidence.recognition.confidence or 0.0)
        if (has_expiry_keyword(text) or has_weak_expiry_keyword(text)) and contains_supported_date_token:
            return parser_conf >= 0.88 and ocr_conf >= 0.55
        if not clean_single_date and not clean_supported_date:
            if not any(self._is_strong_rapidocr_date_tail_evidence(evidence, part) for part in raw_parts + [raw]):
                return False
        if self._is_rapidocr_date_tail_evidence(evidence):
            return parser_conf >= 0.85 and ocr_conf >= 0.80
        if noisy_leading_supported_date:
            return parser_conf >= 0.90 and ocr_conf >= 0.80
        if clean_supported_date:
            return parser_conf >= 0.80 and ocr_conf >= 0.84
        if clean_single_date and contains_supported_date_token:
            return parser_conf >= 0.90 and ocr_conf >= 0.90
        if has_expiry_keyword(text) or has_weak_expiry_keyword(text):
            return parser_conf >= 0.75 and ocr_conf >= 0.75
        return parser_conf >= 0.85 and ocr_conf >= 0.95

    def _is_rapidocr_date_tail_evidence(self, evidence: _DateEvidence) -> bool:
        candidate = evidence.candidate
        crop = evidence.normalized_crop
        if not self._is_rapidocr_gated_evidence(evidence) or crop is None:
            return False
        if crop.bbox_xyxy[0] <= candidate.bbox_xyxy[0] + 4:
            return False
        return self._rapidocr_date_tail_geometry(candidate) is not None

    def _is_strong_rapidocr_date_tail_evidence(self, evidence: _DateEvidence, raw: str) -> bool:
        if not self._is_rapidocr_date_tail_evidence(evidence):
            return False
        cleaned = re.sub(r"^[A-Z]{1,4}(?=\d)", "", (raw or "").strip(), flags=re.IGNORECASE)
        return self._clean_single_full_date_text(cleaned)

    def _is_strong_risky_date_evidence(
        self,
        evidence: _DateEvidence,
        evidence_items: list[_DateEvidence],
        *,
        core_dates: set[date],
    ) -> bool:
        if evidence.parsed.parsed_date is None:
            return False
        if evidence.parsed.parsed_date in core_dates:
            return True
        if self._is_product_rescue_evidence(evidence):
            return self._is_strong_product_rescue_evidence(evidence, evidence_items)
        if self._is_dot_matrix_rescue_evidence(evidence):
            return self._is_strong_dot_matrix_rescue_evidence(evidence, evidence_items)
        if self._is_rapidocr_gated_evidence(evidence):
            return self._is_strong_rapidocr_gated_evidence(evidence, evidence_items)
        if evidence.crop_policy == "legacy_wide_group":
            return self._is_strong_legacy_wide_group_evidence(evidence) or self._same_date_variant_support(evidence, evidence_items) >= 2
        if evidence.crop_policy == "local_group_wide_crop":
            return self._is_strong_local_group_wide_crop_evidence(evidence)
        if evidence.crop_policy == "rotation_wide_crop":
            return self._is_strong_rotation_wide_crop_evidence(evidence)
        if evidence.crop_policy == "partial_day_month_right_expand":
            return self._is_strong_partial_day_month_right_expansion_evidence(evidence)
        if evidence.crop_policy == "anchor_local_sibling":
            return self._is_strong_anchor_local_sibling_evidence(evidence)
        text = self._date_evidence_text(evidence)
        if has_expiry_keyword(text):
            return True
        raw = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        if (
            self._date_precision_score(evidence.parsed) >= 3
            and self._has_clean_full_date_pattern(raw)
            and float(evidence.parsed.confidence or 0.0) >= 0.90
            and float(evidence.recognition.confidence or 0.0) >= 0.90
        ):
            return True
        return self._same_date_variant_support(evidence, evidence_items) >= 2

    def _evaluate_product_cropper_rescue_candidates_direct(
        self,
        image: np.ndarray,
        candidates: list[_YoloCandidate],
        *,
        today: date,
    ) -> list[_EvaluatedCandidate]:
        evaluated: list[_EvaluatedCandidate] = []
        for index, candidate in enumerate(candidates[: self.product_cropper_max_candidates], start=1):
            ranked = self._ranked_candidate_from_product_rescue(candidate, index=index)
            evidence_items: list[_DateEvidence] = []
            representative: _EvaluatedCandidate | None = None
            for crop in self._product_rescue_crops_for_candidate(image, candidate):
                if crop.image.size == 0:
                    continue
                item = self._best_recognition_for_single_crop(
                    crop,
                    today=today,
                    crop_policy="product_cropper_rescue",
                    allowed_variant_names=PRODUCT_CROPPER_RECOGNITION_VARIANTS,
                )
                item.candidate = ranked
                if representative is None:
                    representative = item
                for evidence in item.date_evidence:
                    evidence.candidate = ranked
                    evidence.crop_policy = "product_cropper_rescue"
                    evidence_items.append(evidence)
            accepted = [
                evidence
                for evidence in evidence_items
                if self._is_strong_product_rescue_evidence(evidence, evidence_items)
            ]
            if not accepted:
                continue
            best_evidence = self._select_date_evidence(accepted)
            selected = self._evaluated_from_date_evidence(best_evidence, date_evidence=accepted)
            selected.candidate = ranked
            evaluated.append(selected)
        return evaluated

    @staticmethod
    def _padded_bbox(
        bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
        padding_px: int,
    ) -> tuple[int, int, int, int] | None:
        h, w = image_shape[:2]
        x1 = max(0, int(bbox[0]) - padding_px)
        y1 = max(0, int(bbox[1]) - padding_px)
        x2 = min(w, int(bbox[2]) + padding_px)
        y2 = min(h, int(bbox[3]) + padding_px)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _proposal_polygon_with_offset(proposal: ProposalBox, *, offset_x: int, offset_y: int) -> list[list[float]]:
        if proposal.polygon_xy:
            return [[float(x) + offset_x, float(y) + offset_y] for x, y in proposal.polygon_xy]
        x1, y1, x2, y2 = proposal.bbox_xyxy
        return [
            [float(x1 + offset_x), float(y1 + offset_y)],
            [float(x2 + offset_x), float(y1 + offset_y)],
            [float(x2 + offset_x), float(y2 + offset_y)],
            [float(x1 + offset_x), float(y2 + offset_y)],
        ]

    def _ensure_rapidocr_detector(self) -> RapidOCRTextProposalDetector | None:
        if self._rapidocr_detector is not None or self._rapidocr_error is not None:
            return self._rapidocr_detector
        try:
            self._rapidocr_detector = RapidOCRTextProposalDetector(
                ocr_version=self.rapidocr_ocr_version,
                model_type=self.rapidocr_model_type,
                lang_type=self.rapidocr_lang_type,
                limit_side_len=self.rapidocr_limit_side_len,
                limit_type=self.rapidocr_limit_type,
                max_candidates=self.rapidocr_max_candidates,
                max_boxes=max(
                    self.rapidocr_max_candidates,
                    self.rapidocr_primary_max_boxes_accepted,
                    self.rapidocr_max_boxes_accepted,
                    self.rapidocr_gated_max_boxes_accepted,
                    self.rapidocr_gated_prefilter_probe_top_k,
                ),
                min_confidence=self.rapidocr_min_confidence,
            )
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._rapidocr_error = f"RapidOCR text detector initialization failed: {exc}"
        return self._rapidocr_detector

    def _detect_rapidocr_roi_candidates(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
        *,
        enabled: bool,
        max_rois: int,
        max_boxes: int,
        timeout_seconds: float,
        stage: str,
    ) -> tuple[list[_YoloCandidate], str | None]:
        debug: dict[str, object] = {
            "triggered": False,
            "runtime_ms": 0,
            "roi_count": 0,
            "box_count": 0,
            "accepted_box_count": 0,
            "timed_out": False,
            "reason": None,
        }
        if stage == "primary":
            self._last_rapidocr_primary_debug = debug
        else:
            self._last_rapidocr_rescue_debug = debug
        if not enabled:
            debug["reason"] = f"rapidocr_{stage}_disabled"
            return [], f"rapidocr_{stage}_disabled"
        if not yolo_candidates:
            debug["reason"] = f"rapidocr_{stage}_skipped_no_yolo_roi"
            return [], f"rapidocr_{stage}_skipped_no_yolo_roi"
        if max_rois <= 0:
            debug["reason"] = f"rapidocr_{stage}_skipped_max_rois_zero"
            return [], f"rapidocr_{stage}_skipped_max_rois_zero"

        detector = self._ensure_rapidocr_detector()
        if detector is None:
            reason = self._rapidocr_error or "RapidOCR text detector unavailable"
            debug["reason"] = reason
            return [], reason

        collected: list[_YoloCandidate] = []
        reasons: list[str] = []
        rois = sorted(yolo_candidates, key=lambda item: item.confidence, reverse=True)[:max_rois]
        debug["triggered"] = True
        debug["roi_count"] = len(rois)
        for roi_idx, roi in enumerate(rois, start=1):
            padded = self._padded_bbox(
                roi.bbox_xyxy,
                image_shape=image.shape,
                padding_px=max(self.crop_padding_px, 8),
            )
            if padded is None:
                reasons.append(f"roi_{roi_idx}:invalid_bbox")
                continue
            x1, y1, x2, y2 = padded
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                reasons.append(f"roi_{roi_idx}:empty_crop")
                continue

            started = perf_counter()
            boxes, reason = detector.detect(crop)
            runtime_ms = (perf_counter() - started) * 1000.0
            self._profile_add(
                "rapidocr_primary_ms" if stage == "primary" else "rapidocr_rescue_ms",
                runtime_ms,
            )
            self._profile_inc("num_pp_boxes", len(boxes))
            self._debug_append(
                "rapidocr_text_boxes",
                {
                    "stage": stage,
                    "roi_index": roi_idx,
                    "source_yolo_bbox_xyxy": self._bbox_json(roi.bbox_xyxy),
                    "roi_padded_bbox_xyxy": self._bbox_json(padded),
                    "runtime_ms": round(runtime_ms, 3),
                    "reason": reason,
                    "raw_boxes": [
                        {
                            "bbox_xyxy": self._bbox_json(box.bbox_xyxy),
                            "full_bbox_xyxy": self._bbox_json(
                                (
                                    int(box.bbox_xyxy[0] + x1),
                                    int(box.bbox_xyxy[1] + y1),
                                    int(box.bbox_xyxy[2] + x1),
                                    int(box.bbox_xyxy[3] + y1),
                                )
                            ),
                            "confidence": box.confidence,
                        }
                        for box in boxes
                    ],
                },
            )
            debug["runtime_ms"] = int(debug.get("runtime_ms", 0) or 0) + int(runtime_ms)
            debug["box_count"] = int(debug.get("box_count", 0) or 0) + len(boxes)
            if reason:
                reasons.append(f"roi_{roi_idx}:{reason}")
            if runtime_ms > timeout_seconds * 1000.0:
                reasons.append(f"roi_{roi_idx}:rapidocr_{stage}_timeout_ms={runtime_ms:.0f}")
                debug["timed_out"] = True
                continue

            accepted_from_roi = 0
            for box in boxes:
                if accepted_from_roi >= max_boxes:
                    break
                bx1, by1, bx2, by2 = box.bbox_xyxy
                full_bbox = (
                    int(bx1 + x1),
                    int(by1 + y1),
                    int(bx2 + x1),
                    int(by2 + y1),
                )
                full_bbox = (
                    max(0, min(full_bbox[0], image.shape[1])),
                    max(0, min(full_bbox[1], image.shape[0])),
                    max(0, min(full_bbox[2], image.shape[1])),
                    max(0, min(full_bbox[3], image.shape[0])),
                )
                if full_bbox[2] <= full_bbox[0] or full_bbox[3] <= full_bbox[1]:
                    continue
                if self._bbox_area(full_bbox) < 12:
                    continue
                collected.append(
                    _YoloCandidate(
                        polygon_xy=self._proposal_polygon_with_offset(box, offset_x=x1, offset_y=y1),
                        bbox_xyxy=full_bbox,
                        confidence=float(box.confidence if box.confidence is not None else roi.confidence * 0.75),
                        source="rapidocr_roi",
                        sources=("yolo26s_obb", "rapidocr_ppocrv5"),
                        variant_name=f"rapidocr_{stage}:{self.rapidocr_ocr_version}_{self.rapidocr_model_type}",
                    )
                )
                accepted_from_roi += 1

        collected.sort(key=lambda item: (item.confidence, self._bbox_area(item.bbox_xyxy)), reverse=True)
        deduped = self._dedupe_yolo_candidates(collected)
        accepted = deduped[:max_boxes]
        final_reason = None if accepted else "; ".join(reasons) or "RapidOCR returned no text boxes"
        debug["accepted_box_count"] = len(accepted)
        debug["reason"] = final_reason
        return accepted, final_reason

    def _detect_rapidocr_primary_candidates(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
    ) -> tuple[list[_YoloCandidate], str | None]:
        return self._detect_rapidocr_roi_candidates(
            image,
            yolo_candidates,
            enabled=self.rapidocr_primary_enabled,
            max_rois=self.rapidocr_primary_max_rois_per_scan,
            max_boxes=self.rapidocr_primary_max_boxes_accepted,
            timeout_seconds=self.rapidocr_primary_timeout_seconds,
            stage="primary",
        )

    def _detect_rapidocr_rescue_candidates(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
    ) -> tuple[list[_YoloCandidate], str | None]:
        return self._detect_rapidocr_roi_candidates(
            image,
            yolo_candidates,
            enabled=self.rapidocr_rescue_enabled,
            max_rois=self.rapidocr_max_rois_per_scan,
            max_boxes=self.rapidocr_max_boxes_accepted,
            timeout_seconds=self.rapidocr_timeout_seconds,
            stage="rescue",
        )

    def _collect_advanced_roi_candidates(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
    ) -> list[_MobileROICandidate]:
        rois: list[_MobileROICandidate] = []
        selected = sorted(yolo_candidates, key=lambda item: item.confidence, reverse=True)[: self.rapidocr_primary_max_rois_per_scan]
        for index, candidate in enumerate(selected, start=1):
            padded = self._padded_bbox(
                candidate.bbox_xyxy,
                image_shape=image.shape,
                padding_px=max(self.crop_padding_px, 8),
            )
            if padded is None:
                continue
            x1, y1, x2, y2 = padded
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rois.append(
                _MobileROICandidate(
                    source=f"detector_box_{index}",
                    image=crop,
                    confidence=candidate.confidence,
                    offset_x=x1,
                    offset_y=y1,
                    polygon_xy=candidate.polygon_xy,
                )
            )
        if not rois:
            rois.append(
                _MobileROICandidate(
                    source="full_image",
                    image=image,
                    confidence=yolo_candidates[0].confidence if yolo_candidates else None,
                    offset_x=0,
                    offset_y=0,
                    polygon_xy=yolo_candidates[0].polygon_xy if yolo_candidates else None,
                )
            )
        return rois

    def _detector_variants_for_roi(self, roi: np.ndarray) -> list[ImageVariant]:
        variants = list(self.preprocessor.detector_variants(roi))
        allowed = set(self.expiry_detector_variants)
        if allowed:
            variants = [variant for variant in variants if variant.name in allowed]
        if not variants:
            variants = [ImageVariant("raw", roi, purpose="detector")]
        return variants[: self.expiry_max_detector_variants]

    @classmethod
    def _map_rapidocr_proposals_to_roi_original(
        cls,
        proposals: list[ProposalBox],
        *,
        variant: ImageVariant,
        original_shape: tuple[int, ...],
        detector_variant: str,
    ) -> list[ProposalBox]:
        height, width = original_shape[:2]
        scale_x = float(variant.scale_x or 1.0)
        scale_y = float(variant.scale_y or 1.0)
        mapped: list[ProposalBox] = []
        for proposal in proposals:
            if proposal.polygon_xy:
                polygon = tuple(
                    (
                        max(0.0, min(float(x) / scale_x, float(width))),
                        max(0.0, min(float(y) / scale_y, float(height))),
                    )
                    for x, y in proposal.polygon_xy
                )
                bbox = cls._xyxy_from_polygon([[x, y] for x, y in polygon])
            else:
                x1, y1, x2, y2 = proposal.bbox_xyxy
                bbox = (
                    max(0, min(int(round(x1 / scale_x)), width)),
                    max(0, min(int(round(y1 / scale_y)), height)),
                    max(0, min(int(round(x2 / scale_x)), width)),
                    max(0, min(int(round(y2 / scale_y)), height)),
                )
                polygon = None
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            mapped.append(
                ProposalBox(
                    bbox_xyxy=bbox,
                    confidence=proposal.confidence,
                    source="rapidocr_ppocrv5",
                    sources=("yolo26s_obb", "rapidocr_ppocrv5"),
                    variant_name=detector_variant,
                    polygon_xy=polygon,
                )
            )
        return mapped

    def _detect_rapidocr_primary_proposals_for_roi(
        self,
        roi_image: np.ndarray,
        *,
        variant: ImageVariant,
    ) -> tuple[list[ProposalBox], str | None]:
        detector = self._ensure_rapidocr_detector()
        if detector is None:
            return [], self._rapidocr_error or "RapidOCR text detector unavailable"
        started = perf_counter()
        boxes, reason = detector.detect(variant.image)
        runtime_ms = (perf_counter() - started) * 1000.0
        self._profile_add("rapidocr_primary_ms", runtime_ms)
        self._profile_inc("num_pp_boxes", len(boxes))
        self._debug_append(
            "rapidocr_text_boxes",
            {
                "stage": "primary_roi_variant",
                "variant": variant.name,
                "runtime_ms": round(runtime_ms, 3),
                "timeout_seconds": self.rapidocr_primary_timeout_seconds,
                "reason": reason,
                "raw_boxes": [
                    {
                        "bbox_xyxy": self._bbox_json(box.bbox_xyxy),
                        "confidence": box.confidence,
                    }
                    for box in boxes
                ],
            },
        )
        if runtime_ms > self.rapidocr_primary_timeout_seconds * 1000.0:
            return [], f"rapidocr_primary_timeout_ms={runtime_ms:.0f}"
        detector_variant = f"rapidocr_primary:{self.rapidocr_ocr_version}_{self.rapidocr_model_type}:{variant.name}"
        proposals = self._map_rapidocr_proposals_to_roi_original(
            boxes[: self.rapidocr_primary_max_boxes_accepted],
            variant=variant,
            original_shape=roi_image.shape,
            detector_variant=detector_variant,
        )
        self._debug_append(
            "rapidocr_text_boxes",
            {
                "stage": "primary_roi_variant_after_cap_map",
                "variant": variant.name,
                "max_boxes": self.rapidocr_primary_max_boxes_accepted,
                "mapped_proposals": [self._debug_proposal(proposal) for proposal in proposals],
            },
        )
        return proposals, reason if not proposals else None

    @staticmethod
    def _whole_yolo_roi_proposal(
        roi_image: np.ndarray,
        *,
        roi_confidence: float | None,
    ) -> ProposalBox:
        height, width = roi_image.shape[:2]
        return ProposalBox(
            bbox_xyxy=(0, 0, int(width), int(height)),
            confidence=roi_confidence,
            source="yolo_roi_whole",
            sources=("yolo26s_obb",),
            variant_name="yolo_roi_whole",
            polygon_xy=None,
        )

    def _combined_primary_proposals_for_roi(
        self,
        roi_image: np.ndarray,
        *,
        variant: ImageVariant,
        roi_confidence: float | None,
    ) -> tuple[list[ProposalBox], str | None]:
        proposals = [self._whole_yolo_roi_proposal(roi_image, roi_confidence=roi_confidence)]
        if not self.rapidocr_primary_enabled:
            return proposals, "rapidocr_primary_disabled"
        rapidocr_proposals, reason = self._detect_rapidocr_primary_proposals_for_roi(roi_image, variant=variant)
        proposals.extend(rapidocr_proposals)
        return proposals, reason

    def _detect_rapidocr_full_image_gated_proposals(
        self,
        image: np.ndarray,
        *,
        yolo_candidates: list[_YoloCandidate],
        today: date,
    ) -> tuple[list[ProposalBox], str | None]:
        if not self.rapidocr_primary_enabled or self.rapidocr_primary_mode == "off":
            return [], "ppmobile_gated_disabled"
        detector = self._ensure_rapidocr_detector()
        if detector is None:
            return [], self._rapidocr_error or "RapidOCR text detector unavailable"

        started = perf_counter()
        boxes, reason = detector.detect(image)
        runtime_ms = (perf_counter() - started) * 1000.0
        self._profile_add("pp_mobile_full_image_ms", runtime_ms)
        self._profile_inc("num_pp_boxes", len(boxes))
        mapped = [
            ProposalBox(
                bbox_xyxy=box.bbox_xyxy,
                confidence=box.confidence,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name=f"rapidocr_gated:{self.rapidocr_ocr_version}_{self.rapidocr_model_type}",
                polygon_xy=box.polygon_xy,
            )
            for box in boxes
        ]
        geometry_filtered: list[ProposalBox] = []
        filtered: list[ProposalBox] = []
        if runtime_ms <= self.rapidocr_primary_timeout_seconds * 1000.0:
            geometry_filtered = self._geometry_prefilter_rapidocr_gated_proposals(
                image,
                mapped,
                yolo_candidates=yolo_candidates,
            )
            filtered = self._ocr_prefilter_rapidocr_gated_proposals(image, geometry_filtered, today=today)
        self._debug_append(
            "rapidocr_text_boxes",
            {
                "stage": "gated_full_image",
                "runtime_ms": round(runtime_ms, 3),
                "timeout_seconds": self.rapidocr_primary_timeout_seconds,
                "reason": reason,
                "raw_box_count": len(boxes),
                "geometry_filtered_box_count": len(geometry_filtered),
                "filtered_box_count": len(filtered),
                "max_boxes": self.rapidocr_gated_max_boxes_accepted,
                "raw_boxes": [self._debug_proposal(proposal) for proposal in mapped],
                "geometry_filtered_boxes": [self._debug_proposal(proposal) for proposal in geometry_filtered],
                "filtered_boxes": [self._debug_proposal(proposal) for proposal in filtered],
            },
        )
        if runtime_ms > self.rapidocr_primary_timeout_seconds * 1000.0:
            return [], f"ppmobile_gated_timeout_ms={runtime_ms:.0f}"
        return filtered, reason if not filtered else None

    def _geometry_prefilter_rapidocr_gated_proposals(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
        *,
        yolo_candidates: list[_YoloCandidate],
    ) -> list[ProposalBox]:
        height, width = image.shape[:2]
        image_area = max(1, int(height) * int(width))
        grouped = self._group_rapidocr_same_line_proposals(proposals)
        pool = [*proposals, *grouped]
        self._rapidocr_gated_prefilter_debug = {
            "raw_count": len(proposals),
            "grouped_count": len(grouped),
        }
        scored: list[tuple[float, ProposalBox]] = []
        for proposal in pool:
            bbox = self._clip_bbox_to_image(proposal.bbox_xyxy, image.shape)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            area = box_w * box_h
            area_ratio = area / float(image_area)
            confidence = float(proposal.confidence or 0.0)

            if area_ratio > 0.20:
                continue
            if box_w > width * 0.92 and box_h > height * 0.12:
                continue
            if box_h > height * 0.55 or box_w > width * 0.98:
                continue
            if area < 12:
                continue
            if area < 64 and confidence < 0.70:
                continue
            aspect = box_w / float(box_h)
            inverse_aspect = box_h / float(box_w)
            horizontal_strip = 1.0 if 1.8 <= aspect <= 18.0 and box_h <= height * 0.20 else 0.0
            vertical_strip = 0.85 if 1.8 <= inverse_aspect <= 18.0 and box_w <= width * 0.20 else 0.0
            compact_size = 1.0 - min(area_ratio / 0.08, 1.0)
            proximity = 0.0
            if yolo_candidates:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                image_diag = max(1.0, math.hypot(width, height))
                best_distance = min(
                    math.hypot(cx - ((yx1 + yx2) / 2.0), cy - ((yy1 + yy2) / 2.0))
                    for yx1, yy1, yx2, yy2 in (candidate.bbox_xyxy for candidate in yolo_candidates)
                )
                proximity = max(0.0, 1.0 - best_distance / (image_diag * 0.35))
            score = (
                confidence * 2.0
                + max(horizontal_strip, vertical_strip) * 2.0
                + compact_size
                + proximity * 0.1
                - area_ratio * 3.0
            )
            if bbox != proposal.bbox_xyxy:
                proposal = ProposalBox(
                    bbox_xyxy=bbox,
                    confidence=proposal.confidence,
                    source=proposal.source,
                    sources=proposal.sources,
                    variant_name=proposal.variant_name,
                    polygon_xy=proposal.polygon_xy,
                )
            scored.append((score, proposal))

        scored.sort(key=lambda item: item[0], reverse=True)
        raw_scored = [item for item in scored if not self._is_rapidocr_grouped_proposal(item[1])]
        grouped_scored = [item for item in scored if self._is_rapidocr_grouped_proposal(item[1])]
        budget = self.rapidocr_gated_prefilter_probe_top_k
        raw_reserve = min(len(raw_scored), budget)
        if len(raw_scored) > budget:
            raw_reserve = max(1, int(budget * 0.85))
        ordered_scored = [
            *raw_scored[:raw_reserve],
            *grouped_scored,
            *raw_scored[raw_reserve:],
        ]
        deduped: list[ProposalBox] = []
        for _score, proposal in ordered_scored:
            if any(self._rapidocr_gated_geometry_dedupe_should_skip(proposal, existing) for existing in deduped):
                continue
            deduped.append(proposal)
            if len(deduped) >= self.rapidocr_gated_prefilter_probe_top_k:
                break
        return deduped

    def _prefilter_rapidocr_gated_proposals(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
        *,
        yolo_candidates: list[_YoloCandidate],
    ) -> list[ProposalBox]:
        geometry_filtered = self._geometry_prefilter_rapidocr_gated_proposals(
            image,
            proposals,
            yolo_candidates=yolo_candidates,
        )
        return self._ocr_prefilter_rapidocr_gated_proposals(image, geometry_filtered, today=date.today())

    @staticmethod
    def _rapidocr_probe_metadata_key(proposal: ProposalBox | _MobileRankedCandidate) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in proposal.bbox_xyxy)

    @staticmethod
    def _proposal_angle_degrees(proposal: ProposalBox) -> float:
        polygon = proposal.polygon_xy
        if not polygon or len(polygon) < 2:
            return 0.0
        p0 = polygon[0]
        p1 = polygon[1]
        return float(math.degrees(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0]))))

    def _group_rapidocr_same_line_proposals(self, proposals: list[ProposalBox]) -> list[ProposalBox]:
        candidates = [proposal for proposal in proposals if self._bbox_area(proposal.bbox_xyxy) >= 12]
        if len(candidates) < 2:
            return []
        ordered = sorted(
            candidates,
            key=lambda proposal: (
                (proposal.bbox_xyxy[1] + proposal.bbox_xyxy[3]) / 2.0,
                proposal.bbox_xyxy[0],
            ),
        )
        lines: list[list[ProposalBox]] = []
        for proposal in ordered:
            x1, y1, x2, y2 = proposal.bbox_xyxy
            height = max(1, y2 - y1)
            cy = (y1 + y2) / 2.0
            angle = self._proposal_angle_degrees(proposal)
            placed = False
            for line in lines:
                line_boxes = [item.bbox_xyxy for item in line]
                line_height = sum(max(1, item[3] - item[1]) for item in line_boxes) / len(line_boxes)
                line_cy = sum((item[1] + item[3]) / 2.0 for item in line_boxes) / len(line_boxes)
                line_angle = sum(self._proposal_angle_degrees(item) for item in line) / len(line)
                last = max(line, key=lambda item: item.bbox_xyxy[2])
                gap = max(0, x1 - last.bbox_xyxy[2])
                y_overlap = max(0, min(max(item[3] for item in line_boxes), y2) - max(min(item[1] for item in line_boxes), y1))
                if (
                    abs(cy - line_cy) <= max(height, line_height) * 0.75
                    and y_overlap >= min(height, line_height) * 0.25
                    and gap <= max(height, line_height) * 4.0
                    and abs(angle - line_angle) <= 35.0
                ):
                    line.append(proposal)
                    placed = True
                    break
            if not placed:
                lines.append([proposal])

        grouped: list[ProposalBox] = []
        group_index = 1
        for line in lines:
            if len(line) < 2:
                continue
            line_sorted = sorted(line, key=lambda item: item.bbox_xyxy[0])
            group_windows: list[list[ProposalBox]] = []
            for window_size in (2, 3):
                if len(line_sorted) < window_size:
                    continue
                for start in range(0, len(line_sorted) - window_size + 1):
                    group_windows.append(line_sorted[start : start + window_size])
            if len(line_sorted) > 3:
                group_windows.append(line_sorted)

            for group in group_windows:
                bbox = self._union_many_bboxes([item.bbox_xyxy for item in group])
                if any(self._scan_bboxes_are_duplicates(bbox, item.bbox_xyxy) for item in group):
                    # Keep the group only when it actually extends beyond the single pieces.
                    if self._bbox_area(bbox) <= max(self._bbox_area(item.bbox_xyxy) for item in group) * 1.15:
                        continue
                confidence_values = [float(item.confidence or 0.0) for item in group]
                polygon = self._rapidocr_group_polygon(group)
                grouped.append(
                    ProposalBox(
                        bbox_xyxy=bbox,
                        confidence=max(confidence_values) if confidence_values else None,
                        source="rapidocr_ppocrv5_grouped",
                        sources=("rapidocr_ppocrv5",),
                        variant_name=f"rapidocr_gated_grouped:{group_index}",
                        polygon_xy=polygon,
                    )
                )
                group_index += 1
        return grouped[: max(0, self.rapidocr_gated_prefilter_probe_top_k * 2)]

    @staticmethod
    def _is_rapidocr_grouped_proposal(proposal: ProposalBox) -> bool:
        return (proposal.variant_name or "").startswith("rapidocr_gated_grouped")

    def _rapidocr_group_dedupe_should_keep_both(self, proposal: ProposalBox, existing: ProposalBox) -> bool:
        proposal_grouped = self._is_rapidocr_grouped_proposal(proposal)
        existing_grouped = self._is_rapidocr_grouped_proposal(existing)
        if proposal_grouped != existing_grouped:
            return True
        if not (proposal_grouped and existing_grouped):
            return False
        area_proposal = max(1, self._bbox_area(proposal.bbox_xyxy))
        area_existing = max(1, self._bbox_area(existing.bbox_xyxy))
        area_ratio = max(area_proposal, area_existing) / float(max(1, min(area_proposal, area_existing)))
        return area_ratio > 1.35

    @staticmethod
    def _is_rapidocr_gated_proposal(proposal: ProposalBox) -> bool:
        return (proposal.variant_name or "").startswith("rapidocr_gated")

    def _rapidocr_gated_geometry_dedupe_should_skip(self, proposal: ProposalBox, existing: ProposalBox) -> bool:
        proposal_grouped = self._is_rapidocr_grouped_proposal(proposal)
        existing_grouped = self._is_rapidocr_grouped_proposal(existing)
        if proposal_grouped != existing_grouped:
            return False

        iou, containment = self._bbox_overlap(proposal.bbox_xyxy, existing.bbox_xyxy)
        area_proposal = max(1, self._bbox_area(proposal.bbox_xyxy))
        area_existing = max(1, self._bbox_area(existing.bbox_xyxy))
        area_ratio = max(area_proposal, area_existing) / float(max(1, min(area_proposal, area_existing)))

        if proposal_grouped and existing_grouped:
            return self._scan_bboxes_are_duplicates(
                proposal.bbox_xyxy,
                existing.bbox_xyxy,
            ) and area_ratio <= 1.35

        if self._is_rapidocr_gated_proposal(proposal) and self._is_rapidocr_gated_proposal(existing):
            return iou >= 0.75 or (containment >= 0.92 and area_ratio <= 1.35)

        return self._scan_bboxes_are_duplicates(proposal.bbox_xyxy, existing.bbox_xyxy)

    @staticmethod
    def _rapidocr_group_polygon(proposals: list[ProposalBox]) -> tuple[tuple[float, float], ...] | None:
        points: list[tuple[float, float]] = []
        for proposal in proposals:
            if proposal.polygon_xy:
                points.extend((float(x), float(y)) for x, y in proposal.polygon_xy)
            else:
                x1, y1, x2, y2 = proposal.bbox_xyxy
                points.extend(
                    (
                        (float(x1), float(y1)),
                        (float(x2), float(y1)),
                        (float(x2), float(y2)),
                        (float(x1), float(y2)),
                    )
                )
        if len(points) < 4:
            return None
        rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
        box = cv2.boxPoints(rect)
        return tuple((float(x), float(y)) for x, y in box)

    def _normalized_rapidocr_polygon_probe_crops(
        self,
        image: np.ndarray,
        proposal: ProposalBox,
    ) -> list[NormalizedTextLineCrop]:
        bbox = self._clip_bbox_to_image(proposal.bbox_xyxy, image.shape)
        if bbox is None:
            return []
        polygon = proposal.polygon_xy if proposal.polygon_xy else None
        return normalize_textline_crops(
            image,
            TextLineGeometry(
                bbox_xyxy=bbox,
                polygon_xy=polygon,
            ),
            TextLineCropConfig(padding_px=max(2, min(self.crop_padding_px, 6))),
        )

    def _rapidocr_probe_text_score(
        self,
        text: str,
        confidence: float | None,
        *,
        today: date,
    ) -> tuple[float, str | None]:
        normalized = normalize_recognition_text(self._normalize_ocr_confusions(text or ""))
        if not normalized:
            return 0.0, "empty_text"
        digits = re.sub(r"\D", "", normalized)
        if len(digits) < self.rapidocr_gated_min_probe_digits:
            return 0.0, "too_few_digits"
        letters = sum(ch.isalpha() for ch in normalized)
        date_chars = sum(ch.isdigit() or ch in "/.-:" for ch in normalized)
        compact_len = max(1, len([ch for ch in normalized if not ch.isspace()]))
        if letters > date_chars * 2 and not (has_expiry_keyword(normalized) or has_weak_expiry_keyword(normalized)):
            return 0.0, "mostly_letters"
        if self._is_generic_brand_text(normalized):
            return 0.0, "brand_text"
        if self._is_weight_or_volume_only(normalized):
            return 0.0, "weight_or_volume"

        recognition = _RecognitionOutput(
            raw_text=text or "",
            normalized_text=normalized,
            confidence=confidence,
            reason=None,
            rotation="probe",
        )
        parsed = self._best_parse_for_inputs(self._build_parse_inputs(recognition), today=today)
        if self._rapidocr_has_extra_numeric_date_tail(normalized) and not self._rapidocr_extra_tail_is_expiry_time(
            normalized,
            parsed.parsed_date,
        ):
            return 0.0, "extra_numeric_date_tail"
        conf = float(confidence or 0.0)
        if parsed.parsed_date is not None and self._date_precision_score(parsed) >= 3:
            date_score = 3.0
            if parsed.parsed_date >= today:
                date_score += 0.25
            return date_score + min(conf, 1.0) * 0.25, None

        has_date_pattern = bool(DATE_PATTERN.search(normalized))
        compact = re.sub(r"\s+", "", normalized)
        separator_count = sum(ch in "/.-:" for ch in compact)
        date_ratio = date_chars / float(compact_len)
        if has_date_pattern and separator_count >= 1:
            return 2.0 + min(conf, 1.0) * 0.20, None
        if re.search(r"\d{6,8}", compact):
            return 1.6 + min(conf, 1.0) * 0.15, None
        if len(digits) >= self.rapidocr_gated_min_probe_digits and separator_count >= 1 and date_ratio >= 0.45:
            return 1.0 + min(conf, 1.0) * 0.10, None
        return 0.0, "not_date_like"

    @staticmethod
    def _rapidocr_has_extra_numeric_date_tail(normalized: str) -> bool:
        for token in re.findall(r"\S+", normalized):
            groups = re.findall(r"\d+", token)
            if len(groups) < 4:
                continue
            year_first = re.match(r"^20\d{2}[./:-](\d{1,2})[./:-](\d{1,2})(?:[./:-]\d+)+$", token)
            if year_first:
                month = int(year_first.group(1))
                day = int(year_first.group(2))
                return 1 <= month <= 12 and 1 <= day <= 31
            day_first = re.match(r"^\d{1,2}[./:-](\d{1,2})[./:-](?:20\d{2}|\d{2})(?:[./:-]\d+)+$", token)
            if day_first:
                day = int(groups[0])
                month = int(day_first.group(1))
                return 1 <= day <= 31 and 1 <= month <= 12
            if re.search(r"\d[./:-]\d", token):
                return True
        return False

    @staticmethod
    def _rapidocr_extra_tail_is_expiry_time(normalized: str, parsed_date: date | None) -> bool:
        if parsed_date is None:
            return False
        compact = re.sub(r"\s+", "", normalized)
        pattern = re.compile(r"(\d{1,2})([./:-])(\d{1,2})\2(\d{2}|\d{4}|\d)(?:[./:-]?)(\d{1,2})[:.](\d{2})(?!\d)")
        for match in pattern.finditer(compact):
            context = compact[max(0, match.start() - 12) : min(len(compact), match.end() + 12)]
            prefix = compact[max(0, match.start() - 6) : match.start()]
            prefix_compact = re.sub(r"[^A-Z0-9]", "", prefix.upper())
            has_expiry_time_context = (
                has_expiry_keyword(context)
                or has_weak_expiry_keyword(context)
                or prefix_compact in {"ET", "EY", "EYT"}
            )
            if not has_expiry_time_context:
                continue
            day = int(match.group(1))
            month = int(match.group(3))
            year_raw = match.group(4)
            year = int(year_raw)
            if len(year_raw) == 1:
                year = 2020 + year
            elif len(year_raw) == 2:
                year = 2000 + year if year < 70 else 1900 + year
            hour = int(match.group(5))
            minute = int(match.group(6))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                continue
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate == parsed_date:
                return True
        return False

    def _rapidocr_extract_partial_day_month(self, text: str) -> str | None:
        normalized = normalize_recognition_text(self._normalize_ocr_confusions(text or ""))
        compact = re.sub(r"\s+", "", normalized)
        if re.search(r"[./:-]{2,}", compact):
            return None
        groups = re.findall(r"\d+", compact)
        if len(groups) != 2:
            return None
        match = re.search(r"(?<!\d)(\d{1,2})[./:-](\d{1,2})(?![./:-]?\d)", compact)
        if not match:
            return None
        day = int(match.group(1))
        month = int(match.group(2))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return None
        return f"{day:02d}-{month:02d}"

    def _rapidocr_extract_year_fragment(self, text: str) -> str | None:
        normalized = normalize_recognition_text(self._normalize_ocr_confusions(text or ""))
        compact = re.sub(r"\s+", "", normalized)
        year_match = re.search(r"20\d{2}", compact)
        if year_match:
            return year_match.group(0)
        groups = re.findall(r"\d+", compact)
        if not groups:
            return None
        digits = "".join(groups)
        if len(digits) >= 4 and digits[-4:].startswith("20"):
            return digits[-4:]
        if len(digits) >= 2:
            return digits[-2:]
        return None

    @staticmethod
    def _rapidocr_same_line_right_neighbor(left: ProposalBox, right: ProposalBox) -> bool:
        lx1, ly1, lx2, ly2 = left.bbox_xyxy
        rx1, ry1, rx2, ry2 = right.bbox_xyxy
        left_h = max(1, ly2 - ly1)
        right_h = max(1, ry2 - ry1)
        left_cx = (lx1 + lx2) / 2.0
        right_cx = (rx1 + rx2) / 2.0
        left_cy = (ly1 + ly2) / 2.0
        right_cy = (ry1 + ry2) / 2.0
        y_overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
        gap = max(0, rx1 - lx2)
        return (
            right_cx > left_cx
            and gap <= max(left_h, right_h) * 2.5
            and (
                y_overlap >= min(left_h, right_h) * 0.20
                or abs(left_cy - right_cy) <= max(left_h, right_h) * 0.85
            )
        )

    def _rapidocr_partial_date_completion_candidates(
        self,
        records: list[tuple[ProposalBox, dict[str, object], float]],
        *,
        today: date,
    ) -> list[tuple[float, ProposalBox, dict[str, object]]]:
        completed: list[tuple[float, ProposalBox, dict[str, object]]] = []
        for left_proposal, left_metadata, _left_score in records:
            day_month = self._rapidocr_extract_partial_day_month(
                str(left_metadata.get("probe_normalized_text") or left_metadata.get("probe_text") or "")
            )
            if day_month is None:
                continue
            left_confidence = float(left_metadata.get("probe_confidence") or 0.0)
            if left_confidence < 0.55:
                continue
            best_item: tuple[float, ProposalBox, dict[str, object]] | None = None
            for right_proposal, right_metadata, _right_score in records:
                if right_proposal is left_proposal:
                    continue
                if not self._rapidocr_same_line_right_neighbor(left_proposal, right_proposal):
                    continue
                year = self._rapidocr_extract_year_fragment(
                    str(right_metadata.get("probe_normalized_text") or right_metadata.get("probe_text") or "")
                )
                if year is None:
                    continue
                combined_text = f"{day_month}-{year}"
                combined_confidence = min(
                    left_confidence,
                    float(right_metadata.get("probe_confidence") or left_confidence),
                )
                score, reason = self._rapidocr_probe_text_score(
                    combined_text,
                    combined_confidence,
                    today=today,
                )
                if score < 3.0 or reason is not None:
                    continue
                grouped = ProposalBox(
                    bbox_xyxy=self._union_many_bboxes([left_proposal.bbox_xyxy, right_proposal.bbox_xyxy]),
                    confidence=max(float(left_proposal.confidence or 0.0), float(right_proposal.confidence or 0.0)),
                    source="rapidocr_ppocrv5_grouped",
                    sources=("rapidocr_ppocrv5",),
                    variant_name="rapidocr_gated_grouped:partial_date_completion",
                    polygon_xy=self._rapidocr_group_polygon([left_proposal, right_proposal]),
                )
                metadata = {
                    "probe_text": combined_text,
                    "probe_normalized_text": combined_text,
                    "probe_confidence": combined_confidence,
                    "probe_score": score,
                    "probe_orientation": "combined_same_line",
                    "probe_variant": "partial_date_completion",
                    "probe_polygon_used": bool(grouped.polygon_xy),
                    "probe_transform": "textline_group_completion",
                    "probe_components": [
                        left_metadata.get("probe_text"),
                        right_metadata.get("probe_text"),
                    ],
                }
                if best_item is None or (score, combined_confidence) > (
                    best_item[0],
                    float(best_item[2].get("probe_confidence") or 0.0),
                ):
                    best_item = (score, grouped, metadata)
            if best_item is not None:
                completed.append(best_item)
        return completed

    def _ocr_prefilter_rapidocr_gated_proposals(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
        *,
        today: date,
    ) -> list[ProposalBox]:
        accepted: list[tuple[float, ProposalBox, dict[str, object]]] = []
        probed_records: list[tuple[ProposalBox, dict[str, object], float]] = []
        rejected_reasons: dict[str, int] = {}
        probed_count = 0
        for proposal in proposals[: self.rapidocr_gated_prefilter_probe_top_k]:
            best_score = 0.0
            best_reason = "no_probe_crops"
            best_metadata: dict[str, object] | None = None
            crops = self._normalized_rapidocr_polygon_probe_crops(image, proposal)
            for crop in crops:
                if crop.image.size == 0:
                    continue
                variants = self._recognition_variants_for_crop(
                    crop.image,
                    allowed_names=RAPIDOCR_GATED_PROBE_RECOGNITION_VARIANTS,
                )
                for variant in variants:
                    probed_count += 1
                    rec = self._recognize_variant(
                        variant.image,
                        variant_name=f"rapidocr_gated_probe:{variant.name}",
                        orientation=crop.selected_orientation,
                    )
                    score, reason = self._rapidocr_probe_text_score(
                        rec.normalized_text or rec.raw_text,
                        rec.confidence,
                        today=today,
                    )
                    tie_break = float(rec.confidence or 0.0)
                    if best_metadata is None or (score, tie_break) > (
                        best_score,
                        float(best_metadata.get("probe_confidence") or 0.0),
                    ):
                        best_score = score
                        best_reason = reason or "accepted"
                        best_metadata = {
                            "probe_text": rec.raw_text,
                            "probe_normalized_text": rec.normalized_text,
                            "probe_confidence": rec.confidence,
                            "probe_score": score,
                            "probe_orientation": crop.selected_orientation,
                            "probe_variant": variant.name,
                            "probe_polygon_used": crop.polygon_used,
                            "probe_transform": crop.crop_transform_used,
                        }
            if best_metadata is not None:
                probed_records.append((proposal, best_metadata, best_score))
            if best_score >= self.rapidocr_gated_min_date_likeness_score and best_metadata is not None:
                accepted.append((best_score, proposal, best_metadata))
            else:
                rejected_reasons[best_reason] = rejected_reasons.get(best_reason, 0) + 1

        completed = self._rapidocr_partial_date_completion_candidates(probed_records, today=today)
        if completed:
            accepted.extend(completed)
            rejected_reasons["partial_date_completion_added"] = len(completed)

        accepted.sort(
            key=lambda item: (
                item[0],
                float(item[2].get("probe_confidence") or 0.0),
                float(item[1].confidence or 0.0),
                self._bbox_area(item[1].bbox_xyxy),
            ),
            reverse=True,
        )
        selected: list[ProposalBox] = []
        for _score, proposal, metadata in accepted:
            if any(self._scan_bboxes_are_duplicates(proposal.bbox_xyxy, existing.bbox_xyxy) for existing in selected):
                continue
            selected.append(proposal)
            self._rapidocr_gated_probe_metadata[self._rapidocr_probe_metadata_key(proposal)] = metadata
            if len(selected) >= self.rapidocr_gated_max_boxes_accepted:
                break

        prefilter_debug = getattr(self, "_rapidocr_gated_prefilter_debug", {})
        self._debug_append(
            "ppmobile_gated_prefilter",
            {
                "raw_count": int(prefilter_debug.get("raw_count", 0) or 0),
                "geometry_count": len(proposals),
                "grouped_count": int(prefilter_debug.get("grouped_count", 0) or 0),
                "ocr_probed_count": probed_count,
                "accepted_count": len(selected),
                "accepted": [
                    {
                        "proposal": self._debug_proposal(proposal),
                        **self._rapidocr_gated_probe_metadata.get(self._rapidocr_probe_metadata_key(proposal), {}),
                    }
                    for proposal in selected
                ],
                "rejected_reasons": rejected_reasons,
                "min_digits": self.rapidocr_gated_min_probe_digits,
                "min_score": self.rapidocr_gated_min_date_likeness_score,
            },
        )
        return selected

    @classmethod
    def _map_yolo_candidates_to_original(
        cls,
        candidates: list[_YoloCandidate],
        *,
        variant: ImageVariant,
        original_shape: tuple[int, ...],
    ) -> list[_YoloCandidate]:
        height, width = original_shape[:2]
        scale_x = float(variant.scale_x or 1.0)
        scale_y = float(variant.scale_y or 1.0)
        mapped: list[_YoloCandidate] = []
        for candidate in candidates:
            polygon = [
                [
                    max(0.0, min(float(x) / scale_x, float(width))),
                    max(0.0, min(float(y) / scale_y, float(height))),
                ]
                for x, y in candidate.polygon_xy
            ]
            bbox = cls._xyxy_from_polygon(polygon)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            mapped.append(
                _YoloCandidate(
                    polygon_xy=polygon,
                    bbox_xyxy=bbox,
                    confidence=candidate.confidence,
                    source=candidate.source,
                    sources=candidate.sources,
                    variant_name=variant.name,
                )
            )
        return mapped

    @classmethod
    def _dedupe_yolo_candidates(cls, candidates: list[_YoloCandidate]) -> list[_YoloCandidate]:
        ordered = sorted(
            candidates,
            key=lambda item: (cls._detector_variant_priority(item.variant_name), item.confidence),
            reverse=True,
        )
        kept: list[_YoloCandidate] = []
        for candidate in ordered:
            if any(cls._scan_bboxes_are_duplicates(candidate.bbox_xyxy, existing.bbox_xyxy) for existing in kept):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _detector_variant_priority(variant_name: str | None) -> float:
        if variant_name == "raw":
            return 4.0
        if variant_name == "raw_upscaled":
            return 3.5
        if variant_name in {"luma_clahe", "unsharp", "illumination_normalized"}:
            return 2.0
        return 1.0

    @staticmethod
    def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @classmethod
    def _union_many_bboxes(cls, bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        out = bboxes[0]
        for bbox in bboxes[1:]:
            out = cls._union_bbox(out, bbox)
        return out

    @staticmethod
    def _boxes_are_groupable(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        aw, ah = max(1, a[2] - a[0]), max(1, a[3] - a[1])
        bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
        a_cx, a_cy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        b_cx, b_cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        center_distance = float(np.hypot(a_cx - b_cx, a_cy - b_cy))
        diag_ref = max(np.hypot(aw, ah), np.hypot(bw, bh))
        x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        x_gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        y_gap = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return bool(
            center_distance <= (diag_ref * 2.4)
            and (
                y_overlap >= min(ah, bh) * 0.25
                or x_overlap >= min(aw, bw) * 0.25
                or (x_gap <= max(aw, bw) * 1.5 and y_gap <= max(ah, bh) * 1.5)
            )
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

    def _candidate_from_bbox(
        self,
        *,
        candidate_id: str,
        candidate_type: str,
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, int],
        yolo_candidates: list[_YoloCandidate],
        member_indices: list[int],
        polygon_xy: tuple[tuple[float, float], ...] | None,
        neighbor_counts: dict[int, int] | None = None,
    ) -> _MobileRankedCandidate:
        h, w = image_shape
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        if aspect >= 1.0:
            line_likeness = min(1.0, aspect / 6.0)
        else:
            line_likeness = max(0.0, aspect * 0.4)
        edge_dist = min(bbox[0], bbox[1], max(0, w - bbox[2]), max(0, h - bbox[3]))
        edge_proximity = float(edge_dist) / float(max(1, min(h, w)))
        confidences = [float(yolo_candidates[idx].confidence) for idx in member_indices if idx < len(yolo_candidates)]
        det_conf_avg = sum(confidences) / len(confidences) if confidences else 0.0
        member_neighbor_counts = [neighbor_counts.get(idx, 0) for idx in member_indices] if neighbor_counts else []
        neighbor_avg = float(sum(member_neighbor_counts) / len(member_neighbor_counts)) if member_neighbor_counts else 0.0
        detector_sources = tuple(
            dict.fromkeys(
                source
                for idx in member_indices
                if idx < len(yolo_candidates)
                for source in yolo_candidates[idx].sources
            )
        ) or ("fallback",)
        detector_variants = tuple(
            dict.fromkeys(
                yolo_candidates[idx].variant_name
                for idx in member_indices
                if idx < len(yolo_candidates) and yolo_candidates[idx].variant_name
            )
        )

        size_score = self._geometry_size_score(area_ratio)
        aspect_score = self._geometry_aspect_score(aspect)
        edge_score = self._geometry_edge_score(edge_proximity)
        neighbor_score = min(0.2, neighbor_avg * 0.05)
        det_score = (det_conf_avg - 0.5) * 0.25 if confidences else 0.0
        group_bonus = 0.1 if candidate_type == "group" else 0.0
        fallback_bonus = 0.05 if candidate_type == "fallback" else 0.0
        multi_source_bonus = 0.25 if len(detector_sources) > 1 else 0.0
        geometry_score = size_score + aspect_score + edge_score + neighbor_score + det_score + group_bonus + fallback_bonus + multi_source_bonus
        context_matches = self._context_probe_matches(
            candidate_bbox=bbox,
            member_indices=member_indices,
            yolo_candidates=yolo_candidates,
        )
        context_bboxes = [context_bbox for _idx, context_bbox, _relation in context_matches]
        context_indices = [idx for idx, _context_bbox, _relation in context_matches]
        keyword_relation = context_matches[0][2] if context_matches else None

        return _MobileRankedCandidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            bbox_xyxy=bbox,
            polygon_xy=polygon_xy,
            detector_confidence=det_conf_avg if confidences else None,
            detector_sources=detector_sources,
            detector_variant=detector_variants[0] if detector_variants else None,
            member_indices=member_indices,
            member_bboxes=[yolo_candidates[idx].bbox_xyxy for idx in member_indices if idx < len(yolo_candidates)],
            geometry_features={
                "area_ratio": round(area_ratio, 6),
                "aspect_ratio": round(aspect, 4),
                "line_likeness": round(line_likeness, 4),
                "edge_proximity": round(edge_proximity, 4),
                "neighbor_count": round(neighbor_avg, 2),
                "det_conf_avg": round(det_conf_avg, 4),
            },
            geometry_score=geometry_score,
            score_breakdown={
                "size_score": size_score,
                "aspect_score": aspect_score,
                "edge_score": edge_score,
                "neighbor_score": neighbor_score,
                "det_score": det_score,
                "group_bonus": group_bonus,
                "fallback_bonus": fallback_bonus,
                "multi_source_bonus": multi_source_bonus,
            },
            recognition_bbox=bbox,
            evidence_bbox=self._union_many_bboxes([bbox, *context_bboxes]) if context_bboxes else bbox,
            context_probe_bboxes=context_bboxes,
            context_probe_indices=context_indices,
            keyword_relation=keyword_relation,
            total_score=geometry_score,
        )

    def _cluster_yolo_boxes_into_lines(self, yolo_candidates: list[_YoloCandidate]) -> list[list[int]]:
        ordered = sorted(
            range(len(yolo_candidates)),
            key=lambda idx: (
                (yolo_candidates[idx].bbox_xyxy[1] + yolo_candidates[idx].bbox_xyxy[3]) / 2.0,
                yolo_candidates[idx].bbox_xyxy[0],
            ),
        )
        lines: list[list[int]] = []
        for idx in ordered:
            box = yolo_candidates[idx].bbox_xyxy
            cy = (box[1] + box[3]) / 2.0
            height = max(1, box[3] - box[1])
            placed = False
            for line in lines:
                line_boxes = [yolo_candidates[item].bbox_xyxy for item in line]
                line_cy = sum((item[1] + item[3]) / 2.0 for item in line_boxes) / len(line_boxes)
                line_height = sum(max(1, item[3] - item[1]) for item in line_boxes) / len(line_boxes)
                y_overlap = max(0, min(max(item[3] for item in line_boxes), box[3]) - max(min(item[1] for item in line_boxes), box[1]))
                if abs(cy - line_cy) <= max(height, line_height) * 0.65 or y_overlap >= min(height, line_height) * 0.35:
                    line.append(idx)
                    placed = True
                    break
            if not placed:
                lines.append([idx])
        return [sorted(line, key=lambda idx: yolo_candidates[idx].bbox_xyxy[0]) for line in lines]

    @staticmethod
    def _neighbor_counts(yolo_candidates: list[_YoloCandidate]) -> dict[int, int]:
        counts = {idx: 0 for idx in range(len(yolo_candidates))}
        for i in range(len(yolo_candidates)):
            for j in range(i + 1, len(yolo_candidates)):
                if MobileExpiryPipeline._boxes_are_groupable(yolo_candidates[i].bbox_xyxy, yolo_candidates[j].bbox_xyxy):
                    counts[i] += 1
                    counts[j] += 1
        return counts

    @staticmethod
    def _box_center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        return float(np.hypot(((a[0] + a[2]) / 2.0) - ((b[0] + b[2]) / 2.0), ((a[1] + a[3]) / 2.0) - ((b[1] + b[3]) / 2.0)))

    @staticmethod
    def _box_has_grouping_geometry(bbox: tuple[int, int, int, int], *, image_shape: tuple[int, int]) -> bool:
        h, w = image_shape
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        return bool(0.0003 <= area_ratio <= 0.18 and 0.35 <= aspect <= 14.0)

    @staticmethod
    def _bbox_vertical_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return float(overlap) / float(max(1, min(a[3] - a[1], b[3] - b[1])))

    @staticmethod
    def _bbox_horizontal_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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
        if bx2 <= cx1 and vertical_overlap >= 0.45 and cx1 - bx2 <= 4 * ref_h:
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
        yolo_candidates: list[_YoloCandidate],
    ) -> list[tuple[int, tuple[int, int, int, int], str]]:
        excluded = set(member_indices)
        matches: list[tuple[tuple[int, int, float], int, tuple[int, int, int, int], str]] = []
        cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
        cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
        for idx, yolo in enumerate(yolo_candidates):
            if idx in excluded:
                continue
            bbox = yolo.bbox_xyxy
            relation = self._context_relation(candidate_bbox, bbox)
            if relation is None:
                continue
            bx = (bbox[0] + bbox[2]) / 2.0
            by = (bbox[1] + bbox[3]) / 2.0
            distance = float(np.hypot(cx - bx, cy - by))
            priority = 0 if relation == "left_same_line" else 1
            matches.append(((priority, int(distance), -float(yolo.confidence)), idx, bbox, relation))
        matches.sort(key=lambda item: item[0])
        return [(idx, bbox, relation) for _key, idx, bbox, relation in matches[:2]]

    def _proposal_boxes_from_yolo(self, yolo_candidates: list[_YoloCandidate]) -> list[ProposalBox]:
        return [
            ProposalBox(
                bbox_xyxy=candidate.bbox_xyxy,
                confidence=candidate.confidence,
                source=candidate.source,
                sources=candidate.sources,
                variant_name=candidate.variant_name,
                polygon_xy=tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
            )
            for candidate in yolo_candidates
        ]

    def _expanded_yolo_text_proposals(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
    ) -> list[ProposalBox]:
        h, w = image.shape[:2]
        proposals: list[ProposalBox] = []
        for candidate in yolo_candidates:
            x1, y1, x2, y2 = candidate.bbox_xyxy
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            band_boxes = {
                "yolo_whole": (x1, y1, x2, y2),
                "yolo_band_top": (x1, y1, x2, min(y2, y1 + int(bh * 0.45))),
                "yolo_band_mid": (x1, y1 + int(bh * 0.25), x2, y1 + int(bh * 0.75)),
                "yolo_band_bottom": (x1, max(y1, y2 - int(bh * 0.45)), x2, y2),
                "yolo_left_context": (max(0, x1 - int(bw * 0.35)), y1, x2, y2),
                "yolo_right_context": (x1, y1, min(w, x2 + int(bw * 0.35)), y2),
                "yolo_tall_context": (
                    max(0, x1 - int(bw * 0.12)),
                    max(0, y1 - int(bh * 0.35)),
                    min(w, x2 + int(bw * 0.12)),
                    min(h, y2 + int(bh * 0.35)),
                ),
            }
            for name, bbox in band_boxes.items():
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                proposals.append(
                    ProposalBox(
                        bbox_xyxy=bbox,
                        confidence=candidate.confidence,
                        source=candidate.source,
                        sources=candidate.sources,
                        variant_name=name,
                        polygon_xy=(
                            tuple((float(x), float(y)) for x, y in candidate.polygon_xy)
                            if name == "yolo_whole"
                            else None
                        ),
                    )
                )
        return proposals

    def _build_ranked_candidates(self, image: np.ndarray, yolo_candidates: list[_YoloCandidate]) -> list[_MobileRankedCandidate]:
        return self._build_ranked_candidates_from_proposals(image, self._proposal_boxes_from_yolo(yolo_candidates))

    def _build_ranked_candidates_from_proposals(self, image: np.ndarray, proposals: list[ProposalBox]) -> list[_MobileRankedCandidate]:
        self.candidate_engine.max_group_candidates_per_roi = self.max_group_candidates_per_roi
        self.candidate_engine.geometry_top_n = self.expiry_filter_geometry_top_n
        self.candidate_engine.context_probe_enabled = self.context_probe_enabled
        self.candidate_engine.context_max_boxes_per_candidate = self.context_max_boxes_per_candidate
        self.candidate_engine.require_multiline_for_line_candidates = True
        with self._profile_timer("candidate_build_ms"):
            ranked = self.candidate_engine.build_ranked_candidates(image, proposals)  # type: ignore[assignment]
        self._debug_append(
            "candidate_engine_candidates",
            {
                "proposal_count": len(proposals),
                "candidate_count": len(ranked),
                "proposals": [self._debug_proposal(proposal) for proposal in proposals],
                "candidates": [self._debug_ranked_candidate(candidate) for candidate in ranked],
            },
        )
        return ranked  # type: ignore[return-value]

    def _apply_rapidocr_probe_metadata_to_ranked(self, ranked: list[_MobileRankedCandidate]) -> None:
        if not self._rapidocr_gated_probe_metadata:
            return
        for candidate in ranked:
            keys = [
                self._rapidocr_probe_metadata_key(candidate),
                *(tuple(int(v) for v in bbox) for bbox in candidate.member_bboxes),
            ]
            metadata = next(
                (
                    self._rapidocr_gated_probe_metadata[key]
                    for key in keys
                    if key in self._rapidocr_gated_probe_metadata
                ),
                None,
            )
            if metadata is None:
                continue
            score = float(metadata.get("probe_score") or 0.0)
            candidate.rapidocr_probe_text = str(metadata.get("probe_text") or "")
            candidate.rapidocr_probe_normalized_text = str(metadata.get("probe_normalized_text") or "")
            candidate.rapidocr_probe_confidence = (
                float(metadata["probe_confidence"])
                if metadata.get("probe_confidence") is not None
                else None
            )
            candidate.rapidocr_probe_score = score
            candidate.rapidocr_probe_orientation = str(metadata.get("probe_orientation") or "")
            candidate.rapidocr_probe_polygon_used = bool(metadata.get("probe_polygon_used") or False)
            candidate.probe_text = candidate.rapidocr_probe_text or candidate.probe_text
            candidate.probe_normalized_text = candidate.rapidocr_probe_normalized_text or candidate.probe_normalized_text
            candidate.probe_confidence = candidate.rapidocr_probe_confidence or candidate.probe_confidence
            candidate.score_breakdown["rapidocr_probe_date_likeness_score"] = min(score, 3.5) * 0.75
            candidate.score_breakdown["rapidocr_probe_confidence_score"] = min(
                0.75,
                float(candidate.rapidocr_probe_confidence or 0.0) * 0.75,
            )
            candidate.total_score = candidate.geometry_score + sum(candidate.score_breakdown.values())

    def _normalized_crops_for_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate) -> list[NormalizedTextLineCrop]:
        tight_bbox, policy = self._final_tight_bbox_for_candidate(candidate)
        padded_bbox, padding_px = self._pad_bbox_for_final_crop(tight_bbox, image.shape)
        use_polygon_crop = (
            (
                "dot_matrix_rescue" in candidate.detector_sources
                or (candidate.detector_variant or "").startswith("rapidocr_gated")
            )
            and tight_bbox == candidate.bbox_xyxy
        )
        polygon_xy = candidate.polygon_xy if use_polygon_crop else None
        geometry_bbox = tight_bbox if polygon_xy is not None else padded_bbox
        rapidocr_polygon_crop = (
            polygon_xy is not None
            and (candidate.detector_variant or "").startswith("rapidocr_gated")
        )
        crop_padding_px = (
            max(2, min(padding_px, 6))
            if rapidocr_polygon_crop
            else padding_px if polygon_xy is not None else 0
        )
        crops = normalize_textline_crops(
            image,
            TextLineGeometry(
                bbox_xyxy=geometry_bbox,
                polygon_xy=polygon_xy,
            ),
            TextLineCropConfig(padding_px=crop_padding_px),
        )
        tail_geometry = self._rapidocr_date_tail_geometry(candidate) if rapidocr_polygon_crop else None
        if tail_geometry is not None:
            with self._profile_timer("date_tail_crop_ms"):
                crops.extend(
                    normalize_textline_crops(
                        image,
                        tail_geometry,
                        TextLineCropConfig(padding_px=crop_padding_px),
                    )
                )
        if crops:
            candidate.final_crop_bbox = crops[0].bbox_xyxy
            candidate.final_crop_padding_px = crop_padding_px if polygon_xy is not None else padding_px
            candidate.final_crop_policy = policy
        return crops

    def _rapidocr_date_tail_geometry(self, candidate: _MobileRankedCandidate) -> TextLineGeometry | None:
        text = (
            candidate.rapidocr_probe_normalized_text
            or candidate.rapidocr_probe_text
            or candidate.probe_normalized_text
            or candidate.probe_text
            or ""
        )
        compact = re.sub(r"\s+", "", text.upper())
        if not re.match(r"^[A-Z.:-]{1,6}\d", compact):
            return None
        if not re.search(r"\d{1,3}[./:-]\d", compact):
            return None
        fraction = 0.24
        x1, y1, x2, y2 = candidate.bbox_xyxy
        width = max(1, x2 - x1)
        tail_x1 = int(round(x1 + width * fraction))
        if x2 - tail_x1 < max(24, int(width * 0.35)):
            return None
        polygon = self._trim_polygon_left(candidate.polygon_xy, fraction) if candidate.polygon_xy else None
        bbox = self._xyxy_from_polygon([[x, y] for x, y in polygon]) if polygon is not None else (tail_x1, y1, x2, y2)
        if bbox[0] <= x1:
            bbox = (tail_x1, bbox[1], bbox[2], bbox[3])
        return TextLineGeometry(bbox_xyxy=bbox, polygon_xy=polygon)

    @staticmethod
    def _trim_polygon_left(
        polygon: tuple[tuple[float, float], ...] | None,
        fraction: float,
    ) -> tuple[tuple[float, float], ...] | None:
        if polygon is None or len(polygon) != 4:
            return None
        arr = np.asarray(polygon, dtype=np.float32)
        if arr.shape != (4, 2) or not np.isfinite(arr).all():
            return None
        sums = arr.sum(axis=1)
        diffs = np.diff(arr, axis=1).reshape(-1)
        tl = arr[int(np.argmin(sums))]
        br = arr[int(np.argmax(sums))]
        tr = arr[int(np.argmin(diffs))]
        bl = arr[int(np.argmax(diffs))]
        keep = max(0.0, min(float(fraction), 0.65))
        new_tl = tl + (tr - tl) * keep
        new_bl = bl + (br - bl) * keep
        trimmed = (new_tl, tr, br, new_bl)
        return tuple((float(point[0]), float(point[1])) for point in trimmed)

    def _final_tight_bbox_for_candidate(self, candidate: _MobileRankedCandidate) -> tuple[tuple[int, int, int, int], str]:
        if candidate.recognition_bbox is not None and candidate.recognition_bbox != candidate.bbox_xyxy:
            return candidate.recognition_bbox, "recognition_bbox_tight"
        if "dot_matrix_rescue" in candidate.detector_sources:
            return candidate.bbox_xyxy, "dot_matrix_rescue"
        if candidate.candidate_type == "line":
            line_boxes = self._same_line_member_bboxes(candidate.member_bboxes)
            if line_boxes:
                return self._union_many_bboxes(line_boxes), "line_tight"
            return candidate.bbox_xyxy, "line_tight"
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
        return candidate.bbox_xyxy, f"{candidate.candidate_type}_tight"

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
        if MobileExpiryPipeline._bboxes_share_text_line(bboxes):
            return bboxes
        lines = MobileExpiryPipeline._cluster_bboxes_into_lines(bboxes)
        return max(lines, key=lambda line: (len(line), MobileExpiryPipeline._bbox_area(MobileExpiryPipeline._union_many_bboxes(line)))) if lines else []

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
        candidate: _MobileRankedCandidate,
        lines: list[list[tuple[int, int, int, int]]],
    ) -> list[tuple[int, int, int, int]]:
        probe_lines = [line.strip() for line in re.split(r"[\r\n]+", candidate.probe_text or "") if line.strip()]

        def score(index: int, line: list[tuple[int, int, int, int]]) -> tuple[int, float, float]:
            text = probe_lines[index] if index < len(probe_lines) else ""
            evidence = self._text_evidence_priority(text)
            if has_production_keyword(text) and not has_expiry_keyword(text):
                evidence -= 1
            bbox = self._union_many_bboxes(line)
            aspect = (bbox[2] - bbox[0]) / float(max(1, bbox[3] - bbox[1]))
            return evidence, min(aspect, 12.0), -float(self._bbox_area(bbox))

        indexed = list(enumerate(lines))
        _, selected = max(indexed, key=lambda item: score(item[0], item[1]))
        return selected

    def _recognition_variants_for_crop(
        self,
        crop: np.ndarray,
        *,
        allowed_names: tuple[str, ...] | None = MOBILE_RECOGNITION_VARIANTS,
    ) -> list[ImageVariant]:
        allowed = set(allowed_names or MOBILE_RECOGNITION_VARIANTS)
        variants: list[ImageVariant] = []
        if "original_color_tight" in allowed:
            variants.append(ImageVariant("original_color_tight", crop, purpose="recognition"))
        enhanced_allowed = tuple(name for name in MOBILE_ENHANCED_RECOGNITION_VARIANTS if name in allowed)
        if enhanced_allowed:
            variants.extend(self.preprocessor.recognition_variants(crop, allowed_names=enhanced_allowed))
        return variants

    def _recognize_variant(
        self,
        image: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        recognizer: Any | None = None,
        use_cache: bool = True,
    ) -> _RecognitionOutput:
        active_recognizer = recognizer or self.recognizer
        key: tuple[int, str, str, str, tuple[int, ...], str] | None = None
        if use_cache and self._recognizer_cache_enabled(active_recognizer):
            key = self._svtr_recognition_cache_key(
                image,
                variant_name=variant_name,
                orientation=orientation,
                recognizer=active_recognizer,
            )
            cached = self._svtr_recognition_cache.get(key)
            if cached is not None:
                return cached
        started = perf_counter()
        try:
            result = active_recognizer.recognize(image)
        finally:
            self._record_svtr_profile(
                active_recognizer,
                elapsed_ms=(perf_counter() - started) * 1000.0,
                call_count=1,
            )
        output = _RecognitionOutput(
            raw_text=str(getattr(result, "raw_text", "") or ""),
            normalized_text=normalize_recognition_text(str(getattr(result, "normalized_text", "") or getattr(result, "raw_text", "") or "")),
            confidence=getattr(result, "confidence", None),
            reason=getattr(result, "reason", None),
            rotation=orientation,
        )
        if key is not None:
            self._svtr_recognition_cache[key] = output
        return output

    @staticmethod
    def _coerce_recognition_output(result: object, *, orientation: str) -> _RecognitionOutput:
        return _RecognitionOutput(
            raw_text=str(getattr(result, "raw_text", "") or ""),
            normalized_text=normalize_recognition_text(str(getattr(result, "normalized_text", "") or getattr(result, "raw_text", "") or "")),
            confidence=getattr(result, "confidence", None),
            reason=getattr(result, "reason", None),
            rotation=orientation,
        )

    def _recognize_variants_batch(
        self,
        variants: list[ImageVariant],
        *,
        orientation: str,
        recognizer: Any | None = None,
        use_cache: bool = True,
    ) -> list[tuple[ImageVariant, _RecognitionOutput]]:
        active_recognizer = recognizer or self.recognizer
        outputs: list[_RecognitionOutput | None] = [None for _ in variants]
        pending: list[tuple[int, ImageVariant, tuple[int, str, str, str, tuple[int, ...], str] | None]] = []

        for index, variant in enumerate(variants):
            key: tuple[int, str, str, str, tuple[int, ...], str] | None = None
            if use_cache and self._recognizer_cache_enabled(active_recognizer):
                key = self._svtr_recognition_cache_key(
                    variant.image,
                    variant_name=variant.name,
                    orientation=orientation,
                    recognizer=active_recognizer,
                )
                cached = self._svtr_recognition_cache.get(key)
                if cached is not None:
                    outputs[index] = cached
                    continue
            pending.append((index, variant, key))

        if pending:
            batch_fn = getattr(active_recognizer, "recognize_many", None)
            started = perf_counter()
            if callable(batch_fn):
                images = [variant.image for _index, variant, _key in pending]
                try:
                    raw_outputs = batch_fn(images, batch_size=SVTR_RECOGNITION_BATCH_SIZE)
                finally:
                    self._record_svtr_profile(
                        active_recognizer,
                        elapsed_ms=(perf_counter() - started) * 1000.0,
                        call_count=len(pending),
                    )
            else:
                try:
                    raw_outputs = [active_recognizer.recognize(variant.image) for _index, variant, _key in pending]
                finally:
                    self._record_svtr_profile(
                        active_recognizer,
                        elapsed_ms=(perf_counter() - started) * 1000.0,
                        call_count=len(pending),
                    )

            for (index, _variant, key), raw_output in zip(pending, raw_outputs, strict=False):
                output = self._coerce_recognition_output(raw_output, orientation=orientation)
                outputs[index] = output
                if key is not None:
                    self._svtr_recognition_cache[key] = output

        return [
            (variant, output if output is not None else _RecognitionOutput("", "", None, "recognizer returned no output", orientation))
            for variant, output in zip(variants, outputs, strict=False)
        ]

    @staticmethod
    def _recognizer_supports_batch(recognizer: Any) -> bool:
        return callable(getattr(recognizer, "recognize_many", None))

    def _batch_recognition_enabled(self, recognizer: Any) -> bool:
        if not self.svtr_batch_recognition_enabled:
            return False
        method = getattr(self, "_recognize_variant", None)
        if getattr(method, "__func__", None) is not MobileExpiryPipeline._recognize_variant:
            return False
        return self._recognizer_supports_batch(recognizer)

    @staticmethod
    def _recognizer_cache_enabled(recognizer: Any) -> bool:
        return bool(getattr(recognizer, "cacheable", False)) or isinstance(
            recognizer,
            (SVTRTextRecognizer, SVTROnnxTextRecognizer),
        )

    @staticmethod
    def _svtr_recognition_cache_key(
        image: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        recognizer: Any,
    ) -> tuple[int, str, str, str, tuple[int, ...], str]:
        contiguous = np.ascontiguousarray(image)
        digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
        return (
            id(recognizer),
            variant_name,
            orientation,
            str(contiguous.dtype),
            tuple(int(dim) for dim in contiguous.shape),
            digest,
        )

    def _is_strong_day_date_parse(
        self,
        parsed: ParsedDateData,
        recognition: _RecognitionOutput,
        *,
        text: str = "",
    ) -> bool:
        if parsed.parsed_date is None:
            return False
        if self._date_precision_score(parsed) < 3:
            return False
        if float(parsed.confidence or 0.0) < 0.95:
            return False
        if float(recognition.confidence or 0.0) < 0.85:
            return False
        recognition_text = " ".join(
            part
            for part in (
                recognition.normalized_text,
                recognition.raw_text,
            )
            if part
        )
        recognition_has_expiry = has_expiry_keyword(recognition_text) or has_weak_expiry_keyword(recognition_text)
        if has_production_keyword(recognition_text) and not recognition_has_expiry:
            return False
        context = " ".join(
            part
            for part in (
                text,
                recognition_text,
            )
            if part
        )
        has_expiry_context = has_expiry_keyword(context) or has_weak_expiry_keyword(context)
        if has_production_keyword(context) and not has_expiry_context:
            return False
        if not has_expiry_context and float(recognition.confidence or 0.0) < 0.95:
            return False
        return True

    def _best_probe_for_crop(
        self,
        crop: np.ndarray,
        *,
        today: date,
    ) -> tuple[_RecognitionOutput, str, np.ndarray]:
        best: _RecognitionOutput | None = None
        best_variant = "original"
        best_variant_image = crop
        best_score: tuple[float, float, float, int, float] | None = None
        variants = self._recognition_variants_for_crop(crop, allowed_names=MOBILE_RECOGNITION_VARIANTS)
        with self._svtr_profile_stage("probe_svtr_ms"):
            if self._batch_recognition_enabled(self.recognizer):
                recognized_variants = self._recognize_variants_batch(variants, orientation="original")
            else:
                recognized_variants = [
                    (variant, self._recognize_variant(variant.image, variant_name=variant.name, orientation="original"))
                    for variant in variants
                ]
        for variant, probe in recognized_variants:
            probe_scores = self._probe_signal_scores(probe, candidate=None, today=today)
            parsed = self._best_parse_for_inputs(self._build_parse_inputs(probe), today=today)
            parseable_score = float(parsed.confidence if parsed.parsed_date is not None else 0.0)
            score = (
                float(probe.confidence or 0.0),
                self._score_date_likeness(probe.raw_text),
                parseable_score,
                parsed.parsed_date.toordinal() if parsed.parsed_date is not None else 0,
                float(sum(probe_scores.values())),
            )
            if best_score is None or score > best_score:
                best = probe
                best_variant = variant.name
                best_variant_image = variant.image
                best_score = score
        if best is None:
            return _RecognitionOutput("", "", None, "no recognition variants", "original"), best_variant, best_variant_image
        return best, best_variant, best_variant_image

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
        normalized = normalize_evidence_text(text)
        compact = compact_evidence_text(text)
        if not normalized:
            return False
        if any(token in normalized.split() or token in compact for token in GENERIC_BRAND_TOKENS):
            return True
        letters = sum(ch.isalpha() for ch in compact)
        digits = sum(ch.isdigit() for ch in compact)
        return bool(digits == 0 and letters >= 6)

    @staticmethod
    def _is_weight_or_volume_only(text: str | None) -> bool:
        return bool(WEIGHT_VOLUME_ONLY_PATTERN.match(normalize_evidence_text(text)))

    @staticmethod
    def _is_barcode_like_numeric(text: str | None) -> bool:
        value = text or ""
        digits = re.sub(r"\D", "", value)
        separators = sum(ch in "/.-:" for ch in value)
        return bool(len(digits) >= 8 and separators == 0)

    @staticmethod
    def _score_date_likeness(text: str) -> float:
        compact = [ch for ch in text if not ch.isspace()]
        if not compact:
            return 0.0
        ratio = sum(ch.isdigit() or ch in {"/", "-", ".", ":"} for ch in compact) / len(compact)
        if any(prefix in text.upper() for prefix in ("EXP", "USE BY", "BEST BEFORE", "SKT", "TETT", "BBE")):
            ratio += 0.1
        return ratio

    @classmethod
    def _score_ocr_candidate(cls, recognition: _RecognitionOutput) -> float:
        if not recognition.raw_text:
            return -1.0
        text = recognition.raw_text
        score = float(recognition.confidence or 0.0) * 10.0
        if any(sep in text for sep in ("/", "-", ".")):
            score += 1.0
        if any(char.isdigit() for char in text):
            score += 0.5
        if cls._is_generic_brand_text(text):
            score -= 4.0
        if has_production_keyword(text) and cls._has_date_like_text(text) and not has_expiry_keyword(text):
            score -= 3.0
        if has_expiry_keyword(text) and cls._has_date_like_text(text):
            score += 4.0
        return score

    def _probe_signal_scores(
        self,
        recognition: _RecognitionOutput,
        *,
        candidate: _MobileRankedCandidate | None,
        today: date,
    ) -> dict[str, float]:
        text = (recognition.normalized_text or recognition.raw_text or "").upper()
        if not text:
            return {"probe_empty_penalty": -0.25}
        digits = sum(ch.isdigit() for ch in text)
        separators = sum(ch in "/.-:" for ch in text)
        compact_len = max(1, len([ch for ch in text if not ch.isspace()]))
        date_likeness = (digits + separators) / float(compact_len)
        has_date_pattern = 1.0 if self._has_date_like_text(text) else 0.0
        expiry_keyword = has_expiry_keyword(text)
        production_keyword = has_production_keyword(text)
        negative_hits = sum(1 for token in NEGATIVE_HINTS if token in text)
        production_keyword_penalty = -3.0 if production_keyword and self._has_date_like_text(text) and not expiry_keyword else 0.0
        brand_text_penalty = 0.0
        if self._is_generic_brand_text(text):
            brand_text_penalty -= 4.0
        if self._is_weight_or_volume_only(text):
            brand_text_penalty -= 2.5
        if self._is_barcode_like_numeric(text):
            brand_text_penalty -= 2.0
        parse_inputs = self._build_parse_inputs(recognition)
        parsed = self._best_parse_for_inputs(parse_inputs, today=today)
        parser_probe_bonus = 0.0
        if parsed.parsed_date is not None:
            precision_score = self._date_precision_score(parsed)
            parser_probe_bonus = 2.0 if precision_score >= 3 else 0.25 if precision_score == 2 else 0.0
        expiry_keyword_score = 6.0 if expiry_keyword and self._has_date_like_text(text) else (2.5 if expiry_keyword else 0.0)
        keyword_date_group_bonus = 0.0
        if candidate is not None and expiry_keyword and self._has_date_like_text(text) and candidate.candidate_type in {"group", "line"}:
            keyword_date_group_bonus = 1.5
        return {
            "probe_confidence_score": float(recognition.confidence or 0.0) * 0.75,
            "probe_date_likeness_score": max(0.0, min(1.0, date_likeness)) * 1.2 + (2.0 * has_date_pattern),
            "probe_negative_penalty": -min(2.4, negative_hits * 0.4),
            "probe_lot_only_penalty": -2.0 if LOT_ONLY_PATTERN.search(text) and not self._has_date_like_text(text) else 0.0,
            "date_like_score": max(0.0, min(1.0, date_likeness)) * 1.2 + (2.0 * has_date_pattern),
            "expiry_keyword_score": expiry_keyword_score,
            "production_keyword_penalty": production_keyword_penalty,
            "brand_text_penalty": brand_text_penalty,
            "parser_probe_bonus": parser_probe_bonus,
            "keyword_date_group_bonus": keyword_date_group_bonus,
        }

    def _apply_geometry_shortlist(self, candidates: list[_MobileRankedCandidate]) -> list[_MobileRankedCandidate]:
        if not candidates:
            return []
        primary = candidates[: self.expiry_filter_geometry_top_n]
        cutoff = primary[-1].geometry_score if primary else candidates[0].geometry_score
        secondary_limit = max(self.expiry_filter_geometry_top_n, self.expiry_global_probe_top_k * 4)
        selected_ids = set(id(candidate) for candidate in primary)
        for candidate in candidates[:secondary_limit]:
            if candidate.candidate_type == "line":
                selected_ids.add(id(candidate))
                continue
            if "dot_matrix_rescue" in candidate.detector_sources and candidate.polygon_xy is not None:
                selected_ids.add(id(candidate))
                continue
            if float(candidate.rapidocr_probe_score or 0.0) >= self.rapidocr_gated_min_date_likeness_score:
                selected_ids.add(id(candidate))
                continue
            if candidate.geometry_score >= cutoff - 0.30 and candidate.geometry_features.get("area_ratio", 1.0) <= 0.16:
                selected_ids.add(id(candidate))
        for candidate in candidates:
            candidate.selected_geometry = id(candidate) in selected_ids
            if not candidate.selected_geometry:
                candidate.total_score = candidate.geometry_score
        self._debug_append(
            "geometry_shortlist",
            {
                "input_count": len(candidates),
                "selected_count": sum(1 for candidate in candidates if candidate.selected_geometry),
                "geometry_top_n": self.expiry_filter_geometry_top_n,
                "global_probe_top_k": self.expiry_global_probe_top_k,
                "candidates": [
                    {
                        **self._debug_ranked_candidate(candidate),
                        "rejection_reason": None
                        if candidate.selected_geometry
                        else "outside_geometry_shortlist_or_secondary_rules",
                    }
                    for candidate in candidates
                ],
            },
        )
        return candidates

    @staticmethod
    def _candidate_signal_text(candidate: _MobileRankedCandidate) -> str:
        return " ".join(
            part
            for part in (
                candidate.probe_normalized_text,
                candidate.probe_text,
                candidate.rapidocr_probe_normalized_text,
                candidate.rapidocr_probe_text,
            )
            if part
        )

    def _candidate_evidence_priority(self, candidate: _MobileRankedCandidate) -> int:
        text = self._candidate_signal_text(candidate)
        priority = self._text_evidence_priority(
            text,
            parser_bonus=float(candidate.score_breakdown.get("parser_probe_bonus", 0.0) or 0.0),
            expiry_score=float(candidate.score_breakdown.get("expiry_keyword_score", 0.0) or 0.0),
        )
        if priority >= 3 and candidate.candidate_type == "line":
            return 4
        return priority

    def _force_include_reason(self, candidate: _MobileRankedCandidate) -> str | None:
        if "dot_matrix_rescue" in candidate.detector_sources and candidate.polygon_xy is not None:
            return "dot_matrix_oriented_line"
        text = self._candidate_signal_text(candidate)
        has_date = self._has_date_like_text(text) or candidate.score_breakdown.get("parser_probe_bonus", 0.0) > 0
        has_expiry = has_expiry_keyword(text) or candidate.score_breakdown.get("expiry_keyword_score", 0.0) > 0
        if has_expiry and has_date:
            return "expiry_keyword_date"
        if candidate.score_breakdown.get("keyword_date_group_bonus", 0.0) > 0:
            return "keyword_date_group"
        if has_date and candidate.score_breakdown.get("parser_probe_bonus", 0.0) >= 1.0:
            return "parseable_date_probe"
        if has_date:
            return "date_like_probe"
        return None

    def _select_global_probe_candidates(
        self,
        candidates: list[_MobileGlobalCandidate],
    ) -> list[_MobileGlobalCandidate]:
        for candidate in candidates:
            candidate.ranked.selected_geometry = False
            candidate.ranked.selected_final = False
        deduped = self._dedupe_global_candidates(candidates)
        ordered = sorted(
            deduped,
            key=lambda item: (
                1 if item.ranked.candidate_type == "line" else 0,
                float(item.ranked.rapidocr_probe_score or 0.0),
                item.ranked.geometry_score,
                float(item.ranked.probe_confidence or 0.0),
                -item.ranked.geometry_features.get("area_ratio", 1.0),
            ),
            reverse=True,
        )
        selected = ordered[: self.expiry_global_probe_top_k]
        for candidate in selected:
            candidate.ranked.selected_geometry = True
        return selected

    def _probe_global_candidates(self, candidates: list[_MobileGlobalCandidate], *, today: date) -> None:
        self._profile_inc("num_probe_candidates", len(candidates))
        context_cache: dict[tuple[int, int, int, int], _RecognitionOutput] = {}
        context_candidates_seen = 0
        for global_candidate in candidates:
            candidate = global_candidate.ranked
            crop = candidate.crop(global_candidate.image)
            if crop.size == 0:
                candidate.probe_reason = "candidate crop empty"
                continue
            probe, variant_name, variant_image = self._best_probe_for_crop(crop, today=today)
            scores = self._probe_signal_scores(probe, candidate=candidate, today=today)
            probe_parsed = self._best_parse_for_inputs(self._build_parse_inputs(probe), today=today)
            candidate.probe_text = probe.raw_text
            candidate.probe_normalized_text = probe.normalized_text
            candidate.probe_confidence = probe.confidence
            candidate.probe_reason = probe.reason
            candidate.selected_recognition_variant = variant_name
            candidate.recognition_variant_image = variant_image
            candidate.selected_recognition_output = probe if probe_parsed.parsed_date is not None else None
            candidate.score_breakdown.update(scores)
            candidate.total_score = candidate.geometry_score + sum(scores.values())
            parsed = self._best_parse_for_inputs(self._build_parse_inputs(probe), today=today)
            self._debug_append(
                "probe_ocr",
                {
                    "candidate": self._debug_ranked_candidate(candidate),
                    "variant_name": variant_name,
                    "raw_text": probe.raw_text,
                    "normalized_text": probe.normalized_text,
                    "confidence": probe.confidence,
                    "reason": probe.reason,
                    "parse_result": self._debug_parsed_date(parsed),
                    "score_breakdown": dict(scores),
                },
            )
            if (
                self.context_probe_enabled
                and self.context_max_boxes_per_candidate > 0
                and context_candidates_seen < self.context_max_candidates_per_scan
                and candidate.context_probe_bboxes
            ):
                context_candidates_seen += 1
                context_scores = self._probe_candidate_context(
                    candidate,
                    roi=global_candidate.image,
                    cache=context_cache,
                )
                candidate.score_breakdown.update(context_scores)
                candidate.total_score = candidate.geometry_score + sum(candidate.score_breakdown.values())

    def _probe_candidate_context(
        self,
        candidate: _MobileRankedCandidate,
        *,
        roi: np.ndarray,
        cache: dict[tuple[int, int, int, int], _RecognitionOutput],
    ) -> dict[str, float]:
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
                with self._svtr_profile_stage("context_probe_ms"):
                    rec = self._recognize_variant(crop, variant_name="context", orientation="original")
                cache[bbox] = rec
            text = rec.normalized_text or rec.raw_text
            if not text:
                continue
            texts.append(text)
            confidences.append(rec.confidence)
            if has_expiry_keyword(text):
                expiry_hits += 1
            if has_production_keyword(text):
                production_hits += 1

        if not texts:
            return {}

        candidate.context_probe_texts = texts
        candidate.context_probe_confidences = confidences
        candidate.evidence_bbox = self._union_many_bboxes([candidate.recognition_bbox or candidate.bbox_xyxy, *context_bboxes[: len(texts)]])
        candidate.adjacent_expiry_keyword = expiry_hits > 0
        candidate.adjacent_production_keyword = production_hits > 0
        max_conf = max((float(conf or 0.0) for conf in confidences), default=0.0)
        scores: dict[str, float] = {"context_probe_confidence_score": min(0.25, max_conf * 0.25)}
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
            return max(self.expiry_global_final_top_k, self.expiry_global_debug_final_top_k)
        return self.expiry_global_final_top_k

    def _select_global_final_candidates(
        self,
        candidates: list[_MobileGlobalCandidate],
    ) -> list[_MobileGlobalCandidate]:
        for candidate in candidates:
            candidate.ranked.selected_final = False
            candidate.ranked.force_included_reason = None
            candidate.ranked.final_rank_before_force_include = None
            candidate.ranked.final_rank_after_force_include = None

        deduped = self._dedupe_global_candidates(candidates)
        cap = self._effective_global_final_top_k()
        selected: list[_MobileGlobalCandidate] = []
        selected_ids: set[int] = set()
        per_roi_counts: dict[str, int] = {}
        per_variant_counts: dict[str, int] = {}

        def sort_key(item: _MobileGlobalCandidate) -> tuple[int, float, float, float, float]:
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

        def add(item: _MobileGlobalCandidate, *, force: bool, reason: str | None = None) -> None:
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
        self._debug_append(
            "final_candidate_selection",
            {
                "input_count": len(candidates),
                "deduped_count": len(deduped),
                "cap": cap,
                "ordered": [
                    {
                        "rank": index,
                        "sort_key": list(sort_key(item)),
                        "candidate": self._debug_ranked_candidate(item.ranked),
                    }
                    for index, item in enumerate(ordered, start=1)
                ],
                "selected": [self._debug_ranked_candidate(item.ranked) for item in selected],
            },
        )
        self._profile_inc("num_final_candidates", len(selected))
        return selected

    @classmethod
    def _scan_bboxes_are_duplicates(cls, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        iou, containment = cls._bbox_overlap(a, b)
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        area_ratio = max(area_a, area_b) / float(max(1, min(area_a, area_b)))
        return iou >= 0.35 or (containment >= 0.80 and area_ratio <= 4.0)

    @staticmethod
    def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
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

    def _dedupe_global_candidates(self, candidates: list[_MobileGlobalCandidate]) -> list[_MobileGlobalCandidate]:
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
        kept: list[_MobileGlobalCandidate] = []
        for candidate in ordered:
            if any(
                candidate.ranked.candidate_type == existing.ranked.candidate_type
                and self._scan_bboxes_are_duplicates(candidate.scan_bbox, existing.scan_bbox)
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _normalize_ocr_confusions(text: str) -> str:
        mapping = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})

        def replace_token(match: "re.Match[str]") -> str:
            token = match.group(0)
            has_digit_or_separator = any(ch.isdigit() or ch in "/.-:" for ch in token)
            has_confusable_chars = any(ch in "OILSB" for ch in token)
            return token.translate(mapping) if has_digit_or_separator and has_confusable_chars else token

        return re.sub(r"[A-Z0-9/.\-:]+", replace_token, text.upper())

    @classmethod
    def _build_parse_inputs(cls, recognition: _RecognitionOutput) -> list[str]:
        return ExpiryCandidateEngine.build_parse_inputs_from_text(
            raw_text=recognition.raw_text or "",
            normalized_text=recognition.normalized_text or "",
        )

    def _best_parse_for_inputs(self, parse_inputs: list[str], *, today: date) -> ParsedDateData:
        parsed_inputs: list[tuple[str, ParsedDateData]] = []
        for text in parse_inputs:
            parsed = self.parser.parse(text, reference_date=today)
            if parsed.parsed_date is not None:
                parsed_inputs.append((text, parsed))
        if parsed_inputs:
            return max(parsed_inputs, key=lambda item: self._parse_input_selection_key(item[0], item[1]))[1]
        return self.parser.parse(parse_inputs[0] if parse_inputs else "", reference_date=today)

    @staticmethod
    def _parse_input_selection_key(text: str, parsed: ParsedDateData) -> tuple[int, int, int, int, float]:
        return ExpiryCandidateEngine.parse_input_selection_key(text, parsed)

    def _recognition_result_needs_180_fallback(self, recognition: _RecognitionOutput, parsed: ParsedDateData) -> bool:
        text = recognition.raw_text or recognition.normalized_text or ""
        if parsed.parsed_date is None:
            return True
        return self._is_generic_brand_text(text)

    @staticmethod
    def _rotate_180_normalized_crop(crop: NormalizedTextLineCrop) -> NormalizedTextLineCrop:
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

    def _collect_recognition_evidence(self, crop: np.ndarray, *, today: date) -> list[_RecognitionEvidence]:
        evidence: list[_RecognitionEvidence] = []
        variants = self._recognition_variants_for_crop(crop, allowed_names=MOBILE_RECOGNITION_VARIANTS)
        for variant in variants:
            rec = self._recognize_variant(variant.image, variant_name=variant.name, orientation="original")
            parse_inputs = self._build_parse_inputs(rec)
            parsed = self._best_parse_for_inputs(parse_inputs, today=today)
            evidence.append(
                _RecognitionEvidence(
                    variant_name=variant.name,
                    recognition=rec,
                    parsed=parsed,
                    parse_inputs_count=len(parse_inputs),
                )
            )
            if self._is_strong_day_date_parse(parsed, rec, text=" ".join(parse_inputs)):
                break
        return evidence

    def _best_recognition_for_single_crop(
        self,
        crop: NormalizedTextLineCrop,
        *,
        today: date,
        selected_variant_name: str | None = None,
        selected_variant_image: np.ndarray | None = None,
        selected_variant_output: _RecognitionOutput | None = None,
        crop_policy: str | None = None,
        allowed_variant_names: tuple[str, ...] = MOBILE_RECOGNITION_VARIANTS,
    ) -> _EvaluatedCandidate:
        if crop_policy == "dot_matrix_rescue" and allowed_variant_names == MOBILE_RECOGNITION_VARIANTS:
            allowed_variant_names = DOT_MATRIX_RESCUE_RECOGNITION_VARIANTS
        best_parseable: tuple[tuple[int, int, float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_with_text: tuple[tuple[float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_empty: tuple[_RecognitionOutput, ParsedDateData, int, str] | None = None
        date_evidence: list[_DateEvidence] = []

        variants: list[ImageVariant] = []
        selected_name = selected_variant_name or "original"
        should_prioritize_selected = bool(
            selected_variant_name
            and (
                crop_policy == "dot_matrix_rescue"
                or selected_variant_name in NORMAL_SELECTED_VARIANT_PRIORITY
            )
        )
        if selected_variant_image is not None and should_prioritize_selected:
            variants.append(ImageVariant(selected_name, selected_variant_image, purpose="recognition"))
        elif selected_variant_name and should_prioritize_selected:
            variants.extend(
                variant
                for variant in self._recognition_variants_for_crop(crop.image, allowed_names=(selected_variant_name,))
                if variant.name == selected_name
            )
        for variant in self._recognition_variants_for_crop(crop.image, allowed_names=allowed_variant_names):
            if should_prioritize_selected and variant.name == selected_name and variants:
                continue
            variants.append(variant)

        recognition_jobs: list[tuple[ImageVariant, str, Any | None, _RecognitionOutput | None]] = []
        for index, variant in enumerate(variants):
            precomputed = (
                selected_variant_output
                if index == 0 and should_prioritize_selected and variant.name == selected_name
                else None
            )
            recognition_jobs.append((variant, variant.name, None, precomputed))
        if (
            crop_policy == "dot_matrix_rescue"
            and self.hardcase_recognizer_rescue_enabled
            and self.hardcase_recognizer is not None
        ):
            for variant in self._recognition_variants_for_crop(crop.image, allowed_names=DOT_MATRIX_RESCUE_RECOGNITION_VARIANTS):
                recognition_jobs.append((variant, f"hardcase:{variant.name}", self.hardcase_recognizer, None))

        for variant, variant_name, recognizer_override, precomputed_recognition in recognition_jobs:
            recognize_kwargs: dict[str, Any] = {
                "variant_name": variant_name,
                "orientation": crop.selected_orientation,
            }
            if recognizer_override is not None:
                recognize_kwargs["recognizer"] = recognizer_override
            rec = precomputed_recognition or self._recognize_variant(variant.image, **recognize_kwargs)
            parse_inputs = self._build_parse_inputs(rec)
            parsed = self._best_parse_for_inputs(parse_inputs, today=today)
            self._debug_append(
                "svtr_final_attempts",
                {
                    "crop_bbox_xyxy": self._bbox_json(crop.bbox_xyxy),
                    "crop_policy": crop_policy,
                    "variant_name": variant_name,
                    "orientation": crop.selected_orientation,
                    "raw_text": rec.raw_text,
                    "normalized_text": rec.normalized_text,
                    "confidence": rec.confidence,
                    "reason": rec.reason,
                    "parse_inputs": parse_inputs,
                    "parse_result": self._debug_parsed_date(parsed),
                },
            )
            if parsed.parsed_date is not None:
                date_evidence.append(
                    _DateEvidence(
                        candidate=self._empty_candidate_for_internal_use(),
                        recognition=rec,
                        parsed=parsed,
                        parse_inputs_count=len(parse_inputs),
                        recognition_variant=variant_name,
                        normalized_crop=crop,
                        crop_policy=crop_policy,
                    )
                )
                score = (
                    self._date_precision_score(parsed),
                    parsed.parsed_date.toordinal(),
                    float(parsed.confidence),
                    float(rec.confidence or 0.0),
                    self._score_date_likeness(rec.raw_text),
                )
                item = (score, rec, parsed, len(parse_inputs), variant_name)
                if best_parseable is None or score > best_parseable[0]:
                    best_parseable = item
                if self._is_strong_day_date_parse(parsed, rec, text=" ".join(parse_inputs)):
                    break
                continue
            if rec.raw_text.strip():
                score = (self._score_ocr_candidate(rec), float(rec.confidence or 0.0), self._score_date_likeness(rec.raw_text))
                item = (score, rec, parsed, len(parse_inputs), variant_name)
                if best_with_text is None or score > best_with_text[0]:
                    best_with_text = item
            elif best_empty is None:
                best_empty = (rec, parsed, len(parse_inputs), variant_name)

        if best_parseable is not None:
            _score, rec, parsed, count, variant_name = best_parseable
        elif best_with_text is not None:
            _score, rec, parsed, count, variant_name = best_with_text
        elif best_empty is not None:
            rec, parsed, count, variant_name = best_empty
        else:
            rec = _RecognitionOutput("", "", None, "no recognition variants", crop.selected_orientation)
            parsed = self.parser.parse("", reference_date=today)
            count = 1
            variant_name = "original_padded"

        if crop.selected_orientation == "original" and self._recognition_result_needs_180_fallback(rec, parsed):
            rotated_crop = self._rotate_180_normalized_crop(crop)
            rotated_rec = self._recognize_variant(rotated_crop.image, variant_name="rotate_180", orientation="rotate_180")
            rotated_inputs = self._build_parse_inputs(rotated_rec)
            rotated_parsed = self._best_parse_for_inputs(rotated_inputs, today=today)
            self._debug_append(
                "svtr_final_attempts",
                {
                    "crop_bbox_xyxy": self._bbox_json(rotated_crop.bbox_xyxy),
                    "crop_policy": crop_policy,
                    "variant_name": "rotate_180",
                    "orientation": "rotate_180",
                    "raw_text": rotated_rec.raw_text,
                    "normalized_text": rotated_rec.normalized_text,
                    "confidence": rotated_rec.confidence,
                    "reason": rotated_rec.reason,
                    "parse_inputs": rotated_inputs,
                    "parse_result": self._debug_parsed_date(rotated_parsed),
                },
            )
            rotated_score = (
                self._date_precision_score(rotated_parsed),
                rotated_parsed.parsed_date.toordinal() if rotated_parsed.parsed_date is not None else 0,
                float(rotated_parsed.confidence if rotated_parsed.parsed_date is not None else 0.0),
                self._score_date_likeness(rotated_rec.raw_text),
                float(rotated_rec.confidence or 0.0),
            )
            current_score = (
                self._date_precision_score(parsed),
                parsed.parsed_date.toordinal() if parsed.parsed_date is not None else 0,
                float(parsed.confidence if parsed.parsed_date is not None else 0.0),
                self._score_date_likeness(rec.raw_text),
                float(rec.confidence or 0.0),
            )
            if rotated_score > current_score:
                rotated_evidence = _DateEvidence(
                    candidate=self._empty_candidate_for_internal_use(),
                    recognition=rotated_rec,
                    parsed=rotated_parsed,
                    parse_inputs_count=len(rotated_inputs),
                    recognition_variant="rotate_180",
                    normalized_crop=rotated_crop,
                    crop_policy=crop_policy,
                    original_color_fallback_used=True,
                )
                return _EvaluatedCandidate(
                    candidate=self._empty_candidate_for_internal_use(),
                    recognition=rotated_rec,
                    parsed=rotated_parsed,
                    parse_inputs_count=len(rotated_inputs),
                    recognition_variant="rotate_180",
                    normalized_crop=rotated_crop,
                    original_color_fallback_used=True,
                    date_evidence=[*date_evidence, rotated_evidence],
                )

        return _EvaluatedCandidate(
            candidate=self._empty_candidate_for_internal_use(),
            recognition=rec,
            parsed=parsed,
            parse_inputs_count=count,
            recognition_variant=variant_name,
            normalized_crop=crop,
            date_evidence=date_evidence,
        )

    @staticmethod
    def _empty_candidate_for_internal_use() -> _MobileRankedCandidate:
        return _MobileRankedCandidate(
            candidate_id="internal",
            candidate_type="internal",
            bbox_xyxy=(0, 0, 1, 1),
            polygon_xy=None,
            detector_confidence=None,
            detector_sources=("internal",),
            detector_variant=None,
            member_indices=[],
            member_bboxes=[],
            geometry_features={},
            geometry_score=0.0,
            score_breakdown={},
            total_score=0.0,
        )

    def _legacy_wide_group_crops_for_candidate(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
    ) -> list[NormalizedTextLineCrop]:
        if candidate.candidate_type not in {"group", "fallback", "line"}:
            return []
        if candidate.final_crop_bbox is None:
            return []
        wide_bbox = candidate.evidence_bbox or candidate.bbox_xyxy
        if wide_bbox == candidate.final_crop_bbox:
            return []
        final_area = self._bbox_area(candidate.final_crop_bbox)
        wide_area = self._bbox_area(wide_bbox)
        final_w = max(1, candidate.final_crop_bbox[2] - candidate.final_crop_bbox[0])
        final_h = max(1, candidate.final_crop_bbox[3] - candidate.final_crop_bbox[1])
        wide_w = max(1, wide_bbox[2] - wide_bbox[0])
        wide_h = max(1, wide_bbox[3] - wide_bbox[1])
        if wide_area < final_area * 1.20 and wide_w < final_w * 1.10 and wide_h < final_h * 1.10:
            return []
        return normalize_textline_crops(
            image,
            TextLineGeometry(bbox_xyxy=wide_bbox, polygon_xy=None),
            TextLineCropConfig(padding_px=0),
        )

    def _local_group_wide_bbox_for_candidate(
        self,
        candidate: _MobileRankedCandidate,
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        if candidate.final_crop_bbox is None:
            return None
        if candidate.candidate_type not in {"group", "fallback", "line"} and not candidate.evidence_bbox:
            return None
        base_bbox = candidate.evidence_bbox or candidate.bbox_xyxy
        if base_bbox == candidate.final_crop_bbox:
            return None
        final_w = max(1, candidate.final_crop_bbox[2] - candidate.final_crop_bbox[0])
        final_h = max(1, candidate.final_crop_bbox[3] - candidate.final_crop_bbox[1])
        base_w = max(1, base_bbox[2] - base_bbox[0])
        base_h = max(1, base_bbox[3] - base_bbox[1])
        pad_left = min(48, max(8, int(round(base_w * 0.034))))
        pad_right = min(48, max(8, int(round(base_w * 0.04))))
        pad_top = min(48, max(8, int(round(base_h * 0.08))))
        pad_bottom = min(48, max(8, int(round(base_h * 0.06))))
        wide_bbox = self._clip_bbox_to_image(
            (
                base_bbox[0] - pad_left,
                base_bbox[1] - pad_top,
                base_bbox[2] + pad_right,
                base_bbox[3] + pad_bottom,
            ),
            image_shape,
        )
        if wide_bbox is None or wide_bbox == candidate.final_crop_bbox:
            return None
        wide_w = max(1, wide_bbox[2] - wide_bbox[0])
        wide_h = max(1, wide_bbox[3] - wide_bbox[1])
        if wide_w <= final_w * 1.02 and wide_h <= final_h * 1.02:
            return None
        return wide_bbox

    def _should_run_local_group_wide_crop_evidence(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        date_evidence: list[_DateEvidence],
        *,
        image_shape: tuple[int, ...],
    ) -> bool:
        if not self.local_group_wide_crop_enabled:
            return False
        if self._has_strong_accepted_date_evidence(evaluated, date_evidence):
            return False
        if self._local_group_wide_bbox_for_candidate(candidate, image_shape=image_shape) is None:
            return False
        if evaluated.parsed.parsed_date is None:
            return True
        if self._date_precision_score(evaluated.parsed) < 3:
            return True
        if len({evidence.parsed.parsed_date for evidence in date_evidence if evidence.parsed.parsed_date is not None}) > 1:
            return True
        parser_conf = float(evaluated.parsed.confidence or 0.0)
        ocr_conf = float(evaluated.recognition.confidence or 0.0)
        return parser_conf < 0.90 or ocr_conf < 0.85

    def _collect_local_group_wide_crop_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        *,
        today: date,
    ) -> list[_DateEvidence]:
        bbox = self._local_group_wide_bbox_for_candidate(candidate, image_shape=image.shape)
        if bbox is None:
            return []
        return self._date_evidence_from_bbox(
            image,
            candidate,
            bbox,
            today=today,
            crop_policy="local_group_wide_crop",
            context_texts=tuple(candidate.context_probe_texts or ()),
            metadata={
                "source": "crop_policy_local_group_wide",
                "forced_reason": "weak_or_ambiguous_tight_crop_with_wider_group_bbox",
                "base_bbox": list(candidate.evidence_bbox or candidate.bbox_xyxy),
                "tight_bbox": list(candidate.final_crop_bbox) if candidate.final_crop_bbox else None,
            },
        )

    @staticmethod
    def _is_rotation_rescue_candidate(candidate: _MobileRankedCandidate) -> bool:
        return "rotation_rescue" in set(candidate.detector_sources)

    def _rotation_wide_bbox_for_candidate(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        if not self._is_rotation_rescue_candidate(candidate):
            return None
        if evaluated.normalized_crop is not None:
            base_bbox = evaluated.normalized_crop.bbox_xyxy
        else:
            base_bbox = candidate.final_crop_bbox or candidate.evidence_bbox or candidate.bbox_xyxy
        x1, y1, x2, y2 = base_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(8, int(round(width * 0.35)))
        pad_y = max(4, int(round(height * 0.20)))
        wide_bbox = self._clip_bbox_to_image((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), image_shape)
        if wide_bbox is None or wide_bbox == base_bbox:
            return None
        return wide_bbox

    def _collect_rotation_wide_crop_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if not self.rotation_wide_crop_enabled:
            return []
        if self._has_strong_accepted_date_evidence(evaluated, evaluated.date_evidence):
            return []
        bbox = self._rotation_wide_bbox_for_candidate(candidate, evaluated, image_shape=image.shape)
        if bbox is None:
            return []
        base_bbox = evaluated.normalized_crop.bbox_xyxy if evaluated.normalized_crop is not None else candidate.final_crop_bbox
        return self._date_evidence_from_bbox(
            image,
            candidate,
            bbox,
            today=today,
            crop_policy="rotation_wide_crop",
            context_texts=tuple(candidate.context_probe_texts or ()),
            metadata={
                "source": "crop_policy_rotation_wide",
                "base_bbox": list(base_bbox or candidate.evidence_bbox or candidate.bbox_xyxy),
                "detector_bbox": list(candidate.bbox_xyxy),
            },
        )

    def _candidate_line_bboxes(self, candidate: _MobileRankedCandidate) -> list[tuple[int, int, int, int]]:
        if not candidate.member_bboxes:
            return []
        lines = self._cluster_bboxes_into_lines(candidate.member_bboxes)
        if len(lines) <= 1:
            return []
        return [self._union_many_bboxes(line) for line in lines]

    @staticmethod
    def _bbox_vertical_overlap_ratio(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
        return overlap / float(max(1, min(first[3] - first[1], second[3] - second[1])))

    def _line_bbox_for_crop(
        self,
        candidate: _MobileRankedCandidate,
        crop_bbox: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if crop_bbox is None:
            return None
        lines = self._candidate_line_bboxes(candidate)
        if not lines:
            return None
        return max(lines, key=lambda line: self._bbox_vertical_overlap_ratio(line, crop_bbox))

    def _line_role_context_for_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, ...]:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        prefix_x1 = max(0, x1 - int(round(width * 0.65)))
        prefix_x2 = min(x2, x1 + max(28, int(round(width * 0.18))))
        if prefix_x2 <= prefix_x1:
            return ()
        crop = image[y1:y2, prefix_x1:prefix_x2]
        if crop.size == 0:
            return ()
        texts: list[str] = []
        variants = self._recognition_variants_for_crop(
            crop,
            allowed_names=("original_color_tight", "gray_upscaled", "clahe_gray", "adaptive_binary"),
        )
        context_variants = [
            ImageVariant(f"line_context_{variant.name}", variant.image, purpose=variant.purpose)
            for variant in variants
        ]
        if self._batch_recognition_enabled(self.recognizer):
            recognized_variants = self._recognize_variants_batch(context_variants, orientation="original")
        else:
            recognized_variants = (
                (variant, self._recognize_variant(variant.image, variant_name=variant.name, orientation="original"))
                for variant in context_variants
            )
        for _variant, rec in recognized_variants:
            text = rec.normalized_text or rec.raw_text
            if not text:
                continue
            if has_expiry_keyword(text) or has_weak_expiry_keyword(text) or has_production_keyword(text):
                texts.append(text)
        return tuple(dict.fromkeys(texts))

    @staticmethod
    def _has_expiry_context_signal(context_texts: tuple[str, ...]) -> bool:
        context = " ".join(context_texts)
        return has_expiry_keyword(context) or has_weak_expiry_keyword(context)

    def _line_role_context_for_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        evidence: _DateEvidence,
    ) -> tuple[str, ...]:
        crop_bbox = evidence.normalized_crop.bbox_xyxy if evidence.normalized_crop is not None else candidate.final_crop_bbox
        line_bbox = self._line_bbox_for_crop(candidate, crop_bbox)
        if line_bbox is not None:
            context = self._line_role_context_for_bbox(image, line_bbox)
            if context:
                return context
        if not self.production_anchor_sibling_enabled:
            return ()
        seen: set[tuple[int, int, int, int]] = set()
        for bbox in (
            candidate.recognition_bbox,
            candidate.evidence_bbox,
            candidate.bbox_xyxy,
            candidate.final_crop_bbox,
            crop_bbox,
        ):
            if bbox is None or bbox in seen:
                continue
            seen.add(bbox)
            context = self._line_role_context_for_bbox(image, bbox)
            if context:
                return context
        return ()

    @staticmethod
    def _has_production_anchor_signal(context_texts: tuple[str, ...]) -> bool:
        context = " ".join(context_texts)
        normalized = normalize_evidence_text(context)
        compact = compact_evidence_text(context)
        return (
            has_production_keyword(context)
            or "PKT" in compact
            or bool(re.search(r"\bP\b", normalized))
        )

    @staticmethod
    def _clip_bbox_to_image(
        bbox: tuple[int, int, int, int],
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        h, w = image_shape[:2]
        clipped = (
            max(0, min(w, int(bbox[0]))),
            max(0, min(h, int(bbox[1]))),
            max(0, min(w, int(bbox[2]))),
            max(0, min(h, int(bbox[3]))),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        return clipped

    def _date_evidence_from_bbox(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        bbox: tuple[int, int, int, int],
        *,
        today: date,
        crop_policy: str,
        context_texts: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
        allowed_names: tuple[str, ...] = MOBILE_RECOGNITION_VARIANTS,
        recognizer: Any | None = None,
        recognition_variant_prefix: str | None = None,
    ) -> list[_DateEvidence]:
        clipped = self._clip_bbox_to_image(bbox, image.shape)
        if clipped is None:
            return []
        evidence: list[_DateEvidence] = []
        crops = normalize_textline_crops(
            image,
            TextLineGeometry(bbox_xyxy=clipped, polygon_xy=None),
            TextLineCropConfig(padding_px=0),
        )
        for crop in crops:
            if crop.image.size == 0:
                continue
            variants = self._recognition_variants_for_crop(crop.image, allowed_names=allowed_names)
            for variant in variants:
                variant_name = f"{recognition_variant_prefix}:{variant.name}" if recognition_variant_prefix else variant.name
                recognize_kwargs: dict[str, Any] = {
                    "variant_name": variant_name,
                    "orientation": crop.selected_orientation,
                }
                if recognizer is not None:
                    recognize_kwargs["recognizer"] = recognizer
                rec = self._recognize_variant(variant.image, **recognize_kwargs)
                parse_inputs = self._build_parse_inputs(rec)
                parsed = self._best_parse_for_inputs(parse_inputs, today=today)
                if parsed.parsed_date is None:
                    continue
                evidence.append(
                    _DateEvidence(
                        candidate=candidate,
                        recognition=rec,
                        parsed=parsed,
                        parse_inputs_count=len(parse_inputs),
                        recognition_variant=variant_name,
                        normalized_crop=crop,
                        crop_policy=crop_policy,
                        context_texts=context_texts,
                        metadata=dict(metadata or {}),
                    )
                )
                if self._is_strong_day_date_parse(
                    parsed,
                    rec,
                    text=" ".join([*parse_inputs, *context_texts]),
                ):
                    return evidence
        return evidence

    @staticmethod
    def _has_partial_day_month_text(text: str) -> bool:
        if not text:
            return False
        digit_groups = re.findall(r"\d+", text)
        if len(digit_groups) != 2:
            return False
        if not re.search(r"\d{1,2}\s*[./-]\s*\d{1,2}", text):
            return False
        day = int(digit_groups[0])
        month = int(digit_groups[1])
        return 1 <= day <= 31 and 1 <= month <= 12

    def _is_high_confidence_partial_day_month_eval(self, evaluated: _EvaluatedCandidate) -> bool:
        if evaluated.parsed.parsed_date is not None:
            return False
        if evaluated.normalized_crop is None:
            return False
        if float(evaluated.recognition.confidence or 0.0) < 0.90:
            return False
        return any(self._has_partial_day_month_text(text) for text in self._build_parse_inputs(evaluated.recognition))

    def _has_sibling_full_day_date_evidence(
        self,
        candidate: _MobileRankedCandidate,
        date_evidence: list[_DateEvidence],
    ) -> bool:
        return any(
            evidence.candidate.candidate_id == candidate.candidate_id
            and evidence.parsed.parsed_date is not None
            and self._date_precision_score(evidence.parsed) >= 3
            and not self._should_suppress_date_evidence(evidence)
            for evidence in date_evidence
        )

    @staticmethod
    def _date_evidence_bbox(evidence: _DateEvidence) -> tuple[int, int, int, int] | None:
        if evidence.normalized_crop is not None:
            return evidence.normalized_crop.bbox_xyxy
        return evidence.candidate.final_crop_bbox or evidence.candidate.recognition_bbox or evidence.candidate.bbox_xyxy

    def _is_day_month_year_context_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.parsed.parsed_date is None or self._date_precision_score(evidence.parsed) < 3:
            return False
        if float(evidence.recognition.confidence or 0.0) < 0.70:
            return False
        text = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        digit_groups = re.findall(r"\d+", text)
        digit_count = len(re.findall(r"\d", text))
        return bool(
            self._has_clean_full_date_pattern(text)
            or len(digit_groups) >= 3
            or 6 <= digit_count <= 8
        )

    def _partial_day_month_has_nearby_full_date_context(
        self,
        partial_bbox: tuple[int, int, int, int],
        context_evidence: _DateEvidence,
    ) -> bool:
        context_bbox = self._date_evidence_bbox(context_evidence)
        if context_bbox is None:
            return False
        px1, py1, px2, py2 = partial_bbox
        cx1, cy1, cx2, cy2 = context_bbox
        partial_w = max(1, px2 - px1)
        context_w = max(1, cx2 - cx1)
        partial_h = max(1, py2 - py1)
        context_h = max(1, cy2 - cy1)
        vertical_gap = max(0, max(py1, cy1) - min(py2, cy2))
        vertical_overlap = self._bbox_vertical_overlap_ratio(partial_bbox, context_bbox)
        horizontal_overlap = self._bbox_horizontal_overlap_ratio(partial_bbox, context_bbox)
        partial_cx = (px1 + px2) / 2.0
        context_cx = (cx1 + cx2) / 2.0
        center_aligned = abs(partial_cx - context_cx) <= max(partial_w, context_w) * 0.75
        same_stack = horizontal_overlap >= 0.20 or center_aligned
        nearby_line = vertical_overlap >= 0.10 or vertical_gap <= max(partial_h, context_h) * 1.75
        starts_after_context = px1 > cx2 + max(partial_w, context_w) * 0.25
        return bool(same_stack and nearby_line and not starts_after_context)

    def _partial_day_month_right_expansion_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_right = min(320, max(72, int(round(width * 0.55))))
        pad_y = max(2, int(round(height * 0.08)))
        expanded = self._clip_bbox_to_image((x1, y1 - pad_y, x2 + pad_right, y2 + pad_y), image_shape)
        if expanded is None or expanded == bbox:
            return None
        return expanded

    def _collect_partial_day_month_right_expansion_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        partial_evaluated: list[_EvaluatedCandidate],
        date_evidence: list[_DateEvidence],
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if candidate.candidate_type not in {"group", "fallback", "line"}:
            return []
        if len(self._candidate_line_bboxes(candidate)) <= 1:
            return []
        if not self._has_sibling_full_day_date_evidence(candidate, date_evidence):
            return []
        evidence: list[_DateEvidence] = []
        seen: set[tuple[int, int, int, int]] = set()
        partials = [item for item in partial_evaluated if self._is_high_confidence_partial_day_month_eval(item)]
        for item in partials[:2]:
            crop = item.normalized_crop
            if crop is None:
                continue
            expanded_bbox = self._partial_day_month_right_expansion_bbox(crop.bbox_xyxy, image_shape=image.shape)
            if expanded_bbox is None or expanded_bbox in seen:
                continue
            seen.add(expanded_bbox)
            evidence.extend(
                self._date_evidence_from_bbox(
                    image,
                    candidate,
                    expanded_bbox,
                    today=today,
                    crop_policy="partial_day_month_right_expand",
                    context_texts=tuple(candidate.context_probe_texts or ()),
                    metadata={
                        "source": "partial_day_month_right_expand",
                        "base_bbox": list(crop.bbox_xyxy),
                        "base_raw_text": item.recognition.raw_text,
                        "base_normalized_text": item.recognition.normalized_text,
                        "forced_reason": "partial_day_month_with_sibling_full_date",
                    },
                    allowed_names=PARTIAL_DAY_MONTH_RIGHT_EXPANSION_VARIANTS,
                )
            )
        return evidence

    def _collect_cross_candidate_partial_day_month_right_expansion_evidence(
        self,
        image: np.ndarray,
        evaluated: list[_EvaluatedCandidate],
        *,
        today: date,
    ) -> list[_DateEvidence]:
        context_evidence = [
            evidence
            for item in evaluated
            for evidence in item.date_evidence
            if self._is_day_month_year_context_evidence(evidence)
        ]
        if not context_evidence:
            return []
        partials = [item for item in evaluated if self._is_high_confidence_partial_day_month_eval(item)]
        if not partials:
            return []

        evidence: list[_DateEvidence] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for item in partials[:3]:
            if any(existing.crop_policy == "partial_day_month_right_expand" for existing in item.date_evidence):
                continue
            crop = item.normalized_crop
            if crop is None:
                continue
            nearby_context = [
                context
                for context in context_evidence
                if self._partial_day_month_has_nearby_full_date_context(crop.bbox_xyxy, context)
            ]
            if not nearby_context:
                continue
            expanded_bbox = self._partial_day_month_right_expansion_bbox(crop.bbox_xyxy, image_shape=image.shape)
            if expanded_bbox is None:
                continue
            seen_key = (item.candidate.candidate_id, expanded_bbox)
            if seen_key in seen:
                continue
            seen.add(seen_key)
            context_texts = tuple(
                dict.fromkeys(
                    [
                        *(item.candidate.context_probe_texts or ()),
                        *(
                            context.recognition.raw_text
                            for context in nearby_context[:2]
                            if context.recognition.raw_text
                        ),
                    ]
                )
            )
            evidence.extend(
                self._date_evidence_from_bbox(
                    image,
                    item.candidate,
                    expanded_bbox,
                    today=today,
                    crop_policy="partial_day_month_right_expand",
                    context_texts=context_texts,
                    metadata={
                        "source": "partial_day_month_right_expand",
                        "base_bbox": list(crop.bbox_xyxy),
                        "base_raw_text": item.recognition.raw_text,
                        "base_normalized_text": item.recognition.normalized_text,
                        "forced_reason": "partial_day_month_with_neighbor_full_date",
                        "context_candidate_ids": [context.candidate.candidate_id for context in nearby_context[:2]],
                    },
                    allowed_names=PARTIAL_DAY_MONTH_RIGHT_EXPANSION_VARIANTS,
                )
            )
            if self.hardcase_recognizer_rescue_enabled and self.hardcase_recognizer is not None:
                evidence.extend(
                    self._date_evidence_from_bbox(
                        image,
                        item.candidate,
                        expanded_bbox,
                        today=today,
                        crop_policy="partial_day_month_right_expand",
                        context_texts=context_texts,
                        metadata={
                            "source": "partial_day_month_right_expand",
                            "base_bbox": list(crop.bbox_xyxy),
                            "base_raw_text": item.recognition.raw_text,
                            "base_normalized_text": item.recognition.normalized_text,
                            "forced_reason": "partial_day_month_with_neighbor_full_date",
                            "context_candidate_ids": [context.candidate.candidate_id for context in nearby_context[:2]],
                            "recognizer": "hardcase",
                        },
                        allowed_names=PARTIAL_DAY_MONTH_RIGHT_EXPANSION_VARIANTS,
                        recognizer=self.hardcase_recognizer,
                        recognition_variant_prefix="hardcase",
                    )
                )
        if evidence:
            self._debug_append(
                "partial_day_month_right_expand",
                {
                    "source": "cross_candidate",
                    "evidence_count": len(evidence),
                    "evidence": [self._debug_date_evidence(item) for item in evidence],
                },
            )
        return evidence

    @staticmethod
    def _append_date_evidence_to_evaluated(
        evaluated: list[_EvaluatedCandidate],
        evidence_items: list[_DateEvidence],
    ) -> None:
        if not evidence_items:
            return
        by_candidate_id = {item.candidate.candidate_id: item for item in evaluated}
        for evidence in evidence_items:
            item = by_candidate_id.get(evidence.candidate.candidate_id)
            if item is None:
                continue
            item.date_evidence.append(evidence)

    def _add_cross_candidate_partial_day_month_right_expansion_evidence(
        self,
        image: np.ndarray,
        evaluated: list[_EvaluatedCandidate],
        *,
        today: date,
    ) -> None:
        evidence = self._collect_cross_candidate_partial_day_month_right_expansion_evidence(
            image,
            evaluated,
            today=today,
        )
        self._append_date_evidence_to_evaluated(evaluated, evidence)

    def _collect_line_role_rescue_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if candidate.candidate_type not in {"group", "fallback"}:
            return []
        evidence: list[_DateEvidence] = []
        seen_bboxes: set[tuple[int, int, int, int]] = set()
        for line_bbox in self._candidate_line_bboxes(candidate):
            padded_bbox, _padding = self._pad_bbox_for_final_crop(line_bbox, image.shape)
            if padded_bbox in seen_bboxes:
                continue
            seen_bboxes.add(padded_bbox)
            context_texts = self._line_role_context_for_bbox(image, line_bbox)
            crops = normalize_textline_crops(
                image,
                TextLineGeometry(bbox_xyxy=padded_bbox, polygon_xy=None),
                TextLineCropConfig(padding_px=0),
            )
            for crop in crops:
                if crop.image.size == 0:
                    continue
                variants = self._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS)
                if self._batch_recognition_enabled(self.recognizer):
                    recognized_variants = self._recognize_variants_batch(
                        variants,
                        orientation=crop.selected_orientation,
                        use_cache=False,
                    )
                else:
                    recognized_variants = (
                        (
                            variant,
                            self._recognize_variant(
                                variant.image,
                                variant_name=variant.name,
                                orientation=crop.selected_orientation,
                                use_cache=False,
                            ),
                        )
                        for variant in variants
                    )
                for variant, rec in recognized_variants:
                    parse_inputs = self._build_parse_inputs(rec)
                    parsed = self._best_parse_for_inputs(parse_inputs, today=today)
                    if parsed.parsed_date is None:
                        continue
                    evidence.append(
                        _DateEvidence(
                            candidate=candidate,
                            recognition=rec,
                            parsed=parsed,
                            parse_inputs_count=len(parse_inputs),
                            recognition_variant=variant.name,
                            normalized_crop=crop,
                            crop_policy="line_role_context",
                            context_texts=context_texts,
                        )
                    )
        return evidence

    def _should_run_line_role_rescue(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        date_evidence: list[_DateEvidence],
    ) -> bool:
        if candidate.candidate_type not in {"group", "fallback"}:
            return False
        if len(self._candidate_line_bboxes(candidate)) <= 1:
            return False
        if self._has_strong_accepted_date_evidence(evaluated, date_evidence):
            selected_texts = [
                self._date_evidence_text(evidence)
                for evidence in date_evidence
                if evidence.parsed.parsed_date == evaluated.parsed.parsed_date
            ]
            if not selected_texts:
                selected_texts = [
                    " ".join(
                        part
                        for part in (
                            evaluated.recognition.normalized_text,
                            evaluated.recognition.raw_text,
                            " ".join(evaluated.candidate.context_probe_texts or []),
                        )
                        if part
                    )
                ]
            if any(has_expiry_keyword(text) or has_weak_expiry_keyword(text) for text in selected_texts):
                return False
        if any(has_expiry_keyword(self._date_evidence_text(evidence)) for evidence in date_evidence):
            return False
        return True

    def _production_anchor_contexts(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        date_evidence: list[_DateEvidence],
        *,
        today: date,
    ) -> list[tuple[tuple[int, int, int, int], tuple[str, ...], date | None]]:
        anchors: list[tuple[tuple[int, int, int, int], tuple[str, ...], date | None]] = []
        for evidence in date_evidence:
            bbox = (
                evidence.normalized_crop.bbox_xyxy
                if evidence.normalized_crop is not None
                else candidate.final_crop_bbox or candidate.recognition_bbox or candidate.evidence_bbox or candidate.bbox_xyxy
            )
            if bbox is None:
                continue
            context_texts = evidence.context_texts or tuple(candidate.context_probe_texts or ())
            text = self._date_evidence_text(evidence)
            if has_production_keyword(text) or self._has_production_anchor_signal(context_texts):
                anchors.append((bbox, context_texts, evidence.parsed.parsed_date))
                continue
        representative_bbox = candidate.final_crop_bbox or candidate.recognition_bbox or candidate.evidence_bbox
        if representative_bbox is not None:
            context_texts = tuple(candidate.context_probe_texts or ())
            text = " ".join(
                part
                for part in (
                    candidate.probe_text,
                    candidate.probe_normalized_text,
                    " ".join(candidate.context_probe_texts or []),
                )
                if part
            )
            if has_production_keyword(text) or (context_texts and self._has_production_anchor_signal(context_texts)):
                anchors.append((representative_bbox, context_texts, None))
        deduped: dict[tuple[int, int, int, int], tuple[tuple[str, ...], date | None]] = {}
        for bbox, context_texts, anchor_date in anchors:
            deduped.setdefault(bbox, (context_texts, anchor_date))
        return [(bbox, context_texts, anchor_date) for bbox, (context_texts, anchor_date) in deduped.items()]

    def _production_anchor_sibling_bboxes(
        self,
        anchor_bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        x1, y1, x2, y2 = anchor_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        candidates = [
            (
                "below_date_aligned",
                (
                    x1 + int(round(width * 0.07)),
                    y2 - int(round(height * 0.08)),
                    x2 - int(round(width * 0.23)),
                    y2 + int(round(height * 1.05)),
                ),
            ),
            (
                "right_same_line",
                (
                    x2 + int(round(width * 0.02)),
                    y1 - int(round(height * 0.25)),
                    x2 + int(round(width * 1.15)),
                    y2 + int(round(height * 0.25)),
                ),
            ),
            (
                "above_date_aligned",
                (
                    x1 + int(round(width * 0.07)),
                    y1 - int(round(height * 1.05)),
                    x2 - int(round(width * 0.23)),
                    y1 + int(round(height * 0.08)),
                ),
            ),
            (
                "left_same_line",
                (
                    x1 - int(round(width * 1.15)),
                    y1 - int(round(height * 0.25)),
                    x1 - int(round(width * 0.02)),
                    y2 + int(round(height * 0.25)),
                ),
            ),
            (
                "below_wide",
                (
                    x1 - int(round(width * 0.65)),
                    y2 - int(round(height * 0.20)),
                    x2 + int(round(width * 0.20)),
                    y2 + int(round(height * 1.35)),
                ),
            ),
            (
                "above_wide",
                (
                    x1 - int(round(width * 0.65)),
                    y1 - int(round(height * 1.35)),
                    x2 + int(round(width * 0.20)),
                    y1 + int(round(height * 0.20)),
                ),
            ),
        ]
        out: list[tuple[str, tuple[int, int, int, int]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for relation, bbox in candidates:
            clipped = self._clip_bbox_to_image(bbox, image_shape)
            if clipped is None or clipped in seen:
                continue
            seen.add(clipped)
            out.append((relation, clipped))
        return out

    def _is_strong_production_anchor_sibling_evidence(
        self,
        evidence: _DateEvidence,
        *,
        anchor_date: date | None,
    ) -> bool:
        if evidence.parsed.parsed_date is None:
            return False
        if anchor_date is not None and evidence.parsed.parsed_date <= anchor_date:
            return False
        text = self._date_evidence_text(evidence)
        if has_production_keyword(text) and not (has_expiry_keyword(text) or has_weak_expiry_keyword(text)):
            return False
        return bool(
            self._date_precision_score(evidence.parsed) >= 2
            and float(evidence.parsed.confidence or 0.0) >= 0.70
            and float(evidence.recognition.confidence or 0.0) >= 0.80
        )

    def _collect_production_anchor_sibling_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        date_evidence: list[_DateEvidence],
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if not self.production_anchor_sibling_enabled or not self.context_probe_enabled:
            return []
        if self._has_strong_accepted_date_evidence(evaluated, date_evidence):
            return []
        if any(has_expiry_keyword(self._date_evidence_text(evidence)) for evidence in date_evidence):
            return []
        evidence: list[_DateEvidence] = []
        anchors = self._production_anchor_contexts(image, candidate, date_evidence, today=today)
        max_anchors = 1
        max_directions = 4
        for anchor_bbox, anchor_context, anchor_date in anchors[:max_anchors]:
            for direction_count, (relation, sibling_bbox) in enumerate(
                self._production_anchor_sibling_bboxes(anchor_bbox, image_shape=image.shape),
                start=1,
            ):
                if direction_count > max_directions:
                    break
                sibling_evidence = self._date_evidence_from_bbox(
                    image,
                    candidate,
                    sibling_bbox,
                    today=today,
                    crop_policy="production_anchor_sibling",
                    context_texts=(),
                    metadata={
                        "source": "production_anchor_sibling",
                        "anchor_role": "production",
                        "anchor_bbox": list(anchor_bbox),
                        "sibling_relation": relation,
                        "forced_reason": "production_anchor_sibling",
                        "anchor_context": list(anchor_context),
                        "direction_count": direction_count,
                        "direction_budget": max_directions,
                    },
                    allowed_names=PRODUCTION_ANCHOR_SIBLING_RECOGNITION_VARIANTS,
                )
                evidence.extend(sibling_evidence)
                if any(
                    self._is_strong_production_anchor_sibling_evidence(item, anchor_date=anchor_date)
                    for item in sibling_evidence
                ):
                    return evidence
        return evidence

    def _is_anchor_local_sibling_anchor(self, evidence: _DateEvidence, *, today: date) -> bool:
        if evidence.parsed.parsed_date is None or self._date_precision_score(evidence.parsed) < 3:
            return False
        if float(evidence.recognition.confidence or 0.0) < 0.70:
            return False
        text = self._date_evidence_text(evidence)
        if has_expiry_keyword(text) or has_weak_expiry_keyword(text):
            return False
        if has_production_keyword(text) or self._has_production_anchor_signal(evidence.context_texts):
            return True
        return evidence.parsed.parsed_date <= today

    def _anchor_local_search_bbox(
        self,
        anchor_bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = anchor_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(80, int(round(width * 1.35)))
        pad_up = max(40, int(round(height * 1.40)))
        pad_down = max(80, int(round(height * 2.60)))
        return self._clip_bbox_to_image((x1 - pad_x, y1 - pad_up, x2 + pad_x, y2 + pad_down), image_shape)

    def _anchor_local_synthetic_bboxes(
        self,
        anchor_bbox: tuple[int, int, int, int],
        *,
        image_shape: tuple[int, ...],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        x1, y1, x2, y2 = anchor_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        rows = [
            (
                "local_below_first_line",
                (
                    x1 - int(round(width * 0.40)),
                    y2 - int(round(height * 0.10)),
                    x2 + int(round(width * 0.55)),
                    y2 + int(round(height * 0.95)),
                ),
            ),
            (
                "local_below_second_line",
                (
                    x1 - int(round(width * 0.40)),
                    y2 + int(round(height * 0.65)),
                    x2 + int(round(width * 0.55)),
                    y2 + int(round(height * 1.75)),
                ),
            ),
            (
                "local_below_third_line",
                (
                    x1 - int(round(width * 0.40)),
                    y2 + int(round(height * 1.35)),
                    x2 + int(round(width * 0.55)),
                    y2 + int(round(height * 2.45)),
                ),
            ),
            (
                "local_above_line",
                (
                    x1 - int(round(width * 0.40)),
                    y1 - int(round(height * 1.05)),
                    x2 + int(round(width * 0.55)),
                    y1 + int(round(height * 0.10)),
                ),
            ),
            (
                "local_right_line",
                (
                    x2 - int(round(width * 0.10)),
                    y1 - int(round(height * 0.30)),
                    x2 + int(round(width * 1.35)),
                    y2 + int(round(height * 0.30)),
                ),
            ),
            (
                "local_left_line",
                (
                    x1 - int(round(width * 1.35)),
                    y1 - int(round(height * 0.30)),
                    x1 + int(round(width * 0.10)),
                    y2 + int(round(height * 0.30)),
                ),
            ),
        ]
        out: list[tuple[str, tuple[int, int, int, int]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for relation, bbox in rows:
            clipped = self._clip_bbox_to_image(bbox, image_shape)
            if clipped is None or clipped in seen:
                continue
            seen.add(clipped)
            out.append((relation, clipped))
        return out

    def _anchor_local_bbox_relation(
        self,
        anchor_bbox: tuple[int, int, int, int],
        bbox: tuple[int, int, int, int],
    ) -> str:
        ax1, ay1, ax2, ay2 = anchor_bbox
        bx1, by1, bx2, by2 = bbox
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        if self._bbox_overlap(anchor_bbox, bbox)[1] >= 0.45:
            return "anchor_overlap"
        dx = bcx - acx
        dy = bcy - acy
        if abs(dy) >= abs(dx) * 0.75:
            return "detected_below" if dy > 0 else "detected_above"
        return "detected_right" if dx > 0 else "detected_left"

    def _anchor_local_detected_bboxes(
        self,
        image: np.ndarray,
        anchor_bbox: tuple[int, int, int, int],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        search_bbox = self._anchor_local_search_bbox(anchor_bbox, image_shape=image.shape)
        if search_bbox is None:
            return []
        detector = self._ensure_rapidocr_detector()
        if detector is None:
            return []
        x1, y1, x2, y2 = search_bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        started = perf_counter()
        boxes, reason = detector.detect(crop)
        runtime_ms = (perf_counter() - started) * 1000.0
        mapped: list[tuple[str, tuple[int, int, int, int]]] = []
        for box in boxes[:6]:
            bx1, by1, bx2, by2 = box.bbox_xyxy
            full_bbox = self._clip_bbox_to_image(
                (int(bx1 + x1), int(by1 + y1), int(bx2 + x1), int(by2 + y1)),
                image.shape,
            )
            if full_bbox is None:
                continue
            if self._bbox_overlap(anchor_bbox, full_bbox)[1] >= 0.45:
                continue
            relation = self._anchor_local_bbox_relation(anchor_bbox, full_bbox)
            mapped.append((relation, full_bbox))
        self._debug_append(
            "anchor_local_sibling",
            {
                "stage": "local_text_detection",
                "anchor_bbox": self._bbox_json(anchor_bbox),
                "search_bbox": self._bbox_json(search_bbox),
                "runtime_ms": round(runtime_ms, 3),
                "reason": reason,
                "mapped_bboxes": [
                    {"relation": relation, "bbox_xyxy": self._bbox_json(bbox)}
                    for relation, bbox in mapped
                ],
            },
        )
        return mapped

    def _is_anchor_local_candidate_bbox(
        self,
        anchor_bbox: tuple[int, int, int, int],
        bbox: tuple[int, int, int, int],
    ) -> bool:
        relation = self._anchor_local_bbox_relation(anchor_bbox, bbox)
        if relation == "anchor_overlap":
            return False
        aw = max(1, anchor_bbox[2] - anchor_bbox[0])
        ah = max(1, anchor_bbox[3] - anchor_bbox[1])
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        if bw < max(18, aw * 0.20) or bh < max(8, ah * 0.18):
            return False
        ax = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
        ay = (anchor_bbox[1] + anchor_bbox[3]) / 2.0
        bx = (bbox[0] + bbox[2]) / 2.0
        by = (bbox[1] + bbox[3]) / 2.0
        return bool(
            abs(bx - ax) <= aw * 2.10
            and abs(by - ay) <= ah * 3.10
            and self._bbox_area(bbox) <= self._bbox_area(anchor_bbox) * 4.50
        )

    def _anchor_local_bboxes(
        self,
        image: np.ndarray,
        anchor_bbox: tuple[int, int, int, int],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        rows = [
            *self._anchor_local_detected_bboxes(image, anchor_bbox),
            *self._anchor_local_synthetic_bboxes(anchor_bbox, image_shape=image.shape),
        ]
        out: list[tuple[str, tuple[int, int, int, int]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for relation, bbox in rows:
            if bbox in seen or not self._is_anchor_local_candidate_bbox(anchor_bbox, bbox):
                continue
            seen.add(bbox)
            out.append((relation, bbox))
            if len(out) >= 8:
                break
        return out

    @staticmethod
    def _metadata_anchor_date(evidence: _DateEvidence) -> date | None:
        value = evidence.metadata.get("anchor_date")
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _is_strong_anchor_local_sibling_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "anchor_local_sibling" or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 2:
            return False
        if float(evidence.parsed.confidence or 0.0) < 0.70:
            return False
        compact_token_matches = self._anchor_local_compact_token_matches_date(evidence)
        min_ocr_confidence = 0.55 if compact_token_matches else 0.80
        if float(evidence.recognition.confidence or 0.0) < min_ocr_confidence:
            return False
        text = self._date_evidence_text(evidence)
        if has_production_keyword(text) and not (has_expiry_keyword(text) or has_weak_expiry_keyword(text)):
            return False
        anchor_date = self._metadata_anchor_date(evidence)
        if anchor_date is not None and evidence.parsed.parsed_date <= anchor_date:
            return False
        raw = " ".join(
            dict.fromkeys(
                part
                for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text)
                if part
            )
        )
        digit_groups = re.findall(r"\d+", raw)
        digit_count = len(re.findall(r"\d", raw))
        return bool(
            has_expiry_keyword(text)
            or has_weak_expiry_keyword(text)
            or self._has_clean_full_date_pattern(raw)
            or self._clean_single_full_date_text(raw)
            or compact_token_matches
            or (len(digit_groups) >= 3 and 6 <= digit_count <= 8)
        )

    @staticmethod
    def _anchor_local_compact_token_matches_date(evidence: _DateEvidence) -> bool:
        parsed = evidence.parsed.parsed_date
        if parsed is None:
            return False
        raw = " ".join(
            dict.fromkeys(
                part
                for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text)
                if part
            )
        )
        expected_short = f"{parsed.day:02d}{parsed.month:02d}{parsed.year % 100:02d}"
        expected_full = f"{parsed.day:02d}{parsed.month:02d}{parsed.year:04d}"
        return any(group in {expected_short, expected_full} for group in re.findall(r"\d+", raw))

    def _collect_anchor_local_sibling_evidence(
        self,
        image: np.ndarray,
        evaluated: list[_EvaluatedCandidate],
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if not self.production_anchor_sibling_enabled:
            return []
        accepted_parseable, accepted_evidence = self._accepted_parseable_items(evaluated)
        if not accepted_parseable or any(
            has_expiry_keyword(self._date_evidence_text(evidence)) or has_weak_expiry_keyword(self._date_evidence_text(evidence))
            for evidence in accepted_evidence
        ):
            return []
        anchors = [
            evidence
            for evidence in sorted(accepted_evidence, key=self._score_date_evidence, reverse=True)
            if self._is_anchor_local_sibling_anchor(evidence, today=today)
        ]
        if not anchors:
            return []
        out: list[_DateEvidence] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for anchor in anchors[:2]:
            anchor_bbox = self._date_evidence_bbox(anchor)
            anchor_date = anchor.parsed.parsed_date
            if anchor_bbox is None or anchor_date is None:
                continue
            anchor_text = self._date_evidence_text(anchor)
            for relation, bbox in self._anchor_local_bboxes(image, anchor_bbox):
                seen_key = (anchor.candidate.candidate_id, bbox)
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                sibling_evidence = self._date_evidence_from_bbox(
                    image,
                    anchor.candidate,
                    bbox,
                    today=today,
                    crop_policy="anchor_local_sibling",
                    context_texts=(anchor_text,),
                    metadata={
                        "source": "anchor_local_sibling",
                        "anchor_bbox": list(anchor_bbox),
                        "anchor_date": anchor_date.isoformat(),
                        "anchor_raw_text": anchor.recognition.raw_text,
                        "sibling_relation": relation,
                        "forced_reason": "anchor_local_sibling",
                    },
                    allowed_names=ANCHOR_LOCAL_SIBLING_RECOGNITION_VARIANTS,
                )
                out.extend(sibling_evidence)
                if any(self._is_strong_anchor_local_sibling_evidence(item) for item in sibling_evidence):
                    self._debug_append(
                        "anchor_local_sibling",
                        {
                            "stage": "accepted",
                            "evidence": [self._debug_date_evidence(item) for item in sibling_evidence],
                        },
                    )
                    return out
        if out:
            self._debug_append(
                "anchor_local_sibling",
                {
                    "stage": "collected",
                    "evidence_count": len(out),
                    "evidence": [self._debug_date_evidence(item) for item in out],
                },
            )
        return out

    def _add_anchor_local_sibling_evidence(
        self,
        image: np.ndarray,
        evaluated: list[_EvaluatedCandidate],
        *,
        today: date,
    ) -> None:
        evidence = self._collect_anchor_local_sibling_evidence(image, evaluated, today=today)
        self._append_date_evidence_to_evaluated(evaluated, evidence)

    def _should_run_paired_crop_evidence(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        date_evidence: list[_DateEvidence],
    ) -> bool:
        if not self.paired_crop_evidence_enabled:
            return False
        if self._has_strong_accepted_date_evidence(evaluated, date_evidence):
            return False
        if candidate.candidate_type not in {"group", "fallback", "line"}:
            return False
        if not candidate.member_bboxes:
            return False
        text = " ".join(
            part
            for part in (
                evaluated.recognition.raw_text,
                evaluated.recognition.normalized_text,
                candidate.probe_text,
                candidate.probe_normalized_text,
                " ".join(candidate.context_probe_texts or []),
                " ".join(self._date_evidence_text(evidence) for evidence in date_evidence),
            )
            if part
        )
        parsed_dates = {
            evidence.parsed.parsed_date
            for evidence in date_evidence
            if evidence.parsed.parsed_date is not None
        }
        has_production_context = has_production_keyword(text) or candidate.adjacent_production_keyword
        has_multiple_date_evidence = len(parsed_dates) > 1
        has_multiple_lines = len(self._candidate_line_bboxes(candidate)) > 1
        return bool(has_multiple_lines and (has_production_context or has_multiple_date_evidence))

    def _paired_crop_bboxes(
        self,
        candidate: _MobileRankedCandidate,
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        rows: list[tuple[str, tuple[int, int, int, int]]] = []
        if candidate.recognition_bbox is not None:
            rows.append(("paired_crop_date_only_tight", candidate.recognition_bbox))
        for line_bbox in self._candidate_line_bboxes(candidate):
            x1, y1, x2, y2 = line_bbox
            width = max(1, x2 - x1)
            rows.append(
                (
                    "paired_crop_line_with_left_context",
                    (
                        x1 - int(round(width * 0.45)),
                        y1,
                        x2,
                        y2,
                    ),
                )
            )
        group_bbox = candidate.evidence_bbox or candidate.bbox_xyxy
        rows.append(("paired_crop_local_group", group_bbox))
        deduped: list[tuple[str, tuple[int, int, int, int]]] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for policy, bbox in rows:
            key = (policy, bbox)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((policy, bbox))
        return deduped

    def _collect_paired_crop_evidence(
        self,
        image: np.ndarray,
        candidate: _MobileRankedCandidate,
        *,
        today: date,
    ) -> list[_DateEvidence]:
        evidence: list[_DateEvidence] = []
        for policy, bbox in self._paired_crop_bboxes(candidate):
            clipped = self._clip_bbox_to_image(bbox, image.shape)
            if clipped is None:
                continue
            context_texts = self._line_role_context_for_bbox(image, clipped)
            evidence.extend(
                self._date_evidence_from_bbox(
                    image,
                    candidate,
                    clipped,
                    today=today,
                    crop_policy=policy,
                    context_texts=context_texts,
                    metadata={
                        "source": "paired_crop",
                        "forced_reason": "conditional_paired_crop",
                    },
                )
            )
        return evidence

    def _should_run_legacy_wide_group_rescue(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        date_evidence: list[_DateEvidence],
    ) -> bool:
        if not self.legacy_wide_group_rescue_enabled:
            return False
        if self._has_strong_accepted_date_evidence(evaluated, date_evidence):
            return False
        if candidate.candidate_type not in {"group", "fallback", "line"}:
            return False
        text = " ".join(
            part
            for part in (
                evaluated.recognition.normalized_text,
                evaluated.recognition.raw_text,
                candidate.probe_normalized_text,
                candidate.probe_text,
                " ".join(candidate.context_probe_texts or []),
            )
            if part
        )
        if evaluated.parsed.parsed_date is None:
            return True
        if len({evidence.parsed.parsed_date for evidence in date_evidence if evidence.parsed.parsed_date is not None}) > 1:
            return True
        if has_expiry_keyword(text) or candidate.adjacent_expiry_keyword:
            return False
        parser_conf = float(evaluated.parsed.confidence or 0.0)
        ocr_conf = float(evaluated.recognition.confidence or 0.0)
        return parser_conf < 0.90 or ocr_conf < 0.90

    def _evaluate_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate, *, today: date) -> _EvaluatedCandidate:
        crops = self._normalized_crops_for_candidate(image, candidate)
        best: tuple[tuple[int, int, float, float, float], _EvaluatedCandidate] | None = None
        date_evidence: list[_DateEvidence] = []
        partial_day_month_evaluated: list[_EvaluatedCandidate] = []
        for crop in crops:
            if crop.image.size == 0:
                continue
            selected_variant_image = candidate.recognition_variant_image if crop is crops[0] else None
            selected_variant_output = candidate.selected_recognition_output if crop is crops[0] else None
            evaluated = self._best_recognition_for_single_crop(
                crop,
                today=today,
                selected_variant_name=candidate.selected_recognition_variant or None,
                selected_variant_image=selected_variant_image,
                selected_variant_output=selected_variant_output,
                crop_policy=candidate.final_crop_policy,
            )
            evaluated.candidate = candidate
            for evidence in evaluated.date_evidence:
                evidence.candidate = candidate
                if not evidence.context_texts:
                    evidence.context_texts = tuple(candidate.context_probe_texts or ()) or self._line_role_context_for_evidence(
                        image,
                        candidate,
                        evidence,
                    )
                date_evidence.append(evidence)
            if self._is_high_confidence_partial_day_month_eval(evaluated):
                partial_day_month_evaluated.append(evaluated)
            score = (
                self._date_precision_score(evaluated.parsed),
                evaluated.parsed.parsed_date.toordinal() if evaluated.parsed.parsed_date is not None else 0,
                float(evaluated.parsed.confidence if evaluated.parsed.parsed_date is not None else 0.0),
                self._score_date_likeness(evaluated.recognition.raw_text),
                float(evaluated.recognition.confidence or 0.0),
            )
            if best is None or score > best[0]:
                best = (score, evaluated)
        if best is None:
            rec = _RecognitionOutput("", "", None, "candidate crop empty", "original")
            parsed = self.parser.parse("", reference_date=today)
            best_eval = _EvaluatedCandidate(candidate, rec, parsed, 1, "original_padded", None)
        else:
            best_eval = best[1]
            best_eval.date_evidence = date_evidence
        if candidate.final_crop_bbox is None and best_eval.normalized_crop is not None:
            candidate.final_crop_bbox = best_eval.normalized_crop.bbox_xyxy

        rapidocr_probe_evidence = self._collect_rapidocr_probe_prefilter_evidence(candidate, best_eval, today=today)
        if rapidocr_probe_evidence:
            date_evidence.extend(rapidocr_probe_evidence)
            best_eval.date_evidence = date_evidence

        with self._profile_timer("partial_expansion_ms"):
            partial_day_month_evidence = self._collect_partial_day_month_right_expansion_evidence(
                image,
                candidate,
                partial_day_month_evaluated,
                date_evidence,
                today=today,
            )
        if partial_day_month_evidence:
            date_evidence.extend(partial_day_month_evidence)
            best_eval.date_evidence = date_evidence

        if self._should_run_legacy_wide_group_rescue(candidate, best_eval, date_evidence):
            for crop in self._legacy_wide_group_crops_for_candidate(image, candidate):
                if crop.image.size == 0:
                    continue
                wide_eval = self._best_recognition_for_single_crop(
                    crop,
                    today=today,
                    crop_policy="legacy_wide_group",
                )
                for evidence in wide_eval.date_evidence:
                    evidence.candidate = candidate
                    evidence.crop_policy = "legacy_wide_group"
                    if not evidence.context_texts:
                        evidence.context_texts = tuple(candidate.context_probe_texts or ()) or self._line_role_context_for_evidence(
                            image,
                            candidate,
                            evidence,
                        )
                    date_evidence.append(evidence)
            best_eval.date_evidence = date_evidence

        if self._should_run_line_role_rescue(candidate, best_eval, date_evidence):
            date_evidence.extend(self._collect_line_role_rescue_evidence(image, candidate, today=today))
            best_eval.date_evidence = date_evidence

        if self._should_run_local_group_wide_crop_evidence(candidate, best_eval, date_evidence, image_shape=image.shape):
            date_evidence.extend(self._collect_local_group_wide_crop_evidence(image, candidate, today=today))
            best_eval.date_evidence = date_evidence

        rotation_wide_evidence = self._collect_rotation_wide_crop_evidence(image, candidate, best_eval, today=today)
        if rotation_wide_evidence:
            date_evidence.extend(rotation_wide_evidence)
            best_eval.date_evidence = date_evidence

        if self._should_run_paired_crop_evidence(candidate, best_eval, date_evidence):
            date_evidence.extend(self._collect_paired_crop_evidence(image, candidate, today=today))
            best_eval.date_evidence = date_evidence

        with self._profile_timer("production_sibling_ms"):
            sibling_evidence = self._collect_production_anchor_sibling_evidence(
                image,
                candidate,
                best_eval,
                date_evidence,
                today=today,
            )
        if sibling_evidence:
            date_evidence.extend(sibling_evidence)
            best_eval.date_evidence = date_evidence

        scores = self._probe_signal_scores(best_eval.recognition, candidate=candidate, today=today)
        candidate.probe_text = best_eval.recognition.raw_text
        candidate.probe_normalized_text = best_eval.recognition.normalized_text
        candidate.probe_confidence = best_eval.recognition.confidence
        candidate.probe_reason = best_eval.recognition.reason
        candidate.selected_recognition_variant = best_eval.recognition_variant
        candidate.selected_recognition_output = best_eval.recognition
        candidate.score_breakdown.update(scores)
        candidate.total_score = candidate.geometry_score + sum(scores.values())
        return best_eval

    def _collect_rapidocr_probe_prefilter_evidence(
        self,
        candidate: _MobileRankedCandidate,
        evaluated: _EvaluatedCandidate,
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if float(candidate.rapidocr_probe_score or 0.0) < 3.0:
            return []
        text = candidate.rapidocr_probe_normalized_text or candidate.rapidocr_probe_text or ""
        if not text:
            return []
        recognition = _RecognitionOutput(
            raw_text=candidate.rapidocr_probe_text or text,
            normalized_text=normalize_recognition_text(self._normalize_ocr_confusions(text)),
            confidence=candidate.rapidocr_probe_confidence,
            reason="rapidocr_gated_prefilter_probe",
            rotation=candidate.rapidocr_probe_orientation or "probe",
        )
        parse_inputs = self._build_parse_inputs(recognition)
        parsed = self._best_parse_for_inputs(parse_inputs, today=today)
        if parsed.parsed_date is None or self._date_precision_score(parsed) < 3:
            return []
        if self._rapidocr_prefilter_probe_conflicts_with_final_svtr(
            parsed,
            candidate,
            evaluated.date_evidence,
        ):
            return []
        return [
            _DateEvidence(
                candidate=candidate,
                recognition=recognition,
                parsed=parsed,
                parse_inputs_count=len(parse_inputs),
                recognition_variant=str(candidate.rapidocr_probe_orientation or "rapidocr_probe"),
                normalized_crop=evaluated.normalized_crop,
                crop_policy="rapidocr_gated_prefilter_probe",
                context_texts=tuple(candidate.context_probe_texts or ()),
                metadata={"source": "rapidocr_gated_prefilter"},
            )
        ]

    def _rapidocr_prefilter_probe_conflicts_with_final_svtr(
        self,
        parsed: ParsedDateData,
        candidate: _MobileRankedCandidate,
        evidence_items: list[_DateEvidence],
    ) -> bool:
        if parsed.parsed_date is None or self._date_precision_score(parsed) < 3:
            return False
        for evidence in evidence_items:
            if evidence.candidate is not candidate:
                continue
            if evidence.crop_policy == "rapidocr_gated_prefilter_probe":
                continue
            if evidence.parsed.parsed_date is None or self._date_precision_score(evidence.parsed) < 3:
                continue
            if evidence.parsed.parsed_date == parsed.parsed_date:
                return False
            if (
                evidence.parsed.parsed_date.year == parsed.parsed_date.year
                and evidence.parsed.parsed_date.month == parsed.parsed_date.month
                and evidence.parsed.parsed_date.day != parsed.parsed_date.day
            ):
                quality = min(
                    float(evidence.parsed.confidence or 0.0),
                    float(evidence.recognition.confidence or 0.0),
                )
                if quality >= 0.80:
                    return True
        return False

    def _text_evidence_priority(self, text: str, *, parser_bonus: float = 0.0, expiry_score: float = 0.0) -> int:
        has_date = self._has_date_like_text(text) or parser_bonus > 0
        has_expiry = has_expiry_keyword(text) or expiry_score > 0
        has_production = has_production_keyword(text)
        if has_expiry and has_date:
            return 3
        if has_date and not has_production:
            return 2
        if has_date and has_production:
            return 1
        return 0

    @staticmethod
    def _date_evidence_text(evidence: _DateEvidence) -> str:
        return " ".join(
            part
            for part in (
                evidence.recognition.normalized_text,
                evidence.recognition.raw_text,
                " ".join(evidence.context_texts),
            )
            if part
        )

    def _is_strong_accepted_date_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.parsed.parsed_date is None:
            return False
        if self._should_suppress_date_evidence(evidence):
            return False
        return self._is_strong_day_date_parse(
            evidence.parsed,
            evidence.recognition,
            text=self._date_evidence_text(evidence),
        )

    def _has_strong_accepted_date_evidence(
        self,
        evaluated: _EvaluatedCandidate,
        evidence_items: list[_DateEvidence],
    ) -> bool:
        if evaluated.parsed.parsed_date is None or self._should_suppress_auto_parse(evaluated):
            return False
        selected_evidence = [
            evidence
            for evidence in evidence_items
            if evidence.parsed.parsed_date == evaluated.parsed.parsed_date
        ]
        if selected_evidence:
            return any(self._is_strong_accepted_date_evidence(evidence) for evidence in selected_evidence)
        text = " ".join(
            part
            for part in (
                evaluated.recognition.normalized_text,
                evaluated.recognition.raw_text,
                " ".join(evaluated.candidate.context_probe_texts or []),
            )
            if part
        )
        return self._is_strong_day_date_parse(evaluated.parsed, evaluated.recognition, text=text)

    @classmethod
    def _date_evidence_has_strong_support(cls, evidence: _DateEvidence) -> bool:
        text = cls._date_evidence_text(evidence)
        raw = evidence.recognition.raw_text or evidence.recognition.normalized_text or ""
        return bool(
            has_expiry_keyword(text)
            or any(sep in raw for sep in ("/", "-", "."))
            or (float(evidence.recognition.confidence or 0.0) >= 0.95 and float(evidence.parsed.confidence or 0.0) >= 0.95)
        )

    def _unsupported_compact_future_penalty(self, evidence: _DateEvidence) -> int:
        if evidence.parsed.parsed_date is None or self._date_evidence_has_strong_support(evidence):
            return 0
        raw = evidence.recognition.raw_text or evidence.recognition.normalized_text or ""
        digits = re.sub(r"\D", "", raw)
        separators = sum(ch in "/.-:" for ch in raw)
        if len(digits) >= 6 and separators == 0 and evidence.parsed.parsed_date.year >= date.today().year + 8:
            return -1
        return 0

    def _is_weak_edge_compact_date_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.parsed.parsed_date is None:
            return False
        text = self._date_evidence_text(evidence)
        if has_expiry_keyword(text) or has_weak_expiry_keyword(text):
            return False
        raw = evidence.recognition.raw_text or evidence.recognition.normalized_text or ""
        digits = re.sub(r"\D", "", raw)
        separators = sum(ch in "/.-:" for ch in raw)
        if not (6 <= len(digits) <= 8 and separators == 0):
            return False
        detector_confidence = float(evidence.candidate.detector_confidence or 0.0)
        recognition_confidence = float(evidence.recognition.confidence or 0.0)
        edge_value = evidence.candidate.geometry_features.get("edge_proximity", 1.0)
        edge_proximity = 1.0 if edge_value is None else float(edge_value)
        return bool(detector_confidence < 0.25 and recognition_confidence < 0.75 and edge_proximity <= 0.012)

    @staticmethod
    def _date_evidence_quality_bucket(evidence: _DateEvidence) -> int:
        quality = min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
        return int(max(0.0, min(1.0, quality)) * 20.0)

    @staticmethod
    def _recognition_variant_stability_bucket(evidence: _DateEvidence) -> int:
        variant = (evidence.recognition_variant or "").removeprefix("hardcase:")
        selected_variant = (evidence.candidate.selected_recognition_variant or "").removeprefix("hardcase:")
        if variant and variant == selected_variant:
            return 4
        if evidence.crop_policy == "dot_matrix_rescue":
            return 0
        if variant == "original_color_tight":
            return 4
        if variant == "original_padded":
            return 3
        if variant == "stamp_blackhat":
            return 2
        if variant in {"gray_upscaled", "unsharp_gray"}:
            return 1
        return 0

    def _should_suppress_date_evidence(self, evidence: _DateEvidence) -> bool:
        return self._unsupported_compact_future_penalty(evidence) < 0 or self._is_weak_edge_compact_date_evidence(evidence)

    def _score_date_evidence(self, evidence: _DateEvidence) -> tuple[int, ...] | tuple[float, ...]:
        text = self._date_evidence_text(evidence)
        role = score_date_role(text)
        return (
            self._unsupported_compact_future_penalty(evidence),
            role.priority,
            role.non_production,
            self._date_precision_score(evidence.parsed),
            self._text_evidence_priority(text, parser_bonus=1.0),
            int(self._is_strong_rapidocr_partial_completion_evidence(evidence)),
            self._recognition_variant_stability_bucket(evidence),
            self._date_evidence_quality_bucket(evidence),
            evidence.parsed.parsed_date.toordinal() if evidence.parsed.parsed_date is not None else 0,
            float(evidence.parsed.confidence or 0.0),
            evidence.candidate.total_score,
            self._score_ocr_candidate(evidence.recognition),
            self._score_date_likeness(evidence.recognition.raw_text),
        )

    @staticmethod
    def _date_evidence_group_key(evidence: _DateEvidence) -> tuple[date | None, str | None]:
        return (evidence.parsed.parsed_date, evidence.parsed.date_precision)

    @staticmethod
    def _date_evidence_support_key(evidence: _DateEvidence) -> tuple[object, ...]:
        bbox = evidence.normalized_crop.bbox_xyxy if evidence.normalized_crop is not None else evidence.candidate.final_crop_bbox
        return (evidence.candidate.candidate_id, evidence.recognition_variant, bbox)

    def _date_evidence_group_support(self, evidence_items: list[_DateEvidence]) -> int:
        return len({self._date_evidence_support_key(evidence) for evidence in evidence_items})

    @staticmethod
    def _has_complete_day_month_year_text(text: str) -> bool:
        return bool(re.search(r"\b\d{2}[./:-]\d{2}[./:-]\d{4}\b", text or ""))

    @classmethod
    def _date_evidence_has_complete_day_month_year_text(cls, evidence: _DateEvidence) -> bool:
        text = " ".join(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part)
        return cls._has_complete_day_month_year_text(text)

    def _should_run_hardcase_recognizer_rescue(
        self,
        best: _DateEvidence,
        evidence_items: list[_DateEvidence],
        *,
        today: date,
    ) -> bool:
        if not self.hardcase_recognizer_rescue_enabled or self.hardcase_recognizer is None:
            return False
        if best.parsed.parsed_date is None:
            return False
        if self._date_precision_score(best.parsed) < 3:
            return True
        if best.parsed.parsed_date >= today:
            return False
        hardcase_candidate_types = {"group", "fallback", "line"}
        if best.candidate.candidate_type in hardcase_candidate_types:
            return True
        best_candidate_id = best.candidate.candidate_id
        if any(
            evidence.candidate.candidate_id == best_candidate_id
            and (
                evidence.candidate.candidate_type in hardcase_candidate_types
                or evidence.crop_policy in {"line_role_context", "legacy_wide_group", "local_group_wide_crop"}
            )
            for evidence in evidence_items
        ):
            return True
        return any(
            evidence.parsed.parsed_date == best.parsed.parsed_date
            and (
                evidence.candidate.candidate_type in hardcase_candidate_types
                or evidence.crop_policy in {"line_role_context", "legacy_wide_group", "local_group_wide_crop"}
            )
            for evidence in evidence_items
        )

    def _collect_hardcase_recognizer_rescue_evidence(
        self,
        best: _DateEvidence,
        evidence_items: list[_DateEvidence],
        *,
        today: date,
    ) -> list[_DateEvidence]:
        if self.hardcase_recognizer is None:
            return []
        best_candidate_id = best.candidate.candidate_id
        source_items = [
            evidence
            for evidence in [best, *evidence_items]
            if evidence.normalized_crop is not None
            and (
                evidence.candidate.candidate_id == best_candidate_id
                or (
                    evidence.parsed.parsed_date == best.parsed.parsed_date
                    and (
                        evidence.candidate.candidate_type in {"group", "fallback", "line"}
                        or evidence.crop_policy in {"line_role_context", "legacy_wide_group", "local_group_wide_crop"}
                    )
                )
            )
        ]
        rescue_evidence: list[_DateEvidence] = []
        seen_crops: set[tuple[object, ...]] = set()
        for source in source_items:
            crop = source.normalized_crop
            if crop is None or crop.image.size == 0:
                continue
            crop_key = (
                source.candidate.candidate_id,
                source.crop_policy,
                crop.bbox_xyxy,
                crop.selected_orientation,
            )
            if crop_key in seen_crops:
                continue
            seen_crops.add(crop_key)
            if len(seen_crops) > 4:
                break
            variants = self._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS)
            if self._batch_recognition_enabled(self.hardcase_recognizer):
                recognized_variants = self._recognize_variants_batch(
                    variants,
                    orientation=crop.selected_orientation,
                    recognizer=self.hardcase_recognizer,
                )
            else:
                recognized_variants = (
                    (
                        variant,
                        self._recognize_variant(
                            variant.image,
                            variant_name=variant.name,
                            orientation=crop.selected_orientation,
                            recognizer=self.hardcase_recognizer,
                        ),
                    )
                    for variant in variants
                )
            for variant, rec in recognized_variants:
                parse_inputs = self._build_parse_inputs(rec)
                parsed = self._best_parse_for_inputs(parse_inputs, today=today)
                self._debug_append(
                    "hardcase_recognizer_rescue",
                    {
                        "source_candidate_id": source.candidate.candidate_id,
                        "source_crop_policy": source.crop_policy,
                        "source_raw_text": source.recognition.raw_text,
                        "crop_bbox_xyxy": self._bbox_json(crop.bbox_xyxy),
                        "variant_name": variant.name,
                        "orientation": crop.selected_orientation,
                        "raw_text": rec.raw_text,
                        "normalized_text": rec.normalized_text,
                        "confidence": rec.confidence,
                        "reason": rec.reason,
                        "parse_inputs": parse_inputs,
                        "parse_result": self._debug_parsed_date(parsed),
                    },
                )
                if parsed.parsed_date is None:
                    continue
                rescue_evidence.append(
                    _DateEvidence(
                        candidate=source.candidate,
                        recognition=rec,
                        parsed=parsed,
                        parse_inputs_count=len(parse_inputs),
                        recognition_variant=f"hardcase:{variant.name}",
                        normalized_crop=crop,
                        crop_policy=source.crop_policy,
                        context_texts=source.context_texts,
                        metadata={
                            "recognizer_rescue": "hardcase",
                            "source_raw_text": source.recognition.raw_text,
                            "source_parsed_date": source.parsed.parsed_date.isoformat()
                            if source.parsed.parsed_date is not None
                            else None,
                        },
                    )
                )
        return rescue_evidence

    def _select_hardcase_recognizer_override(
        self,
        primary: _DateEvidence,
        secondary_evidence: list[_DateEvidence],
        *,
        today: date,
    ) -> _DateEvidence | None:
        if primary.parsed.parsed_date is None:
            return None
        primary_text = self._date_evidence_text(primary)
        primary_quality = min(float(primary.parsed.confidence or 0.0), float(primary.recognition.confidence or 0.0))
        if primary_quality >= 0.85 and (
            has_expiry_keyword(primary_text) or has_weak_expiry_keyword(primary_text)
        ):
            primary_date = primary.parsed.parsed_date.isoformat()
            reread_of_primary = any(
                evidence.metadata.get("recognizer_rescue") == "hardcase"
                and evidence.metadata.get("source_parsed_date") == primary_date
                for evidence in secondary_evidence
            )
            if not reread_of_primary:
                return None
        eligible = [
            evidence
            for evidence in secondary_evidence
            if evidence.parsed.parsed_date is not None
            and evidence.parsed.parsed_date > primary.parsed.parsed_date
            and self._date_precision_score(evidence.parsed) >= 3
            and not self._should_suppress_date_evidence(evidence)
            and not (
                evidence.metadata.get("recognizer_rescue") == "hardcase"
                and has_production_keyword(self._date_evidence_text(evidence))
            )
        ]
        groups: dict[tuple[date | None, str | None], list[_DateEvidence]] = {}
        for evidence in eligible:
            groups.setdefault(self._date_evidence_group_key(evidence), []).append(evidence)

        supported: list[tuple[int, int, float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
        for group in groups.values():
            support = self._date_evidence_group_support(group)
            variant_support = len({evidence.recognition_variant for evidence in group})
            if support < 3 or variant_support < 3:
                continue
            quality = max(
                min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                for evidence in group
            )
            if quality < 0.85:
                continue
            group_best = max(group, key=self._score_date_evidence)
            supported.append(
                (
                    support,
                    variant_support,
                    quality,
                    group_best.parsed.parsed_date.toordinal() if group_best.parsed.parsed_date is not None else 0,
                    self._score_date_evidence(group_best),
                    group_best,
                )
            )
        if not supported:
            return None
        return max(supported, key=lambda item: item[:-1])[-1]

    @staticmethod
    def _has_clean_full_date_pattern(text: str) -> bool:
        return bool(re.search(r"\b\d{1,4}[./:-]\d{1,2}[./:-]\d{1,4}\b", text or ""))

    @staticmethod
    def _clean_single_full_date_text(text: str) -> bool:
        if not text:
            return False
        matches = re.findall(r"\b\d{1,4}[./:-]\d{1,2}[./:-]\d{1,4}\b", text)
        if len(matches) != 1:
            return False
        digit_count = len(re.findall(r"\d", text))
        return digit_count <= 10

    @staticmethod
    def _date_token_matches_year(token: int, year: int) -> bool:
        return token == year or (0 <= token < 100 and token == year % 100)

    @classmethod
    def _date_tokens_match_parsed_date(cls, tokens: list[int], parsed_date: date) -> bool:
        if len(tokens) != 3:
            return False
        first, second, third = tokens
        year = parsed_date.year
        day = parsed_date.day
        month = parsed_date.month
        if cls._date_token_matches_year(first, year):
            if second == month and third == day:
                return True
        if cls._date_token_matches_year(third, year):
            return (first == day and second == month) or (first == month and second == day)
        return False

    @classmethod
    def _has_alpha_attached_leading_day_token(cls, text: str, parsed_date: date | None) -> bool:
        if not text or parsed_date is None:
            return False
        pattern = re.compile(r"[A-Z]{2,}\s*(\d{1,2})[./:-](\d{1,2})[./:-](\d{2,4})", re.IGNORECASE)
        for match in pattern.finditer(text):
            day_token, month_token, year_token = match.groups()
            year = int(year_token)
            if year < 100:
                year = 2000 + year if year < 70 else 1900 + year
            if int(day_token) == parsed_date.day and int(month_token) == parsed_date.month and year == parsed_date.year:
                return True
        return False

    @classmethod
    def _clean_supported_rapidocr_full_date_text(cls, text: str, parsed_date: date | None) -> bool:
        if not text or parsed_date is None:
            return False
        if len(re.findall(r"\d", text)) > 12:
            return False
        pattern = re.compile(
            r"(?=(\b(?:[A-Z]{1,4}[./:\-\s]*)?\d{1,4}[./:\-\s]+\d{1,2}[./:\-\s]+\d{2,4}\b))",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            tokens = [int(token) for token in re.findall(r"\d+", match.group(1))]
            if cls._date_tokens_match_parsed_date(tokens, parsed_date):
                return True
        return False

    @classmethod
    def _noisy_leading_day_supported_rapidocr_full_date_text(cls, text: str, parsed_date: date | None) -> bool:
        if not text or parsed_date is None:
            return False
        compact = re.sub(r"\s+", "", text)
        if len(re.findall(r"\d", compact)) > 9:
            return False
        pattern = re.compile(r"(?<!\d)(\d{3})[./:-](\d{1,2})[./:-](\d{2,4})(?![./:-]?\d)")
        for match in pattern.finditer(compact):
            day_token, month_token, year_token = match.groups()
            day = int(day_token[-2:])
            month = int(month_token)
            year = int(year_token)
            if year < 100:
                year = 2000 + year if year < 70 else 1900 + year
            if day == parsed_date.day and month == parsed_date.month and year == parsed_date.year:
                return True
        return False

    @classmethod
    def _contains_supported_full_date_token(cls, text: str, parsed_date: date | None) -> bool:
        if not text or parsed_date is None:
            return False
        pattern = re.compile(
            r"(?<!\d)\d{1,4}[./:\-\s]+\d{1,2}[./:\-\s]+\d{2,4}(?!\d)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            tokens = [int(token) for token in re.findall(r"\d+", match.group(0))]
            if cls._date_tokens_match_parsed_date(tokens, parsed_date):
                return True
        return False

    def _is_strong_legacy_wide_group_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "legacy_wide_group" or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = " ".join(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part)
        return bool(
            self._has_clean_full_date_pattern(text)
            and float(evidence.parsed.confidence or 0.0) >= 0.75
            and float(evidence.recognition.confidence or 0.0) >= 0.75
            and not (has_production_keyword(self._date_evidence_text(evidence)) and not has_expiry_keyword(self._date_evidence_text(evidence)))
        )

    def _is_strong_local_group_wide_crop_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "local_group_wide_crop" or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        clean_supported_date = self._clean_supported_rapidocr_full_date_text(text, evidence.parsed.parsed_date)
        return bool(
            (self._clean_single_full_date_text(text) or clean_supported_date)
            and float(evidence.parsed.confidence or 0.0) >= 0.70
            and float(evidence.recognition.confidence or 0.0) >= 0.70
            and not (has_production_keyword(self._date_evidence_text(evidence)) and not has_expiry_keyword(self._date_evidence_text(evidence)))
        )

    def _is_strong_rotation_wide_crop_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "rotation_wide_crop" or evidence.parsed.parsed_date is None:
            return False
        if float(evidence.parsed.confidence or 0.0) < 0.50:
            return False
        if float(evidence.recognition.confidence or 0.0) < 0.45:
            return False
        text = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        digit_count = len(re.findall(r"\d", text))
        return bool(
            (digit_count <= 10 or self._clean_single_full_date_text(text))
            and not (has_production_keyword(self._date_evidence_text(evidence)) and not has_expiry_keyword(self._date_evidence_text(evidence)))
        )

    def _is_strong_partial_day_month_right_expansion_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "partial_day_month_right_expand" or evidence.parsed.parsed_date is None:
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        return bool(
            self._clean_single_full_date_text(text)
            and float(evidence.parsed.confidence or 0.0) >= 0.75
            and float(evidence.recognition.confidence or 0.0) >= 0.85
            and not (has_production_keyword(self._date_evidence_text(evidence)) and not has_expiry_keyword(self._date_evidence_text(evidence)))
        )

    def _is_strong_rapidocr_partial_completion_evidence(self, evidence: _DateEvidence) -> bool:
        if evidence.crop_policy != "rapidocr_gated_prefilter_probe" or evidence.parsed.parsed_date is None:
            return False
        if not (evidence.candidate.detector_variant or "").startswith("rapidocr_gated_grouped:partial_date_completion"):
            return False
        if self._date_precision_score(evidence.parsed) < 3:
            return False
        text = " ".join(dict.fromkeys(part for part in (evidence.recognition.raw_text, evidence.recognition.normalized_text) if part))
        return bool(
            self._clean_single_full_date_text(text)
            and float(evidence.parsed.confidence or 0.0) >= 0.85
            and float(evidence.recognition.confidence or 0.0) >= 0.75
            and not (has_production_keyword(self._date_evidence_text(evidence)) and not has_expiry_keyword(self._date_evidence_text(evidence)))
        )

    def _select_date_evidence(self, evidence_items: list[_DateEvidence]) -> _DateEvidence:
        best = max(evidence_items, key=self._score_date_evidence)
        groups: dict[tuple[date | None, str | None], list[_DateEvidence]] = {}
        for evidence in evidence_items:
            groups.setdefault(self._date_evidence_group_key(evidence), []).append(evidence)

        best_group = groups.get(self._date_evidence_group_key(best), [best])
        best_support = self._date_evidence_group_support(best_group)
        best_precision = self._date_precision_score(best.parsed)
        best_variant_support = len({evidence.recognition_variant for evidence in best_group})
        best_group_complete = any(self._date_evidence_has_complete_day_month_year_text(evidence) for evidence in best_group)
        best_role = score_date_role(self._date_evidence_text(best))
        best_quality = max(
            min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
            for evidence in best_group
        )
        if (
            best.parsed.parsed_date is not None
            and best.parsed.date_format_detected in {"MM/DD/YY_UNAMBIGUOUS", "MM/DD/YYYY_UNAMBIGUOUS"}
            and not (has_expiry_keyword(self._date_evidence_text(best)) or has_weak_expiry_keyword(self._date_evidence_text(best)))
        ):
            later_month_alternates: list[tuple[int, int, float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) != 2:
                    continue
                if not any(evidence.candidate.candidate_id == best.candidate.candidate_id for evidence in group):
                    continue
                if not any(evidence.parsed.date_format_detected in {"MM/YY", "MM/YYYY"} for evidence in group):
                    continue
                group_role = max(
                    (score_date_role(self._date_evidence_text(evidence)) for evidence in group),
                    key=lambda role: (role.priority, role.non_production),
                )
                if group_role.production_keyword and group_role.priority < 3:
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                if support < 3 or variant_support < 2:
                    continue
                max_ocr_confidence = max(float(evidence.recognition.confidence or 0.0) for evidence in group)
                if max_ocr_confidence < 0.85:
                    continue
                later_month_alternates.append(
                    (
                        support,
                        variant_support,
                        max_ocr_confidence,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if later_month_alternates:
                return max(later_month_alternates, key=lambda item: item[:-1])[-1]
        if best_role.priority < 3:
            stronger_role_alternates: list[
                tuple[int, int, float, tuple[int, ...] | tuple[float, ...], _DateEvidence]
            ] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if self._date_precision_score(group_best.parsed) < best_precision:
                    continue
                group_role = max(
                    (score_date_role(self._date_evidence_text(evidence)) for evidence in group),
                    key=lambda role: (role.priority, role.non_production),
                )
                if group_role.priority <= best_role.priority or group_role.priority < 3:
                    continue
                group_best = max(
                    group,
                    key=lambda evidence: (
                        score_date_role(self._date_evidence_text(evidence)).priority,
                        self._score_date_evidence(evidence),
                    ),
                )
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.85:
                    continue
                stronger_role_alternates.append(
                    (
                        group_role.priority,
                        self._date_evidence_group_support(group),
                        quality,
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if stronger_role_alternates:
                return max(stronger_role_alternates, key=lambda item: item[:-1])[-1]
        if best.parsed.parsed_date is not None and not has_expiry_keyword(self._date_evidence_text(best)):
            same_day_month_later_alternates: list[tuple[float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) != best_precision:
                    continue
                if (
                    group_best.parsed.parsed_date.day != best.parsed.parsed_date.day
                    or group_best.parsed.parsed_date.month != best.parsed.parsed_date.month
                ):
                    continue
                best_role = score_date_role(self._date_evidence_text(best))
                group_role = score_date_role(self._date_evidence_text(group_best))
                if best_role.priority >= 3 and group_role.production_keyword:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.85:
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                later_has_noise_digit_year = any(
                    evidence.parsed.date_format_detected == "DD/MM/YY_NOISE_DIGIT"
                    or bool(
                        re.search(
                            r"\b\d{1,2}[./:-]\d{1,2}[./:-]\d[./:-]\d{2}\b",
                            evidence.recognition.raw_text or evidence.recognition.normalized_text or "",
                        )
                    )
                    for evidence in group
                )
                weak_context_same_day_month_correction = (
                    best_role.role == "weak_expiry_keyword"
                    and later_has_noise_digit_year
                    and quality >= 0.85
                )
                if (support < 2 or variant_support < 2) and not weak_context_same_day_month_correction:
                    continue
                if (
                    support < best_support
                    and variant_support < best_variant_support
                    and not weak_context_same_day_month_correction
                ):
                    continue
                same_day_month_later_alternates.append(
                    (
                        quality,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if same_day_month_later_alternates:
                return max(same_day_month_later_alternates, key=lambda item: item[:-1])[-1]
        if best.parsed.parsed_date is not None and (
            self._is_rapidocr_gated_evidence(best)
            or self._is_product_rescue_evidence(best)
            or self._is_dot_matrix_rescue_evidence(best)
        ):
            stronger_same_day_month_year_alternates: list[
                tuple[int, int, float, tuple[int, ...] | tuple[float, ...], _DateEvidence]
            ] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None:
                    continue
                if self._date_precision_score(group_best.parsed) != best_precision:
                    continue
                if (
                    group_best.parsed.parsed_date.day != best.parsed.parsed_date.day
                    or group_best.parsed.parsed_date.month != best.parsed.parsed_date.month
                ):
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                if support < 3 or variant_support < 3:
                    continue
                if support <= best_support and variant_support <= best_variant_support:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.80:
                    continue
                stronger_same_day_month_year_alternates.append(
                    (
                        support,
                        variant_support,
                        quality,
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if stronger_same_day_month_year_alternates:
                return max(stronger_same_day_month_year_alternates, key=lambda item: item[:-1])[-1]
        if best_role.priority >= 3 and best_quality >= 0.85:
            has_higher_role_alternate = any(
                max(
                    (score_date_role(self._date_evidence_text(evidence)) for evidence in group),
                    key=lambda role: (role.priority, role.non_production),
                ).priority
                > best_role.priority
                for key, group in groups.items()
                if key != self._date_evidence_group_key(best)
            )
            if not has_higher_role_alternate and not (
                self._is_rapidocr_gated_evidence(best)
                or self._is_product_rescue_evidence(best)
                or self._is_dot_matrix_rescue_evidence(best)
            ):
                return best
        if (
            best.parsed.parsed_date is not None
            and any(
                evidence.crop_policy
                in {"same_line_group", "legacy_wide_group", "local_group_wide_crop", "rotation_wide_crop"}
                for evidence in best_group
            )
        ):
            best_quality = max(
                min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                for evidence in best_group
            )
            later_direct_alternates: list[tuple[float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) != best_precision:
                    continue
                if not any(evidence.crop_policy in {"single_tight", "line_role_context"} for evidence in group):
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality + 0.05 < best_quality:
                    continue
                later_direct_alternates.append(
                    (
                        quality,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if later_direct_alternates:
                return max(later_direct_alternates, key=lambda item: item[:-1])[-1]
        if not best_group_complete and best.parsed.parsed_date is not None:
            complete_later_alternates: list[tuple[int, int, float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) < best_precision:
                    continue
                if not any(self._date_evidence_has_complete_day_month_year_text(evidence) for evidence in group):
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                if support < 2 or variant_support < 2:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.85:
                    continue
                complete_later_alternates.append(
                    (
                        support,
                        variant_support,
                        quality,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if complete_later_alternates:
                return max(complete_later_alternates, key=lambda item: item[:-1])[-1]
        if (
            best_precision < 3
            and best.parsed.parsed_date is not None
            and any(evidence.crop_policy == "line_from_multiline_group" for evidence in best_group)
        ):
            later_direct_line_alternates: list[tuple[int, int, float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) != best_precision:
                    continue
                if not any(evidence.crop_policy in {"line_role_context", "single_tight"} for evidence in group):
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                if support < 2 or variant_support < 2:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.70:
                    continue
                later_direct_line_alternates.append(
                    (
                        support,
                        variant_support,
                        quality,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if later_direct_line_alternates:
                return max(later_direct_line_alternates, key=lambda item: item[:-1])[-1]
        if best_precision < 3 and best.parsed.parsed_date is not None:
            stronger_later_same_precision: list[tuple[int, float, int, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if group_best.parsed.parsed_date is None or group_best.parsed.parsed_date <= best.parsed.parsed_date:
                    continue
                if self._date_precision_score(group_best.parsed) != best_precision:
                    continue
                support = self._date_evidence_group_support(group)
                if support <= best_support or support < 3:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.70:
                    continue
                stronger_later_same_precision.append(
                    (
                        support,
                        quality,
                        group_best.parsed.parsed_date.toordinal(),
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if stronger_later_same_precision:
                return max(stronger_later_same_precision, key=lambda item: item[:-1])[-1]
        strong_partial_day_month_expansions = [
            evidence
            for evidence in evidence_items
            if self._is_strong_partial_day_month_right_expansion_evidence(evidence)
            and self._date_precision_score(evidence.parsed) >= best_precision
            and evidence.parsed.parsed_date is not None
            and (
                best.parsed.parsed_date is None
                or evidence.parsed.parsed_date > best.parsed.parsed_date
            )
        ]
        if strong_partial_day_month_expansions:
            return max(
                strong_partial_day_month_expansions,
                key=lambda evidence: (
                    float(evidence.parsed.confidence or 0.0),
                    float(evidence.recognition.confidence or 0.0),
                    evidence.parsed.parsed_date.toordinal() if evidence.parsed.parsed_date is not None else 0,
                    self._score_date_evidence(evidence),
                ),
            )
        strong_rapidocr_partial_completions = [
            evidence
            for evidence in evidence_items
            if self._is_strong_rapidocr_partial_completion_evidence(evidence)
            and self._date_precision_score(evidence.parsed) >= best_precision
            and evidence.parsed.parsed_date is not None
            and (
                best.parsed.parsed_date is None
                or evidence.parsed.parsed_date > best.parsed.parsed_date
            )
        ]
        if strong_rapidocr_partial_completions:
            return max(
                strong_rapidocr_partial_completions,
                key=lambda evidence: (
                    float(evidence.parsed.confidence or 0.0),
                    float(evidence.recognition.confidence or 0.0),
                    evidence.parsed.parsed_date.toordinal() if evidence.parsed.parsed_date is not None else 0,
                    self._score_date_evidence(evidence),
                ),
            )
        strong_anchor_local_siblings = [
            evidence
            for evidence in evidence_items
            if self._is_strong_anchor_local_sibling_evidence(evidence)
            and self._date_precision_score(evidence.parsed) >= best_precision
            and evidence.parsed.parsed_date is not None
            and (
                best.parsed.parsed_date is None
                or evidence.parsed.parsed_date > best.parsed.parsed_date
                or has_expiry_keyword(self._date_evidence_text(evidence))
                or has_weak_expiry_keyword(self._date_evidence_text(evidence))
            )
        ]
        if strong_anchor_local_siblings:
            return max(
                strong_anchor_local_siblings,
                key=lambda evidence: (
                    int(has_expiry_keyword(self._date_evidence_text(evidence)) or has_weak_expiry_keyword(self._date_evidence_text(evidence))),
                    self._date_precision_score(evidence.parsed),
                    float(evidence.parsed.confidence or 0.0),
                    float(evidence.recognition.confidence or 0.0),
                    evidence.parsed.parsed_date.toordinal() if evidence.parsed.parsed_date is not None else 0,
                    self._score_date_evidence(evidence),
                ),
            )
        if best_support >= 3 and best_variant_support >= 3:
            return best
        if self._is_strong_partial_day_month_right_expansion_evidence(best):
            return best
        if self._is_strong_anchor_local_sibling_evidence(best):
            return best

        if (
            (
                self._is_strong_legacy_wide_group_evidence(best)
                or self._is_strong_local_group_wide_crop_evidence(best)
            )
            and best_support < 3
        ):
            supported_non_wide_alternates: list[
                tuple[int, int, float, tuple[int, ...] | tuple[float, ...], _DateEvidence]
            ] = []
            for key, group in groups.items():
                if key == self._date_evidence_group_key(best):
                    continue
                group_best = max(group, key=self._score_date_evidence)
                if self._date_precision_score(group_best.parsed) < best_precision:
                    continue
                if not any(
                    evidence.crop_policy not in {"legacy_wide_group", "local_group_wide_crop"}
                    for evidence in group
                ):
                    continue
                support = self._date_evidence_group_support(group)
                variant_support = len({evidence.recognition_variant for evidence in group})
                if support < 3 or variant_support < 3:
                    continue
                quality = max(
                    min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                    for evidence in group
                )
                if quality < 0.85:
                    continue
                supported_non_wide_alternates.append(
                    (
                        support,
                        variant_support,
                        quality,
                        self._score_date_evidence(group_best),
                        group_best,
                    )
                )
            if supported_non_wide_alternates:
                return max(supported_non_wide_alternates, key=lambda item: item[:-1])[-1]

        strong_wide_alternates = [
            evidence
            for evidence in evidence_items
            if evidence is not best
            and (
                self._is_strong_legacy_wide_group_evidence(evidence)
                or self._is_strong_local_group_wide_crop_evidence(evidence)
            )
            and self._date_precision_score(evidence.parsed) >= best_precision
            and evidence.parsed.parsed_date is not None
            and best.parsed.parsed_date is not None
            and evidence.parsed.parsed_date > best.parsed.parsed_date
        ]
        if strong_wide_alternates:
            return max(strong_wide_alternates, key=self._score_date_evidence)

        supported_alternates: list[tuple[int, int, float, tuple[int, ...] | tuple[float, ...], _DateEvidence]] = []
        for key, group in groups.items():
            if key == self._date_evidence_group_key(best):
                continue
            group_best = max(group, key=self._score_date_evidence)
            if self._date_precision_score(group_best.parsed) < best_precision:
                continue
            if (
                best_group_complete
                and best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date < best.parsed.parsed_date
            ):
                continue
            support = self._date_evidence_group_support(group)
            if support < 3:
                continue
            if (
                best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date < best.parsed.parsed_date
                and (
                    best_support >= 2
                    or any(evidence.crop_policy == "same_line_group" for evidence in group)
                )
                and any(evidence.crop_policy in {"line_role_context", "single_tight"} for evidence in best_group)
                and any(evidence.crop_policy in {"line_from_multiline_group", "same_line_group"} for evidence in group)
            ):
                continue
            if (
                best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date is not None
                and group_best.parsed.parsed_date < best.parsed.parsed_date
                and support <= best_support
            ):
                continue
            quality = max(
                min(float(evidence.parsed.confidence or 0.0), float(evidence.recognition.confidence or 0.0))
                for evidence in group
            )
            supported_alternates.append(
                (
                    support,
                    self._date_precision_score(group_best.parsed),
                    quality,
                    self._score_date_evidence(group_best),
                    group_best,
                )
            )

        if supported_alternates:
            return max(supported_alternates, key=lambda item: item[:-1])[-1]

        return best

    def _evaluated_from_date_evidence(
        self,
        evidence: _DateEvidence,
        *,
        date_evidence: list[_DateEvidence] | None = None,
    ) -> _EvaluatedCandidate:
        return _EvaluatedCandidate(
            candidate=evidence.candidate,
            recognition=evidence.recognition,
            parsed=evidence.parsed,
            parse_inputs_count=evidence.parse_inputs_count,
            recognition_variant=evidence.recognition_variant,
            normalized_crop=evidence.normalized_crop,
            original_color_fallback_used=evidence.original_color_fallback_used,
            date_evidence=date_evidence or [evidence],
            selected_date_evidence=evidence,
        )

    def _score_parseable_candidate(self, evaluated: _EvaluatedCandidate) -> tuple[int, ...] | tuple[float, ...]:
        if evaluated.date_evidence:
            return self._score_date_evidence(self._select_date_evidence(evaluated.date_evidence))
        text = " ".join(part for part in (evaluated.recognition.normalized_text, evaluated.recognition.raw_text) if part)
        role = score_date_role(text)
        return (
            0,
            role.priority,
            role.non_production,
            self._date_precision_score(evaluated.parsed),
            self._text_evidence_priority(text, parser_bonus=1.0),
            int(min(float(evaluated.parsed.confidence or 0.0), float(evaluated.recognition.confidence or 0.0)) * 20.0),
            evaluated.parsed.parsed_date.toordinal() if evaluated.parsed.parsed_date is not None else 0,
            evaluated.parsed.confidence,
            evaluated.candidate.total_score,
            self._score_ocr_candidate(evaluated.recognition),
            self._score_date_likeness(evaluated.recognition.raw_text),
        )

    def _should_suppress_auto_parse(self, evaluated: _EvaluatedCandidate) -> bool:
        if evaluated.parsed.parsed_date is None:
            return False
        detector_confidence = evaluated.candidate.detector_confidence
        recognition_confidence = evaluated.recognition.confidence
        if detector_confidence is None or recognition_confidence is None:
            return False
        text = " ".join(part for part in (evaluated.recognition.normalized_text, evaluated.recognition.raw_text) if part)
        if has_expiry_keyword(text):
            return False
        return bool(detector_confidence < 0.10 and recognition_confidence < 0.85)

    def _has_accepted_date_evidence(self, evaluated: _EvaluatedCandidate) -> bool:
        if not evaluated.date_evidence:
            return evaluated.parsed.parsed_date is not None and not self._should_suppress_auto_parse(evaluated)
        return any(
            evidence.parsed.parsed_date is not None and not self._should_suppress_date_evidence(evidence)
            for evidence in evaluated.date_evidence
        )

    def _accepted_date_evidence_items(self, evidence_items: list[_DateEvidence]) -> list[_DateEvidence]:
        core_dates = {
            evidence.parsed.parsed_date
            for evidence in evidence_items
            if evidence.parsed.parsed_date is not None
            and self._is_core_yolo_evidence(evidence)
            and not self._should_suppress_date_evidence(evidence)
        }
        accepted: list[_DateEvidence] = []
        for evidence in evidence_items:
            rejection_reason: str | None = None
            if evidence.parsed.parsed_date is None:
                rejection_reason = "no_parsed_date"
            elif self._should_suppress_date_evidence(evidence):
                rejection_reason = "suppressed_date_evidence"
            if rejection_reason is not None:
                self._debug_append(
                    "date_evidence",
                    {
                        "accepted": False,
                        "rejection_reason": rejection_reason,
                        "evidence": self._debug_date_evidence(evidence),
                    },
                )
                continue
            risky = (
                self._is_product_rescue_evidence(evidence)
                or self._is_dot_matrix_rescue_evidence(evidence)
                or evidence.crop_policy == "legacy_wide_group"
                or evidence.crop_policy == "local_group_wide_crop"
                or evidence.crop_policy == "rotation_wide_crop"
                or evidence.crop_policy == "anchor_local_sibling"
                or self._is_rapidocr_gated_evidence(evidence)
                or (self.strict_evidence_acceptance_enabled and self._is_rapidocr_evidence(evidence))
            )
            if risky and not self._is_strong_risky_date_evidence(evidence, evidence_items, core_dates=core_dates):
                self._debug_append(
                    "date_evidence",
                    {
                        "accepted": False,
                        "rejection_reason": "risky_evidence_without_strong_support",
                        "evidence": self._debug_date_evidence(evidence),
                    },
                )
                continue
            if (
                self.rapidocr_role_constraint_enabled
                and self._is_rapidocr_evidence(evidence)
                and core_dates
                and evidence.parsed.parsed_date not in core_dates
            ):
                self._debug_append(
                    "date_evidence",
                    {
                        "accepted": False,
                        "rejection_reason": "rapidocr_role_constraint_core_date_mismatch",
                        "evidence": self._debug_date_evidence(evidence),
                    },
                )
                continue
            accepted.append(evidence)
            self._debug_append(
                "date_evidence",
                {
                    "accepted": True,
                    "rejection_reason": None,
                    "evidence": self._debug_date_evidence(evidence),
                },
            )
        return accepted

    def _accepted_parseable_items(
        self,
        evaluated: list[_EvaluatedCandidate],
    ) -> tuple[list[_EvaluatedCandidate], list[_DateEvidence]]:
        all_date_evidence = [
            evidence
            for item in evaluated
            for evidence in item.date_evidence
            if evidence.parsed.parsed_date is not None
        ]
        accepted_evidence = self._accepted_date_evidence_items(all_date_evidence)
        accepted_ids = {id(evidence) for evidence in accepted_evidence}
        parseable: list[_EvaluatedCandidate] = []
        for item in evaluated:
            if item.date_evidence:
                if any(id(evidence) in accepted_ids for evidence in item.date_evidence):
                    parseable.append(item)
            elif item.parsed.parsed_date is not None and not self._should_suppress_auto_parse(item):
                parseable.append(item)
        if parseable and any(item.date_evidence for item in parseable) and not accepted_evidence:
            return [], []
        return parseable, accepted_evidence

    @staticmethod
    def _date_precision_score(parsed: ParsedDateData) -> int:
        if parsed.date_precision == "day" or parsed.parsed_day is not None:
            return 3
        if parsed.date_precision == "month" or parsed.parsed_month is not None:
            return 2
        if parsed.date_precision == "year" or parsed.parsed_year is not None:
            return 1
        return 0

    def _score_unparseable_candidate(self, evaluated: _EvaluatedCandidate) -> tuple[int, float, float, float, float]:
        text = " ".join(part for part in (evaluated.recognition.normalized_text, evaluated.recognition.raw_text) if part)
        return (
            self._text_evidence_priority(text),
            evaluated.candidate.total_score,
            self._score_ocr_candidate(evaluated.recognition),
            self._score_date_likeness(evaluated.recognition.raw_text),
            1.0 if has_expiry_keyword(text) else 0.0,
        )

    @staticmethod
    def _polygon_json(candidate: _MobileRankedCandidate) -> list[list[float]] | None:
        if candidate.polygon_xy is None:
            return None
        return [[float(x), float(y)] for x, y in candidate.polygon_xy]

    @staticmethod
    def _bbox_json(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
        if bbox is None:
            return None
        return [int(value) for value in bbox]

    @staticmethod
    def _bbox_json_with_candidate_offset(
        bbox: tuple[int, int, int, int] | None,
        candidate: _MobileRankedCandidate,
    ) -> list[int] | None:
        if bbox is None:
            return None
        offset_x = int(getattr(candidate, "scan_offset_x", 0) or 0)
        offset_y = int(getattr(candidate, "scan_offset_y", 0) or 0)
        x1, y1, x2, y2 = bbox
        return [int(x1 + offset_x), int(y1 + offset_y), int(x2 + offset_x), int(y2 + offset_y)]

    @staticmethod
    def _final_recognition_bbox_json(evaluated: _EvaluatedCandidate) -> list[int] | None:
        if evaluated.normalized_crop is not None:
            return MobileExpiryPipeline._bbox_json_with_candidate_offset(
                evaluated.normalized_crop.bbox_xyxy,
                evaluated.candidate,
            )
        return MobileExpiryPipeline._bbox_json_with_candidate_offset(
            evaluated.candidate.final_crop_bbox,
            evaluated.candidate,
        )

    @staticmethod
    def _final_recognition_polygon_json(evaluated: _EvaluatedCandidate) -> list[list[float]] | None:
        # NormalizedTextLineCrop currently records whether a polygon crop was used,
        # not the source polygon coordinates. The thin final bbox is the reliable
        # recognizer geometry we can expose for review.
        return None

    @staticmethod
    def _result_reason(evaluated: _EvaluatedCandidate) -> str:
        orientation = evaluated.normalized_crop.selected_orientation if evaluated.normalized_crop is not None else evaluated.recognition.rotation
        sources = ",".join(evaluated.candidate.detector_sources)
        evidence = evaluated.selected_date_evidence
        evidence_bits = ""
        if evidence is not None:
            role = score_date_role(MobileExpiryPipeline._date_evidence_text(evidence))
            context = "|".join(evidence.context_texts)
            evidence_bits = f";date_evidence_selection=1;date_role={role.role};context={context or 'none'}"
        return (
            f"{evaluated.parsed.reason};candidate_id={evaluated.candidate.candidate_id};"
            f"candidate_type={evaluated.candidate.candidate_type};detector_sources={sources};"
            f"detector_variant={evaluated.candidate.detector_variant};variant={evaluated.recognition_variant};"
            f"orientation={orientation};final_rank={evaluated.candidate.final_rank_after_force_include};"
            f"force_include={evaluated.candidate.force_included_reason};parse_inputs={evaluated.parse_inputs_count}"
            f"{evidence_bits}"
        )

    @staticmethod
    def _date_evidence_json(evidence_items: list[_DateEvidence]) -> list[dict[str, object]]:
        def score_item(evidence: _DateEvidence) -> tuple[int, int, float]:
            parsed = evidence.parsed.parsed_date.toordinal() if evidence.parsed.parsed_date is not None else 0
            return (1 if has_expiry_keyword(MobileExpiryPipeline._date_evidence_text(evidence)) else 0, parsed, float(evidence.parsed.confidence or 0.0))

        rows: list[dict[str, object]] = []
        for evidence in sorted(evidence_items, key=score_item, reverse=True)[:12]:
            orientation = evidence.normalized_crop.selected_orientation if evidence.normalized_crop is not None else evidence.recognition.rotation
            rows.append(
                {
                    "parsed_date": evidence.parsed.parsed_date.isoformat() if evidence.parsed.parsed_date is not None else None,
                    "date_precision": evidence.parsed.date_precision,
                    "parser_confidence": evidence.parsed.confidence,
                    "ocr_confidence": evidence.recognition.confidence,
                    "raw_text": evidence.recognition.raw_text,
                    "normalized_text": evidence.recognition.normalized_text,
                    "context_texts": list(evidence.context_texts),
                    "role": score_date_role(MobileExpiryPipeline._date_evidence_text(evidence)).role,
                    "candidate_id": evidence.candidate.candidate_id,
                    "candidate_type": evidence.candidate.candidate_type,
                    "detector_sources": list(evidence.candidate.detector_sources),
                    "detector_variant": evidence.candidate.detector_variant,
                    "recognition_variant": evidence.recognition_variant,
                    "crop_policy": evidence.crop_policy,
                    "orientation": orientation,
                    "final_recognition_bbox_xyxy": MobileExpiryPipeline._bbox_json_with_candidate_offset(
                        evidence.normalized_crop.bbox_xyxy if evidence.normalized_crop is not None else evidence.candidate.final_crop_bbox,
                        evidence.candidate,
                    ),
                    "metadata": evidence.metadata,
                }
            )
        return rows

    def _global_candidates_from_ranked(
        self,
        image: np.ndarray,
        ranked: list[_MobileRankedCandidate],
        *,
        source: str = "mobile_yolo",
        variant: str | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[_MobileGlobalCandidate]:
        global_candidates: list[_MobileGlobalCandidate] = []
        for candidate in ranked:
            if not candidate.selected_geometry:
                continue
            selected_variant = variant or candidate.detector_variant or "yolo26s_obb"
            x1, y1, x2, y2 = candidate.bbox_xyxy
            candidate.scan_offset_x = int(offset_x)
            candidate.scan_offset_y = int(offset_y)
            global_candidates.append(
                _MobileGlobalCandidate(
                    source=source,
                    variant=selected_variant,
                    variant_key=f"{source}:{selected_variant}",
                    image=image,
                    detector_confidence=candidate.detector_confidence,
                    ranked=candidate,
                    scan_bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                )
            )
        return global_candidates

    def _evaluate_architecture_candidates(
        self,
        image: np.ndarray,
        ranked: list[_MobileRankedCandidate],
        *,
        today: date,
    ) -> list[_EvaluatedCandidate]:
        shortlisted = self._apply_geometry_shortlist(ranked)
        global_candidates = self._global_candidates_from_ranked(image, shortlisted)
        probe_candidates = self._select_global_probe_candidates(global_candidates)
        self._probe_global_candidates(probe_candidates, today=today)
        final_candidates = self._select_global_final_candidates(probe_candidates)
        self._attach_local_group_wide_bboxes(final_candidates)
        return [self._evaluate_candidate(image, candidate.ranked, today=today) for candidate in final_candidates]

    def _evaluate_yolo_primary_flow(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
        *,
        today: date,
    ) -> list[_EvaluatedCandidate]:
        ranked = self._build_ranked_candidates(image, yolo_candidates)
        return self._evaluate_architecture_candidates(image, ranked, today=today)

    def _evaluate_ppmobile_gated_flow(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
        *,
        today: date,
    ) -> list[_EvaluatedCandidate]:
        old_probe_top_k = self.expiry_global_probe_top_k
        old_final_top_k = self.expiry_global_final_top_k
        old_debug_final_top_k = self.expiry_global_debug_final_top_k
        try:
            self.expiry_global_probe_top_k = self.rapidocr_gated_probe_top_k
            self.expiry_global_final_top_k = self.rapidocr_gated_final_top_k
            self.expiry_global_debug_final_top_k = self.rapidocr_gated_final_top_k
            ranked = self._build_ranked_candidates_from_proposals(image, proposals)
            self._apply_rapidocr_probe_metadata_to_ranked(ranked)
            shortlisted = self._apply_geometry_shortlist(ranked)
            global_candidates = self._global_candidates_from_ranked(
                image,
                shortlisted,
                source="ppmobile_gated",
                variant="rapidocr_ppocrv5",
            )
            probe_candidates = self._select_global_probe_candidates(global_candidates)
            self._probe_global_candidates(probe_candidates, today=today)
            final_candidates = self._select_global_final_candidates(probe_candidates)
            self._attach_local_group_wide_bboxes(final_candidates)
            self._debug_append(
                "ppmobile_gated_trigger",
                {
                    "stage": "evaluated",
                    "filtered_count": len(proposals),
                    "probe_count": len(probe_candidates),
                    "final_count": len(final_candidates),
                    "probe_top_k": self.rapidocr_gated_probe_top_k,
                    "final_top_k": self.rapidocr_gated_final_top_k,
                },
            )
            return [self._evaluate_candidate(image, candidate.ranked, today=today) for candidate in final_candidates]
        finally:
            self.expiry_global_probe_top_k = old_probe_top_k
            self.expiry_global_final_top_k = old_final_top_k
            self.expiry_global_debug_final_top_k = old_debug_final_top_k

    def _attach_local_group_wide_bboxes(self, candidates: list[_MobileGlobalCandidate]) -> None:
        if not self.local_group_wide_crop_enabled:
            return
        group_candidates = [
            item
            for item in candidates
            if item.ranked.candidate_type in {"group", "fallback", "line"}
        ]
        for item in candidates:
            ranked = item.ranked
            current_bbox = ranked.evidence_bbox or ranked.bbox_xyxy
            current_area = self._bbox_area(current_bbox)
            best_bbox: tuple[int, int, int, int] | None = None
            best_area = current_area
            for group in group_candidates:
                if group is item or group.image is not item.image:
                    continue
                group_bbox = group.ranked.evidence_bbox or group.ranked.bbox_xyxy
                group_area = self._bbox_area(group_bbox)
                if group_area <= best_area * 1.10:
                    continue
                _iou, containment = self._bbox_overlap(current_bbox, group_bbox)
                if containment < 0.60:
                    continue
                best_bbox = group_bbox
                best_area = group_area
            if best_bbox is not None:
                ranked.evidence_bbox = best_bbox

    def _evaluate_rapidocr_primary_flow(
        self,
        image: np.ndarray,
        yolo_candidates: list[_YoloCandidate],
        *,
        today: date,
    ) -> tuple[list[_EvaluatedCandidate], str | None]:
        global_candidates: list[_MobileGlobalCandidate] = []
        reasons: list[str] = []
        if yolo_candidates:
            direct_ranked = self._build_ranked_candidates(image, yolo_candidates)
            direct_ranked = self._apply_geometry_shortlist(direct_ranked)
            global_candidates.extend(
                self._global_candidates_from_ranked(
                    image,
                    direct_ranked,
                    source="mobile_yolo",
                    variant="yolo26s_obb",
                    offset_x=0,
                    offset_y=0,
                )
            )
        for roi in self._collect_advanced_roi_candidates(image, yolo_candidates):
            for variant in self._detector_variants_for_roi(roi.image):
                proposals, reason = self._combined_primary_proposals_for_roi(
                    roi.image,
                    variant=variant,
                    roi_confidence=roi.confidence,
                )
                if reason:
                    reasons.append(f"{roi.source}:{variant.name}:{reason}")
                ranked = self._build_ranked_candidates_from_proposals(roi.image, proposals)
                ranked = self._apply_geometry_shortlist(ranked)
                global_candidates.extend(
                    self._global_candidates_from_ranked(
                        roi.image,
                        ranked,
                        source=roi.source,
                        variant=variant.name,
                        offset_x=roi.offset_x,
                        offset_y=roi.offset_y,
                    )
                )

        if not global_candidates:
            return [], "; ".join(reasons) or "RapidOCR primary returned no candidates"

        probe_candidates = self._select_global_probe_candidates(global_candidates)
        self._probe_global_candidates(probe_candidates, today=today)
        final_candidates = self._select_global_final_candidates(probe_candidates)
        self._attach_local_group_wide_bboxes(final_candidates)
        return [self._evaluate_candidate(candidate.image, candidate.ranked, today=today) for candidate in final_candidates], None

    def _best_parseable_candidate(self, evaluated: list[_EvaluatedCandidate]) -> _EvaluatedCandidate | None:
        parseable = [
            item
            for item in evaluated
            if self._has_accepted_date_evidence(item)
        ]
        if not parseable:
            return None
        return max(parseable, key=self._score_parseable_candidate)

    def _has_day_precision_parseable_candidate(self, evaluated: list[_EvaluatedCandidate]) -> bool:
        for item in evaluated:
            if item.date_evidence:
                if any(self._date_precision_score(evidence.parsed) >= 3 for evidence in self._accepted_date_evidence_items(item.date_evidence)):
                    return True
            elif self._has_accepted_date_evidence(item) and self._date_precision_score(item.parsed) >= 3:
                return True
        return False

    def _has_strong_expiry_answer(self, evaluated: list[_EvaluatedCandidate]) -> bool:
        parseable, accepted_evidence = self._accepted_parseable_items(evaluated)
        for evidence in accepted_evidence:
            text = self._date_evidence_text(evidence)
            has_expiry_context = has_expiry_keyword(text) or has_weak_expiry_keyword(text)
            if has_production_keyword(text) and not has_expiry_context:
                continue
            precision_score = self._date_precision_score(evidence.parsed)
            if precision_score >= 3 and self._is_strong_accepted_date_evidence(evidence):
                return True
            if (
                precision_score >= 2
                and has_expiry_context
                and float(evidence.parsed.confidence or 0.0) >= 0.95
                and float(evidence.recognition.confidence or 0.0) >= 0.90
            ):
                return True

        for item in parseable:
            if item.date_evidence:
                continue
            text = " ".join(
                part
                for part in (
                    item.recognition.normalized_text,
                    item.recognition.raw_text,
                    " ".join(item.candidate.context_probe_texts or []),
                )
                if part
            )
            has_expiry_context = has_expiry_keyword(text) or has_weak_expiry_keyword(text)
            if has_production_keyword(text) and not has_expiry_context:
                continue
            precision_score = self._date_precision_score(item.parsed)
            if precision_score >= 3 and self._is_strong_day_date_parse(item.parsed, item.recognition, text=text):
                return True
            if (
                precision_score >= 2
                and has_expiry_context
                and float(item.parsed.confidence or 0.0) >= 0.95
                and float(item.recognition.confidence or 0.0) >= 0.90
            ):
                return True
        return False

    def _ppmobile_gated_trigger_reason(
        self,
        yolo_candidates: list[_YoloCandidate],
        evaluated: list[_EvaluatedCandidate],
    ) -> str | None:
        if self.rapidocr_primary_mode == "off":
            return None
        if not yolo_candidates:
            return "no_yolo_boxes"
        if self._has_strong_expiry_answer(evaluated):
            return None
        parseable, accepted_evidence = self._accepted_parseable_items(evaluated)
        if not parseable:
            return "no_parseable_yolo_result"
        if any(self._date_precision_score(evidence.parsed) < 3 for evidence in accepted_evidence):
            return "partial_yolo_date"
        for evidence in accepted_evidence:
            text = self._date_evidence_text(evidence)
            if has_production_keyword(text) and not (has_expiry_keyword(text) or has_weak_expiry_keyword(text)):
                return "production_only_yolo_result"
        return "weak_yolo_result"

    def _should_run_ppmobile_proposal_rescue(
        self,
        yolo_candidates: list[_YoloCandidate],
        evaluated: list[_EvaluatedCandidate],
    ) -> bool:
        return self._ppmobile_gated_trigger_reason(yolo_candidates, evaluated) is not None

    def _should_run_rapidocr_rescue(self, evaluated: list[_EvaluatedCandidate]) -> bool:
        if not self.rapidocr_rescue_enabled:
            return False
        best = self._best_parseable_candidate(evaluated)
        if best is None:
            if evaluated:
                best_unparseable = max(evaluated, key=self._score_unparseable_candidate)
                text = " ".join(
                    part
                    for part in (
                        best_unparseable.recognition.normalized_text,
                        best_unparseable.recognition.raw_text,
                    )
                    if part
                )
                date_likeness = self._score_date_likeness(best_unparseable.recognition.raw_text)
                ocr_score = self._score_ocr_candidate(best_unparseable.recognition)
                if ocr_score >= 0.80 and (date_likeness >= 0.50 or DATE_PATTERN.search(text)):
                    return False
            return True
        if self._date_precision_score(best.parsed) < 3:
            return True
        return False

    def _accepted_rapidocr_rescue_evaluations(
        self,
        primary_evaluated: list[_EvaluatedCandidate],
        rescue_evaluated: list[_EvaluatedCandidate],
    ) -> list[_EvaluatedCandidate]:
        primary_best = self._best_parseable_candidate(primary_evaluated)
        if primary_best is None:
            return rescue_evaluated

        primary_ocr = self._score_ocr_candidate(primary_best.recognition)
        primary_likeness = self._score_date_likeness(primary_best.recognition.raw_text)
        accepted: list[_EvaluatedCandidate] = []
        for item in rescue_evaluated:
            if item.parsed.parsed_date is None:
                continue
            rescue_ocr = self._score_ocr_candidate(item.recognition)
            rescue_likeness = self._score_date_likeness(item.recognition.raw_text)
            if rescue_ocr >= primary_ocr + 0.08 and rescue_likeness >= primary_likeness:
                accepted.append(item)
        return accepted

    def run(self, image_bytes: bytes, *, today: date) -> MobileExpiryPipelineResult:
        started = perf_counter()
        self._reset_debug_records()
        with self._profile_timer("decode_ms"):
            image = self.decode_image(image_bytes)
        if image is None:
            return MobileExpiryPipelineResult(
                status="failed",
                detected_expiry_date=None,
                raw_text=None,
                normalized_text=None,
                recognition_confidence=None,
                detector_confidence=None,
                reason="image unreadable",
                detection_polygon_json=None,
                runtime_ms=int((perf_counter() - started) * 1000),
                final_recognition_bbox_xyxy=None,
                final_recognition_polygon_json=None,
                final_crop_policy=None,
                final_crop_padding_px=None,
                date_evidence_json=None,
                debug_profile=self._finish_debug_profile(started),
                debug_candidates_json_bytes=self._debug_bytes(),
            )

        with self._profile_timer("yolo_ms"):
            yolo_candidates, detection_reason = self._detect_candidates(image)
        if not yolo_candidates:
            with self._profile_timer("detector_variant_rescue_ms"):
                rotation_candidates, rotation_reason = self._detect_full_image_rotation_rescue_candidates(image)
            if rotation_candidates:
                yolo_candidates = self._dedupe_yolo_candidates([*yolo_candidates, *rotation_candidates])[
                    : self.full_image_rotation_detector_rescue_max_candidates
                ]
                detection_reason = None
            elif rotation_reason and detection_reason:
                detection_reason = f"{detection_reason}; {rotation_reason}"
            elif rotation_reason:
                detection_reason = rotation_reason
        self._profile_set("num_yolo_boxes", len(yolo_candidates))
        if self.rapidocr_primary_enabled and self.rapidocr_primary_mode == "always":
            evaluated, rapidocr_reason = self._evaluate_rapidocr_primary_flow(image, yolo_candidates, today=today)
            if rapidocr_reason and detection_reason:
                detection_reason = f"{detection_reason}; {rapidocr_reason}"
            elif rapidocr_reason:
                detection_reason = rapidocr_reason
        else:
            evaluated = self._evaluate_yolo_primary_flow(image, yolo_candidates, today=today)
            ppmobile_reason = (
                self._ppmobile_gated_trigger_reason(yolo_candidates, evaluated)
                if self.rapidocr_primary_enabled
                else None
            )
            if ppmobile_reason is not None:
                self._debug_append(
                    "ppmobile_gated_trigger",
                    {
                        "stage": "trigger",
                        "triggered": True,
                        "reason": ppmobile_reason,
                        "yolo_count": len(yolo_candidates),
                        "yolo_evaluated_count": len(evaluated),
                    },
                )
                ppmobile_proposals, rapidocr_reason = self._detect_rapidocr_full_image_gated_proposals(
                    image,
                    yolo_candidates=yolo_candidates,
                    today=today,
                )
                if ppmobile_proposals:
                    evaluated.extend(self._evaluate_ppmobile_gated_flow(image, ppmobile_proposals, today=today))
                elif rapidocr_reason and detection_reason:
                    detection_reason = f"{detection_reason}; {rapidocr_reason}"
                elif rapidocr_reason:
                    detection_reason = rapidocr_reason
            else:
                self._debug_append(
                    "ppmobile_gated_trigger",
                    {
                        "stage": "trigger",
                        "triggered": False,
                        "reason": "strong_yolo_expiry_or_disabled",
                        "yolo_count": len(yolo_candidates),
                        "yolo_evaluated_count": len(evaluated),
                    },
                )
        primary_evaluated = list(evaluated)
        if self._should_run_rapidocr_rescue(primary_evaluated):
            proposal_candidates, proposal_reason = self._detect_rapidocr_rescue_candidates(image, yolo_candidates)
            if proposal_candidates:
                proposal_ranked = self._build_ranked_candidates(image, proposal_candidates)
                proposal_evaluated = self._evaluate_architecture_candidates(image, proposal_ranked, today=today)
                evaluated.extend(self._accepted_rapidocr_rescue_evaluations(primary_evaluated, proposal_evaluated))
            elif proposal_reason and detection_reason:
                detection_reason = f"{detection_reason}; {proposal_reason}"
            elif proposal_reason:
                detection_reason = proposal_reason
        if not self._has_day_precision_parseable_candidate(evaluated):
            with self._profile_timer("detector_variant_rescue_ms"):
                rescue_candidates, rescue_reason = self._detect_variant_rescue_candidates(image)
            if rescue_candidates:
                rescue_ranked = self._build_ranked_candidates(image, rescue_candidates)
                evaluated.extend(self._evaluate_architecture_candidates(image, rescue_ranked, today=today))
            elif rescue_reason and detection_reason:
                detection_reason = f"{detection_reason}; {rescue_reason}"
            elif rescue_reason:
                detection_reason = rescue_reason
        if not self._has_day_precision_parseable_candidate(evaluated):
            with self._profile_timer("dot_matrix_ms"):
                dot_matrix_proposals, dot_matrix_reason = self._detect_dot_matrix_rescue_proposals(image)
            if dot_matrix_proposals:
                dot_matrix_ranked = self._build_ranked_candidates_from_proposals(image, dot_matrix_proposals)
                evaluated.extend(self._evaluate_architecture_candidates(image, dot_matrix_ranked, today=today))
            elif dot_matrix_reason and detection_reason:
                detection_reason = f"{detection_reason}; {dot_matrix_reason}"
            elif dot_matrix_reason:
                detection_reason = dot_matrix_reason
        with self._profile_timer("partial_expansion_ms"):
            self._add_cross_candidate_partial_day_month_right_expansion_evidence(image, evaluated, today=today)
        with self._profile_timer("anchor_local_sibling_ms"):
            self._add_anchor_local_sibling_evidence(image, evaluated, today=today)
        accepted_before_product, _ = self._accepted_parseable_items(evaluated)
        if self.product_cropper_rescue_enabled and (
            not yolo_candidates or not accepted_before_product or not self._has_day_precision_parseable_candidate(evaluated)
        ):
            product_candidates, product_reason = self._detect_product_cropper_rescue_candidates(image)
            if product_candidates:
                evaluated.extend(
                    self._evaluate_product_cropper_rescue_candidates_direct(
                        image,
                        product_candidates,
                        today=today,
                    )
                )
            elif product_reason and detection_reason:
                detection_reason = f"{detection_reason}; {product_reason}"
            elif product_reason:
                detection_reason = product_reason
        if yolo_candidates and not self._has_day_precision_parseable_candidate(evaluated):
            fallback_ranked = self._build_ranked_candidates(image, [])
            evaluated.extend(self._evaluate_architecture_candidates(image, fallback_ranked, today=today))
        with self._profile_timer("partial_expansion_ms"):
            self._add_cross_candidate_partial_day_month_right_expansion_evidence(image, evaluated, today=today)
        with self._profile_timer("anchor_local_sibling_ms"):
            self._add_anchor_local_sibling_evidence(image, evaluated, today=today)

        if not evaluated:
            return MobileExpiryPipelineResult(
                status="manual_review_required",
                detected_expiry_date=None,
                raw_text=None,
                normalized_text=None,
                recognition_confidence=None,
                detector_confidence=yolo_candidates[0].confidence if yolo_candidates else None,
                reason=detection_reason or "recognizer returned no candidates",
                detection_polygon_json=yolo_candidates[0].polygon_xy if yolo_candidates else None,
                runtime_ms=int((perf_counter() - started) * 1000),
                final_recognition_bbox_xyxy=None,
                final_recognition_polygon_json=None,
                final_crop_policy=None,
                final_crop_padding_px=None,
                date_evidence_json=None,
                debug_profile=self._finish_debug_profile(started),
                debug_candidates_json_bytes=self._debug_bytes(),
            )

        parseable, date_evidence = self._accepted_parseable_items(evaluated)
        if not parseable:
            best_unparseable = max(evaluated, key=self._score_unparseable_candidate)
            return MobileExpiryPipelineResult(
                status="manual_review_required",
                detected_expiry_date=None,
                raw_text=best_unparseable.recognition.raw_text or None,
                normalized_text=best_unparseable.recognition.normalized_text or None,
                recognition_confidence=best_unparseable.recognition.confidence,
                detector_confidence=best_unparseable.candidate.detector_confidence,
                reason=best_unparseable.recognition.reason or best_unparseable.parsed.reason or "no valid date parsed",
                detection_polygon_json=yolo_candidates[0].polygon_xy if yolo_candidates else self._polygon_json(best_unparseable.candidate),
                runtime_ms=int((perf_counter() - started) * 1000),
                final_recognition_bbox_xyxy=self._final_recognition_bbox_json(best_unparseable),
                final_recognition_polygon_json=self._final_recognition_polygon_json(best_unparseable),
                final_crop_policy=best_unparseable.candidate.final_crop_policy,
                final_crop_padding_px=best_unparseable.candidate.final_crop_padding_px,
                date_evidence_json=self._date_evidence_json(
                    [
                        evidence
                        for item in evaluated
                        for evidence in item.date_evidence
                        if evidence.parsed.parsed_date is not None
                    ]
                ),
                debug_profile=self._finish_debug_profile(started),
                debug_candidates_json_bytes=self._debug_bytes(),
            )

        if date_evidence:
            best_evidence = self._select_date_evidence(date_evidence)
            if self._should_run_hardcase_recognizer_rescue(best_evidence, date_evidence, today=today):
                hardcase_evidence = self._collect_hardcase_recognizer_rescue_evidence(
                    best_evidence,
                    date_evidence,
                    today=today,
                )
                hardcase_override = self._select_hardcase_recognizer_override(
                    best_evidence,
                    hardcase_evidence,
                    today=today,
                )
                if hardcase_override is not None:
                    date_evidence = [*date_evidence, *hardcase_evidence]
                    best_evidence = hardcase_override
            best = self._evaluated_from_date_evidence(best_evidence, date_evidence=date_evidence)
        else:
            best = max(parseable, key=self._score_parseable_candidate)
        if best.selected_date_evidence is not None:
            self._debug_append(
                "final_selected_evidence",
                {
                    "source": "date_evidence",
                    "evidence": self._debug_date_evidence(best.selected_date_evidence),
                    "reason": self._result_reason(best),
                },
            )
        else:
            self._debug_append(
                "final_selected_evidence",
                {
                    "source": "evaluated_candidate",
                    "candidate": self._debug_ranked_candidate(best.candidate),
                    "recognition": {
                        "raw_text": best.recognition.raw_text,
                        "normalized_text": best.recognition.normalized_text,
                        "confidence": best.recognition.confidence,
                        "reason": best.recognition.reason,
                    },
                    "parsed": self._debug_parsed_date(best.parsed),
                    "reason": self._result_reason(best),
                },
            )
        return MobileExpiryPipelineResult(
            status="parsed_success",
            detected_expiry_date=best.parsed.parsed_date,
            raw_text=best.recognition.raw_text or None,
            normalized_text=best.recognition.normalized_text or None,
            recognition_confidence=best.recognition.confidence,
            detector_confidence=best.candidate.detector_confidence,
            reason=self._result_reason(best),
            detection_polygon_json=yolo_candidates[0].polygon_xy if yolo_candidates else self._polygon_json(best.candidate),
            runtime_ms=int((perf_counter() - started) * 1000),
            final_recognition_bbox_xyxy=self._final_recognition_bbox_json(best),
            final_recognition_polygon_json=self._final_recognition_polygon_json(best),
            final_crop_policy=best.candidate.final_crop_policy,
            final_crop_padding_px=best.candidate.final_crop_padding_px,
            date_evidence_json=self._date_evidence_json(date_evidence),
            debug_profile=self._finish_debug_profile(started),
            debug_candidates_json_bytes=self._debug_bytes(),
        )
