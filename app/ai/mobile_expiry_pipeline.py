from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
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
class _EvaluatedCandidate:
    candidate: _MobileRankedCandidate
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int
    recognition_variant: str
    normalized_crop: NormalizedTextLineCrop | None
    original_color_fallback_used: bool = False


@dataclass(slots=True)
class _RecognitionEvidence:
    variant_name: str
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int


class SVTRTextRecognizer:
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

    def recognize(self, image: np.ndarray) -> _RecognitionOutput:
        self._ensure_loaded()
        if self._load_error:
            return _RecognitionOutput("", "", None, self._load_error, "original")
        if self._model is None:
            return _RecognitionOutput("", "", None, "SVTR recognizer unavailable", "original")

        if image.ndim == 2:
            recognizer_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            recognizer_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            recognizer_input = image

        try:
            result = self._model.predict(recognizer_input)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return _RecognitionOutput("", "", None, f"SVTR recognition failed: {exc}", "original")

        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            return _RecognitionOutput("", "", None, "SVTR recognizer returned no result", "original")

        entry = result[0]
        raw = str(entry.get("rec_text", "") or "").strip()
        confidence: float | None = None
        try:
            confidence = float(entry["rec_score"]) if entry.get("rec_score") is not None else None
        except Exception:
            confidence = None

        if not raw:
            return _RecognitionOutput("", "", confidence, "SVTR recognizer returned empty text", "original")
        return _RecognitionOutput(raw, normalize_recognition_text(raw), confidence, None, "original")


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
        svtr_backend: str = "paddle",
        svtr_onnx_model_path: Path | None = None,
        proposal_rescue_backend: str = "rapidocr_ppocrv5",
        rapidocr_primary_enabled: bool = True,
        rapidocr_rescue_enabled: bool = True,
        rapidocr_ocr_version: str = "PP-OCRv5",
        rapidocr_model_type: str = "mobile",
        rapidocr_lang_type: str = "ch",
        rapidocr_limit_side_len: int = 512,
        rapidocr_limit_type: str = "max",
        rapidocr_max_candidates: int = 16,
        rapidocr_primary_max_rois_per_scan: int = 12,
        rapidocr_primary_max_boxes_accepted: int = 40,
        rapidocr_primary_timeout_seconds: float = 12.0,
        rapidocr_max_rois_per_scan: int = 1,
        rapidocr_max_boxes_accepted: int = 5,
        rapidocr_timeout_seconds: float = 0.75,
        rapidocr_min_confidence: float = 0.10,
    ) -> None:
        self.detector_model_path = detector_model_path
        self.detector_backend = detector_backend.lower().strip()
        self.detector_onnx_path = detector_onnx_path
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_imgsz = detector_imgsz
        self.max_candidates = max(1, int(max_candidates))
        self.crop_padding_px = max(0, int(crop_padding_px))
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
        self.rapidocr_max_rois_per_scan = max(0, int(rapidocr_max_rois_per_scan))
        self.rapidocr_max_boxes_accepted = max(1, int(rapidocr_max_boxes_accepted))
        self.rapidocr_timeout_seconds = max(0.1, float(rapidocr_timeout_seconds))
        self.rapidocr_min_confidence = max(0.0, min(float(rapidocr_min_confidence), 1.0))
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
        self._detector_model: Any | None = None
        self._onnx_detector: YoloObbOnnxDetector | None = None
        self._detector_error: str | None = None
        self._rapidocr_detector: RapidOCRTextProposalDetector | None = None
        self._rapidocr_error: str | None = None
        self._last_rapidocr_primary_debug: dict[str, object] = {}
        self._last_rapidocr_rescue_debug: dict[str, object] = {}

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
            return [
                _YoloCandidate(
                    polygon_xy=item.polygon_xy,
                    bbox_xyxy=item.bbox_xyxy,
                    confidence=item.confidence,
                    variant_name=variant_name,
                )
                for item in detections
            ], reason
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
        candidates: list[_YoloCandidate] = []
        height, width = image.shape[:2]
        for polygon, confidence in zip(polygons, confidences, strict=False):
            conf = float(confidence)
            if conf < self.detector_confidence_threshold:
                continue
            cleaned = [
                [max(0.0, min(float(x), float(width))), max(0.0, min(float(y), float(height)))]
                for x, y in polygon
            ]
            bbox = self._xyxy_from_polygon(cleaned)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            candidates.append(_YoloCandidate(polygon_xy=cleaned, bbox_xyxy=bbox, confidence=conf, variant_name=variant_name))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[: self.max_candidates], None if candidates else "no candidate passed confidence threshold"

    def _detect_candidates(self, image: np.ndarray) -> tuple[list[_YoloCandidate], str | None]:
        candidates, reason = self._detect_candidates_for_variant(image, variant_name="yolo26s_obb")
        return candidates, reason

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
                max_boxes=max(self.rapidocr_primary_max_boxes_accepted, self.rapidocr_max_boxes_accepted),
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
        if runtime_ms > self.rapidocr_primary_timeout_seconds * 1000.0:
            return [], f"rapidocr_primary_timeout_ms={runtime_ms:.0f}"
        detector_variant = f"rapidocr_primary:{self.rapidocr_ocr_version}_{self.rapidocr_model_type}:{variant.name}"
        proposals = self._map_rapidocr_proposals_to_roi_original(
            boxes[: self.rapidocr_primary_max_boxes_accepted],
            variant=variant,
            original_shape=roi_image.shape,
            detector_variant=detector_variant,
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
        return self.candidate_engine.build_ranked_candidates(image, proposals)  # type: ignore[return-value]

    def _normalized_crops_for_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate) -> list[NormalizedTextLineCrop]:
        tight_bbox, policy = self._final_tight_bbox_for_candidate(candidate)
        padded_bbox, padding_px = self._pad_bbox_for_final_crop(tight_bbox, image.shape)
        polygon_xy = candidate.polygon_xy if tight_bbox == candidate.bbox_xyxy and padded_bbox == tight_bbox else None
        crops = normalize_textline_crops(
            image,
            TextLineGeometry(
                bbox_xyxy=padded_bbox,
                polygon_xy=polygon_xy,
            ),
            TextLineCropConfig(padding_px=0),
        )
        if crops:
            candidate.final_crop_bbox = crops[0].bbox_xyxy
            candidate.final_crop_padding_px = padding_px
            candidate.final_crop_policy = policy
        return crops

    def _final_tight_bbox_for_candidate(self, candidate: _MobileRankedCandidate) -> tuple[tuple[int, int, int, int], str]:
        if candidate.recognition_bbox is not None and candidate.recognition_bbox != candidate.bbox_xyxy:
            return candidate.recognition_bbox, "recognition_bbox_tight"
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
        variants.extend(self.preprocessor.recognition_variants(crop, allowed_names=enhanced_allowed))
        return variants

    def _recognize_variant(self, image: np.ndarray, *, variant_name: str, orientation: str) -> _RecognitionOutput:
        result = self.recognizer.recognize(image)
        return _RecognitionOutput(
            raw_text=str(getattr(result, "raw_text", "") or ""),
            normalized_text=normalize_recognition_text(str(getattr(result, "normalized_text", "") or getattr(result, "raw_text", "") or "")),
            confidence=getattr(result, "confidence", None),
            reason=getattr(result, "reason", None),
            rotation=orientation,
        )

    def _best_probe_for_crop(
        self,
        crop: np.ndarray,
        *,
        today: date,
    ) -> tuple[_RecognitionOutput, str, np.ndarray]:
        best: _RecognitionOutput | None = None
        best_variant = "original"
        best_variant_image = crop
        best_score: tuple[float, float, float, float] | None = None
        for variant in self._recognition_variants_for_crop(crop, allowed_names=MOBILE_RECOGNITION_VARIANTS):
            probe = self._recognize_variant(variant.image, variant_name=variant.name, orientation="original")
            probe_scores = self._probe_signal_scores(probe, candidate=None, today=today)
            parsed = self._best_parse_for_inputs(self._build_parse_inputs(probe), today=today)
            parseable_score = float(parsed.confidence if parsed.parsed_date is not None else 0.0)
            score = (
                float(probe.confidence or 0.0),
                self._score_date_likeness(probe.raw_text),
                parseable_score,
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
        parser_probe_bonus = 2.0 if parsed.parsed_date is not None else 0.0
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
            if candidate.geometry_score >= cutoff - 0.30 and candidate.geometry_features.get("area_ratio", 1.0) <= 0.16:
                selected_ids.add(id(candidate))
        for candidate in candidates:
            candidate.selected_geometry = id(candidate) in selected_ids
            if not candidate.selected_geometry:
                candidate.total_score = candidate.geometry_score
        return candidates

    @staticmethod
    def _candidate_signal_text(candidate: _MobileRankedCandidate) -> str:
        return " ".join(part for part in (candidate.probe_normalized_text, candidate.probe_text) if part)

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
        text = self._candidate_signal_text(candidate)
        has_date = self._has_date_like_text(text) or candidate.score_breakdown.get("parser_probe_bonus", 0.0) > 0
        has_expiry = has_expiry_keyword(text) or candidate.score_breakdown.get("expiry_keyword_score", 0.0) > 0
        if has_expiry and has_date:
            return "expiry_keyword_date"
        if candidate.score_breakdown.get("keyword_date_group_bonus", 0.0) > 0:
            return "keyword_date_group"
        if has_date and candidate.score_breakdown.get("parser_probe_bonus", 0.0) > 0:
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
            candidate.probe_text = probe.raw_text
            candidate.probe_normalized_text = probe.normalized_text
            candidate.probe_confidence = probe.confidence
            candidate.probe_reason = probe.reason
            candidate.selected_recognition_variant = variant_name
            candidate.recognition_variant_image = variant_image
            candidate.score_breakdown.update(scores)
            candidate.total_score = candidate.geometry_score + sum(scores.values())
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
        for variant in self._recognition_variants_for_crop(crop, allowed_names=MOBILE_RECOGNITION_VARIANTS):
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
        return evidence

    def _best_recognition_for_single_crop(
        self,
        crop: NormalizedTextLineCrop,
        *,
        today: date,
        selected_variant_name: str | None = None,
        selected_variant_image: np.ndarray | None = None,
    ) -> _EvaluatedCandidate:
        best_parseable: tuple[tuple[int, int, float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_with_text: tuple[tuple[float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_empty: tuple[_RecognitionOutput, ParsedDateData, int, str] | None = None

        variants: list[ImageVariant] = []
        selected_name = selected_variant_name or "original"
        if selected_variant_image is not None:
            variants.append(ImageVariant(selected_name, selected_variant_image, purpose="recognition"))
        elif selected_variant_name:
            variants.extend(self._recognition_variants_for_crop(crop.image, allowed_names=(selected_variant_name,)))
        for variant in self._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS):
            if variant.name == selected_name and variants:
                continue
            variants.append(variant)

        for variant in variants:
            rec = self._recognize_variant(variant.image, variant_name=variant.name, orientation=crop.selected_orientation)
            parse_inputs = self._build_parse_inputs(rec)
            parsed = self._best_parse_for_inputs(parse_inputs, today=today)
            if parsed.parsed_date is not None:
                score = (
                    self._date_precision_score(parsed),
                    parsed.parsed_date.toordinal(),
                    float(parsed.confidence),
                    float(rec.confidence or 0.0),
                    self._score_date_likeness(rec.raw_text),
                )
                item = (score, rec, parsed, len(parse_inputs), variant.name)
                if best_parseable is None or score > best_parseable[0]:
                    best_parseable = item
                continue
            if rec.raw_text.strip():
                score = (self._score_ocr_candidate(rec), float(rec.confidence or 0.0), self._score_date_likeness(rec.raw_text))
                item = (score, rec, parsed, len(parse_inputs), variant.name)
                if best_with_text is None or score > best_with_text[0]:
                    best_with_text = item
            elif best_empty is None:
                best_empty = (rec, parsed, len(parse_inputs), variant.name)

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
                return _EvaluatedCandidate(
                    candidate=self._empty_candidate_for_internal_use(),
                    recognition=rotated_rec,
                    parsed=rotated_parsed,
                    parse_inputs_count=len(rotated_inputs),
                    recognition_variant="rotate_180",
                    normalized_crop=rotated_crop,
                    original_color_fallback_used=True,
                )

        return _EvaluatedCandidate(
            candidate=self._empty_candidate_for_internal_use(),
            recognition=rec,
            parsed=parsed,
            parse_inputs_count=count,
            recognition_variant=variant_name,
            normalized_crop=crop,
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

    def _evaluate_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate, *, today: date) -> _EvaluatedCandidate:
        crops = self._normalized_crops_for_candidate(image, candidate)
        best: tuple[tuple[int, int, float, float, float], _EvaluatedCandidate] | None = None
        for crop in crops:
            if crop.image.size == 0:
                continue
            selected_variant_image = candidate.recognition_variant_image if crop is crops[0] else None
            evaluated = self._best_recognition_for_single_crop(
                crop,
                today=today,
                selected_variant_name=candidate.selected_recognition_variant or None,
                selected_variant_image=selected_variant_image,
            )
            evaluated.candidate = candidate
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

        scores = self._probe_signal_scores(best_eval.recognition, candidate=candidate, today=today)
        candidate.probe_text = best_eval.recognition.raw_text
        candidate.probe_normalized_text = best_eval.recognition.normalized_text
        candidate.probe_confidence = best_eval.recognition.confidence
        candidate.probe_reason = best_eval.recognition.reason
        candidate.selected_recognition_variant = best_eval.recognition_variant
        candidate.score_breakdown.update(scores)
        candidate.total_score = candidate.geometry_score + sum(scores.values())
        return best_eval

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

    def _score_parseable_candidate(self, evaluated: _EvaluatedCandidate) -> tuple[int, int, int, int, int, float, float, float, float]:
        text = " ".join(part for part in (evaluated.recognition.normalized_text, evaluated.recognition.raw_text) if part)
        role = score_date_role(text)
        return (
            role.priority,
            role.non_production,
            self._date_precision_score(evaluated.parsed),
            evaluated.parsed.parsed_date.toordinal() if evaluated.parsed.parsed_date is not None else 0,
            self._text_evidence_priority(text, parser_bonus=1.0),
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
        return (
            f"{evaluated.parsed.reason};candidate_id={evaluated.candidate.candidate_id};"
            f"candidate_type={evaluated.candidate.candidate_type};detector_sources={sources};"
            f"detector_variant={evaluated.candidate.detector_variant};variant={evaluated.recognition_variant};"
            f"orientation={orientation};final_rank={evaluated.candidate.final_rank_after_force_include};"
            f"force_include={evaluated.candidate.force_included_reason};parse_inputs={evaluated.parse_inputs_count}"
        )

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
        return [self._evaluate_candidate(image, candidate.ranked, today=today) for candidate in final_candidates]

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
        return [self._evaluate_candidate(candidate.image, candidate.ranked, today=today) for candidate in final_candidates], None

    def _best_parseable_candidate(self, evaluated: list[_EvaluatedCandidate]) -> _EvaluatedCandidate | None:
        parseable = [
            item
            for item in evaluated
            if item.parsed.parsed_date is not None and not self._should_suppress_auto_parse(item)
        ]
        if not parseable:
            return None
        return max(parseable, key=self._score_parseable_candidate)

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
            )

        yolo_candidates, detection_reason = self._detect_candidates(image)
        if self.rapidocr_primary_enabled:
            evaluated, rapidocr_reason = self._evaluate_rapidocr_primary_flow(image, yolo_candidates, today=today)
            if rapidocr_reason and detection_reason:
                detection_reason = f"{detection_reason}; {rapidocr_reason}"
            elif rapidocr_reason:
                detection_reason = rapidocr_reason
        else:
            ranked = self._build_ranked_candidates(image, yolo_candidates)
            evaluated = self._evaluate_architecture_candidates(image, ranked, today=today)
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
        if not any(item.parsed.parsed_date is not None for item in evaluated):
            rescue_candidates, rescue_reason = self._detect_variant_rescue_candidates(image)
            if rescue_candidates:
                rescue_ranked = self._build_ranked_candidates(image, rescue_candidates)
                evaluated.extend(self._evaluate_architecture_candidates(image, rescue_ranked, today=today))
            elif rescue_reason and detection_reason:
                detection_reason = f"{detection_reason}; {rescue_reason}"
            elif rescue_reason:
                detection_reason = rescue_reason
        if yolo_candidates and not any(item.parsed.parsed_date is not None for item in evaluated):
            fallback_ranked = self._build_ranked_candidates(image, [])
            evaluated.extend(self._evaluate_architecture_candidates(image, fallback_ranked, today=today))

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
            )

        parseable = [
            item
            for item in evaluated
            if item.parsed.parsed_date is not None and not self._should_suppress_auto_parse(item)
        ]
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
            )

        best = max(parseable, key=self._score_parseable_candidate)
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
        )
