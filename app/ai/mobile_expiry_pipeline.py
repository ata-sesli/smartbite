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
from app.ai.onnx_inference import SVTROnnxTextRecognizer, YoloObbOnnxDetector
from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ImageVariant, ROIImagePreprocessor
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
MOBILE_RECOGNITION_VARIANTS = ("original", "clahe_gray", "adaptive_binary", "unsharp_gray")
MOBILE_ENHANCED_RECOGNITION_VARIANTS = ("clahe_gray", "adaptive_binary", "unsharp_gray")


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


@dataclass(slots=True)
class _YoloCandidate:
    polygon_xy: list[list[float]]
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


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
    member_indices: list[int]
    member_bboxes: list[tuple[int, int, int, int]]
    geometry_score: float
    score_breakdown: dict[str, float]
    total_score: float
    probe_text: str = ""
    probe_normalized_text: str = ""
    probe_confidence: float | None = None
    probe_reason: str | None = None
    selected_recognition_variant: str = ""


@dataclass(slots=True)
class _EvaluatedCandidate:
    candidate: _MobileRankedCandidate
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int
    recognition_variant: str
    normalized_crop: NormalizedTextLineCrop | None
    original_color_fallback_used: bool = False


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
    ) -> None:
        self.detector_model_path = detector_model_path
        self.detector_backend = detector_backend.lower().strip()
        self.detector_onnx_path = detector_onnx_path
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_imgsz = detector_imgsz
        self.max_candidates = max(1, int(max_candidates))
        self.crop_padding_px = max(0, int(crop_padding_px))
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

    def _detect_candidates(self, image: np.ndarray) -> tuple[list[_YoloCandidate], str | None]:
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
            candidates.append(_YoloCandidate(polygon_xy=cleaned, bbox_xyxy=bbox, confidence=conf))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[: self.max_candidates], None if candidates else "no candidate passed confidence threshold"

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
    ) -> _MobileRankedCandidate:
        h, w = image_shape
        bw = max(1, bbox[2] - bbox[0])
        bh = max(1, bbox[3] - bbox[1])
        area_ratio = float(bw * bh) / float(max(1, h * w))
        aspect = float(bw) / float(bh)
        edge_dist = min(bbox[0], bbox[1], max(0, w - bbox[2]), max(0, h - bbox[3]))
        edge_proximity = float(edge_dist) / float(max(1, min(h, w)))
        confidences = [float(yolo_candidates[idx].confidence) for idx in member_indices if idx < len(yolo_candidates)]
        det_conf_avg = sum(confidences) / len(confidences) if confidences else 0.0

        size_score = self._geometry_size_score(area_ratio)
        aspect_score = self._geometry_aspect_score(aspect)
        edge_score = self._geometry_edge_score(edge_proximity)
        det_score = (det_conf_avg - 0.5) * 0.25 if confidences else 0.0
        group_bonus = 0.1 if candidate_type == "group" else 0.0
        line_bonus = 0.12 if candidate_type == "line" else 0.0
        fallback_bonus = 0.05 if candidate_type == "fallback" else 0.0
        geometry_score = size_score + aspect_score + edge_score + det_score + group_bonus + line_bonus + fallback_bonus

        return _MobileRankedCandidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            bbox_xyxy=bbox,
            polygon_xy=polygon_xy,
            detector_confidence=det_conf_avg if confidences else None,
            member_indices=member_indices,
            member_bboxes=[yolo_candidates[idx].bbox_xyxy for idx in member_indices if idx < len(yolo_candidates)],
            geometry_score=geometry_score,
            score_breakdown={
                "size_score": size_score,
                "aspect_score": aspect_score,
                "edge_score": edge_score,
                "det_score": det_score,
                "group_bonus": group_bonus,
                "line_bonus": line_bonus,
                "fallback_bonus": fallback_bonus,
            },
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

    def _build_ranked_candidates(self, image: np.ndarray, yolo_candidates: list[_YoloCandidate]) -> list[_MobileRankedCandidate]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []
        if not yolo_candidates:
            return [
                self._candidate_from_bbox(
                    candidate_id="fallback_full",
                    candidate_type="fallback",
                    bbox=(0, 0, w, h),
                    image_shape=(h, w),
                    yolo_candidates=[],
                    member_indices=[],
                    polygon_xy=None,
                )
            ]

        candidates: list[_MobileRankedCandidate] = []
        seen: set[tuple[int, int, int, int, tuple[int, ...], str]] = set()

        def add(candidate_type: str, bbox: tuple[int, int, int, int], member_indices: list[int], polygon: tuple[tuple[float, float], ...] | None) -> None:
            key = (*bbox, tuple(member_indices), candidate_type)
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                self._candidate_from_bbox(
                    candidate_id=f"cand_{len(candidates) + 1}",
                    candidate_type=candidate_type,
                    bbox=bbox,
                    image_shape=(h, w),
                    yolo_candidates=yolo_candidates,
                    member_indices=member_indices,
                    polygon_xy=polygon,
                )
            )

        for idx, yolo in enumerate(yolo_candidates):
            polygon = tuple((float(x), float(y)) for x, y in yolo.polygon_xy)
            add("single", yolo.bbox_xyxy, [idx], polygon)

        for line in self._cluster_yolo_boxes_into_lines(yolo_candidates):
            if len(line) >= 2:
                add("line", self._union_many_bboxes([yolo_candidates[idx].bbox_xyxy for idx in line]), line, None)

        group_count = 0
        for i in range(len(yolo_candidates)):
            if group_count >= max(4, self.max_candidates):
                break
            neighbors = [
                (self._bbox_area(self._union_bbox(yolo_candidates[i].bbox_xyxy, yolo_candidates[j].bbox_xyxy)), j)
                for j in range(len(yolo_candidates))
                if j != i and self._boxes_are_groupable(yolo_candidates[i].bbox_xyxy, yolo_candidates[j].bbox_xyxy)
            ]
            neighbors.sort(key=lambda item: item[0])
            for _area, j in neighbors[:2]:
                if i >= j:
                    continue
                add("group", self._union_bbox(yolo_candidates[i].bbox_xyxy, yolo_candidates[j].bbox_xyxy), [i, j], None)
                group_count += 1

        candidates.sort(key=lambda item: item.geometry_score, reverse=True)
        for idx, candidate in enumerate(candidates, start=1):
            candidate.candidate_id = f"cand_{idx}"
        return candidates[: max(self.max_candidates, self.max_candidates * 2)]

    def _normalized_crops_for_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate) -> list[NormalizedTextLineCrop]:
        return normalize_textline_crops(
            image,
            TextLineGeometry(
                bbox_xyxy=candidate.bbox_xyxy,
                polygon_xy=candidate.polygon_xy,
            ),
            TextLineCropConfig(padding_px=self.crop_padding_px),
        )

    def _recognition_variants_for_crop(
        self,
        crop: np.ndarray,
        *,
        allowed_names: tuple[str, ...] | None = MOBILE_RECOGNITION_VARIANTS,
    ) -> list[ImageVariant]:
        allowed = set(allowed_names or MOBILE_RECOGNITION_VARIANTS)
        variants: list[ImageVariant] = []
        if "original" in allowed:
            variants.append(ImageVariant("original", crop, purpose="recognition"))
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
        return sum(ch.isdigit() or ch in {"/", "-", ".", ":"} for ch in compact) / len(compact)

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
        base = cls._normalize_ocr_confusions(recognition.normalized_text or "")
        raw = cls._normalize_ocr_confusions(recognition.raw_text or "")
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
        tokens = [token for token in raw.split() if any(ch.isdigit() for ch in token)]
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
            parsed = self.parser.parse(text, reference_date=today)
            if parsed.parsed_date is not None:
                parsed_inputs.append((text, parsed))
        if parsed_inputs:
            return max(parsed_inputs, key=lambda item: self._parse_input_selection_key(item[0], item[1]))[1]
        return self.parser.parse(parse_inputs[0] if parse_inputs else "", reference_date=today)

    @staticmethod
    def _parse_input_selection_key(text: str, parsed: ParsedDateData) -> tuple[int, int, int, int, float]:
        return role_selection_key(evidence=score_date_role(text), parsed_date=parsed.parsed_date, specificity=1, confidence=parsed.confidence)

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

    def _best_recognition_for_single_crop(
        self,
        crop: NormalizedTextLineCrop,
        *,
        today: date,
    ) -> _EvaluatedCandidate:
        best_parseable: tuple[tuple[float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_with_text: tuple[tuple[float, float, float], _RecognitionOutput, ParsedDateData, int, str] | None = None
        best_empty: tuple[_RecognitionOutput, ParsedDateData, int, str] | None = None

        for variant in self._recognition_variants_for_crop(crop.image, allowed_names=MOBILE_RECOGNITION_VARIANTS):
            rec = self._recognize_variant(variant.image, variant_name=variant.name, orientation=crop.selected_orientation)
            parse_inputs = self._build_parse_inputs(rec)
            parsed = self._best_parse_for_inputs(parse_inputs, today=today)
            if parsed.parsed_date is not None:
                score = (float(parsed.confidence), float(rec.confidence or 0.0), self._score_date_likeness(rec.raw_text))
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
                float(rotated_parsed.confidence if rotated_parsed.parsed_date is not None else 0.0),
                self._score_date_likeness(rotated_rec.raw_text),
                float(rotated_rec.confidence or 0.0),
            )
            current_score = (
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
            member_indices=[],
            member_bboxes=[],
            geometry_score=0.0,
            score_breakdown={},
            total_score=0.0,
        )

    def _evaluate_candidate(self, image: np.ndarray, candidate: _MobileRankedCandidate, *, today: date) -> _EvaluatedCandidate:
        crops = self._normalized_crops_for_candidate(image, candidate)
        best: tuple[tuple[float, float, float], _EvaluatedCandidate] | None = None
        for crop in crops:
            if crop.image.size == 0:
                continue
            evaluated = self._best_recognition_for_single_crop(crop, today=today)
            evaluated.candidate = candidate
            score = (
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

    def _score_parseable_candidate(self, evaluated: _EvaluatedCandidate) -> tuple[int, int, int, int, float, float, float, float]:
        text = " ".join(part for part in (evaluated.recognition.normalized_text, evaluated.recognition.raw_text) if part)
        role = score_date_role(text)
        return (
            role.priority,
            role.non_production,
            evaluated.parsed.parsed_date.toordinal() if evaluated.parsed.parsed_date is not None else 0,
            self._text_evidence_priority(text, parser_bonus=1.0),
            evaluated.parsed.confidence,
            evaluated.candidate.total_score,
            self._score_ocr_candidate(evaluated.recognition),
            self._score_date_likeness(evaluated.recognition.raw_text),
        )

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
    def _result_reason(evaluated: _EvaluatedCandidate) -> str:
        orientation = evaluated.normalized_crop.selected_orientation if evaluated.normalized_crop is not None else evaluated.recognition.rotation
        return (
            f"{evaluated.parsed.reason};candidate_id={evaluated.candidate.candidate_id};"
            f"candidate_type={evaluated.candidate.candidate_type};variant={evaluated.recognition_variant};"
            f"orientation={orientation};parse_inputs={evaluated.parse_inputs_count}"
        )

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
            )

        yolo_candidates, detection_reason = self._detect_candidates(image)
        ranked = self._build_ranked_candidates(image, yolo_candidates)
        evaluated = [self._evaluate_candidate(image, candidate, today=today) for candidate in ranked]
        if yolo_candidates and not any(item.parsed.parsed_date is not None for item in evaluated):
            fallback_ranked = self._build_ranked_candidates(image, [])
            evaluated.extend(self._evaluate_candidate(image, candidate, today=today) for candidate in fallback_ranked)

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
            )

        parseable = [item for item in evaluated if item.parsed.parsed_date is not None]
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
                detection_polygon_json=self._polygon_json(best_unparseable.candidate),
                runtime_ms=int((perf_counter() - started) * 1000),
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
            detection_polygon_json=self._polygon_json(best.candidate),
            runtime_ms=int((perf_counter() - started) * 1000),
        )
