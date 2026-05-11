from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ai.crop_normalization import TextLineCropConfig, TextLineGeometry, normalize_textline_crop
from app.ai.parser import ExpiryDateParser
from app.infra.settings import PROJECT_ROOT


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
class _ScoredCandidate:
    yolo: _YoloCandidate
    recognition: _RecognitionOutput
    parsed_date: date | None
    parse_confidence: float
    parse_reason: str


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
    ) -> None:
        self.detector_model_path = detector_model_path
        self.detector_confidence_threshold = detector_confidence_threshold
        self.detector_imgsz = detector_imgsz
        self.max_candidates = max(1, int(max_candidates))
        self.crop_padding_px = max(0, int(crop_padding_px))
        self.parser = ExpiryDateParser(min_candidate_confidence=parser_min_candidate_confidence)
        self.recognizer = SVTRTextRecognizer(
            model_name=svtr_model_name,
            model_dir=svtr_model_dir,
            device=svtr_device,
        )
        self._detector_model: Any | None = None
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
    def _rotation_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
        return [
            ("original", image),
            ("rotate_90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            ("rotate_180", cv2.rotate(image, cv2.ROTATE_180)),
            ("rotate_90_ccw", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

    def _crop_candidate(self, image: np.ndarray, candidate: _YoloCandidate) -> np.ndarray:
        crop = normalize_textline_crop(
            image,
            TextLineGeometry(
                bbox_xyxy=candidate.bbox_xyxy,
                polygon_xy=tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
            ),
            TextLineCropConfig(padding_px=self.crop_padding_px),
        )
        return crop.image

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

        candidates, detection_reason = self._detect_candidates(image)
        if not candidates:
            return MobileExpiryPipelineResult(
                status="manual_review_required",
                detected_expiry_date=None,
                raw_text=None,
                normalized_text=None,
                recognition_confidence=None,
                detector_confidence=None,
                reason=detection_reason,
                detection_polygon_json=None,
                runtime_ms=int((perf_counter() - started) * 1000),
            )

        scored: list[_ScoredCandidate] = []
        recognition_reasons: list[str] = []
        for candidate in candidates:
            crop = self._crop_candidate(image, candidate)
            if crop.size == 0:
                recognition_reasons.append("candidate crop empty")
                continue
            for rotation_name, rotated in self._rotation_variants(crop):
                rec = self.recognizer.recognize(rotated)
                rec.rotation = rotation_name
                if rec.reason:
                    recognition_reasons.append(rec.reason)
                parsed = self.parser.parse(rec.normalized_text or rec.raw_text, reference_date=today)
                scored.append(
                    _ScoredCandidate(
                        yolo=candidate,
                        recognition=rec,
                        parsed_date=parsed.parsed_date,
                        parse_confidence=parsed.confidence,
                        parse_reason=parsed.reason,
                    )
                )

        if not scored:
            return MobileExpiryPipelineResult(
                status="manual_review_required",
                detected_expiry_date=None,
                raw_text=None,
                normalized_text=None,
                recognition_confidence=None,
                detector_confidence=candidates[0].confidence,
                reason="; ".join(dict.fromkeys(recognition_reasons)) or "recognizer returned no candidates",
                detection_polygon_json=candidates[0].polygon_xy,
                runtime_ms=int((perf_counter() - started) * 1000),
            )

        def rank_key(item: _ScoredCandidate) -> tuple[int, float, float, float]:
            return (
                1 if item.parsed_date is not None else 0,
                float(item.parse_confidence or 0.0),
                float(item.recognition.confidence or 0.0),
                float(item.yolo.confidence or 0.0),
            )

        best = max(scored, key=rank_key)
        if best.parsed_date is None:
            return MobileExpiryPipelineResult(
                status="manual_review_required",
                detected_expiry_date=None,
                raw_text=best.recognition.raw_text or None,
                normalized_text=best.recognition.normalized_text or None,
                recognition_confidence=best.recognition.confidence,
                detector_confidence=best.yolo.confidence,
                reason=best.recognition.reason or best.parse_reason or "no valid date parsed",
                detection_polygon_json=best.yolo.polygon_xy,
                runtime_ms=int((perf_counter() - started) * 1000),
            )

        return MobileExpiryPipelineResult(
            status="parsed_success",
            detected_expiry_date=best.parsed_date,
            raw_text=best.recognition.raw_text,
            normalized_text=best.recognition.normalized_text,
            recognition_confidence=best.recognition.confidence,
            detector_confidence=best.yolo.confidence,
            reason=f"{best.parse_reason};rotation={best.recognition.rotation}",
            detection_polygon_json=best.yolo.polygon_xy,
            runtime_ms=int((perf_counter() - started) * 1000),
        )
