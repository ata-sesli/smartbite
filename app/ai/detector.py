from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ai.types import BoundingBox, DetectionResult


class ExpiryRegionDetector:
    def __init__(self, model_path: Path, confidence_threshold: float = 0.15) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_error: str | None = None

    def _resolve_model_source(self) -> str | None:
        ref = str(self.model_path).strip()
        if not ref:
            self._load_error = "detector model path is empty"
            return None

        candidate = Path(ref)
        if candidate.exists():
            return str(candidate)

        # Ultralytics supports named model refs (for example: yolo26n.pt) and can
        # auto-download them. We allow that when the ref is not an explicit filepath.
        if candidate.is_absolute() or "/" in ref or "\\" in ref:
            self._load_error = f"detector model not found: {candidate}"
            return None
        return ref

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        model_source = self._resolve_model_source()
        if model_source is None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(model_source)
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            self._load_error = f"failed to load detector model: {exc}"

    def detect(self, image: np.ndarray) -> DetectionResult:
        self._ensure_model()
        if self._load_error:
            return DetectionResult(False, [], None, None, self._load_error)

        if self._model is None:
            return DetectionResult(False, [], None, None, "detector unavailable")

        try:
            results = self._model.predict(image, verbose=False)
            if not results:
                return DetectionResult(False, [], None, None, "detector returned no result")
            boxes_raw = results[0].boxes
            if boxes_raw is None or len(boxes_raw) == 0:
                return DetectionResult(False, [], None, None, "no expiry region detected")

            candidates: list[BoundingBox] = []
            for row in boxes_raw:
                conf = float(row.conf.item())
                if conf < self.confidence_threshold:
                    continue
                xyxy = row.xyxy[0].tolist()
                candidates.append(
                    BoundingBox(
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3]),
                        confidence=conf,
                    )
                )
            if not candidates:
                return DetectionResult(False, [], None, None, "no candidate passed confidence threshold")

            candidates.sort(key=lambda b: b.confidence, reverse=True)
            best = candidates[0]
            return DetectionResult(True, candidates, best, best.confidence, None)
        except Exception as exc:  # pragma: no cover - model runtime path
            return DetectionResult(False, [], None, None, f"detector inference error: {exc}")
