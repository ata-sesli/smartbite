from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.ai.runtime_device import resolve_runtime_device


GENERAL_TEXT_ENGINE_NAME = "ppocrv5_general_text"
DEFAULT_TEXT_DETECTION_MODEL = "PP-OCRv5_server_det"
DEFAULT_TEXT_RECOGNITION_MODEL = "PP-OCRv5_server_rec"


@dataclass(slots=True)
class GeneralTextLine:
    text: str
    confidence: float | None
    bbox_xyxy: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneralTextExtraction:
    lines: list[GeneralTextLine]
    full_text: str
    confidence: float | None
    engine_name: str
    runtime_device: str
    reason: str | None
    inference_ms: int


class PPOCRV5GeneralTextRunner:
    def __init__(
        self,
        *,
        device_mode: str = "auto",
        text_detection_model_name: str = DEFAULT_TEXT_DETECTION_MODEL,
        text_recognition_model_name: str = DEFAULT_TEXT_RECOGNITION_MODEL,
        max_image_side: int = 1600,
    ) -> None:
        requested_device = resolve_runtime_device(device_mode)
        self.runtime_device = "cuda" if requested_device == "cuda" else "cpu"
        self._paddle_device = "gpu:0" if self.runtime_device == "cuda" else "cpu"
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.max_image_side = max_image_side
        self.engine_name = GENERAL_TEXT_ENGINE_NAME
        self._model: Any | None = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"ppocrv5 general OCR import failed: {exc}"
            return

        try:
            self._model = PaddleOCR(
                text_detection_model_name=self.text_detection_model_name,
                text_recognition_model_name=self.text_recognition_model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device=self._paddle_device,
            )
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"ppocrv5 general OCR initialization failed: {exc}"

    @staticmethod
    def _result_payload(item: object) -> dict[str, Any] | None:
        if isinstance(item, dict):
            if isinstance(item.get("res"), dict):
                return item["res"]
            return item
        res = getattr(item, "res", None)
        if isinstance(res, dict):
            return res
        json_value = getattr(item, "json", None)
        if isinstance(json_value, dict):
            payload = json_value.get("res")
            return payload if isinstance(payload, dict) else json_value
        return None

    @staticmethod
    def _bbox_from_poly(poly: object) -> list[int] | None:
        if poly is None:
            return None
        try:
            arr = np.asarray(poly, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 1 and arr.size >= 8:
            arr = arr.reshape(-1, 2)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [
            int(np.floor(xs.min())),
            int(np.floor(ys.min())),
            int(np.ceil(xs.max())),
            int(np.ceil(ys.max())),
        ]

    @classmethod
    def parse_result(cls, raw_result: object) -> list[GeneralTextLine]:
        if raw_result is None:
            return []
        items = raw_result if isinstance(raw_result, list) else [raw_result]
        lines: list[GeneralTextLine] = []
        for item in items:
            payload = cls._result_payload(item)
            if payload is None:
                continue

            texts = payload.get("rec_texts") or payload.get("texts") or []
            scores = payload.get("rec_scores") or payload.get("scores") or []
            polys = payload.get("rec_polys") or payload.get("dt_polys") or payload.get("polys") or []
            for index, text_raw in enumerate(texts):
                text = str(text_raw or "").strip()
                if not text:
                    continue
                confidence: float | None = None
                if index < len(scores):
                    try:
                        confidence = float(scores[index])
                    except Exception:
                        confidence = None
                bbox = cls._bbox_from_poly(polys[index]) if index < len(polys) else None
                lines.append(GeneralTextLine(text=text, confidence=confidence, bbox_xyxy=bbox))

        def sort_key(line: GeneralTextLine) -> tuple[int, int, str]:
            bbox = line.bbox_xyxy
            if bbox is None:
                return (10**9, 10**9, line.text)
            return (bbox[1], bbox[0], line.text)

        lines.sort(key=sort_key)
        return lines

    @staticmethod
    def join_lines(lines: list[GeneralTextLine]) -> str:
        return "\n".join(line.text for line in lines if line.text.strip())

    @staticmethod
    def average_confidence(lines: list[GeneralTextLine]) -> float | None:
        scores = [line.confidence for line in lines if line.confidence is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    @staticmethod
    def _image_to_array(self, image_bytes: bytes) -> np.ndarray:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            longest_side = max(image.size)
            if self.max_image_side > 0 and longest_side > self.max_image_side:
                scale = self.max_image_side / float(longest_side)
                size = (
                    max(1, int(round(image.width * scale))),
                    max(1, int(round(image.height * scale))),
                )
                image = image.resize(size, Image.Resampling.LANCZOS)
            return np.asarray(image)

    def extract(self, image_bytes: bytes) -> GeneralTextExtraction:
        start = perf_counter()
        self._ensure_loaded()
        if self._load_error is not None:
            return GeneralTextExtraction(
                lines=[],
                full_text="",
                confidence=None,
                engine_name=self.engine_name,
                runtime_device=self.runtime_device,
                reason=self._load_error,
                inference_ms=int((perf_counter() - start) * 1000),
            )
        if self._model is None:
            return GeneralTextExtraction(
                lines=[],
                full_text="",
                confidence=None,
                engine_name=self.engine_name,
                runtime_device=self.runtime_device,
                reason="ppocrv5 general OCR unavailable",
                inference_ms=int((perf_counter() - start) * 1000),
            )

        try:
            image = self._image_to_array(image_bytes)
            raw_result = self._model.predict(image)
            lines = self.parse_result(raw_result)
            full_text = self.join_lines(lines)
            reason = None if full_text else "ppocrv5 general OCR returned no text"
            return GeneralTextExtraction(
                lines=lines,
                full_text=full_text,
                confidence=self.average_confidence(lines),
                engine_name=self.engine_name,
                runtime_device=self.runtime_device,
                reason=reason,
                inference_ms=int((perf_counter() - start) * 1000),
            )
        except Exception as exc:  # pragma: no cover - runtime inference path
            return GeneralTextExtraction(
                lines=[],
                full_text="",
                confidence=None,
                engine_name=self.engine_name,
                runtime_device=self.runtime_device,
                reason=f"ppocrv5 general OCR failed: {exc}",
                inference_ms=int((perf_counter() - start) * 1000),
            )
