from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ai.types import OCRResultData


def normalize_ocr_text(raw_text: str) -> str:
    return " ".join(raw_text.strip().upper().split())


def detect_runtime_device(mode: str) -> str:
    explicit = mode.lower()
    if explicit in {"cpu", "mps", "cuda"}:
        return explicit

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass(slots=True)
class OCRConfig:
    onnx_model_path: Path
    onnx_charset_path: Path | None
    paddle_lang: str
    device_mode: str


class ONNXLiteOCRRunner:
    def __init__(self, model_path: Path, charset_path: Path | None) -> None:
        self.model_path = model_path
        self.charset_path = charset_path
        self._session = None
        self._load_error: str | None = None
        self._charset: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._session is not None or self._load_error is not None:
            return
        if not self.model_path.exists():
            self._load_error = f"onnx model not found: {self.model_path}"
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            if self.charset_path and self.charset_path.exists():
                self._charset = [line.strip() for line in self.charset_path.read_text().splitlines() if line.strip()]
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to initialize onnx runner: {exc}"

    def _decode(self, logits: np.ndarray) -> str:
        token_ids = logits.argmax(axis=-1).tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        text_tokens: list[str] = []
        last = -1
        for token in token_ids:
            if token == last:
                continue
            last = token
            if token == 0:
                continue
            if self._charset and token - 1 < len(self._charset):
                text_tokens.append(self._charset[token - 1])
            else:
                text_tokens.append(str(token))
        return "".join(text_tokens)

    def run(self, image: np.ndarray) -> OCRResultData:
        self._ensure_loaded()
        if self._load_error:
            return OCRResultData("", "", None, "onnx_lite", "cpu", self._load_error)
        if self._session is None:
            return OCRResultData("", "", None, "onnx_lite", "cpu", "onnx runner unavailable")

        try:
            import cv2

            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            resized = cv2.resize(gray, (320, 48), interpolation=cv2.INTER_LINEAR)
            input_tensor = resized.astype("float32") / 255.0
            input_tensor = np.expand_dims(np.expand_dims(input_tensor, axis=0), axis=0)

            input_name = self._session.get_inputs()[0].name
            output = self._session.run(None, {input_name: input_tensor})[0]
            raw_text = self._decode(output)
            normalized = normalize_ocr_text(raw_text)
            return OCRResultData(raw_text, normalized, None, "onnx_lite", "cpu", None)
        except Exception as exc:  # pragma: no cover
            return OCRResultData("", "", None, "onnx_lite", "cpu", f"onnx inference failed: {exc}")


class PaddleFullOCRRunner:
    def __init__(self, language: str, runtime_device: str) -> None:
        self.language = language
        self.runtime_device = runtime_device
        self._model = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        try:
            from paddleocr import PaddleOCR

            use_gpu = self.runtime_device in {"cuda", "mps"}
            self._model = PaddleOCR(use_angle_cls=True, lang=self.language, use_gpu=use_gpu)
        except Exception as exc:  # pragma: no cover
            self._load_error = f"failed to initialize paddleocr: {exc}"

    def run(self, image: np.ndarray) -> OCRResultData:
        self._ensure_loaded()
        if self._load_error:
            return OCRResultData("", "", None, "paddle_full", self.runtime_device, self._load_error)
        if self._model is None:
            return OCRResultData("", "", None, "paddle_full", self.runtime_device, "paddle runner unavailable")

        try:
            result = self._model.ocr(image, cls=True)
            lines = result[0] if result else []
            texts: list[str] = []
            confs: list[float] = []
            for line in lines:
                if len(line) < 2:
                    continue
                txt, conf = line[1]
                if txt:
                    texts.append(txt)
                    confs.append(float(conf))

            raw_text = " ".join(texts)
            normalized = normalize_ocr_text(raw_text)
            confidence = sum(confs) / len(confs) if confs else None
            reason = None if raw_text else "paddleocr returned empty text"
            return OCRResultData(raw_text, normalized, confidence, "paddle_full", self.runtime_device, reason)
        except Exception as exc:  # pragma: no cover
            return OCRResultData("", "", None, "paddle_full", self.runtime_device, f"paddle inference failed: {exc}")


class OCRRouter:
    def __init__(self, config: OCRConfig) -> None:
        self.runtime_device = detect_runtime_device(config.device_mode)
        self._onnx_runner = ONNXLiteOCRRunner(config.onnx_model_path, config.onnx_charset_path)
        self._paddle_runner = PaddleFullOCRRunner(config.paddle_lang, self.runtime_device)

    def run(self, roi_image: np.ndarray) -> OCRResultData:
        if self.runtime_device == "cpu":
            return self._onnx_runner.run(roi_image)
        return self._paddle_runner.run(roi_image)
