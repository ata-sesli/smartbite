from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.infra.settings import PROJECT_ROOT


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_text(raw_text: str) -> str:
    return " ".join(raw_text.strip().upper().split())


def _simple_character_dict_parse(text: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    in_character_dict = False
    for line in lines:
        stripped = line.strip()
        if stripped == "character_dict:":
            in_character_dict = True
            continue
        if in_character_dict and stripped.startswith("- "):
            value = stripped[2:].strip()
            if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            out.append(value)
            continue
        if in_character_dict and stripped and not stripped.startswith("- "):
            break
    return out


def load_svtr_character_dict(model_dir: Path) -> list[str]:
    yml_path = _resolve_path(model_dir) / "inference.yml"
    if not yml_path.exists():
        raise FileNotFoundError(f"SVTR inference.yml not found: {yml_path}")
    text = yml_path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
        postprocess = payload.get("PostProcess") if isinstance(payload, dict) else None
        chars = postprocess.get("character_dict") if isinstance(postprocess, dict) else None
        if isinstance(chars, list) and all(isinstance(item, str) for item in chars):
            return list(chars)
    except Exception:
        pass
    chars = _simple_character_dict_parse(text)
    if not chars:
        raise ValueError(f"SVTR character_dict not found in {yml_path}")
    return chars


def build_svtr_ctc_character_dict(model_dir: Path) -> list[str]:
    chars = load_svtr_character_dict(model_dir)
    if " " not in chars:
        chars = [*chars, " "]
    return chars


def prepare_svtr_input(
    image: np.ndarray,
    *,
    image_shape: tuple[int, int, int] = (3, 48, 320),
    max_width: int = 3200,
) -> np.ndarray:
    channels, target_h, base_w = image_shape
    if channels != 3:
        raise ValueError("SVTR ONNX preprocessing currently supports 3-channel recognition models")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unsupported SVTR input image shape: {image.shape}")

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("SVTR input image is empty")
    wh_ratio = w / float(h)
    max_wh_ratio = max(base_w / float(target_h), wh_ratio)
    target_w = min(max_width, int(target_h * max_wh_ratio))
    resized_w = min(target_w, max(1, int(np.ceil(target_h * (w / float(h))))))
    resized = cv2.resize(image, (resized_w, target_h), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - 0.5) / 0.5
    chw = np.transpose(normalized, (2, 0, 1))
    padded = np.zeros((channels, target_h, target_w), dtype=np.float32)
    padded[:, :, :resized_w] = chw
    return padded[np.newaxis, :]


def decode_ctc(output: np.ndarray, character_dict: list[str]) -> tuple[str, float | None]:
    probs = np.asarray(output)
    if probs.ndim == 3:
        probs = probs[0]
    if probs.ndim != 2:
        raise ValueError(f"expected SVTR output with shape (T, C) or (1, T, C), got {output.shape}")

    indexes = probs.argmax(axis=1).astype(int).tolist()
    max_probs = probs.max(axis=1).astype(np.float32).tolist()
    chars: list[str] = []
    confidences: list[float] = []
    previous = -1
    for index, confidence in zip(indexes, max_probs, strict=False):
        if index != previous and index > 0:
            char_index = index - 1
            if char_index < len(character_dict):
                chars.append(character_dict[char_index])
                confidences.append(float(confidence))
        previous = index
    if not chars:
        return "", None
    return "".join(chars), float(np.mean(np.asarray(confidences, dtype=np.float32)))


@dataclass(slots=True)
class OnnxRecognitionOutput:
    raw_text: str
    normalized_text: str
    confidence: float | None
    reason: str | None
    rotation: str = "original"


class SVTROnnxTextRecognizer:
    def __init__(
        self,
        *,
        model_path: Path,
        paddle_model_dir: Path,
        runtime_device: str,
        engine_name: str = "svtrv2_onnx",
    ) -> None:
        self.model_path = model_path
        self.paddle_model_dir = paddle_model_dir
        self.runtime_device = runtime_device
        self.engine_name = engine_name
        self.model_name = "svtrv2_onnx"
        self.character_dict: list[str] | None = None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._load_error: str | None = None

    def _providers(self) -> list[str]:
        if self.runtime_device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _ensure_loaded(self) -> None:
        if self._session is not None or self._load_error is not None:
            return
        model_path = _resolve_path(self.model_path)
        if not model_path.exists():
            self._load_error = f"SVTR ONNX model not found: {model_path}"
            return
        try:
            import onnxruntime as ort

            self.character_dict = build_svtr_ctc_character_dict(self.paddle_model_dir)
            self._session = ort.InferenceSession(str(model_path), providers=self._providers())
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"SVTR ONNX recognizer initialization failed: {exc}"

    def recognize(self, image: np.ndarray) -> OnnxRecognitionOutput:
        self._ensure_loaded()
        if self._load_error:
            return OnnxRecognitionOutput("", "", None, self._load_error)
        if self._session is None or self._input_name is None or self.character_dict is None:
            return OnnxRecognitionOutput("", "", None, "SVTR ONNX recognizer unavailable")
        try:
            tensor = prepare_svtr_input(image)
            outputs = self._session.run(None, {self._input_name: tensor})
            if not outputs:
                return OnnxRecognitionOutput("", "", None, "SVTR ONNX recognizer returned no output")
            raw, confidence = decode_ctc(np.asarray(outputs[0]), self.character_dict)
        except Exception as exc:  # pragma: no cover - runtime inference path
            return OnnxRecognitionOutput("", "", None, f"SVTR ONNX recognition failed: {exc}")
        if not raw:
            return OnnxRecognitionOutput("", "", confidence, "SVTR ONNX recognizer returned empty text")
        return OnnxRecognitionOutput(raw, _normalize_text(raw), confidence, None)

    def runtime_info(self) -> dict[str, object]:
        self._ensure_loaded()
        return {
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "paddle_model_dir": str(self.paddle_model_dir),
            "runtime_device": self.runtime_device,
            "backend": "onnxruntime",
            "model_loaded": self._session is not None,
            "load_error": self._load_error,
        }

    def warmup(self) -> dict[str, object]:
        return self.runtime_info()


@dataclass(slots=True)
class OnnxObbDetection:
    polygon_xy: list[list[float]]
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float


def obb_xywhr_to_polygon(
    *,
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_rad: float,
) -> list[list[float]]:
    half_w = width / 2.0
    half_h = height / 2.0
    corners = np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float32,
    )
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rotated = corners @ rotation.T
    rotated[:, 0] += float(cx)
    rotated[:, 1] += float(cy)
    return [[round(float(x), 6), round(float(y), 6)] for x, y in rotated.tolist()]


class YoloObbOnnxDetector:
    def __init__(
        self,
        *,
        model_path: Path,
        confidence_threshold: float,
        imgsz: int,
        max_candidates: int,
        runtime_device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.imgsz = int(imgsz)
        self.max_candidates = max(1, int(max_candidates))
        self.runtime_device = runtime_device
        self._session: Any | None = None
        self._input_name: str | None = None
        self._load_error: str | None = None

    def _providers(self) -> list[str]:
        if self.runtime_device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _ensure_loaded(self) -> None:
        if self._session is not None or self._load_error is not None:
            return
        model_path = _resolve_path(self.model_path)
        if not model_path.exists():
            self._load_error = f"YOLO OBB ONNX model not found: {model_path}"
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(model_path), providers=self._providers())
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"YOLO OBB ONNX detector initialization failed: {exc}"

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("YOLO OBB ONNX input image is empty")
        ratio = min(self.imgsz / float(h), self.imgsz / float(w))
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        pad_x = (self.imgsz - new_w) / 2.0
        pad_y = (self.imgsz - new_h) / 2.0
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top : top + new_h, left : left + new_w] = resized
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :]
        return tensor, ratio, float(left), float(top)

    @staticmethod
    def _rows_from_output(output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"expected YOLO OBB ONNX output with 2 or 3 dimensions, got {output.shape}")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
            arr = arr.T
        return arr

    def _decode(
        self,
        output: np.ndarray,
        *,
        original_shape: tuple[int, ...],
        ratio: float,
        pad_x: float,
        pad_y: float,
    ) -> list[OnnxObbDetection]:
        height, width = original_shape[:2]
        rows = self._rows_from_output(output)
        detections: list[OnnxObbDetection] = []
        for row_raw in rows:
            row = np.asarray(row_raw, dtype=np.float32)
            if row.size < 6:
                continue
            cx, cy, box_w, box_h = [float(value) for value in row[:4]]
            angle = float(row[-1])
            scores = row[4:-1]
            confidence = float(scores.max()) if scores.size else float(row[4])
            if confidence < self.confidence_threshold:
                continue
            cx = (cx - pad_x) / ratio
            cy = (cy - pad_y) / ratio
            box_w = box_w / ratio
            box_h = box_h / ratio
            polygon = obb_xywhr_to_polygon(cx=cx, cy=cy, width=box_w, height=box_h, angle_rad=angle)
            clipped = [[max(0.0, min(float(x), float(width))), max(0.0, min(float(y), float(height)))] for x, y in polygon]
            xs = [point[0] for point in clipped]
            ys = [point[1] for point in clipped]
            bbox = (int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys))))
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            detections.append(OnnxObbDetection(polygon_xy=clipped, bbox_xyxy=bbox, confidence=confidence))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.max_candidates]

    def detect(self, image: np.ndarray) -> tuple[list[OnnxObbDetection], str | None]:
        self._ensure_loaded()
        if self._load_error:
            return [], self._load_error
        if self._session is None or self._input_name is None:
            return [], "YOLO OBB ONNX detector unavailable"
        try:
            tensor, ratio, pad_x, pad_y = self._preprocess(image)
            outputs = self._session.run(None, {self._input_name: tensor})
            if not outputs:
                return [], "YOLO OBB ONNX detector returned no output"
            detections = self._decode(outputs[0], original_shape=image.shape, ratio=ratio, pad_x=pad_x, pad_y=pad_y)
            return detections, None if detections else "no ONNX candidate passed confidence threshold"
        except Exception as exc:  # pragma: no cover - runtime inference path
            return [], f"YOLO OBB ONNX detector inference failed: {exc}"
