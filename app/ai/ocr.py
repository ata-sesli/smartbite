from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.ai.parseq import ParSeqRecognizer, resolve_runtime_device
from app.ai.types import OCRResultData


SERVER_TEXT_DETECTOR_MODEL_NAME = "PP-OCRv5_server_det"
PPOCRV5_SERVER_DETECTOR_SOURCE = "ppocrv5_server"
CRAFT_DETECTOR_SOURCE = "craft"
ENSEMBLE_DETECTOR_SOURCE = "ensemble"
MOBILE_TEXT_RECOGNIZER_MODEL_NAME = "PP-OCRv5_mobile_rec"
SVTRV2_RECOGNIZER_MODEL_NAME = "ch_SVTRv2_rec"
TEXT_DETECTOR_MODES = ("ppocrv5_server", "craft", "ensemble")
FINAL_RECOGNIZER_MODES = ("svtrv2", "parseq")


def normalize_ocr_text(raw_text: str) -> str:
    return " ".join(raw_text.strip().upper().split())


def detect_runtime_device(mode: str) -> str:
    return resolve_runtime_device(mode)


@dataclass(slots=True)
class OCRConfig:
    parseq_model_dir: Path
    paddle_lang: str
    device_mode: str
    parseq_device_mode: str
    ppocrv5_use_angle_cls: bool
    ppocrv5_det_db_thresh: float
    ppocrv5_det_db_box_thresh: float = 0.5
    ppocrv5_det_limit_side_len: int | None = None
    ppocrv5_det_limit_type: str | None = None
    ppocrv5_det_db_unclip_ratio: float | None = None
    expiry_recognizer: str = "svtrv2"
    svtrv2_rec_model_name: str = SVTRV2_RECOGNIZER_MODEL_NAME
    svtrv2_rec_model_dir: Path | None = Path("models/svtrv2/smartbite_svtrv2_expdate_rec")
    svtrv2_device_mode: str = "cpu"
    context_recognizer: str = "svtrv2"
    context_svtrv2_rec_model_name: str = SVTRV2_RECOGNIZER_MODEL_NAME
    context_svtrv2_rec_model_dir: Path | None = Path("models/svtrv2/ch_SVTRv2_rec")
    context_svtrv2_device_mode: str = "cpu"
    text_detector_mode: str = "ensemble"
    ppocrv5_text_det_model_dir: Path | None = None
    ppocrv5_text_det_model_name: str = SERVER_TEXT_DETECTOR_MODEL_NAME
    craft_enabled: bool = True
    craft_model_path: Path = Path("models/craft/craft_mlt_25k.pth")
    craft_text_threshold: float = 0.7
    craft_link_threshold: float = 0.4
    craft_low_text: float = 0.4
    craft_canvas_size: int = 1280
    craft_mag_ratio: float = 1.5
    craft_variant_names: tuple[str, ...] = ("raw", "raw_upscaled", "luma_clahe")
    expiry_probe_rec_model_name: str = MOBILE_TEXT_RECOGNIZER_MODEL_NAME
    expiry_probe_mobile_rec_model_dir: Path | None = Path("models/ppocrv5/mobile_rec")
    # Kept for prefetch/asset compatibility; intentionally unused in runtime pipeline.
    expiry_probe_mobile_det_model_dir: Path | None = Path("models/ppocrv5/mobile_det")
    # Deprecated/compat placeholders kept to avoid breaking older callers.
    ppocrv5_main_model_dir: Path | None = None
    ppocrv5_main_char_dict_path: Path | None = None
    substitute_config_path: Path | None = None
    enable_substitute_model: bool = False


@dataclass(slots=True)
class TextDetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float | None
    source: str = PPOCRV5_SERVER_DETECTOR_SOURCE
    sources: tuple[str, ...] = (PPOCRV5_SERVER_DETECTOR_SOURCE,)
    variant_name: str | None = None
    polygon_xy: tuple[tuple[float, float], ...] | None = None

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class RecognitionData:
    raw_text: str
    normalized_text: str
    confidence: float | None
    reason: str | None


class FinalTextRecognizer(Protocol):
    engine_name: str
    model_name: str
    runtime_device: str

    def recognize(self, image: np.ndarray) -> RecognitionData:
        ...

    def runtime_info(self) -> dict[str, object]:
        ...

    def warmup(self) -> dict[str, object]:
        ...


class TextDetector(Protocol):
    source_name: str

    def detect(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        ...


class PPOCRV5TextDetector:
    source_name = PPOCRV5_SERVER_DETECTOR_SOURCE

    def __init__(
        self,
        *,
        model_dir: Path | None,
        model_name: str,
        language: str,
        runtime_device: str,
        det_db_thresh: float,
        det_db_box_thresh: float,
        det_limit_side_len: int | None = None,
        det_limit_type: str | None = None,
        det_db_unclip_ratio: float | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.model_name = model_name
        self.language = language
        self.runtime_device = runtime_device
        self._paddle_device = "gpu:0" if runtime_device == "cuda" else "cpu"
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self._model = None
        self._load_error: str | None = None

    @staticmethod
    def _is_inference_dir(path: Path) -> bool:
        return (path / "inference.yml").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            from paddleocr import TextDetection
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"ppocr text detector import failed: {exc}"
            return

        kwargs: dict[str, object] = {
            "device": self._paddle_device,
            "thresh": self.det_db_thresh,
            "box_thresh": self.det_db_box_thresh,
        }
        if self.det_limit_side_len is not None:
            kwargs["limit_side_len"] = self.det_limit_side_len
        if self.det_limit_type is not None:
            kwargs["limit_type"] = self.det_limit_type
        if self.det_db_unclip_ratio is not None:
            kwargs["unclip_ratio"] = self.det_db_unclip_ratio
        if self.model_dir is not None:
            resolved = self.model_dir
            if resolved.is_file() and resolved.name.endswith(".pdparams"):
                resolved = resolved.parent
            if not resolved.exists():
                self._load_error = f"ppocr text detector model directory not found: {resolved}"
                return
            if not self._is_inference_dir(resolved):
                self._load_error = (
                    "ppocr text detector model must be an exported inference directory "
                    f"(missing inference.yml): {resolved}"
                )
                return
            kwargs["model_dir"] = str(resolved)
        elif self.model_name:
            kwargs["model_name"] = self.model_name

        try:
            self._model = TextDetection(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"ppocr text detector initialization failed: {exc}"

    @staticmethod
    def _extract_boxes(det_result: object, image_shape: tuple[int, ...]) -> list[TextDetectionBox]:
        if det_result is None:
            return []
        if not isinstance(det_result, list) or not det_result:
            return []
        entry = det_result[0]
        if not isinstance(entry, dict):
            return []
        polys = entry.get("dt_polys")
        if polys is None:
            return []

        scores: list[float | None] = []
        scores_raw = entry.get("dt_scores")
        if scores_raw is not None:
            try:
                scores = [float(x) for x in np.asarray(scores_raw).reshape(-1)]
            except Exception:
                scores = []

        h = int(image_shape[0])
        w = int(image_shape[1])
        boxes: list[TextDetectionBox] = []
        for idx, poly in enumerate(polys):
            try:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.ndim == 1 and arr.size >= 8:
                    arr = arr.reshape(-1, 2)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                x1 = max(0, min(int(np.floor(xs.min())), w - 1))
                y1 = max(0, min(int(np.floor(ys.min())), h - 1))
                x2 = max(0, min(int(np.ceil(xs.max())), w))
                y2 = max(0, min(int(np.ceil(ys.max())), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                score = scores[idx] if idx < len(scores) else None
                polygon = tuple(
                    (float(max(0.0, min(float(x), float(w)))), float(max(0.0, min(float(y), float(h)))))
                    for x, y in arr[:4]
                ) if arr.shape[0] >= 4 else None
                boxes.append(
                    TextDetectionBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=score,
                        source=PPOCRV5_SERVER_DETECTOR_SOURCE,
                        sources=(PPOCRV5_SERVER_DETECTOR_SOURCE,),
                        polygon_xy=polygon,
                    )
                )
            except Exception:
                continue
        boxes.sort(key=lambda b: (b.y1, b.x1))
        return boxes

    def detect(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        self._ensure_loaded()
        if self._load_error:
            return [], self._load_error
        if self._model is None:
            return [], "ppocr text detector unavailable"

        try:
            result = self._model.predict(image)
            boxes = self._extract_boxes(result, image.shape)
            boxes = [
                TextDetectionBox(
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    confidence=box.confidence,
                    source=PPOCRV5_SERVER_DETECTOR_SOURCE,
                    sources=(PPOCRV5_SERVER_DETECTOR_SOURCE,),
                    variant_name=variant_name,
                    polygon_xy=box.polygon_xy,
                )
                for box in boxes
            ]
            return boxes, None
        except Exception as exc:  # pragma: no cover - runtime inference path
            return [], f"ppocr text detection failed: {exc}"


class CRAFTTextDetector:
    source_name = CRAFT_DETECTOR_SOURCE

    def __init__(
        self,
        *,
        model_path: Path,
        runtime_device: str,
        enabled: bool,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        variant_names: tuple[str, ...] = ("raw", "raw_upscaled", "luma_clahe"),
    ) -> None:
        self.model_path = model_path
        self.runtime_device = runtime_device
        self.enabled = enabled
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.variant_names = tuple(v.strip() for v in variant_names if v.strip())
        self._craft_net = None
        self._refine_net = None
        self._get_prediction = None
        self._load_error: str | None = None

    def _should_run_for_variant(self, variant_name: str | None) -> bool:
        if not self.variant_names:
            return True
        return (variant_name or "raw") in self.variant_names

    def _ensure_loaded(self) -> None:
        if self._craft_net is not None or self._load_error is not None:
            return
        if not self.enabled:
            self._load_error = "craft detector disabled"
            return
        if not self.model_path.exists():
            self._load_error = f"craft model not found: {self.model_path}"
            return

        try:
            import torch
            from craft_detector import CRAFT
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"craft_detector import failed: {exc}"
            return

        try:
            device_name = self.runtime_device if self.runtime_device in {"cuda", "mps"} else "cpu"
            device = torch.device(device_name)
            net = CRAFT()
            state = torch.load(self.model_path, map_location=device)
            state_dict = self._extract_state_dict(state)
            net.load_state_dict(state_dict, strict=False)
            net.to(device)
            net.eval()
            self._craft_net = net
            self._refine_net = None
            self._get_prediction = self._predict_with_craft_detector
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"craft detector initialization failed: {exc}"

    @staticmethod
    def _extract_state_dict(state: object) -> dict[str, object]:
        if isinstance(state, dict):
            for key in ("state_dict", "model", "craft"):
                nested = state.get(key)
                if isinstance(nested, dict):
                    state = nested
                    break
        if not isinstance(state, dict):
            raise TypeError("CRAFT weights did not contain a state dict")
        return {str(key).removeprefix("module."): value for key, value in state.items()}

    def _predict_with_craft_detector(self, image: np.ndarray) -> dict[str, object]:
        import cv2
        from craft_detector.craft_utils import adjustResultCoordinates, detect_textbox
        from craft_detector.imgproc import resize_aspect_ratio

        if self._craft_net is None:
            return {"boxes": []}

        image_rgb = image
        if image.ndim == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized, target_ratio = resize_aspect_ratio(
            image_rgb,
            square_size=self.canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=self.mag_ratio,
        )
        ratio_h = ratio_w = 1.0 / target_ratio
        device_name = self.runtime_device if self.runtime_device in {"cuda", "mps"} else "cpu"
        _score_text, _score_link, boxes, polys = detect_textbox(
            resized,
            self._craft_net,
            self._refine_net,
            self.text_threshold,
            self.link_threshold,
            self.low_text,
            False,
            device_name,
        )
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        return {"boxes": [poly if poly is not None else box for box, poly in zip(boxes, polys)]}

    @staticmethod
    def _extract_boxes(prediction: object, image_shape: tuple[int, ...], variant_name: str | None) -> list[TextDetectionBox]:
        if not isinstance(prediction, dict):
            return []
        raw_boxes = prediction.get("boxes") or prediction.get("polys") or []
        h = int(image_shape[0])
        w = int(image_shape[1])
        boxes: list[TextDetectionBox] = []
        for raw_box in raw_boxes:
            try:
                arr = np.asarray(raw_box, dtype=np.float32)
                if arr.ndim == 1 and arr.size >= 8:
                    arr = arr.reshape(-1, 2)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                x1 = max(0, min(int(np.floor(xs.min())), w - 1))
                y1 = max(0, min(int(np.floor(ys.min())), h - 1))
                x2 = max(0, min(int(np.ceil(xs.max())), w))
                y2 = max(0, min(int(np.ceil(ys.max())), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                polygon = tuple(
                    (float(max(0.0, min(float(x), float(w)))), float(max(0.0, min(float(y), float(h)))))
                    for x, y in arr[:4]
                ) if arr.shape[0] >= 4 else None
                boxes.append(
                    TextDetectionBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=None,
                        source=CRAFT_DETECTOR_SOURCE,
                        sources=(CRAFT_DETECTOR_SOURCE,),
                        variant_name=variant_name,
                        polygon_xy=polygon,
                    )
                )
            except Exception:
                continue
        boxes.sort(key=lambda b: (b.y1, b.x1))
        return boxes

    def detect(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        if not self._should_run_for_variant(variant_name):
            return [], None
        self._ensure_loaded()
        if self._load_error:
            return [], self._load_error
        if self._craft_net is None or self._get_prediction is None:
            return [], "craft detector unavailable"

        try:
            prediction = self._get_prediction(image)
            return self._extract_boxes(prediction, image.shape, variant_name), None
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return [], f"craft text detection failed: {exc}"


class EnsembleTextDetector:
    source_name = ENSEMBLE_DETECTOR_SOURCE

    def __init__(
        self,
        *,
        ppocr_detector: TextDetector,
        craft_detector: TextDetector,
        iou_threshold: float = 0.35,
        containment_threshold: float = 0.80,
        max_boxes: int = 120,
    ) -> None:
        self.ppocr_detector = ppocr_detector
        self.craft_detector = craft_detector
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.max_boxes = max(1, int(max_boxes))

    @staticmethod
    def _area(box: TextDetectionBox) -> int:
        return max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)

    @classmethod
    def _overlap(cls, a: TextDetectionBox, b: TextDetectionBox) -> tuple[float, float]:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0, 0.0
        area_a = cls._area(a)
        area_b = cls._area(b)
        union = max(1, area_a + area_b - inter)
        containment = inter / float(max(1, min(area_a, area_b)))
        return inter / float(union), containment

    def _merge_pair(self, a: TextDetectionBox, b: TextDetectionBox) -> TextDetectionBox:
        sources = tuple(dict.fromkeys((*a.sources, *b.sources)))
        confidence_values = [v for v in (a.confidence, b.confidence) if v is not None]
        confidence = max(confidence_values) if confidence_values else None
        return TextDetectionBox(
            x1=min(a.x1, b.x1),
            y1=min(a.y1, b.y1),
            x2=max(a.x2, b.x2),
            y2=max(a.y2, b.y2),
            confidence=confidence,
            source=ENSEMBLE_DETECTOR_SOURCE if len(sources) > 1 else sources[0],
            sources=sources,
            variant_name=a.variant_name or b.variant_name,
            polygon_xy=a.polygon_xy if a.polygon_xy == b.polygon_xy else None,
        )

    def _merge_boxes(self, boxes: list[TextDetectionBox]) -> list[TextDetectionBox]:
        merged: list[TextDetectionBox] = []
        for box in boxes:
            candidate = box
            absorbed = True
            while absorbed:
                absorbed = False
                for idx, existing in enumerate(merged):
                    iou, containment = self._overlap(existing, candidate)
                    if iou >= self.iou_threshold or containment >= self.containment_threshold:
                        candidate = self._merge_pair(existing, candidate)
                        merged.pop(idx)
                        absorbed = True
                        break
            merged.append(candidate)

        def rank_key(box: TextDetectionBox) -> tuple[int, float, int]:
            return (
                len(box.sources),
                float(box.confidence or 0.0),
                -self._area(box),
            )

        merged.sort(key=rank_key, reverse=True)
        return merged[: self.max_boxes]

    def detect(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        ppocr_boxes, ppocr_reason = self.ppocr_detector.detect(image, variant_name=variant_name)
        craft_boxes, craft_reason = self.craft_detector.detect(image, variant_name=variant_name)
        all_boxes = [
            TextDetectionBox(
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                confidence=box.confidence,
                source=box.source,
                sources=box.sources,
                variant_name=box.variant_name or variant_name,
            )
            for box in (*ppocr_boxes, *craft_boxes)
        ]
        reasons = [reason for reason in (ppocr_reason, craft_reason) if reason]
        return self._merge_boxes(all_boxes), "; ".join(dict.fromkeys(reasons)) or None


class PaddleTextRecognizer:
    def __init__(
        self,
        *,
        model_name: str,
        model_dir: Path | None,
        runtime_device: str,
        engine_name: str = "paddle_text_recognition",
    ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.runtime_device = runtime_device
        self.engine_name = engine_name
        self._paddle_device = "gpu:0" if runtime_device == "cuda" else "cpu"
        self._model = None
        self._load_error: str | None = None

    @staticmethod
    def _is_inference_dir(path: Path) -> bool:
        return (path / "inference.yml").exists()

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            from paddleocr import TextRecognition
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"paddle text recognizer import failed: {exc}"
            return

        kwargs: dict[str, object] = {
            "device": self._paddle_device,
            "model_name": self.model_name,
        }
        if self.model_dir is not None:
            resolved = self.model_dir
            if not resolved.exists():
                self._load_error = f"paddle text recognizer model directory not found: {resolved}"
                return
            if not self._is_inference_dir(resolved):
                self._load_error = (
                    "paddle text recognizer model must be an exported inference directory "
                    f"(missing inference.yml): {resolved}"
                )
                return
            kwargs["model_dir"] = str(resolved)

        try:
            self._model = TextRecognition(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"paddle text recognizer initialization failed: {exc}"

    def recognize(self, image: np.ndarray) -> RecognitionData:
        self._ensure_loaded()
        if self._load_error:
            return RecognitionData(raw_text="", normalized_text="", confidence=None, reason=self._load_error)
        if self._model is None:
            return RecognitionData(raw_text="", normalized_text="", confidence=None, reason="paddle text recognizer unavailable")

        try:
            if image.ndim == 2:
                recognizer_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 1:
                recognizer_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                recognizer_input = image
            result = self._model.predict(recognizer_input)
            if not isinstance(result, list) or not result:
                return RecognitionData(raw_text="", normalized_text="", confidence=None, reason="paddle text recognizer returned no result")

            entry = result[0]
            if not isinstance(entry, dict):
                return RecognitionData(raw_text="", normalized_text="", confidence=None, reason="paddle text recognizer returned invalid result")

            raw = str(entry.get("rec_text", "") or "").strip()
            score_raw = entry.get("rec_score")
            score: float | None = None
            if score_raw is not None:
                try:
                    score = float(score_raw)
                except Exception:
                    score = None

            if not raw:
                return RecognitionData(raw_text="", normalized_text="", confidence=score, reason="paddle text recognizer returned empty text")

            normalized = normalize_ocr_text(raw)
            return RecognitionData(raw_text=raw, normalized_text=normalized, confidence=score, reason=None)
        except Exception as exc:  # pragma: no cover - runtime inference path
            return RecognitionData(raw_text="", normalized_text="", confidence=None, reason=f"paddle text recognizer failed: {exc}")

    def runtime_info(self) -> dict[str, object]:
        self._ensure_loaded()
        return {
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "model_dir": str(self.model_dir) if self.model_dir is not None else None,
            "runtime_device": self.runtime_device,
            "backend": "paddleocr.TextRecognition",
            "model_loaded": self._model is not None,
            "load_error": self._load_error,
        }

    def warmup(self) -> dict[str, object]:
        return self.runtime_info()


PPOCRV5MobileRecognizer = PaddleTextRecognizer


class ParSeqTextRecognizer:
    engine_name = "parseq_small"
    model_name = "parseq"

    def __init__(self, recognizer: ParSeqRecognizer) -> None:
        self.recognizer = recognizer
        self.runtime_device = recognizer.runtime_device

    def recognize(self, image: np.ndarray) -> RecognitionData:
        output = self.recognizer.recognize(image)
        self.runtime_device = self.recognizer.runtime_device
        if not output.text:
            return RecognitionData(raw_text="", normalized_text="", confidence=output.confidence, reason=output.reason)
        return RecognitionData(
            raw_text=output.text,
            normalized_text=normalize_ocr_text(output.text),
            confidence=output.confidence,
            reason=output.reason,
        )

    def runtime_info(self) -> dict[str, object]:
        return self.recognizer.runtime_info()

    def warmup(self) -> dict[str, object]:
        self.recognizer._ensure_loaded()
        self.runtime_device = self.recognizer.runtime_device
        return self.recognizer.runtime_info()


class ExpiryOCRRunner:
    def __init__(
        self,
        *,
        text_detector: TextDetector,
        probe_recognizer: PPOCRV5MobileRecognizer,
        context_recognizer: FinalTextRecognizer,
        final_recognizer: FinalTextRecognizer,
        use_angle_cls: bool,
        text_detector_mode: str,
    ) -> None:
        self.text_detector = text_detector
        self.probe_recognizer = probe_recognizer
        self.context_recognizer = context_recognizer
        self.final_recognizer = final_recognizer
        self.use_angle_cls = use_angle_cls
        self.text_detector_mode = text_detector_mode
        self.engine_name = final_recognizer.engine_name
        self.model_name = final_recognizer.model_name
        self.probe_engine_name = MOBILE_TEXT_RECOGNIZER_MODEL_NAME
        self.context_engine_name = context_recognizer.engine_name
        self.runtime_device = final_recognizer.runtime_device

    @staticmethod
    def _normalize_input(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if image.ndim == 3 and image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        return image

    @staticmethod
    def _rotation_variants(crop: np.ndarray) -> list[np.ndarray]:
        return [
            crop,
            np.ascontiguousarray(np.rot90(crop, 1)),
            np.ascontiguousarray(np.rot90(crop, 2)),
            np.ascontiguousarray(np.rot90(crop, 3)),
        ]

    def detect_text_boxes(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        ocr_input = self._normalize_input(image)
        return self.text_detector.detect(ocr_input, variant_name=variant_name)

    def recognize_with_probe(self, crop: np.ndarray) -> RecognitionData:
        probe_input = self._normalize_input(crop)
        return self.probe_recognizer.recognize(probe_input)

    def recognize_with_context(self, crop: np.ndarray) -> RecognitionData:
        context_input = self._normalize_input(crop)
        return self.context_recognizer.recognize(context_input)

    def recognize_with_final(self, crop: np.ndarray) -> RecognitionData:
        parse_input = self._normalize_input(crop)
        variants = self._rotation_variants(parse_input) if self.use_angle_cls else [parse_input]
        best = RecognitionData(raw_text="", normalized_text="", confidence=None, reason=None)

        for variant in variants:
            output = self.final_recognizer.recognize(variant)
            self.runtime_device = self.final_recognizer.runtime_device
            if not output.raw_text:
                if best.reason is None:
                    best = output
                continue

            if best.raw_text and (best.confidence or 0.0) >= (output.confidence or 0.0):
                continue
            best = RecognitionData(
                raw_text=output.raw_text,
                normalized_text=output.normalized_text or normalize_ocr_text(output.raw_text),
                confidence=output.confidence,
                reason=output.reason,
            )

        if not best.raw_text and best.reason is None:
            return RecognitionData(raw_text="", normalized_text="", confidence=None, reason="final recognizer returned empty text")
        return best

    def recognize_with_parseq(self, crop: np.ndarray) -> RecognitionData:
        return self.recognize_with_final(crop)

    def run(self, image: np.ndarray) -> OCRResultData:
        ocr_input = self._normalize_input(image)
        boxes, det_reason = self.text_detector.detect(ocr_input, variant_name="raw")

        texts: list[str] = []
        confidences: list[float] = []
        reasons: list[str] = []

        for box in boxes:
            crop = ocr_input[box.y1:box.y2, box.x1:box.x2]
            if crop.size == 0:
                continue
            rec = self.recognize_with_final(crop)
            if rec.raw_text:
                texts.append(rec.raw_text)
                if rec.confidence is not None:
                    confidences.append(rec.confidence)
            elif rec.reason:
                reasons.append(rec.reason)

        # Compatibility fallback to whole ROI recognition.
        if not texts:
            rec = self.recognize_with_final(ocr_input)
            if rec.raw_text:
                texts = [rec.raw_text]
                if rec.confidence is not None:
                    confidences = [rec.confidence]
            elif rec.reason:
                reasons.append(rec.reason)

        raw_text = "\n".join(texts).strip()
        normalized = normalize_ocr_text(raw_text)
        confidence = float(sum(confidences) / len(confidences)) if confidences else None

        reason_parts = [part for part in (det_reason, *reasons) if part]
        reason = None
        if not raw_text:
            reason = "; ".join(dict.fromkeys(reason_parts)) if reason_parts else "ocr returned empty text"

        return OCRResultData(
            raw_text=raw_text,
            normalized_text=normalized,
            confidence=confidence,
            engine_name=self.engine_name,
            runtime_device=self.runtime_device,
            reason=reason,
        )


class OCRRouter:
    def __init__(self, config: OCRConfig) -> None:
        det_runtime = detect_runtime_device(config.device_mode)
        text_detector_mode = config.text_detector_mode.lower().strip()
        if text_detector_mode not in TEXT_DETECTOR_MODES:
            raise ValueError(f"text_detector_mode must be one of {TEXT_DETECTOR_MODES}")
        final_recognizer_mode = config.expiry_recognizer.lower().strip()
        if final_recognizer_mode not in FINAL_RECOGNIZER_MODES:
            raise ValueError(f"expiry_recognizer must be one of {FINAL_RECOGNIZER_MODES}")
        if final_recognizer_mode == "parseq":
            final_recognizer: FinalTextRecognizer = ParSeqTextRecognizer(
                ParSeqRecognizer(
                    model_dir=config.parseq_model_dir,
                    device_mode=config.parseq_device_mode,
                )
            )
        else:
            final_recognizer = PaddleTextRecognizer(
                model_name=config.svtrv2_rec_model_name,
                model_dir=config.svtrv2_rec_model_dir,
                runtime_device=detect_runtime_device(config.svtrv2_device_mode),
                engine_name="svtrv2",
            )
        context_recognizer = PaddleTextRecognizer(
            model_name=config.context_svtrv2_rec_model_name,
            model_dir=config.context_svtrv2_rec_model_dir,
            runtime_device=detect_runtime_device(config.context_svtrv2_device_mode),
            engine_name="context_svtrv2",
        )
        ppocr_detector = PPOCRV5TextDetector(
            model_dir=config.ppocrv5_text_det_model_dir,
            model_name=config.ppocrv5_text_det_model_name,
            language=config.paddle_lang,
            runtime_device=det_runtime,
            det_db_thresh=config.ppocrv5_det_db_thresh,
            det_db_box_thresh=config.ppocrv5_det_db_box_thresh,
            det_limit_side_len=config.ppocrv5_det_limit_side_len,
            det_limit_type=config.ppocrv5_det_limit_type,
            det_db_unclip_ratio=config.ppocrv5_det_db_unclip_ratio,
        )
        craft_detector = CRAFTTextDetector(
            model_path=config.craft_model_path,
            runtime_device=det_runtime,
            enabled=config.craft_enabled,
            text_threshold=config.craft_text_threshold,
            link_threshold=config.craft_link_threshold,
            low_text=config.craft_low_text,
            canvas_size=config.craft_canvas_size,
            mag_ratio=config.craft_mag_ratio,
            variant_names=config.craft_variant_names,
        )
        merge_detector = EnsembleTextDetector(ppocr_detector=ppocr_detector, craft_detector=craft_detector)
        if text_detector_mode == "craft":
            text_detector = craft_detector
        else:
            text_detector = ppocr_detector
        probe_recognizer = PPOCRV5MobileRecognizer(
            model_name=config.expiry_probe_rec_model_name,
            model_dir=config.expiry_probe_mobile_rec_model_dir,
            runtime_device=det_runtime,
            engine_name=config.expiry_probe_rec_model_name,
        )
        self._runner = ExpiryOCRRunner(
            text_detector=text_detector,
            probe_recognizer=probe_recognizer,
            context_recognizer=context_recognizer,
            final_recognizer=final_recognizer,
            use_angle_cls=config.ppocrv5_use_angle_cls,
            text_detector_mode=text_detector_mode,
        )
        self._final_recognizer = final_recognizer
        self._context_recognizer = context_recognizer
        self._parseq_recognizer = final_recognizer.recognizer if isinstance(final_recognizer, ParSeqTextRecognizer) else None
        self._text_detector = text_detector
        self._ppocr_detector = ppocr_detector
        self._craft_detector = craft_detector
        self._merge_detector = merge_detector
        self.active_engine_name = self._runner.engine_name
        self.final_recognizer_name = final_recognizer_mode
        self.final_recognizer_model_name = self._runner.model_name
        self.probe_engine_name = self._runner.probe_engine_name
        self.context_recognizer_name = config.context_recognizer
        self.context_recognizer_model_name = context_recognizer.model_name
        self.text_detector_mode = text_detector_mode
        self.parseq_runtime_device = self._runner.runtime_device
        self.final_recognition_runtime_device = self._runner.runtime_device
        self.substitute_available = False

    def detect_text_boxes(
        self,
        roi_image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        return self._runner.detect_text_boxes(roi_image, variant_name=variant_name)

    def detect_ppocr_text_boxes(
        self,
        roi_image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        ocr_input = self._runner._normalize_input(roi_image)
        return self._ppocr_detector.detect(ocr_input, variant_name=variant_name)

    def detect_craft_text_boxes(
        self,
        roi_image: np.ndarray,
        *,
        variant_name: str | None = None,
        canvas_size: int | None = None,
        mag_ratio: float | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        ocr_input = self._runner._normalize_input(roi_image)
        old_canvas = getattr(self._craft_detector, "canvas_size", None)
        old_mag = getattr(self._craft_detector, "mag_ratio", None)
        try:
            if canvas_size is not None and hasattr(self._craft_detector, "canvas_size"):
                self._craft_detector.canvas_size = int(canvas_size)
            if mag_ratio is not None and hasattr(self._craft_detector, "mag_ratio"):
                self._craft_detector.mag_ratio = float(mag_ratio)
            return self._craft_detector.detect(ocr_input, variant_name=variant_name)
        finally:
            if old_canvas is not None and hasattr(self._craft_detector, "canvas_size"):
                self._craft_detector.canvas_size = old_canvas
            if old_mag is not None and hasattr(self._craft_detector, "mag_ratio"):
                self._craft_detector.mag_ratio = old_mag

    def merge_text_boxes(
        self,
        ppocr_boxes: list[TextDetectionBox],
        craft_boxes: list[TextDetectionBox],
    ) -> list[TextDetectionBox]:
        return self._merge_detector._merge_boxes([*ppocr_boxes, *craft_boxes])

    def recognize_probe(self, crop_image: np.ndarray) -> RecognitionData:
        return self._runner.recognize_with_probe(crop_image)

    def recognize_context(self, crop_image: np.ndarray) -> RecognitionData:
        return self._runner.recognize_with_context(crop_image)

    def recognize_final(self, crop_image: np.ndarray) -> RecognitionData:
        rec = self._runner.recognize_with_final(crop_image)
        self.final_recognition_runtime_device = self._runner.runtime_device = self._final_recognizer.runtime_device
        # Deprecated compatibility name used by old debug reports and scripts.
        self.parseq_runtime_device = self.final_recognition_runtime_device
        return rec

    def recognize_parseq(self, crop_image: np.ndarray) -> RecognitionData:
        return self.recognize_final(crop_image)

    def run(self, roi_image: np.ndarray) -> OCRResultData:
        return self._runner.run(roi_image)

    def final_recognition_runtime_info(self) -> dict[str, object]:
        return self._final_recognizer.runtime_info()

    def context_recognition_runtime_info(self) -> dict[str, object]:
        return self._context_recognizer.runtime_info()

    def parseq_runtime_info(self) -> dict[str, object]:
        return self.final_recognition_runtime_info()

    def warmup_final_recognizer(self) -> dict[str, object]:
        info = self._final_recognizer.warmup()
        self.final_recognition_runtime_device = self._runner.runtime_device = self._final_recognizer.runtime_device
        self.parseq_runtime_device = self.final_recognition_runtime_device
        return info

    def warmup_parseq(self) -> dict[str, object]:
        return self.warmup_final_recognizer()
