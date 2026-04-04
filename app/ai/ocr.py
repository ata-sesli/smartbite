from __future__ import annotations

import inspect
import json
import os
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
    ppocrv5_main_model_dir: Path
    ppocrv5_main_char_dict_path: Path | None
    paddle_lang: str
    device_mode: str
    ppocrv5_use_angle_cls: bool
    ppocrv5_det_db_thresh: float
    substitute_config_path: Path | None
    ppocrv5_det_db_box_thresh: float = 0.5
    enable_substitute_model: bool = False


@dataclass(slots=True)
class SubstituteModelConfig:
    engine_name: str
    model_dir: Path
    char_dict_path: Path | None
    use_angle_cls: bool
    det_db_thresh: float

    @classmethod
    def load(cls, path: Path) -> SubstituteModelConfig:
        payload = json.loads(path.read_text())
        model_dir = Path(payload["model_dir"])
        char_dict = payload.get("char_dict_path")
        return cls(
            engine_name=str(payload.get("engine_name", "ppocrv5_substitute")),
            model_dir=model_dir,
            char_dict_path=Path(char_dict) if char_dict else None,
            use_angle_cls=bool(payload.get("use_angle_cls", True)),
            det_db_thresh=float(payload.get("det_db_thresh", 0.3)),
        )


class PPOCRV5Runner:
    def __init__(
        self,
        *,
        model_dir: Path,
        char_dict_path: Path | None,
        language: str,
        runtime_device: str,
        use_angle_cls: bool,
        det_db_thresh: float,
        det_db_box_thresh: float,
        engine_name: str,
    ) -> None:
        self.model_dir = model_dir
        self.char_dict_path = char_dict_path
        self.language = language
        self.runtime_device = runtime_device
        self.effective_runtime_device = "cuda" if runtime_device == "cuda" else "cpu"
        self._paddle_device = "gpu:0" if runtime_device == "cuda" else "cpu"
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.engine_name = engine_name
        self._model = None
        self._det_model = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            self._load_error = f"ppocrv5 model directory not found: {self.model_dir}"
            return

        if self.char_dict_path is not None and not self.char_dict_path.exists():
            self._load_error = f"ppocrv5 char dict file not found: {self.char_dict_path}"
            return

        try:
            # We use local model directories; skip remote model-source reachability checks
            # to avoid startup delays and noisy network warnings.
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddleocr import PaddleOCR

            try:
                from paddleocr import TextDetection, TextRecognition
            except Exception:
                TextDetection = None
                TextRecognition = None

            resolved_model_dir = self.model_dir

            # Convenience: allow pointing to parent directory containing an `inference/` export.
            nested_infer_dir = self.model_dir / "inference"
            if not (resolved_model_dir / "inference.yml").exists() and (nested_infer_dir / "inference.yml").exists():
                resolved_model_dir = nested_infer_dir

            # Prefer recognition-only API on PaddleOCR 3.x to avoid loading det/orientation
            # models for already-cropped ROI input.
            if TextRecognition is not None:
                if not (resolved_model_dir / "inference.yml").exists() and (self.model_dir / "model.pdparams").exists():
                    self._load_error = (
                        "ppocrv5 model path points to a training checkpoint (found model.pdparams), "
                        "but PaddleOCR 3.x requires exported inference files (missing inference.yml). "
                        "Export the checkpoint to an inference directory and set SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR "
                        "to that exported folder."
                    )
                    return

                text_rec_init_params = set(inspect.signature(TextRecognition.__init__).parameters.keys())
                rec_kwargs: dict[str, object] = {
                    "model_dir": str(resolved_model_dir),
                    "device": self._paddle_device,
                }
                if self.char_dict_path is not None:
                    for key in (
                        "text_recognition_char_dict_path",
                        "rec_char_dict_path",
                        "character_dict_path",
                        "char_dict_path",
                    ):
                        if key in text_rec_init_params:
                            rec_kwargs[key] = str(self.char_dict_path)
                            break
                self._model = TextRecognition(**rec_kwargs)
                if TextDetection is not None:
                    det_kwargs: dict[str, object] = {
                        "device": self._paddle_device,
                        "thresh": self.det_db_thresh,
                        "box_thresh": self.det_db_box_thresh,
                    }
                    self._det_model = TextDetection(**det_kwargs)
                return

            init_params = set(inspect.signature(PaddleOCR.__init__).parameters.keys())
            modern_api = "text_recognition_model_dir" in init_params and "use_gpu" not in init_params
            if not modern_api:
                # PaddleOCR 2.x compatibility path.
                kwargs = {
                    "use_angle_cls": self.use_angle_cls,
                    "lang": self.language,
                    "use_gpu": self.runtime_device == "cuda",
                    "rec_model_dir": str(resolved_model_dir),
                    "det_db_thresh": self.det_db_thresh,
                    "det_db_box_thresh": self.det_db_box_thresh,
                }
                if self.char_dict_path is not None:
                    kwargs["rec_char_dict_path"] = str(self.char_dict_path)
                self._model = PaddleOCR(**kwargs)
                return

            # Final fallback for modern runtimes without TextRecognition class.
            if not (resolved_model_dir / "inference.yml").exists() and (self.model_dir / "model.pdparams").exists():
                self._load_error = (
                    "ppocrv5 model path points to a training checkpoint (found model.pdparams), "
                    "but PaddleOCR 3.x requires exported inference files (missing inference.yml). "
                    "Export the checkpoint to an inference directory and set SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR "
                    "to that exported folder."
                )
                return

            kwargs: dict[str, object] = {
                "text_recognition_model_dir": str(resolved_model_dir),
                "lang": self.language,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": self.use_angle_cls,
                "text_det_thresh": self.det_db_thresh,
                "text_det_box_thresh": self.det_db_box_thresh,
                "device": self._paddle_device,
            }
            if self.char_dict_path is not None:
                if "text_recognition_char_dict_path" in init_params:
                    kwargs["text_recognition_char_dict_path"] = str(self.char_dict_path)
                elif "rec_char_dict_path" in init_params:
                    kwargs["rec_char_dict_path"] = str(self.char_dict_path)
            self._model = PaddleOCR(**kwargs)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"failed to initialize {self.engine_name}: {exc}"

    @staticmethod
    def _extract_text_and_conf(result: object) -> tuple[list[str], list[float]]:
        texts: list[str] = []
        confs: list[float] = []

        if result is None:
            return texts, confs

        entries: list[object]
        if isinstance(result, list):
            entries = result
        else:
            entries = [result]

        # PaddleOCR 3.x: each entry is a dict-like OCRResult containing rec_texts/rec_scores.
        for entry in entries:
            if isinstance(entry, dict) and "rec_texts" in entry:
                rec_texts = entry.get("rec_texts") or []
                rec_scores = entry.get("rec_scores") or []
                for idx, text in enumerate(rec_texts):
                    text_s = str(text).strip()
                    if not text_s:
                        continue
                    texts.append(text_s)
                    if idx < len(rec_scores):
                        try:
                            confs.append(float(rec_scores[idx]))
                        except (TypeError, ValueError):
                            pass
                if texts:
                    return texts, confs

            # PaddleOCR 3.x TextRecognition result:
            # {'rec_text': '...', 'rec_score': 0.xx, ...}
            if isinstance(entry, dict) and "rec_text" in entry:
                text_s = str(entry.get("rec_text") or "").strip()
                if text_s:
                    texts.append(text_s)
                    score = entry.get("rec_score")
                    if score is not None:
                        try:
                            confs.append(float(score))
                        except (TypeError, ValueError):
                            pass
                if texts:
                    return texts, confs

        # PaddleOCR 2.x fallback: [[box, (text, conf)], ...]
        if isinstance(result, list) and result and isinstance(result[0], list):
            lines = result[0]
            for line in lines:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                rec = line[1]
                if not isinstance(rec, (list, tuple)) or not rec:
                    continue
                text_s = str(rec[0]).strip()
                if not text_s:
                    continue
                texts.append(text_s)
                if len(rec) > 1:
                    try:
                        confs.append(float(rec[1]))
                    except (TypeError, ValueError):
                        pass

        return texts, confs

    @staticmethod
    def _extract_detection_boxes(det_result: object, image_shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
        if det_result is None:
            return []
        if not isinstance(det_result, list) or not det_result:
            return []
        first = det_result[0]
        if not isinstance(first, dict):
            return []
        polys = first.get("dt_polys")
        if polys is None:
            return []

        h = int(image_shape[0])
        w = int(image_shape[1])
        boxes: list[tuple[int, int, int, int]] = []
        for poly in polys:
            try:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.size < 8:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                x1 = max(0, min(int(np.floor(xs.min())), w - 1))
                y1 = max(0, min(int(np.floor(ys.min())), h - 1))
                x2 = max(0, min(int(np.ceil(xs.max())), w))
                y2 = max(0, min(int(np.ceil(ys.max())), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((x1, y1, x2, y2))
            except Exception:
                continue
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    def _run_recognition_model(self, crop: np.ndarray) -> tuple[list[str], list[float]]:
        if hasattr(self._model, "predict"):
            rec_result = self._model.predict(crop)
        else:
            rec_result = self._model.ocr(crop, cls=self.use_angle_cls)
        return self._extract_text_and_conf(rec_result)

    def _recognize_with_rotation_search(self, crop: np.ndarray) -> tuple[list[str], list[float]]:
        if not self.use_angle_cls:
            return self._run_recognition_model(crop)

        variants: list[np.ndarray] = [crop]
        # Approximate angle classification by trying cardinal rotations and picking
        # the highest-confidence recognition output.
        variants.append(np.ascontiguousarray(np.rot90(crop, 1)))
        variants.append(np.ascontiguousarray(np.rot90(crop, 2)))
        variants.append(np.ascontiguousarray(np.rot90(crop, 3)))

        best_texts: list[str] = []
        best_confs: list[float] = []
        best_score = -1.0

        for variant in variants:
            texts, confs = self._run_recognition_model(variant)
            if not texts:
                continue
            if confs:
                score = sum(confs) / len(confs)
            else:
                score = 0.0
            if score > best_score:
                best_score = score
                best_texts = texts
                best_confs = confs

        return best_texts, best_confs

    def run(self, image: np.ndarray) -> OCRResultData:
        self._ensure_loaded()

        if self._load_error:
            return OCRResultData("", "", None, self.engine_name, self.effective_runtime_device, self._load_error)
        if self._model is None:
            return OCRResultData(
                "", "", None, self.engine_name, self.effective_runtime_device, "ocr runner unavailable"
            )

        try:
            ocr_input = image
            # TextRecognition expects a 3-channel image; convert grayscale ROI when needed.
            if isinstance(ocr_input, np.ndarray):
                if ocr_input.ndim == 2:
                    ocr_input = np.repeat(ocr_input[:, :, np.newaxis], 3, axis=2)
                elif ocr_input.ndim == 3 and ocr_input.shape[2] == 1:
                    ocr_input = np.repeat(ocr_input, 3, axis=2)

            texts: list[str] = []
            confs: list[float] = []

            # If detection model is available, first detect text regions and run recognition
            # on each region. This improves recall on product ROI images.
            if self._det_model is not None and hasattr(self._det_model, "predict"):
                det_result = self._det_model.predict(ocr_input)
                boxes = self._extract_detection_boxes(det_result, ocr_input.shape)
                for x1, y1, x2, y2 in boxes:
                    crop = ocr_input[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    rec_texts, rec_confs = self._recognize_with_rotation_search(crop)
                    texts.extend(rec_texts)
                    confs.extend(rec_confs)

            # Fallback to whole ROI recognition when detection path yields nothing.
            if not texts:
                texts, confs = self._recognize_with_rotation_search(ocr_input)

            raw_text = " ".join(texts)
            normalized = normalize_ocr_text(raw_text)
            confidence = sum(confs) / len(confs) if confs else None
            reason = None if raw_text else f"{self.engine_name} returned empty text"
            return OCRResultData(raw_text, normalized, confidence, self.engine_name, self.effective_runtime_device, reason)
        except Exception as exc:  # pragma: no cover - runtime inference path
            return OCRResultData(
                "",
                "",
                None,
                self.engine_name,
                self.effective_runtime_device,
                f"{self.engine_name} inference failed: {exc}",
            )


class StaticFailureRunner:
    def __init__(self, *, engine_name: str, runtime_device: str, reason: str) -> None:
        self.engine_name = engine_name
        self.runtime_device = runtime_device
        self.reason = reason

    def run(self, image: np.ndarray) -> OCRResultData:
        _ = image
        return OCRResultData("", "", None, self.engine_name, self.runtime_device, self.reason)


class OCRRouter:
    def __init__(self, config: OCRConfig) -> None:
        self.runtime_device = detect_runtime_device(config.device_mode)
        self.substitute_available = bool(config.substitute_config_path and config.substitute_config_path.exists())

        runner = PPOCRV5Runner(
            model_dir=config.ppocrv5_main_model_dir,
            char_dict_path=config.ppocrv5_main_char_dict_path,
            language=config.paddle_lang,
            runtime_device=self.runtime_device,
            use_angle_cls=config.ppocrv5_use_angle_cls,
            det_db_thresh=config.ppocrv5_det_db_thresh,
            det_db_box_thresh=config.ppocrv5_det_db_box_thresh,
            engine_name="ppocrv5_main",
        )

        if config.enable_substitute_model:
            substitute_cfg, error = self._load_substitute_config(config.substitute_config_path)
            if substitute_cfg is not None:
                runner = PPOCRV5Runner(
                    model_dir=substitute_cfg.model_dir,
                    char_dict_path=substitute_cfg.char_dict_path,
                    language=config.paddle_lang,
                    runtime_device=self.runtime_device,
                    use_angle_cls=substitute_cfg.use_angle_cls,
                    det_db_thresh=substitute_cfg.det_db_thresh,
                    det_db_box_thresh=config.ppocrv5_det_db_box_thresh,
                    engine_name=substitute_cfg.engine_name,
                )
            else:
                runner = StaticFailureRunner(
                    engine_name="ppocrv5_substitute",
                    runtime_device=self.runtime_device,
                    reason=error or "substitute model enabled but unavailable",
                )

        self._runner = runner
        self.active_engine_name = runner.engine_name

    @staticmethod
    def _load_substitute_config(path: Path | None) -> tuple[SubstituteModelConfig | None, str | None]:
        if path is None:
            return None, "substitute model enabled but SMARTBITE_OCR_SUBSTITUTE_CONFIG_PATH is not set"
        if not path.exists():
            return None, f"substitute model enabled but config file not found: {path}"
        try:
            return SubstituteModelConfig.load(path), None
        except Exception as exc:
            return None, f"substitute model config invalid at {path}: {exc}"

    def run(self, roi_image: np.ndarray) -> OCRResultData:
        return self._runner.run(roi_image)
