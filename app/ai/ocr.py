from __future__ import annotations

import json
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
        engine_name: str,
    ) -> None:
        self.model_dir = model_dir
        self.char_dict_path = char_dict_path
        self.language = language
        self.runtime_device = runtime_device
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.engine_name = engine_name
        self._model = None
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
            from paddleocr import PaddleOCR

            # PaddleOCR uses CUDA toggle. MPS selection stays on the ppocrv5 path and
            # runs through CPU execution unless CUDA is available.
            use_gpu = self.runtime_device == "cuda"
            kwargs = {
                "use_angle_cls": self.use_angle_cls,
                "lang": self.language,
                "use_gpu": use_gpu,
                "rec_model_dir": str(self.model_dir),
            }
            if self.char_dict_path is not None:
                kwargs["rec_char_dict_path"] = str(self.char_dict_path)

            self._model = PaddleOCR(**kwargs)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"failed to initialize {self.engine_name}: {exc}"

    def run(self, image: np.ndarray) -> OCRResultData:
        self._ensure_loaded()

        if self._load_error:
            return OCRResultData("", "", None, self.engine_name, self.runtime_device, self._load_error)
        if self._model is None:
            return OCRResultData("", "", None, self.engine_name, self.runtime_device, "ocr runner unavailable")

        try:
            result = self._model.ocr(image, cls=self.use_angle_cls)
            lines = result[0] if result else []

            texts: list[str] = []
            confs: list[float] = []
            for line in lines:
                if len(line) < 2:
                    continue
                text, conf = line[1]
                if text:
                    texts.append(text)
                    confs.append(float(conf))

            raw_text = " ".join(texts)
            normalized = normalize_ocr_text(raw_text)
            confidence = sum(confs) / len(confs) if confs else None
            reason = None if raw_text else f"{self.engine_name} returned empty text"
            return OCRResultData(raw_text, normalized, confidence, self.engine_name, self.runtime_device, reason)
        except Exception as exc:  # pragma: no cover - runtime inference path
            return OCRResultData(
                "", "", None, self.engine_name, self.runtime_device, f"{self.engine_name} inference failed: {exc}"
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
