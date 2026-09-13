from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.expiry_candidate_engine import ProposalBox


@dataclass(slots=True)
class RapidOCRTextProposalDetector:
    ocr_version: str = "PP-OCRv5"
    model_type: str = "server"
    lang_type: str = "ch"
    engine_type: str = "onnxruntime"
    model_path: Path | None = None
    model_dir: Path | None = None
    limit_side_len: int = 512
    limit_type: str = "max"
    max_candidates: int = 16
    max_boxes: int = 5
    min_confidence: float = 0.10
    min_box_area: int = 12
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    thresh: float | None = None
    box_thresh: float | None = None
    unclip_ratio: float | None = None
    use_dilation: bool | None = None
    score_mode: str | None = None
    text_detector: Any | None = None
    _load_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.ocr_version = self.ocr_version.strip() or "PP-OCRv5"
        self.model_type = self.model_type.strip().lower() or "server"
        self.lang_type = self.lang_type.strip().lower() or "ch"
        self.engine_type = self.engine_type.strip().lower() or "onnxruntime"
        self.limit_side_len = max(128, int(self.limit_side_len))
        self.limit_type = self.limit_type.strip().lower() or "max"
        self.max_candidates = max(1, int(self.max_candidates))
        self.max_boxes = max(1, int(self.max_boxes))
        self.min_confidence = max(0.0, min(float(self.min_confidence), 1.0))
        self.min_box_area = max(1, int(self.min_box_area))

    def _ensure_text_detector(self) -> Any | None:
        if self.text_detector is not None or self._load_error is not None:
            return self.text_detector
        try:
            from rapidocr import EngineType, LangDet, ModelType, OCRVersion
            from rapidocr.ch_ppocr_det.main import TextDetector
            from rapidocr.main import DEFAULT_CFG_PATH, ParseParams, root_dir

            engine_type_map = {
                "onnx": EngineType.ONNXRUNTIME,
                "onnxruntime": EngineType.ONNXRUNTIME,
                "paddle": EngineType.PADDLE,
            }
            model_type = ModelType.MOBILE if self.model_type == "mobile" else ModelType.SERVER
            ocr_version = OCRVersion.PPOCRV5 if self.ocr_version.upper().replace("_", "-") == "PP-OCRV5" else OCRVersion.PPOCRV4
            lang_type = LangDet.CH if self.lang_type == "ch" else LangDet.EN
            params = {
                "Det.engine_type": engine_type_map.get(self.engine_type, EngineType.ONNXRUNTIME),
                "Det.model_type": model_type,
                "Det.ocr_version": ocr_version,
                "Det.lang_type": lang_type,
                "Det.limit_side_len": self.limit_side_len,
                "Det.limit_type": self.limit_type,
                "Det.max_candidates": self.max_candidates,
            }
            if self.model_path is not None:
                params["Det.model_path"] = Path(self.model_path).resolve()
            if self.model_dir is not None:
                params["Det.model_dir"] = Path(self.model_dir).resolve()
            if self.mean is not None:
                params["Det.mean"] = list(self.mean)
            if self.std is not None:
                params["Det.std"] = list(self.std)
            if self.thresh is not None:
                params["Det.thresh"] = float(self.thresh)
            if self.box_thresh is not None:
                params["Det.box_thresh"] = float(self.box_thresh)
            if self.unclip_ratio is not None:
                params["Det.unclip_ratio"] = float(self.unclip_ratio)
            if self.use_dilation is not None:
                params["Det.use_dilation"] = bool(self.use_dilation)
            if self.score_mode is not None:
                params["Det.score_mode"] = self.score_mode
            cfg = ParseParams.load(DEFAULT_CFG_PATH)
            cfg = ParseParams.update_batch(cfg, params)
            if cfg.Global.model_root_dir is None:
                cfg.Global.model_root_dir = root_dir / "models"
            cfg.Det.engine_cfg = cfg.EngineConfig[cfg.Det.engine_type.value]
            cfg.Det.model_root_dir = cfg.Global.model_root_dir
            self.text_detector = TextDetector(cfg.Det)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            self._load_error = f"RapidOCR text detector initialization failed: {exc}"
        return self.text_detector

    @staticmethod
    def _polygon_bbox(points: np.ndarray, *, width: int, height: int) -> tuple[int, int, int, int] | None:
        if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
            return None
        xs = points[:, 0]
        ys = points[:, 1]
        x1 = max(0, min(int(np.floor(float(np.min(xs)))), width))
        y1 = max(0, min(int(np.floor(float(np.min(ys)))), height))
        x2 = max(0, min(int(np.ceil(float(np.max(xs)))), width))
        y2 = max(0, min(int(np.ceil(float(np.max(ys)))), height))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _polygon_tuple(points: np.ndarray) -> tuple[tuple[float, float], ...]:
        return tuple((float(x), float(y)) for x, y in points[:, :2])

    def detect(self, image: np.ndarray) -> tuple[list[ProposalBox], str | None]:
        if image.size == 0:
            return [], "RapidOCR received empty ROI"
        text_detector = self._ensure_text_detector()
        if text_detector is None:
            return [], self._load_error or "RapidOCR text detector unavailable"

        try:
            output = text_detector(image)
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            return [], f"RapidOCR text detection failed: {exc}"

        boxes_raw = getattr(output, "boxes", None)
        if boxes_raw is None:
            return [], "RapidOCR returned no text boxes"
        boxes_arr = np.asarray(boxes_raw, dtype=np.float32)
        if boxes_arr.size == 0:
            return [], "RapidOCR returned no text boxes"
        scores_raw = getattr(output, "scores", None)
        scores_arr = np.asarray(scores_raw, dtype=np.float32) if scores_raw is not None else np.ones((len(boxes_arr),), dtype=np.float32)

        height, width = image.shape[:2]
        proposals: list[ProposalBox] = []
        for idx, box in enumerate(boxes_arr):
            score = float(scores_arr[idx]) if idx < len(scores_arr) else 1.0
            if score < self.min_confidence:
                continue
            bbox = self._polygon_bbox(box, width=width, height=height)
            if bbox is None:
                continue
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < self.min_box_area:
                continue
            proposals.append(
                ProposalBox(
                    bbox_xyxy=bbox,
                    confidence=score,
                    source="rapidocr_ppocrv5",
                    sources=("rapidocr_ppocrv5",),
                    variant_name=f"{self.ocr_version}_{self.model_type}",
                    polygon_xy=self._polygon_tuple(box),
                )
            )

        proposals.sort(
            key=lambda item: (
                float(item.confidence or 0.0),
                (item.bbox_xyxy[2] - item.bbox_xyxy[0]) * (item.bbox_xyxy[3] - item.bbox_xyxy[1]),
            ),
            reverse=True,
        )
        accepted = proposals[: self.max_boxes]
        return accepted, None if accepted else "RapidOCR returned no accepted text boxes"
