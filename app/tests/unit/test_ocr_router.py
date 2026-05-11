from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from app.ai.ocr import (
    CRAFTTextDetector,
    EnsembleTextDetector,
    OCRConfig,
    OCRRouter,
    PPOCRV5TextDetector,
    PPOCRV5_SERVER_DETECTOR_SOURCE,
    RecognitionData,
    TextDetectionBox,
    detect_runtime_device,
)


def test_detect_runtime_explicit_override() -> None:
    assert detect_runtime_device("cpu") == "cpu"
    assert detect_runtime_device("mps") == "mps"
    assert detect_runtime_device("cuda") == "cuda"


class StubFinalRecognizer:
    engine_name = "svtrv2"
    model_name = "ch_SVTRv2_rec"

    def __init__(self) -> None:
        self.runtime_device = "cpu"
        self.calls = 0

    def recognize(self, image: np.ndarray) -> RecognitionData:
        _ = image
        self.calls += 1
        return RecognitionData(
            raw_text="SVTR 12/12/2027",
            normalized_text="SVTR 12/12/2027",
            confidence=0.91,
            reason=None,
        )

    def runtime_info(self) -> dict[str, object]:
        return {
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "runtime_device": self.runtime_device,
            "model_loaded": True,
        }

    def warmup(self) -> dict[str, object]:
        return self.runtime_info()


class StubNamedRecognizer(StubFinalRecognizer):
    def __init__(self, *, engine_name: str, model_name: str, text: str) -> None:
        super().__init__()
        self.engine_name = engine_name
        self.model_name = model_name
        self.text = text

    def recognize(self, image: np.ndarray) -> RecognitionData:
        _ = image
        self.calls += 1
        return RecognitionData(
            raw_text=self.text,
            normalized_text=self.text,
            confidence=0.91,
            reason=None,
        )


def test_router_defaults_to_svtrv2_final_recognizer_without_parseq(tmp_path: Path, monkeypatch) -> None:
    stub = StubFinalRecognizer()
    monkeypatch.setattr("app.ai.ocr.PaddleTextRecognizer", lambda **kwargs: stub, raising=False)

    def fail_parseq(*args, **kwargs):
        raise AssertionError("PARSeq should not be initialized when SVTRv2 is configured")

    monkeypatch.setattr("app.ai.ocr.ParSeqRecognizer", fail_parseq)

    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=tmp_path / "missing-parseq",
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=False,
            ppocrv5_det_db_thresh=0.3,
        )
    )

    assert router.active_engine_name == "svtrv2"
    assert router.final_recognizer_name == "svtrv2"
    assert router.final_recognizer_model_name == "ch_SVTRv2_rec"

    sample = np.zeros((32, 128, 3), dtype=np.uint8)
    result = router.recognize_final(sample)
    legacy_result = router.recognize_parseq(sample)
    assert result.raw_text == "SVTR 12/12/2027"
    assert legacy_result.raw_text == "SVTR 12/12/2027"
    assert stub.calls == 2
    assert router.final_recognition_runtime_info()["engine_name"] == "svtrv2"


def test_router_exposes_separate_context_recognizer_role(tmp_path: Path, monkeypatch) -> None:
    final_stub = StubNamedRecognizer(engine_name="svtrv2", model_name="fine_tuned_svtrv2", text="13/02/2031")
    context_stub = StubNamedRecognizer(engine_name="context_svtrv2", model_name="ch_SVTRv2_rec", text="TETT")
    created: list[tuple[str, str]] = []

    def make_recognizer(**kwargs):
        engine_name = str(kwargs["engine_name"])
        model_name = str(kwargs["model_name"])
        created.append((engine_name, model_name))
        if engine_name == "context_svtrv2":
            return context_stub
        return final_stub

    monkeypatch.setattr("app.ai.ocr.PaddleTextRecognizer", make_recognizer, raising=False)

    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=tmp_path / "missing-parseq",
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=False,
            ppocrv5_det_db_thresh=0.3,
            svtrv2_rec_model_name="fine_tuned_svtrv2",
            svtrv2_rec_model_dir=tmp_path / "fine-tuned",
            context_svtrv2_rec_model_name="ch_SVTRv2_rec",
            context_svtrv2_rec_model_dir=tmp_path / "default-svtr",
        )
    )

    sample = np.zeros((32, 128, 3), dtype=np.uint8)
    assert router.recognize_final(sample).raw_text == "13/02/2031"
    assert router.recognize_context(sample).raw_text == "TETT"
    assert final_stub.calls == 1
    assert context_stub.calls == 1
    assert ("svtrv2", "fine_tuned_svtrv2") in created
    assert ("context_svtrv2", "ch_SVTRv2_rec") in created
    assert router.context_recognizer_model_name == "ch_SVTRv2_rec"
    assert router.context_recognition_runtime_info()["engine_name"] == "context_svtrv2"


def test_router_exposes_parseq_engine_name_when_configured(tmp_path: Path) -> None:
    parseq_dir = tmp_path / "parseq"
    parseq_dir.mkdir()
    # Touch a placeholder file to satisfy directory checks in model validation path.
    (parseq_dir / "pytorch_model.bin").write_bytes(b"not-a-real-model")

    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=parseq_dir,
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            expiry_recognizer="parseq",
        )
    )
    assert router.active_engine_name == "parseq_small"
    assert router.substitute_available is False
    assert router.text_detector_mode == "ensemble"


def test_ppocrv5_text_detector_forwards_non_null_db_knobs(monkeypatch, tmp_path: Path) -> None:
    created_kwargs: dict[str, object] = {}

    class FakeTextDetection:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs.update(kwargs)

        def predict(self, image: np.ndarray) -> list[dict[str, object]]:
            _ = image
            return [
                {
                    "dt_polys": np.array([[[1, 2], [12, 2], [12, 8], [1, 8]]], dtype=np.float32),
                    "dt_scores": np.array([0.88], dtype=np.float32),
                }
            ]

    model_dir = tmp_path / "det"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text("model: fake\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=FakeTextDetection))

    detector = PPOCRV5TextDetector(
        model_dir=model_dir,
        model_name="PP-OCRv5_server_det",
        language="en",
        runtime_device="cpu",
        det_db_thresh=0.25,
        det_db_box_thresh=0.35,
        det_limit_side_len=1216,
        det_limit_type="max",
        det_db_unclip_ratio=1.8,
    )

    boxes, reason = detector.detect(np.zeros((16, 24, 3), dtype=np.uint8), variant_name="raw")

    assert reason is None
    assert len(boxes) == 1
    assert created_kwargs["device"] == "cpu"
    assert created_kwargs["thresh"] == 0.25
    assert created_kwargs["box_thresh"] == 0.35
    assert created_kwargs["limit_side_len"] == 1216
    assert created_kwargs["limit_type"] == "max"
    assert created_kwargs["unclip_ratio"] == 1.8


def test_ppocrv5_text_detector_omits_null_optional_db_knobs(monkeypatch, tmp_path: Path) -> None:
    created_kwargs: dict[str, object] = {}

    class FakeTextDetection:
        def __init__(self, **kwargs: object) -> None:
            created_kwargs.update(kwargs)

        def predict(self, image: np.ndarray) -> list[dict[str, object]]:
            _ = image
            return [{"dt_polys": []}]

    model_dir = tmp_path / "det"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text("model: fake\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextDetection=FakeTextDetection))

    detector = PPOCRV5TextDetector(
        model_dir=model_dir,
        model_name="PP-OCRv5_server_det",
        language="en",
        runtime_device="cpu",
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_limit_side_len=None,
        det_limit_type=None,
        det_db_unclip_ratio=None,
    )

    detector.detect(np.zeros((16, 24, 3), dtype=np.uint8), variant_name="raw")

    assert "limit_side_len" not in created_kwargs
    assert "limit_type" not in created_kwargs
    assert "unclip_ratio" not in created_kwargs


def test_missing_parseq_assets_return_explicit_reason(tmp_path: Path) -> None:
    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=tmp_path / "missing-parseq",
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            expiry_recognizer="parseq",
        )
    )

    import numpy as np

    sample = np.zeros((32, 128, 3), dtype=np.uint8)
    result = router.run(sample)
    assert result.engine_name == "parseq_small"
    assert result.reason is not None
    assert "PARSeq model directory not found" in result.reason


class StubDetector:
    def __init__(
        self,
        boxes: list[TextDetectionBox],
        reason: str | None = None,
        *,
        source_name: str,
    ) -> None:
        self.boxes = boxes
        self.reason = reason
        self.source_name = source_name
        self.calls: list[str | None] = []

    def detect(self, image: np.ndarray, *, variant_name: str | None = None) -> tuple[list[TextDetectionBox], str | None]:
        _ = image
        self.calls.append(variant_name)
        return self.boxes, self.reason


def test_ensemble_returns_ppocr_boxes_when_craft_unavailable() -> None:
    ppocr = StubDetector(
        [TextDetectionBox(10, 10, 60, 30, 0.9, source="ppocrv5_server", sources=("ppocrv5_server",))],
        source_name="ppocrv5_server",
    )
    craft = StubDetector([], "craft unavailable", source_name="craft")
    ensemble = EnsembleTextDetector(ppocr_detector=ppocr, craft_detector=craft)

    boxes, reason = ensemble.detect(np.zeros((80, 120, 3), dtype=np.uint8), variant_name="raw")

    assert len(boxes) == 1
    assert boxes[0].source == "ppocrv5_server"
    assert boxes[0].sources == ("ppocrv5_server",)
    assert boxes[0].variant_name == "raw"
    assert reason == "craft unavailable"


def test_ensemble_merges_overlapping_ppocr_and_craft_boxes() -> None:
    ppocr = StubDetector(
        [TextDetectionBox(10, 10, 60, 30, 0.8, source="ppocrv5_server", sources=("ppocrv5_server",))],
        source_name="ppocrv5_server",
    )
    craft = StubDetector(
        [TextDetectionBox(14, 8, 62, 32, 0.7, source="craft", sources=("craft",))],
        source_name="craft",
    )
    ensemble = EnsembleTextDetector(ppocr_detector=ppocr, craft_detector=craft)

    boxes, reason = ensemble.detect(np.zeros((80, 120, 3), dtype=np.uint8), variant_name="raw")

    assert reason is None
    assert len(boxes) == 1
    assert boxes[0].x1 == 10
    assert boxes[0].y1 == 8
    assert boxes[0].x2 == 62
    assert boxes[0].y2 == 32
    assert boxes[0].source == "ensemble"
    assert boxes[0].sources == ("ppocrv5_server", "craft")


def test_ensemble_keeps_craft_only_boxes() -> None:
    ppocr = StubDetector([], source_name="ppocrv5_server")
    craft = StubDetector(
        [TextDetectionBox(20, 20, 80, 42, 0.72, source="craft", sources=("craft",))],
        source_name="craft",
    )
    ensemble = EnsembleTextDetector(ppocr_detector=ppocr, craft_detector=craft)

    boxes, _reason = ensemble.detect(np.zeros((80, 120, 3), dtype=np.uint8), variant_name="raw")

    assert len(boxes) == 1
    assert boxes[0].source == "craft"
    assert boxes[0].sources == ("craft",)


def test_router_ensemble_detect_text_boxes_runs_ppocr_only(tmp_path: Path, monkeypatch) -> None:
    parseq_dir = tmp_path / "parseq"
    parseq_dir.mkdir()

    ppocr = StubDetector(
        [TextDetectionBox(1, 2, 30, 14, 0.91, source="ppocrv5_server", sources=("ppocrv5_server",))],
        source_name="ppocrv5_server",
    )
    craft = StubDetector(
        [TextDetectionBox(5, 6, 42, 18, 0.7, source="craft", sources=("craft",))],
        source_name="craft",
    )

    monkeypatch.setattr("app.ai.ocr.PPOCRV5TextDetector", lambda **kwargs: ppocr)
    monkeypatch.setattr("app.ai.ocr.CRAFTTextDetector", lambda **kwargs: craft)

    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=parseq_dir,
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            text_detector_mode="ensemble",
        )
    )

    boxes, reason = router.detect_text_boxes(np.zeros((40, 80, 3), dtype=np.uint8), variant_name="raw")

    assert reason is None
    assert len(boxes) == 1
    assert boxes[0].source == PPOCRV5_SERVER_DETECTOR_SOURCE
    assert ppocr.calls == ["raw"]
    assert craft.calls == []


def test_router_exposes_explicit_craft_detection(tmp_path: Path, monkeypatch) -> None:
    parseq_dir = tmp_path / "parseq"
    parseq_dir.mkdir()

    ppocr = StubDetector([], source_name="ppocrv5_server")
    craft = StubDetector(
        [TextDetectionBox(5, 6, 42, 18, 0.7, source="craft", sources=("craft",))],
        source_name="craft",
    )

    monkeypatch.setattr("app.ai.ocr.PPOCRV5TextDetector", lambda **kwargs: ppocr)
    monkeypatch.setattr("app.ai.ocr.CRAFTTextDetector", lambda **kwargs: craft)

    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=parseq_dir,
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            text_detector_mode="ensemble",
        )
    )

    boxes, reason = router.detect_craft_text_boxes(np.zeros((40, 80, 3), dtype=np.uint8), variant_name="raw")

    assert reason is None
    assert len(boxes) == 1
    assert boxes[0].source == "craft"
    assert ppocr.calls == []
    assert craft.calls == ["raw"]


def test_router_merge_text_boxes_preserves_multi_source(tmp_path: Path, monkeypatch) -> None:
    parseq_dir = tmp_path / "parseq"
    parseq_dir.mkdir()

    monkeypatch.setattr("app.ai.ocr.PPOCRV5TextDetector", lambda **kwargs: StubDetector([], source_name="ppocrv5_server"))
    monkeypatch.setattr("app.ai.ocr.CRAFTTextDetector", lambda **kwargs: StubDetector([], source_name="craft"))
    router = OCRRouter(
        OCRConfig(
            parseq_model_dir=parseq_dir,
            paddle_lang="en",
            device_mode="cpu",
            parseq_device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            text_detector_mode="ensemble",
        )
    )

    boxes = router.merge_text_boxes(
        [TextDetectionBox(10, 10, 60, 30, 0.9, source="ppocrv5_server", sources=("ppocrv5_server",))],
        [TextDetectionBox(14, 8, 62, 32, 0.7, source="craft", sources=("craft",))],
    )

    assert len(boxes) == 1
    assert boxes[0].source == "ensemble"
    assert boxes[0].sources == ("ppocrv5_server", "craft")


def test_craft_detector_respects_variant_allowlist(tmp_path: Path) -> None:
    detector = CRAFTTextDetector(
        model_path=tmp_path / "missing.pth",
        runtime_device="cpu",
        enabled=True,
        variant_names=("raw", "luma_clahe"),
    )

    boxes, reason = detector.detect(np.zeros((80, 120, 3), dtype=np.uint8), variant_name="unsharp")

    assert boxes == []
    assert reason is None


def test_craft_detector_targets_craft_detector_package(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "craft_mlt_25k.pth"
    model_path.write_bytes(b"not-real-weights")
    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "craft_detector" or name.startswith("craft_detector."):
            raise ImportError("blocked craft_detector")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    detector = CRAFTTextDetector(
        model_path=model_path,
        runtime_device="cpu",
        enabled=True,
        variant_names=("raw",),
    )

    boxes, reason = detector.detect(np.zeros((80, 120, 3), dtype=np.uint8), variant_name="raw")

    assert boxes == []
    assert reason is not None
    assert "craft_detector import failed" in reason
