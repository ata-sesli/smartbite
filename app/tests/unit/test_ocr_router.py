from __future__ import annotations

from pathlib import Path

from app.ai.ocr import OCRConfig, OCRRouter, detect_runtime_device


def test_detect_runtime_explicit_override() -> None:
    assert detect_runtime_device("cpu") == "cpu"
    assert detect_runtime_device("mps") == "mps"
    assert detect_runtime_device("cuda") == "cuda"


def test_router_selects_onnx_on_cpu(tmp_path: Path) -> None:
    router = OCRRouter(
        OCRConfig(
            onnx_model_path=tmp_path / "missing.onnx",
            onnx_charset_path=None,
            paddle_lang="en",
            device_mode="cpu",
        )
    )
    assert router.runtime_device == "cpu"
