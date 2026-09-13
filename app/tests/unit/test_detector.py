from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ai.detector import ExpiryRegionDetector


def test_named_model_ref_is_allowed() -> None:
    detector = ExpiryRegionDetector(Path("yolo26n.pt"))
    assert detector._resolve_model_source() == "yolo26n.pt"


def test_missing_explicit_model_path_returns_reason(tmp_path: Path) -> None:
    detector = ExpiryRegionDetector(tmp_path / "missing.pt")
    sample = np.zeros((32, 32, 3), dtype=np.uint8)
    result = detector.detect(sample)

    assert result.detected is False
    assert result.reason is not None
    assert "detector model not found" in result.reason
