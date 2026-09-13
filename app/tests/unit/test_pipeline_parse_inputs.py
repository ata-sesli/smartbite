from __future__ import annotations

from app.ai.ocr import TextDetectionBox
from app.ai.pipeline import ExpiryPipeline
from app.ai.types import OCRResultData


def test_build_parse_inputs_keeps_line_level_date_like_text() -> None:
    ocr = OCRResultData(
        raw_text="SON TUKETIM TARIHI\n23/07/26 ESK\nYF1 03 05:17",
        normalized_text="SON TUKETIM TARIHI 23/07/26 ESK YF1 03 05:17",
        confidence=0.84,
        engine_name="ppocrv5_main",
        runtime_device="cpu",
        reason=None,
    )

    parse_inputs = ExpiryPipeline._build_parse_inputs(ocr)

    assert "23/07/26 ESK" in parse_inputs
    assert "23/07/26" in parse_inputs


def test_build_parse_inputs_normalizes_common_ocr_confusions() -> None:
    ocr = OCRResultData(
        raw_text="EXP O7/I1/2S",
        normalized_text="EXP O7/I1/2S",
        confidence=0.81,
        engine_name="ppocrv5_main",
        runtime_device="cpu",
        reason=None,
    )

    parse_inputs = ExpiryPipeline._build_parse_inputs(ocr)

    assert any("07/11/25" in value for value in parse_inputs)


def test_detector_box_debug_uses_scan_coordinates_and_keeps_roi_coordinates() -> None:
    box = TextDetectionBox(
        x1=10,
        y1=20,
        x2=110,
        y2=50,
        confidence=0.91,
        source="ppocrv5_server",
        sources=("ppocrv5_server",),
        variant_name="raw",
        polygon_xy=((10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0)),
    )

    payload = ExpiryPipeline._detector_box_to_debug_dict(box, offset_x=300, offset_y=400)

    assert payload["bbox_xyxy"] == [310.0, 420.0, 410.0, 450.0]
    assert payload["roi_bbox_xyxy"] == [10.0, 20.0, 110.0, 50.0]
    assert payload["polygon_xy"] == [[310.0, 420.0], [410.0, 420.0], [410.0, 450.0], [310.0, 450.0]]
    assert payload["roi_polygon_xy"] == [[10.0, 20.0], [110.0, 20.0], [110.0, 50.0], [10.0, 50.0]]
