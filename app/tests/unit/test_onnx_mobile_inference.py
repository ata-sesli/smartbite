from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.mobile_expiry_pipeline import MobileExpiryPipeline, SVTRTextRecognizer, _RecognitionOutput, _YoloCandidate
from app.ai.onnx_inference import (
    SVTROnnxTextRecognizer,
    build_svtr_ctc_character_dict,
    decode_ctc,
    load_svtr_character_dict,
    obb_xywhr_to_polygon,
    prepare_svtr_input,
)
from app.ai.preprocess import ImageVariant
from app.infra.settings import Settings
from app.scripts.export_onnx_models import build_svtr_export_command, build_yolo_export_kwargs
from app.scripts.mobile_test64_upload_benchmark import _detected_matches_expected, _expected_label


def _image_bytes() -> bytes:
    image = Image.fromarray(np.full((120, 360, 3), 255, dtype=np.uint8), mode="RGB")
    out = BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()


def _pipeline_for_unit_tests() -> MobileExpiryPipeline:
    return MobileExpiryPipeline(
        detector_model_path=Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt"),
        detector_confidence_threshold=0.05,
        detector_imgsz=1024,
        max_candidates=12,
        crop_padding_px=4,
        svtr_model_name="ch_SVTRv2_rec",
        svtr_model_dir=Path("models/svtrv2/smartbite_svtrv2_expdate_rec"),
        svtr_device="cpu",
        parser_min_candidate_confidence=0.4,
    )


class _QueueRecognizer:
    def __init__(self, outputs: list[_RecognitionOutput]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image: np.ndarray) -> _RecognitionOutput:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        if not self.outputs:
            return _RecognitionOutput("", "", None, "empty queue", "original")
        return self.outputs.pop(0)


def test_mobile_rich_ranking_prefers_expiry_date_over_higher_confidence_mfg(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_padded", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [170.0, 20.0], [170.0, 54.0], [20.0, 54.0]],
                    bbox_xyxy=(20, 20, 170, 54),
                    confidence=0.96,
                ),
                _YoloCandidate(
                    polygon_xy=[[20.0, 70.0], [170.0, 70.0], [170.0, 104.0], [20.0, 104.0]],
                    bbox_xyxy=(20, 70, 170, 104),
                    confidence=0.62,
                ),
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("MFG 01/01/2026", "MFG 01/01/2026", 0.99, None, "original"),
            _RecognitionOutput("EXP 01/01/2027", "EXP 01/01/2027", 0.72, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 1, 1)
    assert result.normalized_text == "EXP 01/01/2027"
    assert "candidate_type=single" in (result.reason or "")


def test_mobile_parse_inputs_recover_ocr_confused_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_padded", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [170.0, 20.0], [170.0, 54.0], [20.0, 54.0]],
                    bbox_xyxy=(20, 20, 170, 54),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [_RecognitionOutput("EXP 12/O5/2O27", "EXP 12/O5/2O27", 0.84, None, "original")]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert "parse_inputs=" in (result.reason or "")


def test_mobile_conditional_180_fallback_runs_after_unparseable_original(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_padded", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [170.0, 20.0], [170.0, 54.0], [20.0, 54.0]],
                    bbox_xyxy=(20, 20, 170, 54),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("COMPANY", "COMPANY", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.82, None, "rotate_180"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert "orientation=rotate_180" in (result.reason or "")


def test_mobile_recognition_variant_can_win_when_original_is_empty(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [
            ImageVariant("original_padded", crop, purpose="recognition"),
            ImageVariant("clahe_gray", crop, purpose="recognition"),
        ],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [170.0, 20.0], [170.0, 54.0], [20.0, 54.0]],
                    bbox_xyxy=(20, 20, 170, 54),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("", "", None, "SVTR recognizer returned empty text", "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.82, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert "variant=clahe_gray" in (result.reason or "")


def test_mobile_default_recognition_variants_keep_raw_crop_as_original() -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((24, 80, 3), 127, dtype=np.uint8)

    variants = pipeline._recognition_variants_for_crop(crop)

    assert variants[0].name == "original"
    assert np.array_equal(variants[0].image, crop)
    assert {"clahe_gray", "adaptive_binary", "unsharp_gray"} <= {variant.name for variant in variants}


def test_settings_default_to_existing_runtime_backends() -> None:
    settings = Settings(_env_file=None)

    assert settings.mobile_expiry_detector_backend == "ultralytics"
    assert settings.svtrv2_rec_backend == "paddle"
    assert settings.mobile_expiry_detector_onnx_path == Path(
        "models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.onnx"
    )
    assert settings.svtrv2_rec_onnx_path == Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx")


def test_mobile_pipeline_selects_paddle_fallback_by_default() -> None:
    pipeline = MobileExpiryPipeline(
        detector_model_path=Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt"),
        detector_confidence_threshold=0.05,
        detector_imgsz=1024,
        max_candidates=12,
        crop_padding_px=4,
        svtr_model_name="ch_SVTRv2_rec",
        svtr_model_dir=Path("models/svtrv2/smartbite_svtrv2_expdate_rec"),
        svtr_device="cpu",
        parser_min_candidate_confidence=0.4,
    )

    assert isinstance(pipeline.recognizer, SVTRTextRecognizer)
    assert pipeline.detector_backend == "ultralytics"


def test_mobile_pipeline_can_select_onnx_recognizer_backend() -> None:
    pipeline = MobileExpiryPipeline(
        detector_model_path=Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt"),
        detector_confidence_threshold=0.05,
        detector_imgsz=1024,
        max_candidates=12,
        crop_padding_px=4,
        svtr_model_name="ch_SVTRv2_rec",
        svtr_model_dir=Path("models/svtrv2/smartbite_svtrv2_expdate_rec"),
        svtr_device="cpu",
        parser_min_candidate_confidence=0.4,
        svtr_backend="onnx",
        svtr_onnx_model_path=Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx"),
    )

    assert isinstance(pipeline.recognizer, SVTROnnxTextRecognizer)


def test_load_svtr_character_dict_uses_paddle_inference_yml_order(tmp_path: Path) -> None:
    model_dir = tmp_path / "svtr"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text(
        "\n".join(
            [
                "PostProcess:",
                "  name: CTCLabelDecode",
                "  character_dict:",
                "  - '0'",
                "  - '1'",
                "  - /",
            ]
        ),
        encoding="utf-8",
    )

    assert load_svtr_character_dict(model_dir) == ["0", "1", "/"]


def test_build_svtr_ctc_character_dict_appends_paddle_space_class(tmp_path: Path) -> None:
    model_dir = tmp_path / "svtr"
    model_dir.mkdir()
    (model_dir / "inference.yml").write_text(
        "\n".join(
            [
                "PostProcess:",
                "  name: CTCLabelDecode",
                "  character_dict:",
                "  - '0'",
                "  - '1'",
                "  - /",
            ]
        ),
        encoding="utf-8",
    )

    assert build_svtr_ctc_character_dict(model_dir) == ["0", "1", "/", " "]


def test_decode_ctc_collapses_repeats_and_ignores_blank() -> None:
    # CTCLabelDecode reserves index 0 for blank; character index 1 maps to "0".
    probs = np.array(
        [
            [
                [0.01, 0.90, 0.05, 0.04],
                [0.01, 0.91, 0.04, 0.04],
                [0.01, 0.04, 0.92, 0.03],
                [0.95, 0.02, 0.02, 0.01],
                [0.01, 0.03, 0.04, 0.92],
            ]
        ],
        dtype=np.float32,
    )

    text, confidence = decode_ctc(probs, ["0", "1", "/"])

    assert text == "01/"
    assert confidence == np.float32((0.90 + 0.92 + 0.92) / 3).item()


def test_prepare_svtr_input_pads_to_paddle_shape() -> None:
    image = np.full((24, 60, 3), 255, dtype=np.uint8)

    tensor = prepare_svtr_input(image)

    assert tensor.shape == (1, 3, 48, 320)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= -1.0


def test_prepare_svtr_input_uses_dynamic_width_for_wide_crops() -> None:
    image = np.full((100, 1000, 3), 255, dtype=np.uint8)

    tensor = prepare_svtr_input(image)

    assert tensor.shape == (1, 3, 48, 480)


def test_obb_xywhr_to_polygon_returns_four_points() -> None:
    polygon = obb_xywhr_to_polygon(cx=20.0, cy=10.0, width=8.0, height=4.0, angle_rad=0.0)

    assert polygon == [[16.0, 8.0], [24.0, 8.0], [24.0, 12.0], [16.0, 12.0]]


def test_export_helpers_lock_expected_model_paths() -> None:
    yolo_kwargs = build_yolo_export_kwargs(
        model_path=Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt"),
        imgsz=1024,
    )

    assert yolo_kwargs == {
        "format": "onnx",
        "imgsz": 1024,
        "opset": 17,
        "simplify": True,
        "dynamic": False,
        "nms": False,
    }

    command = build_svtr_export_command(
        model_dir=Path("models/svtrv2/smartbite_svtrv2_expdate_rec"),
        save_file=Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx"),
    )

    assert command[:5] == [
        "paddle2onnx",
        "--model_dir",
        "models/svtrv2/smartbite_svtrv2_expdate_rec",
        "--model_filename",
        "inference.json",
    ]
    assert "--enable_onnx_checker" in command


def test_mobile_test64_upload_scoring_accepts_month_precision_expected_label() -> None:
    expected = _expected_label(
        {
            "expected_day": None,
            "expected_month": 10,
            "expected_year": 2026,
        }
    )

    assert expected["label"] == "2026-10"
    assert expected["precision"] == "month"
    assert _detected_matches_expected("2026-10-31", expected)
    assert not _detected_matches_expected("2026-11-01", expected)


def test_mobile_test64_upload_scoring_requires_day_when_label_has_day() -> None:
    expected = _expected_label(
        {
            "expected_day": 11,
            "expected_month": 4,
            "expected_year": 2026,
        }
    )

    assert expected["label"] == "2026-04-11"
    assert expected["precision"] == "day"
    assert _detected_matches_expected("2026-04-11", expected)
    assert not _detected_matches_expected("2026-04-12", expected)
