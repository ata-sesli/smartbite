from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.mobile_expiry_pipeline import (
    MOBILE_RECOGNITION_VARIANTS,
    MobileExpiryPipeline,
    SVTRTextRecognizer,
    _EvaluatedCandidate,
    _MobileGlobalCandidate,
    _MobileRankedCandidate,
    _RecognitionOutput,
    _YoloCandidate,
)
from app.ai.crop_normalization import NormalizedTextLineCrop
from app.ai.expiry_candidate_engine import ProposalBox
from app.ai.ocr import TextDetectionBox
from app.ai.pipeline import ExpiryPipeline
from app.ai.rapidocr_text_detector import RapidOCRTextProposalDetector
from app.ai.onnx_inference import (
    SVTROnnxTextRecognizer,
    build_svtr_ctc_character_dict,
    decode_ctc,
    load_svtr_character_dict,
    obb_xywhr_to_polygon,
    prepare_svtr_input,
)
from app.ai.preprocess import ImageVariant
from app.ai.types import ParsedDateData
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
        rapidocr_primary_enabled=False,
        rapidocr_rescue_enabled=False,
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


class _FakeRapidOCRTextDetector:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def detect(self, image: np.ndarray) -> tuple[list[ProposalBox], str | None]:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        return [
            ProposalBox(
                bbox_xyxy=(4, 5, 84, 23),
                confidence=0.91,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name="PP-OCRv5_mobile",
                polygon_xy=((4.0, 5.0), (84.0, 5.0), (84.0, 23.0), (4.0, 23.0)),
            )
        ], None


class _FakeMultiRapidOCRTextDetector:
    def __init__(self, boxes: list[ProposalBox]) -> None:
        self.boxes = boxes
        self.calls: list[tuple[int, int]] = []

    def detect(self, image: np.ndarray) -> tuple[list[ProposalBox], str | None]:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        return self.boxes, None


class _WidthAwareRecognizer:
    def __init__(self, *, width_threshold: int, narrow: _RecognitionOutput, wide: _RecognitionOutput) -> None:
        self.width_threshold = width_threshold
        self.narrow = narrow
        self.wide = wide
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image: np.ndarray) -> _RecognitionOutput:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        width = int(image.shape[1]) if image.ndim >= 2 else 0
        return self.wide if width >= self.width_threshold else self.narrow


def _mobile_ranked_candidate(
    *,
    candidate_type: str = "single",
    bbox_xyxy: tuple[int, int, int, int] = (10, 10, 80, 30),
    member_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> _MobileRankedCandidate:
    return _MobileRankedCandidate(
        candidate_id="cand_1",
        candidate_type=candidate_type,
        bbox_xyxy=bbox_xyxy,
        polygon_xy=None,
        detector_confidence=0.9,
        detector_sources=("yolo26s_obb", "rapidocr_ppocrv5"),
        detector_variant="rapidocr_primary:PP-OCRv5_server",
        member_indices=[0],
        member_bboxes=member_bboxes or [bbox_xyxy],
        geometry_features={"area_ratio": 0.01},
        geometry_score=1.0,
        score_breakdown={},
        total_score=1.0,
        recognition_bbox=bbox_xyxy,
        evidence_bbox=bbox_xyxy,
    )


def _evaluated_candidate_for_score(
    *,
    text: str,
    parsed_date: date,
    precision: str,
    confidence: float,
) -> _EvaluatedCandidate:
    return _EvaluatedCandidate(
        candidate=_mobile_ranked_candidate(),
        recognition=_RecognitionOutput(text, text, 0.9, None, "original"),
        parsed=ParsedDateData(
            parsed_date=parsed_date,
            date_format_detected="test",
            confidence=confidence,
            candidates=[text],
            reason="selected_test",
            date_precision=precision,
            parsed_day=parsed_date.day if precision == "day" else None,
            parsed_month=parsed_date.month,
            parsed_year=parsed_date.year,
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=None,
    )


def test_rapidocr_ppocrv5_detector_maps_boxes_to_proposals() -> None:
    class FakeDetOutput:
        boxes = np.array([[[4, 5], [84, 5], [84, 23], [4, 23]]], dtype=np.float32)
        scores = np.array([0.91], dtype=np.float32)

    class FakeTextDetector:
        def __call__(self, image: np.ndarray) -> FakeDetOutput:
            assert image.shape == (40, 100, 3)
            return FakeDetOutput()

    detector = RapidOCRTextProposalDetector(text_detector=FakeTextDetector(), max_boxes=5)

    proposals, reason = detector.detect(np.full((40, 100, 3), 255, dtype=np.uint8))

    assert reason is None
    assert len(proposals) == 1
    assert proposals[0].bbox_xyxy == (4, 5, 84, 23)
    assert proposals[0].sources == ("rapidocr_ppocrv5",)


def test_rapidocr_ppocrv5_detector_accepts_mobile_and_server_model_types() -> None:
    class FakeDetOutput:
        boxes = np.array([[[4, 5], [84, 5], [84, 23], [4, 23]]], dtype=np.float32)
        scores = np.array([0.91], dtype=np.float32)

    class FakeTextDetector:
        def __call__(self, _image: np.ndarray) -> FakeDetOutput:
            return FakeDetOutput()

    mobile = RapidOCRTextProposalDetector(text_detector=FakeTextDetector(), model_type="mobile")
    server = RapidOCRTextProposalDetector(text_detector=FakeTextDetector(), model_type="server")

    mobile_proposals, mobile_reason = mobile.detect(np.full((40, 100, 3), 255, dtype=np.uint8))
    server_proposals, server_reason = server.detect(np.full((40, 100, 3), 255, dtype=np.uint8))

    assert mobile_reason is None
    assert server_reason is None
    assert mobile.model_type == "mobile"
    assert server.model_type == "server"
    assert mobile_proposals[0].variant_name == "PP-OCRv5_mobile"
    assert server_proposals[0].variant_name == "PP-OCRv5_server"


def test_mobile_rich_ranking_prefers_expiry_date_over_higher_confidence_mfg(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.max_group_candidates_per_roi = 0
    pipeline.expiry_global_probe_top_k = 2
    pipeline.expiry_global_final_top_k = 2
    pipeline.expiry_global_debug_final_top_k = 2
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
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
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
            _RecognitionOutput("EXP 12/O5/2O27", "EXP 12/O5/2O27", 0.84, None, "original"),
            _RecognitionOutput("EXP 12/O5/2O27", "EXP 12/O5/2O27", 0.84, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert "parse_inputs=" in (result.reason or "")


def test_mobile_conditional_180_fallback_runs_after_unparseable_original(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
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
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
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
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.82, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert "variant=clahe_gray" in (result.reason or "")


def test_mobile_recognition_variant_prefers_later_parseable_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
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
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.99, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.81, None, "original"),
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.99, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.81, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 1, 1)
    assert "variant=clahe_gray" in (result.reason or "")


def test_mobile_candidate_crop_selection_prefers_later_parseable_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    crops = [
        NormalizedTextLineCrop(
            image=np.full((24, 96, 3), 255, dtype=np.uint8),
            bbox_xyxy=(10, 10, 80, 30),
            polygon_used=False,
            crop_transform_used="bbox_raw",
            selected_orientation="original",
            original_crop_shape=[24, 96, 3],
            normalized_crop_shape=[24, 96, 3],
            orientation_candidates_tried=["original"],
            selected_transform_reason="unit_test",
        ),
        NormalizedTextLineCrop(
            image=np.full((24, 104, 3), 255, dtype=np.uint8),
            bbox_xyxy=(10, 32, 88, 52),
            polygon_used=False,
            crop_transform_used="bbox_raw",
            selected_orientation="original",
            original_crop_shape=[24, 104, 3],
            normalized_crop_shape=[24, 104, 3],
            orientation_candidates_tried=["original"],
            selected_transform_reason="unit_test",
        ),
    ]
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: crops)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_color_tight", crop, purpose="recognition")],
        raising=False,
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.99, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.81, None, "original"),
        ]
    )

    evaluated = pipeline._evaluate_candidate(np.full((120, 360, 3), 255, dtype=np.uint8), candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2027, 1, 1)
    assert evaluated.recognition.raw_text == "01/01/2027"


def test_mobile_records_all_variant_evidence_for_a_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((32, 128, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda _crop, *, allowed_names=None: [
            ImageVariant("original", crop, purpose="recognition"),
            ImageVariant("clahe_gray", crop, purpose="recognition"),
        ],
        raising=False,
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("020", "020", 0.7, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.9, None, "original"),
        ]
    )

    evidence = pipeline._collect_recognition_evidence(crop, today=date(2026, 1, 1))

    assert [item.variant_name for item in evidence] == ["original", "clahe_gray"]
    assert evidence[1].parsed.parsed_date == date(2027, 5, 12)


def test_mobile_rapidocr_ppocrv5_rescue_runs_only_after_primary_failure(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = False
    pipeline.proposal_rescue_backend = "rapidocr_ppocrv5"
    pipeline.rapidocr_rescue_enabled = True
    pipeline.rapidocr_max_rois_per_scan = 1
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
    pipeline.max_group_candidates_per_roi = 0
    fake_rapid = _FakeRapidOCRTextDetector()
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [180.0, 20.0], [180.0, 70.0], [20.0, 70.0]],
                    bbox_xyxy=(20, 20, 180, 70),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("BRAND", "BRAND", 0.92, None, "original"),
            _RecognitionOutput("BRAND", "BRAND", 0.92, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert fake_rapid.calls == [(66, 176)]
    assert "detector_sources=yolo26s_obb,rapidocr_ppocrv5" in (result.reason or "")


def test_mobile_rapidocr_ppocrv5_primary_proposals_feed_advanced_engine(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_max_rois_per_scan = 1
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
    pipeline.max_group_candidates_per_roi = 0
    fake_rapid = _FakeRapidOCRTextDetector()
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [180.0, 20.0], [180.0, 70.0], [20.0, 70.0]],
                    bbox_xyxy=(20, 20, 180, 70),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert fake_rapid.calls == [(66, 176)]
    assert "detector_sources=yolo26s_obb,rapidocr_ppocrv5" in (result.reason or "")
    assert "variant=original" in (result.reason or "")


def test_mobile_combined_primary_pool_includes_whole_yolo_roi_and_ppocr_boxes() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    fake_rapid = _FakeRapidOCRTextDetector()
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    roi = np.full((66, 176, 3), 255, dtype=np.uint8)

    proposals, reason = pipeline._combined_primary_proposals_for_roi(
        roi,
        variant=ImageVariant("raw", roi, purpose="detector"),
        roi_confidence=0.88,
    )

    assert reason is None
    assert fake_rapid.calls == [(66, 176)]
    assert proposals[0].variant_name == "yolo_roi_whole"
    assert proposals[0].bbox_xyxy == (0, 0, 176, 66)
    assert proposals[0].sources == ("yolo26s_obb",)
    assert any(proposal.source == "rapidocr_ppocrv5" for proposal in proposals[1:])


def test_mobile_combined_primary_allows_yolo_crop_to_beat_wrong_ppocr_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_primary_max_rois_per_scan = 1
    pipeline.expiry_global_probe_top_k = 4
    pipeline.expiry_global_final_top_k = 4
    pipeline.expiry_global_debug_final_top_k = 4
    pipeline.max_group_candidates_per_roi = 0
    pipeline.context_probe_enabled = False
    fake_rapid = _FakeRapidOCRTextDetector()
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [180.0, 20.0], [180.0, 70.0], [20.0, 70.0]],
                    bbox_xyxy=(20, 20, 180, 70),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _WidthAwareRecognizer(
        width_threshold=120,
        narrow=_RecognitionOutput("MFG 01/01/2026", "MFG 01/01/2026", 0.99, None, "original"),
        wide=_RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.82, None, "original"),
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert result.normalized_text == "EXP 12/05/2027"
    assert result.final_recognition_bbox_xyxy == [14, 14, 186, 76]
    assert "detector_sources=yolo26s_obb" in (result.reason or "")
    assert "detector_variant=yolo26s_obb" in (result.reason or "")


def test_mobile_rapidocr_primary_uses_top_twelve_yolo_rois(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_primary_max_rois_per_scan = 12
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
    pipeline.max_group_candidates_per_roi = 0
    fake_rapid = _FakeRapidOCRTextDetector()
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[float(10 + i * 10), 20.0], [float(90 + i * 10), 20.0], [float(90 + i * 10), 50.0], [float(10 + i * 10), 50.0]],
                    bbox_xyxy=(10 + i * 10, 20, 90 + i * 10, 50),
                    confidence=1.0 - i * 0.01,
                )
                for i in range(13)
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert len(fake_rapid.calls) == 12


def test_mobile_advanced_primary_uses_server_ppocr_text_boxes_without_yolo_bands(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_model_type = "server"
    pipeline.rapidocr_primary_max_rois_per_scan = 1
    pipeline.rapidocr_primary_max_boxes_accepted = 12
    pipeline.expiry_global_probe_top_k = 2
    pipeline.expiry_global_final_top_k = 2
    pipeline.expiry_global_debug_final_top_k = 2
    pipeline.max_group_candidates_per_roi = 0
    fake_rapid = _FakeMultiRapidOCRTextDetector(
        [
            ProposalBox(
                bbox_xyxy=(4, 5, 84, 23),
                confidence=0.91,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name="PP-OCRv5_server",
                polygon_xy=((4.0, 5.0), (84.0, 5.0), (84.0, 23.0), (4.0, 23.0)),
            ),
            ProposalBox(
                bbox_xyxy=(4, 31, 120, 49),
                confidence=0.88,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name="PP-OCRv5_server",
                polygon_xy=((4.0, 31.0), (120.0, 31.0), (120.0, 49.0), (4.0, 49.0)),
            ),
        ]
    )
    pipeline._rapidocr_detector = fake_rapid  # type: ignore[assignment]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[20.0, 20.0], [180.0, 20.0], [180.0, 70.0], [20.0, 70.0]],
                    bbox_xyxy=(20, 20, 180, 70),
                    confidence=0.88,
                )
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("EXP", "EXP", 0.80, None, "original"),
            _RecognitionOutput("12/05/2027", "12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.94, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.94, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert fake_rapid.calls == [(66, 176)]
    assert "detector_sources=yolo26s_obb,rapidocr_ppocrv5" in (result.reason or "")
    assert "rapidocr_primary:PP-OCRv5_server" in (result.reason or "")
    assert "yolo_band_" not in (result.reason or "")


def test_mobile_context_probe_scores_adjacent_expiry_and_production_keywords(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    candidate.context_probe_bboxes = [(1, 10, 8, 30), (82, 10, 115, 30)]
    global_candidate = _MobileGlobalCandidate(
        source="detector_box_1",
        variant="raw",
        variant_key="detector_box_1:raw",
        image=np.full((50, 130, 3), 255, dtype=np.uint8),
        detector_confidence=0.9,
        ranked=candidate,
        scan_bbox=candidate.bbox_xyxy,
    )
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("12/05/2027", "12/05/2027", 0.90, None, "original"),
            _RecognitionOutput("EXP", "EXP", 0.88, None, "original"),
            _RecognitionOutput("MFG", "MFG", 0.86, None, "original"),
        ]
    )

    pipeline._probe_global_candidates([global_candidate], today=date(2026, 1, 1))

    assert candidate.context_probe_texts == ["EXP", "MFG"]
    assert candidate.adjacent_expiry_keyword is True
    assert candidate.adjacent_production_keyword is True
    assert candidate.score_breakdown["adjacent_expiry_keyword_bonus"] == 4.0
    assert candidate.score_breakdown["keyword_date_group_bonus"] == 2.0


def test_mobile_final_selection_prefers_day_precision_over_later_month_only_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    exact_day = _evaluated_candidate_for_score(
        text="10/09/2026",
        parsed_date=date(2026, 9, 10),
        precision="day",
        confidence=0.80,
    )
    month_only = _evaluated_candidate_for_score(
        text="09/2026",
        parsed_date=date(2026, 9, 30),
        precision="month",
        confidence=0.99,
    )

    selected = max([exact_day, month_only], key=pipeline._score_parseable_candidate)

    assert selected is exact_day


def test_mobile_suppresses_weak_low_detector_auto_parse() -> None:
    pipeline = _pipeline_for_unit_tests()
    weak = _EvaluatedCandidate(
        candidate=_mobile_ranked_candidate(),
        recognition=_RecognitionOutput("203 26 210123", "203 26 210123", 0.76, None, "original"),
        parsed=ParsedDateData(
            parsed_date=date(2023, 1, 21),
            date_format_detected="DDMMYY",
            confidence=0.55,
            candidates=["203 26 210123"],
            reason="selected_DDMMYY",
        ),
        parse_inputs_count=3,
        recognition_variant="unsharp_gray",
        normalized_crop=None,
    )
    weak.candidate.detector_confidence = 0.05

    strong = _EvaluatedCandidate(
        candidate=_mobile_ranked_candidate(),
        recognition=_RecognitionOutput("15/03/2022", "15/03/2022", 0.99, None, "original"),
        parsed=ParsedDateData(
            parsed_date=date(2022, 3, 15),
            date_format_detected="DD/MM/YYYY",
            confidence=0.95,
            candidates=["15/03/2022"],
            reason="selected_DD/MM/YYYY",
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=None,
    )
    strong.candidate.detector_confidence = 0.05

    assert pipeline._should_suppress_auto_parse(weak)
    assert not pipeline._should_suppress_auto_parse(strong)


def test_mobile_final_crop_tightens_multiline_group_like_advanced() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((80, 160, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(10, 10, 130, 60),
        member_bboxes=[(10, 10, 130, 25), (10, 42, 120, 58)],
    )
    candidate.probe_text = "EXP 12/05/2027\nMFG 01/01/2026"

    crops = pipeline._normalized_crops_for_candidate(image, candidate)

    assert crops
    assert candidate.final_crop_policy == "line_from_multiline_group"
    assert candidate.final_crop_bbox is not None
    assert candidate.final_crop_bbox[3] < 42
    assert candidate.final_crop_padding_px == 5


def test_mobile_default_recognition_variants_keep_raw_crop_as_original() -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((24, 80, 3), 127, dtype=np.uint8)

    variants = pipeline._recognition_variants_for_crop(crop)

    assert tuple(variant.name for variant in variants) == MOBILE_RECOGNITION_VARIANTS
    assert variants[0].name == "original_color_tight"
    assert np.array_equal(variants[0].image, crop)
    assert len({variant.name for variant in variants}) == len(MOBILE_RECOGNITION_VARIANTS)


def test_mobile_expands_yolo_roi_into_text_evidence_proposals() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo = _YoloCandidate(
        polygon_xy=[[20.0, 20.0], [220.0, 20.0], [220.0, 100.0], [20.0, 100.0]],
        bbox_xyxy=(20, 20, 220, 100),
        confidence=0.9,
    )

    proposals = pipeline._expanded_yolo_text_proposals(np.full((160, 320, 3), 255, dtype=np.uint8), [yolo])
    names = {proposal.variant_name for proposal in proposals}

    assert "yolo_whole" in names
    assert "yolo_band_top" in names
    assert "yolo_band_mid" in names
    assert "yolo_band_bottom" in names
    assert "yolo_left_context" in names
    assert "yolo_right_context" in names


def test_mobile_yolo_adapter_builds_advanced_candidate_types() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((160, 320, 3), 255, dtype=np.uint8)
    yolo_candidates = [
        _YoloCandidate(
            polygon_xy=[[20.0, 30.0], [92.0, 30.0], [92.0, 50.0], [20.0, 50.0]],
            bbox_xyxy=(20, 30, 92, 50),
            confidence=0.91,
        ),
        _YoloCandidate(
            polygon_xy=[[100.0, 30.0], [176.0, 30.0], [176.0, 50.0], [100.0, 50.0]],
            bbox_xyxy=(100, 30, 176, 50),
            confidence=0.88,
        ),
        _YoloCandidate(
            polygon_xy=[[22.0, 64.0], [160.0, 64.0], [160.0, 86.0], [22.0, 86.0]],
            bbox_xyxy=(22, 64, 160, 86),
            confidence=0.77,
        ),
    ]

    mobile = pipeline._build_ranked_candidates(image, yolo_candidates)
    advanced_pipeline = ExpiryPipeline(
        detector=object(),
        preprocessor=pipeline.preprocessor,
        ocr_router=object(),
        parser=pipeline.parser,
        decision=object(),
    )
    advanced = advanced_pipeline._build_ranked_candidates(
        image,
        [
            TextDetectionBox(
                *candidate.bbox_xyxy,
                candidate.confidence,
                source="yolo26s_obb",
                sources=("yolo26s_obb",),
                variant_name="yolo26s_obb",
                polygon_xy=tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
            )
            for candidate in yolo_candidates
        ],
    )

    assert {candidate.candidate_type for candidate in mobile} == {candidate.candidate_type for candidate in advanced}
    assert all(candidate.detector_sources == ("yolo26s_obb",) for candidate in mobile)
    assert any(candidate.detector_variant == "yolo26s_obb" for candidate in mobile)
    assert not any(str(candidate.detector_variant).startswith("yolo_band_") for candidate in mobile)


def test_mobile_uses_advanced_probe_then_final_selection_cap(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.expiry_global_probe_top_k = 6
    pipeline.expiry_global_final_top_k = 2
    pipeline.expiry_global_debug_final_top_k = 2
    pipeline.context_probe_enabled = False
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_detect_candidates",
        lambda _image: (
            [
                _YoloCandidate(
                    polygon_xy=[[float(10 + i * 12), 20.0], [float(18 + i * 12), 20.0], [float(18 + i * 12), 34.0], [float(10 + i * 12), 34.0]],
                    bbox_xyxy=(10 + i * 12, 20, 18 + i * 12, 34),
                    confidence=0.9 - i * 0.01,
                )
                for i in range(6)
            ],
            None,
        ),
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            *[_RecognitionOutput("PROBE", "PROBE", 0.4, None, "original") for _ in range(6)],
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("MFG 01/01/2026", "MFG 01/01/2026", 0.99, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)
    assert len(pipeline.recognizer.calls) == 8
    assert "final_rank=" in (result.reason or "")
    assert "detector_sources=yolo26s_obb" in (result.reason or "")


def test_settings_default_to_existing_runtime_backends() -> None:
    settings = Settings(_env_file=None)

    assert settings.mobile_expiry_detector_backend == "ultralytics"
    assert settings.svtrv2_rec_backend == "paddle"
    assert settings.mobile_expiry_detector_onnx_path == Path(
        "models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.onnx"
    )
    assert settings.svtrv2_rec_onnx_path == Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx")
    assert settings.mobile_proposal_rescue_backend == "rapidocr_ppocrv5"
    assert settings.mobile_rapidocr_primary_enabled is True
    assert settings.mobile_rapidocr_ocr_version == "PP-OCRv5"
    assert settings.mobile_rapidocr_model_type == "mobile"
    assert settings.mobile_rapidocr_primary_max_rois_per_scan == 12
    assert settings.mobile_rapidocr_primary_max_boxes_accepted == 40


def test_settings_accept_ppocrv5_server_override() -> None:
    settings = Settings(_env_file=None, mobile_rapidocr_model_type="server")

    assert settings.mobile_rapidocr_model_type == "server"


def test_mobile_pipeline_has_no_craft_or_doctr_fast_runtime_hooks() -> None:
    pipeline = _pipeline_for_unit_tests()

    assert not hasattr(pipeline, "_doctr_fast_detector")
    assert not hasattr(pipeline, "_detect_doctr_fast_rescue_candidates")
    assert not hasattr(pipeline, "_should_run_doctr_fast_rescue")
    assert not hasattr(pipeline, "craft_rescue_enabled")


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
