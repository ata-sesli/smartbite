from __future__ import annotations

from datetime import date
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.mobile_expiry_pipeline import (
    MOBILE_RECOGNITION_VARIANTS,
    MobileExpiryPipeline,
    SVTRTextRecognizer,
    _DateEvidence,
    _EvaluatedCandidate,
    _MobileGlobalCandidate,
    _ProductCropperRoi,
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


def test_mobile_debug_profile_exposes_stage_timings_and_counts() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline._reset_debug_records()

    payload = json.loads(pipeline._debug_bytes())

    assert set(payload["profile"]) >= {
        "decode_ms",
        "yolo_ms",
        "rapidocr_primary_ms",
        "candidate_build_ms",
        "probe_svtr_ms",
        "context_probe_ms",
        "final_svtr_primary_ms",
        "hardcase_svtr_ms",
        "dot_matrix_ms",
        "pp_mobile_full_image_ms",
        "date_tail_crop_ms",
        "partial_expansion_ms",
        "production_sibling_ms",
        "anchor_local_sibling_ms",
        "detector_variant_rescue_ms",
        "rapidocr_rescue_ms",
        "total_ms",
        "num_yolo_boxes",
        "num_pp_boxes",
        "num_probe_candidates",
        "num_final_candidates",
        "num_primary_svtr_calls",
        "num_hardcase_svtr_calls",
        "num_dot_matrix_crops",
    }
    assert payload["profile"]["decode_ms"] == 0
    assert payload["profile"]["num_yolo_boxes"] == 0

    pipeline._profile_add("yolo_ms", 1.2345)
    pipeline._profile_inc("num_yolo_boxes", 3)
    payload = json.loads(pipeline._debug_bytes())
    assert payload["profile"]["yolo_ms"] == 1.234
    assert payload["profile"]["num_yolo_boxes"] == 3

    result = pipeline.run(b"not an image", today=date(2026, 5, 30))
    assert result.debug_profile is not None
    assert result.debug_profile["decode_ms"] >= 0
    assert result.debug_profile["total_ms"] >= result.debug_profile["decode_ms"]
    assert json.loads(result.debug_candidates_json_bytes or b"{}")["profile"] == result.debug_profile


class _QueueRecognizer:
    def __init__(self, outputs: list[_RecognitionOutput]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[int, int]] = []

    def recognize(self, image: np.ndarray) -> _RecognitionOutput:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        if not self.outputs:
            return _RecognitionOutput("", "", None, "empty queue", "original")
        return self.outputs.pop(0)


class _BatchQueueRecognizer(_QueueRecognizer):
    cacheable = True

    def __init__(self, outputs: list[_RecognitionOutput]) -> None:
        super().__init__(outputs)
        self.batch_calls: list[list[tuple[int, int]]] = []

    def recognize_many(self, images: list[np.ndarray], *, batch_size: int) -> list[_RecognitionOutput]:
        self.batch_calls.append([(int(image.shape[0]), int(image.shape[1])) for image in images])
        out: list[_RecognitionOutput] = []
        for _image in images:
            if not self.outputs:
                out.append(_RecognitionOutput("", "", None, "empty queue", "original"))
            else:
                out.append(self.outputs.pop(0))
        return out


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


class _FakeProductBoxRow:
    def __init__(self, xyxy: tuple[int, int, int, int], confidence: float) -> None:
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]
        self.conf = np.asarray(confidence, dtype=np.float32)


class _FakeProductBoxes:
    def __init__(self, rows: list[_FakeProductBoxRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class _FakeProductResult:
    def __init__(self, rows: list[_FakeProductBoxRow]) -> None:
        self.obb = None
        self.boxes = _FakeProductBoxes(rows)


class _FakeProductDetector:
    def __init__(self, rows: list[_FakeProductBoxRow]) -> None:
        self.rows = rows
        self.calls: list[tuple[int, int, float, int]] = []

    def predict(self, image: np.ndarray, *, conf: float, imgsz: int, verbose: bool):
        self.calls.append((int(image.shape[0]), int(image.shape[1]), float(conf), int(imgsz)))
        return [_FakeProductResult(self.rows)]


def _mobile_ranked_candidate(
    *,
    candidate_id: str = "cand_1",
    candidate_type: str = "single",
    bbox_xyxy: tuple[int, int, int, int] = (10, 10, 80, 30),
    polygon_xy: tuple[tuple[float, float], ...] | None = None,
    member_bboxes: list[tuple[int, int, int, int]] | None = None,
) -> _MobileRankedCandidate:
    return _MobileRankedCandidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        bbox_xyxy=bbox_xyxy,
        polygon_xy=polygon_xy,
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


def _date_evidence_for_score(
    *,
    text: str,
    parsed_date: date,
    precision: str = "day",
    parser_confidence: float = 0.9,
    ocr_confidence: float = 0.9,
    date_format_detected: str = "test",
    candidate: _MobileRankedCandidate | None = None,
    normalized_crop: NormalizedTextLineCrop | None = None,
    context_texts: tuple[str, ...] = (),
) -> _DateEvidence:
    return _DateEvidence(
        candidate=candidate or _mobile_ranked_candidate(),
        recognition=_RecognitionOutput(text, text, ocr_confidence, None, "original"),
        parsed=ParsedDateData(
            parsed_date=parsed_date,
            date_format_detected=date_format_detected,
            confidence=parser_confidence,
            candidates=[text],
            reason="selected_test",
            date_precision=precision,
            parsed_day=parsed_date.day if precision == "day" else None,
            parsed_month=parsed_date.month,
            parsed_year=parsed_date.year,
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=normalized_crop,
        context_texts=context_texts,
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
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.90, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.90, None, "original"),
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.90, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.90, None, "original"),
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
            _RecognitionOutput("01/01/2026", "01/01/2026", 0.90, None, "original"),
            _RecognitionOutput("01/01/2027", "01/01/2027", 0.90, None, "original"),
        ]
    )

    evaluated = pipeline._evaluate_candidate(np.full((120, 360, 3), 255, dtype=np.uint8), candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2027, 1, 1)
    assert evaluated.recognition.raw_text == "01/01/2027"


def test_mobile_evaluate_candidate_preserves_representative_when_collecting_date_evidence(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    crop = NormalizedTextLineCrop(
        image=np.full((24, 96, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 10, 80, 30),
        polygon_used=False,
        crop_transform_used="bbox_raw",
        selected_orientation="original",
        original_crop_shape=[24, 96, 3],
        normalized_crop_shape=[24, 96, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="unit_test",
    )
    representative = _evaluated_candidate_for_score(
        text="01/01/2026",
        parsed_date=date(2026, 1, 1),
        precision="day",
        confidence=0.99,
    )
    evidence = _date_evidence_for_score(
        text="01/01/2027",
        parsed_date=date(2027, 1, 1),
        parser_confidence=0.90,
        ocr_confidence=0.90,
    )
    representative.date_evidence = [evidence]

    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)

    evaluated = pipeline._evaluate_candidate(np.full((120, 360, 3), 255, dtype=np.uint8), candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2026, 1, 1)
    assert evaluated.recognition.raw_text == "01/01/2026"
    assert [item.parsed.parsed_date for item in evaluated.date_evidence] == [date(2027, 1, 1)]


def test_mobile_legacy_wide_group_crop_adds_date_evidence_without_replacing_representative(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.legacy_wide_group_rescue_enabled = True
    pipeline.expiry_final_crop_max_padding_px = 0
    image = np.full((140, 260, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(20, 20, 220, 100),
        member_bboxes=[(24, 24, 160, 44), (24, 70, 210, 96)],
    )
    candidate.probe_text = "07/03/2026\n07/03/29.28"
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_color_tight", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    pipeline.recognizer = _WidthAwareRecognizer(
        width_threshold=195,
        narrow=_RecognitionOutput("O7/03/2026", "O7/03/2026", 0.70, None, "original"),
        wide=_RecognitionOutput("07/03/29.28", "07/03/29.28", 0.82, None, "original"),
    )

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2026, 3, 7)
    assert evaluated.recognition.raw_text == "O7/03/2026"
    assert any(evidence.parsed.parsed_date == date(2029, 3, 7) for evidence in evaluated.date_evidence)
    assert any(evidence.crop_policy == "legacy_wide_group" for evidence in evaluated.date_evidence)


def test_mobile_legacy_wide_group_evidence_can_beat_tight_wrong_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    tight = _date_evidence_for_score(
        text="O7/03/2026",
        parsed_date=date(2026, 3, 7),
        parser_confidence=0.90,
        ocr_confidence=0.90,
    )
    wide = _date_evidence_for_score(
        text="07/03/29.28",
        parsed_date=date(2029, 3, 7),
        parser_confidence=0.82,
        ocr_confidence=0.82,
    )
    wide.crop_policy = "legacy_wide_group"

    selected = pipeline._select_date_evidence([tight, wide])

    assert selected is wide


def test_mobile_production_anchor_sibling_adds_expiry_evidence_without_replacing_representative(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.production_anchor_sibling_enabled = True
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="single",
        bbox_xyxy=(80, 40, 180, 68),
        member_bboxes=[(80, 40, 180, 68)],
    )
    crop = NormalizedTextLineCrop(
        image=np.full((28, 100, 3), 255, dtype=np.uint8),
        bbox_xyxy=(80, 40, 180, 68),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[28, 100, 3],
        normalized_crop_shape=[28, 100, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    representative = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("28.02.2023", "28.02.2023", 0.98, None, "original"),
        parsed=ParsedDateData(
            parsed_date=date(2023, 2, 28),
            date_format_detected="DD/MM/YYYY",
            confidence=0.95,
            candidates=["28.02.2023"],
            reason="selected_test",
            date_precision="day",
            parsed_day=28,
            parsed_month=2,
            parsed_year=2023,
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=crop,
        date_evidence=[
            _date_evidence_for_score(
                text="28.02.2023",
                parsed_date=date(2023, 2, 28),
                parser_confidence=0.95,
                ocr_confidence=0.98,
                candidate=candidate,
                context_texts=("URT",),
            )
        ],
    )
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)
    monkeypatch.setattr(pipeline, "_line_role_context_for_bbox", lambda _image, _bbox: ("URT",))
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )

    def recognize_by_height(
        image_in: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        use_cache: bool = True,
    ) -> _RecognitionOutput:
        return _RecognitionOutput("EY.02.2028", "EY.02.2028", 0.84, None, orientation)

    monkeypatch.setattr(pipeline, "_recognize_variant", recognize_by_height)

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2023, 2, 28)
    assert any(
        evidence.crop_policy == "production_anchor_sibling"
        and evidence.parsed.parsed_date == date(2028, 2, 29)
        for evidence in evaluated.date_evidence
    )


def test_mobile_production_anchor_sibling_uses_detector_context_for_single_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.production_anchor_sibling_enabled = True
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="single",
        bbox_xyxy=(80, 40, 180, 68),
        member_bboxes=[(80, 40, 180, 68)],
    )
    crop = NormalizedTextLineCrop(
        image=np.full((40, 120, 3), 255, dtype=np.uint8),
        bbox_xyxy=(70, 34, 190, 76),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[40, 120, 3],
        normalized_crop_shape=[40, 120, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    initial_evidence = _date_evidence_for_score(
        text="28.02.2023",
        parsed_date=date(2023, 2, 28),
        parser_confidence=0.95,
        ocr_confidence=0.98,
        candidate=candidate,
    )
    initial_evidence.normalized_crop = crop
    initial_evidence.crop_policy = "single_tight"
    representative = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("28.02.2023", "28.02.2023", 0.98, None, "original"),
        parsed=initial_evidence.parsed,
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=crop,
        date_evidence=[initial_evidence],
    )
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)

    def context_for_bbox(_image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, ...]:
        return ("URT",) if bbox == candidate.recognition_bbox else ()

    monkeypatch.setattr(pipeline, "_line_role_context_for_bbox", context_for_bbox)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_recognize_variant",
        lambda *_args, **_kwargs: _RecognitionOutput("02.2028", "02.2028", 0.94, None, "original"),
    )

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 5, 27))
    selected = pipeline._select_date_evidence(evaluated.date_evidence)

    assert initial_evidence.context_texts == ("URT",)
    assert any(
        evidence.crop_policy == "production_anchor_sibling"
        and evidence.parsed.parsed_date == date(2028, 2, 29)
        and evidence.parsed.date_precision == "month"
        for evidence in evaluated.date_evidence
    )
    assert selected.crop_policy == "production_anchor_sibling"
    assert selected.parsed.parsed_date == date(2028, 2, 29)


def test_mobile_production_anchor_sibling_includes_tight_below_date_window() -> None:
    pipeline = _pipeline_for_unit_tests()

    windows = pipeline._production_anchor_sibling_bboxes(
        (823, 2248, 1687, 2405),
        image_shape=(3000, 2500, 3),
    )

    assert [name for name, _bbox in windows[:4]] == [
        "below_date_aligned",
        "right_same_line",
        "above_date_aligned",
        "left_same_line",
    ]
    assert dict(windows)["below_date_aligned"] == (883, 2392, 1488, 2570)


def test_mobile_production_anchor_sibling_respects_direction_budget(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.production_anchor_sibling_enabled = True
    image = np.full((220, 320, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="single",
        bbox_xyxy=(90, 70, 190, 100),
        member_bboxes=[(90, 70, 190, 100)],
    )
    crop = NormalizedTextLineCrop(
        image=np.full((30, 100, 3), 255, dtype=np.uint8),
        bbox_xyxy=(90, 70, 190, 100),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[30, 100, 3],
        normalized_crop_shape=[30, 100, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    prod_evidence = _date_evidence_for_score(
        text="28.02.2023",
        parsed_date=date(2023, 2, 28),
        parser_confidence=0.95,
        ocr_confidence=0.98,
        candidate=candidate,
        context_texts=("URT",),
    )
    prod_evidence.normalized_crop = crop
    evaluated = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("28.02.2023", "28.02.2023", 0.98, None, "original"),
        parsed=prod_evidence.parsed,
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=crop,
        date_evidence=[prod_evidence],
    )
    monkeypatch.setattr(
        pipeline,
        "_production_anchor_sibling_bboxes",
        lambda *_args, **_kwargs: [
            (f"side_{index}", (10 + index, 20, 40 + index, 60))
            for index in range(6)
        ],
    )
    calls: list[str] = []

    def collect_empty_bbox(*_args, metadata=None, **_kwargs):
        calls.append(str((metadata or {}).get("sibling_relation")))
        return []

    monkeypatch.setattr(pipeline, "_date_evidence_from_bbox", collect_empty_bbox)

    sibling_evidence = pipeline._collect_production_anchor_sibling_evidence(
        image,
        candidate,
        evaluated,
        [prod_evidence],
        today=date(2026, 5, 27),
    )

    assert sibling_evidence == []
    assert calls == ["side_0", "side_1", "side_2", "side_3"]


def test_mobile_anchor_local_sibling_discovers_expiry_below_strong_past_anchor(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.production_anchor_sibling_enabled = True
    image = np.full((420, 520, 3), 255, dtype=np.uint8)
    anchor_candidate = _mobile_ranked_candidate(
        candidate_id="anchor",
        candidate_type="single",
        bbox_xyxy=(220, 180, 350, 230),
        member_bboxes=[(220, 180, 350, 230)],
    )
    anchor_crop = NormalizedTextLineCrop(
        image=np.full((50, 130, 3), 255, dtype=np.uint8),
        bbox_xyxy=(220, 180, 350, 230),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[50, 130, 3],
        normalized_crop_shape=[50, 130, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    anchor_evidence = _date_evidence_for_score(
        text="18/11/2025",
        parsed_date=date(2025, 11, 18),
        parser_confidence=0.95,
        ocr_confidence=0.94,
        candidate=anchor_candidate,
    )
    anchor_evidence.normalized_crop = anchor_crop
    anchor_evaluated = _EvaluatedCandidate(
        candidate=anchor_candidate,
        recognition=anchor_evidence.recognition,
        parsed=anchor_evidence.parsed,
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=anchor_crop,
        date_evidence=[anchor_evidence],
    )
    pipeline._rapidocr_detector = _FakeMultiRapidOCRTextDetector(
        [
            ProposalBox(
                bbox_xyxy=(120, 170, 255, 214),
                confidence=0.86,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name="PP-OCRv5_mobile",
                polygon_xy=None,
            )
        ]
    )
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_recognize_variant",
        lambda *_args, **_kwargs: _RecognitionOutput("E14/09/2026", "E14/09/2026", 0.91, None, "original"),
    )

    sibling_evidence = pipeline._collect_anchor_local_sibling_evidence(
        image,
        [anchor_evaluated],
        today=date(2026, 5, 27),
    )
    anchor_evaluated.date_evidence.extend(sibling_evidence)
    _parseable, accepted = pipeline._accepted_parseable_items([anchor_evaluated])
    selected = pipeline._select_date_evidence(accepted)

    assert pipeline._rapidocr_detector.calls
    assert any(
        evidence.crop_policy == "anchor_local_sibling"
        and evidence.parsed.parsed_date == date(2026, 9, 14)
        and evidence.metadata.get("anchor_date") == "2025-11-18"
        for evidence in sibling_evidence
    )
    assert selected.crop_policy == "anchor_local_sibling"
    assert selected.parsed.parsed_date == date(2026, 9, 14)


def test_mobile_paired_crop_adds_line_context_evidence_for_multiline_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.paired_crop_evidence_enabled = True
    image = np.full((160, 280, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(40, 30, 230, 110),
        member_bboxes=[(72, 34, 180, 56), (72, 78, 215, 104)],
    )
    crop = NormalizedTextLineCrop(
        image=np.full((80, 190, 3), 255, dtype=np.uint8),
        bbox_xyxy=(40, 30, 230, 110),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[80, 190, 3],
        normalized_crop_shape=[80, 190, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    representative = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("URT 10/01/2026", "URT 10/01/2026", 0.95, None, "original"),
        parsed=ParsedDateData(
            parsed_date=date(2026, 1, 10),
            date_format_detected="DD/MM/YYYY",
            confidence=0.95,
            candidates=["10/01/2026"],
            reason="selected_test",
            date_precision="day",
            parsed_day=10,
            parsed_month=1,
            parsed_year=2026,
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=crop,
        date_evidence=[
            _date_evidence_for_score(
                text="URT 10/01/2026",
                parsed_date=date(2026, 1, 10),
                parser_confidence=0.95,
                ocr_confidence=0.95,
                candidate=candidate,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )

    def context_for_bbox(_image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, ...]:
        return ("EXP",) if bbox[1] > 60 else ("URT",)

    def recognize_by_line(
        image_in: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        use_cache: bool = True,
    ) -> _RecognitionOutput:
        if image_in.shape[0] <= 35:
            return _RecognitionOutput("12/05/2027", "12/05/2027", 0.83, None, orientation)
        return _RecognitionOutput("URT 10/01/2026", "URT 10/01/2026", 0.95, None, orientation)

    monkeypatch.setattr(pipeline, "_line_role_context_for_bbox", context_for_bbox)
    monkeypatch.setattr(pipeline, "_recognize_variant", recognize_by_line)

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 1, 1))

    assert evaluated.parsed.parsed_date == date(2026, 1, 10)
    assert any(
        evidence.crop_policy == "paired_crop_line_with_left_context"
        and evidence.parsed.parsed_date == date(2027, 5, 12)
        and evidence.context_texts == ("EXP",)
        for evidence in evaluated.date_evidence
    )


def test_mobile_local_group_wide_crop_adds_late_evidence_without_replacing_tight_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.local_group_wide_crop_enabled = True
    image = np.full((180, 280, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(50, 40, 220, 110),
        member_bboxes=[(80, 48, 155, 68), (88, 76, 205, 100)],
    )
    candidate.evidence_bbox = (50, 40, 220, 110)
    tight_crop = NormalizedTextLineCrop(
        image=np.full((24, 90, 3), 255, dtype=np.uint8),
        bbox_xyxy=(80, 48, 170, 72),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[24, 90, 3],
        normalized_crop_shape=[24, 90, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    representative = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("O7/03/2026", "O7/03/2026", 0.80, None, "original"),
        parsed=ParsedDateData(
            parsed_date=date(2026, 3, 7),
            date_format_detected="DD/MM/YYYY",
            confidence=0.95,
            candidates=["O7/03/2026"],
            reason="selected_test",
            date_precision="day",
            parsed_day=7,
            parsed_month=3,
            parsed_year=2026,
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=tight_crop,
        date_evidence=[
            _date_evidence_for_score(
                text="O7/03/2026",
                parsed_date=date(2026, 3, 7),
                parser_confidence=0.95,
                ocr_confidence=0.80,
                candidate=candidate,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [tight_crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )

    def recognize_by_crop_size(
        image_in: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        use_cache: bool = True,
    ) -> _RecognitionOutput:
        if image_in.shape[1] > 130:
            return _RecognitionOutput("07/03/29.28", "07/03/29.28", 0.72, None, orientation)
        return _RecognitionOutput("O7/03/2026", "O7/03/2026", 0.80, None, orientation)

    monkeypatch.setattr(pipeline, "_recognize_variant", recognize_by_crop_size)

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 1, 1))
    selected = pipeline._select_date_evidence(evaluated.date_evidence)

    assert evaluated.parsed.parsed_date == date(2026, 3, 7)
    assert evaluated.normalized_crop is tight_crop
    assert any(
        evidence.crop_policy == "local_group_wide_crop"
        and evidence.metadata.get("source") == "crop_policy_local_group_wide"
        and evidence.parsed.parsed_date == date(2029, 3, 7)
        for evidence in evaluated.date_evidence
    )
    assert selected.crop_policy == "local_group_wide_crop"
    assert selected.parsed.parsed_date == date(2029, 3, 7)


def test_mobile_rotation_wide_crop_adds_late_evidence_without_replacing_tight_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rotation_wide_crop_enabled = True
    image = np.full((120, 260, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="single",
        bbox_xyxy=(90, 40, 140, 62),
        member_bboxes=[(90, 40, 140, 62)],
    )
    candidate.detector_sources = ("yolo26s_obb", "rotation_rescue")
    candidate.detector_variant = "rot90_ccw"
    tight_crop = NormalizedTextLineCrop(
        image=np.full((22, 50, 3), 255, dtype=np.uint8),
        bbox_xyxy=(90, 40, 140, 62),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[22, 50, 3],
        normalized_crop_shape=[22, 50, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    representative = _EvaluatedCandidate(
        candidate=candidate,
        recognition=_RecognitionOutput("020", "020", 0.73, None, "original"),
        parsed=ParsedDateData(
            parsed_date=None,
            date_format_detected=None,
            confidence=0.0,
            candidates=[],
            reason="no valid date parsed",
        ),
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=tight_crop,
        date_evidence=[],
    )
    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [tight_crop])
    monkeypatch.setattr(pipeline, "_best_recognition_for_single_crop", lambda *_args, **_kwargs: representative)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop_image, *, allowed_names=None: [ImageVariant("original_color_tight", crop_image, purpose="recognition")],
        raising=False,
    )

    def recognize_by_crop_width(
        image_in: np.ndarray,
        *,
        variant_name: str,
        orientation: str,
        use_cache: bool = True,
    ) -> _RecognitionOutput:
        if image_in.shape[1] > 70:
            return _RecognitionOutput("06/05/2026", "06/05/2026", 0.50, None, orientation)
        return _RecognitionOutput("020", "020", 0.73, None, orientation)

    monkeypatch.setattr(pipeline, "_recognize_variant", recognize_by_crop_width)

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 1, 1))
    accepted = pipeline._accepted_date_evidence_items(evaluated.date_evidence)

    assert evaluated.recognition.raw_text == "020"
    assert any(
        evidence.crop_policy == "rotation_wide_crop"
        and evidence.metadata.get("source") == "crop_policy_rotation_wide"
        and evidence.parsed.parsed_date == date(2026, 5, 6)
        for evidence in accepted
    )


def test_mobile_local_group_wide_crop_rejects_noisy_multidate_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.local_group_wide_crop_enabled = True
    candidate = _mobile_ranked_candidate(candidate_type="group")
    stable = _date_evidence_for_score(
        text="09/12/2026",
        parsed_date=date(2026, 12, 9),
        parser_confidence=0.95,
        ocr_confidence=0.95,
        candidate=candidate,
    )
    noisy_wide = _date_evidence_for_score(
        text="109/02/206 31T09/12/2028",
        parsed_date=date(2028, 12, 9),
        parser_confidence=0.95,
        ocr_confidence=0.82,
        candidate=candidate,
    )
    noisy_wide.crop_policy = "local_group_wide_crop"

    accepted = pipeline._accepted_date_evidence_items([stable, noisy_wide])

    assert stable in accepted
    assert noisy_wide not in accepted


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


def test_mobile_product_cropper_rescue_maps_expiry_boxes_from_top_product_roi(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.product_cropper_rescue_enabled = True
    pipeline.product_cropper_padding_ratio = 0.0
    pipeline.product_cropper_max_candidates = 3
    fake_product = _FakeProductDetector([_FakeProductBoxRow((40, 20, 240, 90), 0.86)])
    expiry_calls: list[tuple[tuple[int, int, int], str]] = []

    monkeypatch.setattr(pipeline, "_ensure_product_cropper_model", lambda: fake_product)

    def fake_detect_expiry(crop: np.ndarray, *, variant_name: str):
        expiry_calls.append((crop.shape, variant_name))
        return [
            _YoloCandidate(
                polygon_xy=[[10.0, 5.0], [90.0, 5.0], [90.0, 25.0], [10.0, 25.0]],
                bbox_xyxy=(10, 5, 90, 25),
                confidence=0.77,
                variant_name=variant_name,
            )
        ], None

    monkeypatch.setattr(pipeline, "_detect_candidates_for_variant", fake_detect_expiry)

    candidates, reason = pipeline._detect_product_cropper_rescue_candidates(np.full((120, 300, 3), 255, dtype=np.uint8))

    assert reason is None
    assert fake_product.calls == [(120, 300, 0.15, 1024)]
    assert expiry_calls == [((70, 200, 3), "product_cropper_rescue")]
    assert len(candidates) == 1
    assert candidates[0].bbox_xyxy == (50, 25, 130, 45)
    assert candidates[0].source == "product_cropper_rescue"
    assert candidates[0].sources == ("product_yolo", "yolo26s_obb")
    assert candidates[0].variant_name == "product_cropper_rescue"


def test_mobile_product_cropper_direct_rescue_uses_limited_variants_and_accepts_multivariant_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.product_cropper_rescue_enabled = True
    seen_allowed: list[tuple[str, ...] | None] = []

    def limited_variants(crop: np.ndarray, *, allowed_names=None):
        seen_allowed.append(tuple(allowed_names) if allowed_names is not None else None)
        return [ImageVariant("original_color_tight", crop, purpose="recognition"), ImageVariant("clahe_gray", crop, purpose="recognition")]

    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        limited_variants,
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("15/03/2022", "15/03/2022", 0.82, None, "original"),
            _RecognitionOutput("15/03/2022", "15/03/2022", 0.83, None, "original"),
        ]
    )
    yolo = _YoloCandidate(
        polygon_xy=[[40.0, 30.0], [190.0, 30.0], [190.0, 72.0], [40.0, 72.0]],
        bbox_xyxy=(40, 30, 190, 72),
        confidence=0.82,
        source="product_cropper_rescue",
        sources=("product_yolo", "yolo26s_obb"),
        variant_name="product_cropper_rescue",
    )

    evaluated = pipeline._evaluate_product_cropper_rescue_candidates_direct(
        np.full((140, 260, 3), 255, dtype=np.uint8),
        [yolo],
        today=date(2026, 1, 1),
    )

    assert len(evaluated) == 1
    assert evaluated[0].parsed.parsed_date == date(2022, 3, 15)
    assert {evidence.crop_policy for evidence in evaluated[0].date_evidence} == {"product_cropper_rescue"}
    assert seen_allowed and seen_allowed[0] == ("original_color_tight", "original_padded", "clahe_gray", "adaptive_binary")


def test_mobile_product_cropper_direct_rescue_rejects_weak_single_variant_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.product_cropper_rescue_enabled = True
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_color_tight", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    pipeline.recognizer = _QueueRecognizer(
        [_RecognitionOutput("20/10/2028", "20/10/2028", 0.76, None, "original")]
    )
    yolo = _YoloCandidate(
        polygon_xy=[[40.0, 30.0], [190.0, 30.0], [190.0, 72.0], [40.0, 72.0]],
        bbox_xyxy=(40, 30, 190, 72),
        confidence=0.82,
        source="product_cropper_rescue",
        sources=("product_yolo", "yolo26s_obb"),
        variant_name="product_cropper_rescue",
    )

    evaluated = pipeline._evaluate_product_cropper_rescue_candidates_direct(
        np.full((140, 260, 3), 255, dtype=np.uint8),
        [yolo],
        today=date(2026, 1, 1),
    )

    assert evaluated == []


def test_mobile_product_cropper_direct_rescue_accepts_repeated_spaced_short_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.product_cropper_rescue_enabled = True
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [
            ImageVariant("original_color_tight", crop, purpose="recognition"),
            ImageVariant("original_padded", crop, purpose="recognition"),
            ImageVariant("clahe_gray", crop, purpose="recognition"),
        ],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("S1T 15 09 26", "S1T 15 09 26", 0.83, None, "original"),
            _RecognitionOutput("S1T 15 09 26", "S1T 15 09 26", 0.87, None, "original"),
            _RecognitionOutput("S1T 15 09 26", "S1T 15 09 26", 0.79, None, "original"),
        ]
    )
    yolo = _YoloCandidate(
        polygon_xy=[[40.0, 30.0], [190.0, 30.0], [190.0, 72.0], [40.0, 72.0]],
        bbox_xyxy=(40, 30, 190, 72),
        confidence=0.82,
        source="product_cropper_rescue",
        sources=("product_yolo", "yolo26s_obb"),
        variant_name="product_cropper_rescue",
    )

    evaluated = pipeline._evaluate_product_cropper_rescue_candidates_direct(
        np.full((140, 260, 3), 255, dtype=np.uint8),
        [yolo],
        today=date(2026, 1, 1),
    )

    assert len(evaluated) == 1
    assert evaluated[0].parsed.parsed_date == date(2026, 9, 15)


def test_mobile_product_cropper_rescue_rejects_alpha_attached_leading_day() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    candidate.detector_sources = ("product_yolo", "yolo26s_obb")
    candidate.detector_variant = "product_cropper_rescue"
    evidence_items = []
    for index, confidence in enumerate((0.884, 0.786, 0.764), start=1):
        evidence = _date_evidence_for_score(
            text="STJ1/05/26",
            parsed_date=date(2026, 5, 1),
            parser_confidence=0.76,
            ocr_confidence=confidence,
            candidate=candidate,
        )
        evidence.crop_policy = "product_cropper_rescue"
        evidence.recognition_variant = f"variant_{index}"
        evidence_items.append(evidence)

    accepted = pipeline._accepted_date_evidence_items(evidence_items)

    assert accepted == []


def test_mobile_product_cropper_rescue_skips_when_primary_has_accepted_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.product_cropper_rescue_enabled = True
    pipeline.rapidocr_primary_enabled = False
    pipeline.rapidocr_rescue_enabled = False
    pipeline.expiry_global_probe_top_k = 1
    pipeline.expiry_global_final_top_k = 1
    pipeline.expiry_global_debug_final_top_k = 1
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
    monkeypatch.setattr(pipeline, "_detect_variant_rescue_candidates", lambda _image: ([], "no variant expiry"))
    monkeypatch.setattr(
        pipeline,
        "_detect_product_cropper_rescue_candidates",
        lambda _image: (_ for _ in ()).throw(AssertionError("product cropper should not run after an accepted primary date")),
    )
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: [ImageVariant("original_color_tight", crop, purpose="recognition")],
        raising=False,
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.91, None, "original"),
        ]
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2027, 5, 12)


def test_mobile_rapidocr_ppocrv5_primary_proposals_feed_advanced_engine(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_primary_mode = "always"
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
    pipeline.rapidocr_primary_mode = "always"
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
    pipeline.rapidocr_primary_mode = "always"
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
    pipeline.rapidocr_primary_mode = "always"
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


def test_mobile_probe_bonus_keeps_month_precision_below_full_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    month_scores = pipeline._probe_signal_scores(
        _RecognitionOutput("10/28", "10/28", 0.76, None, "original"),
        candidate=None,
        today=date(2026, 1, 1),
    )
    day_scores = pipeline._probe_signal_scores(
        _RecognitionOutput("09/12/2026", "09/12/2026", 0.72, None, "original"),
        candidate=None,
        today=date(2026, 1, 1),
    )
    month_candidate = _mobile_ranked_candidate()
    month_candidate.score_breakdown = month_scores

    assert month_scores["parser_probe_bonus"] < day_scores["parser_probe_bonus"]
    assert pipeline._force_include_reason(month_candidate) == "date_like_probe"


def test_mobile_month_precision_parseable_still_runs_rescue() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_rescue_enabled = True
    evidence = _date_evidence_for_score(
        text="10/28",
        parsed_date=date(2028, 10, 31),
        precision="month",
        parser_confidence=0.74,
    )
    month_only = pipeline._evaluated_from_date_evidence(evidence, date_evidence=[evidence])

    assert pipeline._should_run_rapidocr_rescue([month_only])


def test_mobile_day_precision_guard_ignores_rejected_risky_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(candidate_id="rapid_risky", bbox_xyxy=(100, 100, 140, 220))
    candidate.detector_sources = ("rapidocr_ppocrv5",)
    candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="92EPO20232-/1",
        parsed_date=date(2032, 2, 2),
        parser_confidence=0.72,
        ocr_confidence=0.412,
        candidate=candidate,
    )
    evidence.crop_policy = "single_tight"
    evaluated = pipeline._evaluated_from_date_evidence(evidence, date_evidence=[evidence])

    assert pipeline._accepted_date_evidence_items([evidence]) == []
    assert not pipeline._has_day_precision_parseable_candidate([evaluated])


def test_mobile_date_evidence_prefers_exp_context_over_higher_confidence_mfg() -> None:
    pipeline = _pipeline_for_unit_tests()
    production = _date_evidence_for_score(
        text="MFG 01/01/2027",
        parsed_date=date(2027, 1, 1),
        parser_confidence=0.99,
        ocr_confidence=0.99,
    )
    expiry = _date_evidence_for_score(
        text="12/05/2027",
        parsed_date=date(2027, 5, 12),
        parser_confidence=0.82,
        ocr_confidence=0.78,
        context_texts=("EXP",),
    )

    selected = max([production, expiry], key=pipeline._score_date_evidence)

    assert selected is expiry


def test_mobile_date_evidence_prefers_adjacent_expiry_over_naked_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    naked = _date_evidence_for_score(
        text="01/01/2027",
        parsed_date=date(2027, 1, 1),
        parser_confidence=0.99,
        ocr_confidence=0.99,
    )
    expiry_context = _date_evidence_for_score(
        text="12/05/2027",
        parsed_date=date(2027, 5, 12),
        parser_confidence=0.80,
        ocr_confidence=0.75,
        context_texts=("BEST BEFORE",),
    )

    selected = max([naked, expiry_context], key=pipeline._score_date_evidence)

    assert selected is expiry_context


def test_mobile_date_evidence_prefers_tett_context_over_urt_context() -> None:
    pipeline = _pipeline_for_unit_tests()
    production = _date_evidence_for_score(
        text="14.02.2026",
        parsed_date=date(2026, 2, 14),
        parser_confidence=0.95,
        ocr_confidence=0.96,
        context_texts=("UFT",),
    )
    expiry = _date_evidence_for_score(
        text="13.02.2031",
        parsed_date=date(2031, 2, 13),
        parser_confidence=0.82,
        ocr_confidence=0.74,
        context_texts=("TETT",),
    )

    selected = max([production, expiry], key=pipeline._score_date_evidence)

    assert selected is expiry


def test_mobile_accepts_anchor_local_sibling_when_compact_token_matches_future_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(candidate_id="anchor_sibling", bbox_xyxy=(100, 100, 180, 140))
    evidence = _date_evidence_for_score(
        text="1 040427 22",
        parsed_date=date(2027, 4, 4),
        parser_confidence=0.72,
        ocr_confidence=0.586,
        candidate=candidate,
    )
    evidence.crop_policy = "anchor_local_sibling"
    evidence.parsed.date_format_detected = "DDMMYY"
    evidence.metadata["anchor_date"] = date(2021, 4, 4)

    assert pipeline._is_strong_anchor_local_sibling_evidence(evidence)


def test_mobile_date_evidence_prefers_supported_weak_expiry_group_over_production_group() -> None:
    pipeline = _pipeline_for_unit_tests()
    production = [
        _date_evidence_for_score(
            text="14/04/2028",
            parsed_date=date(2028, 4, 14),
            parser_confidence=0.95,
            ocr_confidence=0.97,
            context_texts=("URT.11",),
        )
    ]
    weak_expiry = [
        _date_evidence_for_score(
            text="11/04/2025",
            parsed_date=date(2025, 4, 11),
            parser_confidence=0.95,
            ocr_confidence=0.94 + index * 0.005,
            context_texts=("1E.T.1",),
        )
        for index in range(3)
    ]
    for index, evidence in enumerate(weak_expiry):
        evidence.recognition_variant = f"expiry_variant_{index}"

    selected = pipeline._select_date_evidence([*production, *weak_expiry])

    assert selected.parsed.parsed_date == date(2025, 4, 11)


def test_mobile_date_evidence_prefers_weak_expiry_even_when_production_group_has_later_high_confidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    production = [
        _date_evidence_for_score(
            text="14/04/2028",
            parsed_date=date(2028, 4, 14),
            parser_confidence=0.95,
            ocr_confidence=0.974,
            context_texts=("URT.11",),
        )
        for _ in range(3)
    ]
    for index, evidence in enumerate(production):
        evidence.crop_policy = "line_from_multiline_group" if index == 0 else "line_role_context"
        evidence.recognition_variant = f"prod_variant_{index}"
    weak_expiry = [
        _date_evidence_for_score(
            text="11/04/2025",
            parsed_date=date(2025, 4, 11),
            parser_confidence=0.95,
            ocr_confidence=0.95 + index * 0.01,
            context_texts=("1E.T.1",),
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(weak_expiry):
        evidence.crop_policy = "line_role_context" if index < 3 else "single_tight"
        evidence.recognition_variant = f"expiry_variant_{index}"

    selected = pipeline._select_date_evidence([*production, *weak_expiry])

    assert selected.parsed.parsed_date == date(2025, 4, 11)


def test_mobile_date_evidence_prefers_later_unknown_same_precision() -> None:
    pipeline = _pipeline_for_unit_tests()
    earlier = _date_evidence_for_score(text="01/01/2026", parsed_date=date(2026, 1, 1), ocr_confidence=0.90)
    later = _date_evidence_for_score(text="01/01/2027", parsed_date=date(2027, 1, 1), ocr_confidence=0.90)

    selected = max([earlier, later], key=pipeline._score_date_evidence)

    assert selected is later


def test_mobile_date_evidence_prefers_later_same_day_month_when_earlier_only_has_weak_context() -> None:
    pipeline = _pipeline_for_unit_tests()
    earlier = _date_evidence_for_score(
        text="4.03.2026",
        parsed_date=date(2026, 3, 4),
        parser_confidence=0.95,
        ocr_confidence=0.98,
        context_texts=("FEYT-014",),
    )
    later = _date_evidence_for_score(
        text="04.03.2.28",
        parsed_date=date(2028, 3, 4),
        parser_confidence=0.86,
        ocr_confidence=0.925,
    )

    selected = pipeline._select_date_evidence([earlier, later])

    assert selected is later


def test_mobile_date_evidence_prefers_clean_high_confidence_over_low_confidence_later_unknown() -> None:
    pipeline = _pipeline_for_unit_tests()
    clean = _date_evidence_for_score(
        text="12.06.2026",
        parsed_date=date(2026, 6, 12),
        parser_confidence=0.95,
        ocr_confidence=0.95,
    )
    later_noisy = _date_evidence_for_score(
        text="22/3/27",
        parsed_date=date(2027, 3, 22),
        parser_confidence=0.76,
        ocr_confidence=0.79,
    )

    selected = max([clean, later_noisy], key=pipeline._score_date_evidence)

    assert selected is clean


def test_mobile_accepts_clean_high_confidence_rapidocr_gated_date_over_noisy_yolo_dates() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo_candidate = _mobile_ranked_candidate()
    yolo_candidate.detector_sources = ("yolo26s_obb",)
    yolo_candidate.detector_variant = "yolo26s_obb"
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"

    noisy_later = _date_evidence_for_score(
        text="ANO-2-08-29",
        parsed_date=date(2029, 8, 2),
        parser_confidence=0.93,
        ocr_confidence=0.756,
        candidate=yolo_candidate,
    )
    noisy_later.crop_policy = "line_role_context"
    noisy_prefixed_rapid = _date_evidence_for_score(
        text="1002.08.2026",
        parsed_date=date(2026, 8, 2),
        parser_confidence=0.95,
        ocr_confidence=0.90,
        candidate=rapid_candidate,
    )
    noisy_prefixed_rapid.crop_policy = "single_tight"
    clean_rapid = _date_evidence_for_score(
        text="26.08.2026",
        parsed_date=date(2026, 8, 26),
        parser_confidence=0.93,
        ocr_confidence=0.926,
        candidate=rapid_candidate,
    )
    clean_rapid.crop_policy = "rapidocr_gated_prefilter_probe"

    accepted = pipeline._accepted_date_evidence_items([noisy_later, noisy_prefixed_rapid, clean_rapid])
    selected = pipeline._select_date_evidence(accepted)

    assert noisy_prefixed_rapid not in accepted
    assert clean_rapid in accepted
    assert selected.parsed.parsed_date == date(2026, 8, 26)


def test_mobile_risky_rapidocr_best_does_not_block_supported_later_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    early_candidate = _mobile_ranked_candidate(candidate_id="rapid_early")
    early_candidate.detector_sources = ("rapidocr_ppocrv5",)
    early_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    later_candidate = _mobile_ranked_candidate(candidate_id="rapid_later")
    later_candidate.detector_sources = ("rapidocr_ppocrv5",)
    later_candidate.detector_variant = "rapidocr_gated_grouped:partial_date_completion"

    evidence_items = []
    for index, variant in enumerate(["original_color_tight", "gray_upscaled", "clahe_gray", "unsharp_gray"], start=1):
        evidence = _date_evidence_for_score(
            text="28-04-26",
            parsed_date=date(2026, 4, 28),
            parser_confidence=0.88,
            ocr_confidence=0.98,
            candidate=early_candidate,
        )
        evidence.crop_policy = "single_tight"
        evidence.recognition_variant = variant
        evidence_items.append(evidence)
    for index, variant in enumerate(["gray_upscaled", "clahe_gray", "unsharp_gray", "adaptive_binary"], start=1):
        evidence = _date_evidence_for_score(
            text="03-05-26",
            parsed_date=date(2026, 5, 3),
            parser_confidence=0.88,
            ocr_confidence=0.91,
            candidate=later_candidate,
        )
        evidence.crop_policy = "single_tight"
        evidence.recognition_variant = variant
        evidence_items.append(evidence)
    completion = _date_evidence_for_score(
        text="03-05-2026",
        parsed_date=date(2026, 5, 3),
        parser_confidence=0.95,
        ocr_confidence=0.81,
        candidate=later_candidate,
    )
    completion.crop_policy = "rapidocr_gated_prefilter_probe"
    completion.recognition_variant = "combined_same_line"
    evidence_items.append(completion)
    single_variant_wrong_year = _date_evidence_for_score(
        text="03-05-28",
        parsed_date=date(2028, 5, 3),
        parser_confidence=0.88,
        ocr_confidence=0.96,
        candidate=later_candidate,
    )
    single_variant_wrong_year.crop_policy = "single_tight"
    single_variant_wrong_year.recognition_variant = "stamp_blackhat"
    evidence_items.append(single_variant_wrong_year)

    accepted = pipeline._accepted_date_evidence_items(evidence_items)
    selected = pipeline._select_date_evidence(accepted)

    assert selected.parsed.parsed_date == date(2026, 5, 3)


def test_mobile_date_evidence_prefers_clean_direct_crop_over_weaker_later_group_crop() -> None:
    pipeline = _pipeline_for_unit_tests()
    direct = _date_evidence_for_score(
        text="11/04/2025",
        parsed_date=date(2025, 4, 11),
        parser_confidence=0.95,
        ocr_confidence=0.994,
    )
    direct.crop_policy = "single_tight"
    direct.recognition_variant = "original_color_tight"
    grouped_later = _date_evidence_for_score(
        text="11/04/2028",
        parsed_date=date(2028, 4, 11),
        parser_confidence=0.95,
        ocr_confidence=0.866,
    )
    grouped_later.crop_policy = "line_from_multiline_group"
    grouped_later.recognition_variant = "hardcase:original_color_tight"

    selected = pipeline._select_date_evidence([direct, grouped_later])

    assert selected is direct


def test_mobile_date_evidence_prefers_direct_crop_over_weaker_later_local_group_wide() -> None:
    pipeline = _pipeline_for_unit_tests()
    direct = _date_evidence_for_score(
        text="07 01 2029",
        parsed_date=date(2029, 1, 7),
        parser_confidence=0.95,
        ocr_confidence=0.917,
    )
    direct.crop_policy = "single_tight"
    direct.recognition_variant = "original_color_tight"
    wide_earlier = _date_evidence_for_score(
        text="07 01 2028",
        parsed_date=date(2028, 1, 7),
        parser_confidence=0.95,
        ocr_confidence=0.880,
    )
    wide_earlier.crop_policy = "local_group_wide_crop"
    wide_earlier.recognition_variant = "original_color_tight"

    selected = pipeline._select_date_evidence([direct, wide_earlier])

    assert selected is direct


def test_mobile_date_evidence_does_not_let_same_line_group_support_override_later_direct_crop() -> None:
    pipeline = _pipeline_for_unit_tests()
    direct = _date_evidence_for_score(
        text="07 01 2029",
        parsed_date=date(2029, 1, 7),
        parser_confidence=0.95,
        ocr_confidence=0.917,
    )
    direct.crop_policy = "single_tight"
    direct.recognition_variant = "original_color_tight"
    repeated_group = []
    for index in range(4):
        evidence = _date_evidence_for_score(
            text="07 01 2028",
            parsed_date=date(2028, 1, 7),
            parser_confidence=0.95,
            ocr_confidence=0.90 + index * 0.005,
        )
        evidence.crop_policy = "same_line_group"
        evidence.recognition_variant = f"group_variant_{index}"
        repeated_group.append(evidence)

    selected = pipeline._select_date_evidence([direct, *repeated_group])

    assert selected is direct


def test_mobile_date_evidence_suppresses_unsupported_compact_far_future_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    clean = _date_evidence_for_score(
        text="12/09/2026",
        parsed_date=date(2026, 9, 12),
        parser_confidence=0.84,
        ocr_confidence=0.72,
    )
    compact = _date_evidence_for_score(
        text="090637 05013",
        parsed_date=date(2037, 6, 9),
        parser_confidence=0.95,
        ocr_confidence=0.84,
    )

    selected = max([clean, compact], key=pipeline._score_date_evidence)

    assert selected is clean


def test_mobile_date_evidence_cluster_overrides_single_weak_same_precision_variant() -> None:
    pipeline = _pipeline_for_unit_tests()
    correct_cluster = [
        _date_evidence_for_score(
            text="09 05 26",
            parsed_date=date(2026, 5, 9),
            parser_confidence=0.86,
            ocr_confidence=0.86 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(correct_cluster):
        evidence.recognition_variant = f"variant_{index}"
    wrong_single = _date_evidence_for_score(
        text="09/05/25",
        parsed_date=date(2025, 5, 9),
        parser_confidence=0.95,
        ocr_confidence=0.95,
    )

    selected = pipeline._select_date_evidence([*correct_cluster, wrong_single])

    assert selected.parsed.parsed_date == date(2026, 5, 9)


def test_mobile_date_evidence_cluster_blocks_later_local_group_wide_override() -> None:
    pipeline = _pipeline_for_unit_tests()
    correct_cluster = [
        _date_evidence_for_score(
            text="13.02.2031",
            parsed_date=date(2031, 2, 13),
            parser_confidence=0.95,
            ocr_confidence=0.90 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(correct_cluster):
        evidence.recognition_variant = f"variant_{index}"
    later_wide = _date_evidence_for_score(
        text="13.02.2036",
        parsed_date=date(2036, 2, 13),
        parser_confidence=0.95,
        ocr_confidence=0.74,
    )
    later_wide.crop_policy = "local_group_wide_crop"

    selected = pipeline._select_date_evidence([*correct_cluster, later_wide])

    assert selected.parsed.parsed_date == date(2031, 2, 13)


def test_mobile_date_evidence_prefers_complete_full_date_over_shorter_misread_cluster() -> None:
    pipeline = _pipeline_for_unit_tests()
    complete = [
        _date_evidence_for_score(
            text="22.06.2028",
            parsed_date=date(2028, 6, 22),
            parser_confidence=0.95,
            ocr_confidence=0.88 + index * 0.08,
        )
        for index in range(2)
    ]
    for index, evidence in enumerate(complete):
        evidence.recognition_variant = f"complete_variant_{index}"
    shorter_misread = [
        _date_evidence_for_score(
            text="2.06.2023",
            parsed_date=date(2023, 6, 2),
            parser_confidence=0.95,
            ocr_confidence=0.91 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(shorter_misread):
        evidence.recognition_variant = f"shorter_variant_{index}"

    selected = pipeline._select_date_evidence([*complete, *shorter_misread])

    assert selected.parsed.parsed_date == date(2028, 6, 22)


def test_mobile_date_evidence_prefers_stable_normal_variant_over_higher_confidence_enhanced_variant() -> None:
    pipeline = _pipeline_for_unit_tests()
    stable = _date_evidence_for_score(
        text="125/02/2029",
        parsed_date=date(2029, 2, 25),
        parser_confidence=0.95,
        ocr_confidence=0.79,
    )
    stable.recognition_variant = "original_color_tight"
    enhanced = _date_evidence_for_score(
        text="125/02/2023",
        parsed_date=date(2023, 2, 25),
        parser_confidence=0.95,
        ocr_confidence=0.82,
        candidate=stable.candidate,
    )
    enhanced.recognition_variant = "gray_upscaled"

    selected = pipeline._select_date_evidence([stable, enhanced])

    assert selected.parsed.parsed_date == date(2029, 2, 25)


def test_mobile_normal_yolo_candidate_uses_padded_bbox_crop_not_polygon_warp() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(bbox_xyxy=(40, 50, 100, 70))
    candidate.detector_sources = ("yolo26s_obb",)
    candidate.detector_variant = "yolo26s_obb"
    candidate.polygon_xy = ((42.0, 50.0), (100.0, 52.0), (98.0, 70.0), (40.0, 68.0))

    crops = pipeline._normalized_crops_for_candidate(image, candidate)

    assert crops
    assert crops[0].polygon_used is False
    assert crops[0].bbox_xyxy == (37, 47, 103, 73)
    assert candidate.final_crop_bbox == (37, 47, 103, 73)
    assert candidate.final_crop_padding_px == 3


def test_mobile_date_evidence_prefers_later_direct_month_over_older_multiline_split() -> None:
    pipeline = _pipeline_for_unit_tests()
    older = []
    for index in range(4):
        evidence = _date_evidence_for_score(
            text="10/2025",
            parsed_date=date(2025, 10, 31),
            precision="month",
            parser_confidence=0.75,
            ocr_confidence=0.90,
            candidate=_mobile_ranked_candidate(candidate_id="older"),
        )
        evidence.recognition_variant = f"older_variant_{index}"
        evidence.crop_policy = "line_from_multiline_group"
        older.append(evidence)
    later = []
    for index in range(2):
        evidence = _date_evidence_for_score(
            text="10/2026",
            parsed_date=date(2026, 10, 31),
            precision="month",
            parser_confidence=0.75,
            ocr_confidence=0.76,
            candidate=_mobile_ranked_candidate(candidate_id="older"),
        )
        evidence.recognition_variant = f"later_variant_{index}"
        evidence.crop_policy = "line_role_context"
        later.append(evidence)

    selected = pipeline._select_date_evidence([*older, *later])

    assert selected.parsed.parsed_date == date(2026, 10, 31)


def test_mobile_svtr_variant_recognition_is_cached_per_request() -> None:
    pipeline = _pipeline_for_unit_tests()
    recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("09/12/2026", "09/12/2026", 0.96, None, "original"),
            _RecognitionOutput("10/12/2026", "10/12/2026", 0.96, None, "original"),
        ]
    )
    recognizer.cacheable = True
    pipeline.recognizer = recognizer
    image = np.full((16, 64, 3), 255, dtype=np.uint8)

    first = pipeline._recognize_variant(image, variant_name="original_color_tight", orientation="original")
    second = pipeline._recognize_variant(image.copy(), variant_name="original_color_tight", orientation="original")
    third = pipeline._recognize_variant(image, variant_name="gray_upscaled", orientation="original")

    assert first.raw_text == "09/12/2026"
    assert second.raw_text == "09/12/2026"
    assert third.raw_text == "10/12/2026"
    assert len(recognizer.calls) == 2


def test_mobile_probe_variant_scoring_uses_batch_recognition(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.svtr_batch_recognition_enabled = True
    recognizer = _BatchQueueRecognizer(
        [
            _RecognitionOutput("NO DATE", "NO DATE", 0.91, None, "original"),
            _RecognitionOutput("WORD 11/12/2026", "WORD 11/12/2026", 0.82, None, "original"),
            _RecognitionOutput("WORD 12/12/2026", "WORD 12/12/2026", 0.83, None, "original"),
        ]
    )
    pipeline.recognizer = recognizer
    image = np.full((24, 120, 3), 255, dtype=np.uint8)
    variants = [
        ImageVariant("original_color_tight", image, purpose="recognition"),
        ImageVariant("gray_upscaled", image[:, :80].copy(), purpose="recognition"),
        ImageVariant("unsharp_gray", image[:, :60].copy(), purpose="recognition"),
    ]
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, *, allowed_names=None: variants,
        raising=False,
    )

    best, variant_name, variant_image = pipeline._best_probe_for_crop(image, today=date(2026, 5, 27))

    assert best.raw_text == "NO DATE"
    assert variant_name == "original_color_tight"
    assert variant_image is image
    assert recognizer.calls == []
    assert recognizer.batch_calls == [[(24, 120), (24, 80), (24, 60)]]


def test_mobile_batch_recognition_uses_cached_variants() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.svtr_batch_recognition_enabled = True
    recognizer = _BatchQueueRecognizer(
        [
            _RecognitionOutput("11/12/2026", "11/12/2026", 0.92, None, "original"),
            _RecognitionOutput("12/12/2026", "12/12/2026", 0.93, None, "original"),
        ]
    )
    pipeline.recognizer = recognizer
    image = np.full((24, 120, 3), 255, dtype=np.uint8)
    variants = [
        ImageVariant("original_color_tight", image, purpose="recognition"),
        ImageVariant("gray_upscaled", image[:, :80].copy(), purpose="recognition"),
    ]

    first = pipeline._recognize_variants_batch(variants, orientation="original")
    second = pipeline._recognize_variants_batch(variants, orientation="original")

    assert [rec.raw_text for _variant, rec in first] == ["11/12/2026", "12/12/2026"]
    assert [rec.raw_text for _variant, rec in second] == ["11/12/2026", "12/12/2026"]
    assert recognizer.calls == []
    assert recognizer.batch_calls == [[(24, 120), (24, 80)]]


def test_mobile_strong_day_parse_stops_single_crop_variant_fanout() -> None:
    pipeline = _pipeline_for_unit_tests()
    recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("09/12/2026", "09/12/2026", 0.96, None, "original"),
            _RecognitionOutput("SHOULD_NOT_USE", "SHOULD_NOT_USE", 0.99, None, "original"),
        ]
    )
    pipeline.recognizer = recognizer
    image = np.full((24, 120, 3), 255, dtype=np.uint8)
    crop = NormalizedTextLineCrop(
        image=image,
        bbox_xyxy=(4, 5, 124, 29),
        polygon_used=None,
        crop_transform_used="unit",
        selected_orientation="original",
        original_crop_shape=[24, 120, 3],
        normalized_crop_shape=[24, 120, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="unit",
    )

    evaluated = pipeline._best_recognition_for_single_crop(crop, today=date(2026, 5, 27))

    assert evaluated.parsed.parsed_date == date(2026, 12, 9)
    assert evaluated.recognition.raw_text == "09/12/2026"
    assert len(recognizer.calls) == 1


def test_mobile_normal_crop_does_not_prioritize_aggressive_probe_variant() -> None:
    pipeline = _pipeline_for_unit_tests()

    class MeanAwareRecognizer:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def recognize(self, image: np.ndarray) -> _RecognitionOutput:
            mean_value = int(image.mean())
            self.calls.append(mean_value)
            if mean_value == 220:
                return _RecognitionOutput("03/02/2024", "03/02/2024", 0.96, None, "original")
            return _RecognitionOutput("03/12/2024", "03/12/2024", 0.78, None, "original")

    recognizer = MeanAwareRecognizer()
    pipeline.recognizer = recognizer
    image = np.full((24, 120, 3), 255, dtype=np.uint8)
    selected_variant_image = np.full((24, 120, 3), 220, dtype=np.uint8)
    crop = NormalizedTextLineCrop(
        image=image,
        bbox_xyxy=(4, 5, 124, 29),
        polygon_used=None,
        crop_transform_used="unit",
        selected_orientation="original",
        original_crop_shape=[24, 120, 3],
        normalized_crop_shape=[24, 120, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="unit",
    )

    evaluated = pipeline._best_recognition_for_single_crop(
        crop,
        today=date(2026, 5, 27),
        selected_variant_name="adaptive_binary",
        selected_variant_image=selected_variant_image,
        crop_policy="single_tight",
    )

    assert evaluated.parsed.parsed_date == date(2024, 12, 3)
    assert evaluated.recognition.raw_text == "03/12/2024"


def test_mobile_strong_day_date_evidence_blocks_expensive_rescues() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.legacy_wide_group_rescue_enabled = True
    pipeline.local_group_wide_crop_enabled = True
    pipeline.paired_crop_evidence_enabled = True
    pipeline.production_anchor_sibling_enabled = True
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(10, 10, 120, 60),
        member_bboxes=[(10, 10, 120, 28), (10, 32, 120, 50)],
    )
    candidate.final_crop_bbox = (18, 15, 100, 45)
    strong = _date_evidence_for_score(
        text="EXP 09/12/2026",
        parsed_date=date(2026, 12, 9),
        parser_confidence=0.95,
        ocr_confidence=0.96,
        candidate=candidate,
    )
    conflicting = _date_evidence_for_score(
        text="10/12/2026",
        parsed_date=date(2026, 12, 10),
        parser_confidence=0.75,
        ocr_confidence=0.70,
        candidate=candidate,
    )
    evaluated = _EvaluatedCandidate(
        candidate=candidate,
        recognition=strong.recognition,
        parsed=strong.parsed,
        parse_inputs_count=strong.parse_inputs_count,
        recognition_variant=strong.recognition_variant,
        normalized_crop=None,
        date_evidence=[strong, conflicting],
    )

    assert not pipeline._should_run_legacy_wide_group_rescue(candidate, evaluated, [strong, conflicting])
    assert not pipeline._should_run_line_role_rescue(candidate, evaluated, [strong, conflicting])
    assert not pipeline._should_run_local_group_wide_crop_evidence(
        candidate,
        evaluated,
        [strong, conflicting],
        image_shape=(100, 160, 3),
    )
    assert not pipeline._should_run_paired_crop_evidence(candidate, evaluated, [strong, conflicting])


def test_mobile_hardcase_recognizer_rescue_selects_later_supported_day_cluster() -> None:
    pipeline = _pipeline_for_unit_tests()
    primary_candidate = _mobile_ranked_candidate(
        candidate_type="group",
        member_bboxes=[(10, 10, 90, 28), (12, 30, 92, 48)],
    )
    primary = _date_evidence_for_score(
        text="4.02.2026",
        parsed_date=date(2026, 2, 4),
        candidate=primary_candidate,
        parser_confidence=0.95,
        ocr_confidence=0.96,
    )
    secondary = [
        _date_evidence_for_score(
            text="13.02.2031",
            parsed_date=date(2031, 2, 13),
            candidate=primary_candidate,
            parser_confidence=0.95,
            ocr_confidence=0.90 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(secondary):
        evidence.recognition_variant = f"hardcase_variant_{index}"
        evidence.metadata["recognizer_rescue"] = "hardcase"

    selected = pipeline._select_hardcase_recognizer_override(primary, secondary, today=date(2026, 5, 27))

    assert selected is not None
    assert selected.parsed.parsed_date == date(2031, 2, 13)


def test_mobile_hardcase_recognizer_rescue_rejects_earlier_specialist_cluster() -> None:
    pipeline = _pipeline_for_unit_tests()
    primary = _date_evidence_for_score(
        text="22.06.2028",
        parsed_date=date(2028, 6, 22),
        parser_confidence=0.95,
        ocr_confidence=0.96,
    )
    secondary = [
        _date_evidence_for_score(
            text="2.06.2023",
            parsed_date=date(2023, 6, 2),
            parser_confidence=0.95,
            ocr_confidence=0.90 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(secondary):
        evidence.recognition_variant = f"hardcase_variant_{index}"
        evidence.metadata["recognizer_rescue"] = "hardcase"

    selected = pipeline._select_hardcase_recognizer_override(primary, secondary, today=date(2026, 5, 27))

    assert selected is None


def test_mobile_hardcase_recognizer_rescue_does_not_override_strong_weak_expiry_primary() -> None:
    pipeline = _pipeline_for_unit_tests()
    primary = _date_evidence_for_score(
        text="11/04/2025",
        parsed_date=date(2025, 4, 11),
        parser_confidence=0.95,
        ocr_confidence=0.95,
        context_texts=("1E.T.1",),
    )
    secondary = [
        _date_evidence_for_score(
            text="11/04/2028",
            parsed_date=date(2028, 4, 11),
            parser_confidence=0.95,
            ocr_confidence=0.88,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(secondary):
        evidence.recognition_variant = f"hardcase_variant_{index}"
        evidence.metadata["recognizer_rescue"] = "hardcase"

    selected = pipeline._select_hardcase_recognizer_override(primary, secondary, today=date(2026, 5, 27))

    assert selected is None


def test_mobile_hardcase_recognizer_rescue_rejects_production_context_reread() -> None:
    pipeline = _pipeline_for_unit_tests()
    primary = _date_evidence_for_score(
        text="11/04/2025",
        parsed_date=date(2025, 4, 11),
        parser_confidence=0.95,
        ocr_confidence=0.95,
        context_texts=("1E.T.1",),
    )
    secondary = [
        _date_evidence_for_score(
            text="11/04/2028",
            parsed_date=date(2028, 4, 11),
            parser_confidence=0.95,
            ocr_confidence=0.88 + index * 0.01,
            context_texts=("URT.11",),
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(secondary):
        evidence.recognition_variant = f"hardcase_variant_{index}"
        evidence.metadata["recognizer_rescue"] = "hardcase"
        evidence.metadata["source_parsed_date"] = "2028-04-11"

    selected = pipeline._select_hardcase_recognizer_override(primary, secondary, today=date(2026, 5, 31))

    assert selected is None


def test_mobile_hardcase_recognizer_rescue_allows_same_crop_year_correction() -> None:
    pipeline = _pipeline_for_unit_tests()
    primary = _date_evidence_for_score(
        text="13.02.2021",
        parsed_date=date(2021, 2, 13),
        parser_confidence=0.95,
        ocr_confidence=0.92,
        context_texts=("1ETT.1", "1ETT.13"),
    )
    secondary = [
        _date_evidence_for_score(
            text="13.02.2031",
            parsed_date=date(2031, 2, 13),
            parser_confidence=0.95,
            ocr_confidence=0.88 + index * 0.01,
        )
        for index in range(4)
    ]
    for index, evidence in enumerate(secondary):
        evidence.recognition_variant = f"hardcase_variant_{index}"
        evidence.metadata["recognizer_rescue"] = "hardcase"
        evidence.metadata["source_parsed_date"] = "2021-02-13"

    selected = pipeline._select_hardcase_recognizer_override(primary, secondary, today=date(2026, 5, 31))

    assert selected is not None
    assert selected.parsed.parsed_date == date(2031, 2, 13)


def test_mobile_hardcase_rescue_uses_same_date_multiline_sibling(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    hardcase_recognizer = object()
    pipeline.hardcase_recognizer_rescue_enabled = True
    pipeline.hardcase_recognizer = hardcase_recognizer
    best_candidate = _mobile_ranked_candidate(candidate_id="cand_single", candidate_type="single")
    group_candidate = _mobile_ranked_candidate(
        candidate_id="cand_group",
        candidate_type="group",
        bbox_xyxy=(10, 10, 120, 60),
        member_bboxes=[(10, 10, 120, 30), (10, 34, 120, 60)],
    )
    best = _date_evidence_for_score(
        text="13.02.2021",
        parsed_date=date(2021, 2, 13),
        candidate=best_candidate,
        parser_confidence=0.95,
        ocr_confidence=0.92,
    )
    sibling = _date_evidence_for_score(
        text="13.02.2021",
        parsed_date=date(2021, 2, 13),
        candidate=group_candidate,
        parser_confidence=0.95,
        ocr_confidence=0.92,
    )
    sibling.crop_policy = "line_role_context"
    sibling.normalized_crop = NormalizedTextLineCrop(
        image=np.full((24, 120, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 34, 120, 60),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[24, 120, 3],
        normalized_crop_shape=[24, 120, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )

    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, allowed_names=MOBILE_RECOGNITION_VARIANTS: [
            ImageVariant("original_color_tight", crop, purpose="recognition")
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "_recognize_variant",
        lambda *_args, **_kwargs: _RecognitionOutput("13.02.2031", "13.02.2031", 0.91, None, "original"),
    )

    assert pipeline._should_run_hardcase_recognizer_rescue(best, [best, sibling], today=date(2026, 5, 27))
    rescue = pipeline._collect_hardcase_recognizer_rescue_evidence(best, [best, sibling], today=date(2026, 5, 27))

    assert rescue
    assert rescue[0].parsed.parsed_date == date(2031, 2, 13)
    assert rescue[0].candidate.candidate_id == "cand_group"


def test_mobile_date_evidence_cluster_does_not_let_month_override_day_precision() -> None:
    pipeline = _pipeline_for_unit_tests()
    day_single = _date_evidence_for_score(
        text="07/03/2026",
        parsed_date=date(2026, 3, 7),
        precision="day",
        parser_confidence=0.92,
        ocr_confidence=0.92,
    )
    month_cluster = [
        _date_evidence_for_score(
            text="03/2026",
            parsed_date=date(2026, 3, 31),
            precision="month",
            parser_confidence=0.95,
            ocr_confidence=0.95,
        )
        for _ in range(4)
    ]
    for index, evidence in enumerate(month_cluster):
        evidence.recognition_variant = f"variant_{index}"

    selected = pipeline._select_date_evidence([day_single, *month_cluster])

    assert selected is day_single


def test_mobile_date_evidence_prefers_supported_later_month_over_naked_us_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(candidate_id="cand_group")
    day_single = _date_evidence_for_score(
        text="2/15/2025",
        parsed_date=date(2025, 2, 15),
        precision="day",
        parser_confidence=0.93,
        ocr_confidence=0.79,
        date_format_detected="MM/DD/YYYY_UNAMBIGUOUS",
        candidate=candidate,
    )
    month_cluster = [
        _date_evidence_for_score(
            text="10/2026",
            parsed_date=date(2026, 10, 31),
            precision="month",
            parser_confidence=0.75,
            ocr_confidence=0.90,
            date_format_detected="MM/YYYY",
            candidate=candidate,
        )
        for _ in range(4)
    ]
    for index, evidence in enumerate(month_cluster):
        evidence.recognition_variant = f"variant_{index}"

    selected = pipeline._select_date_evidence([day_single, *month_cluster])

    assert selected.parsed.parsed_date == date(2026, 10, 31)


def test_mobile_suppresses_unsupported_compact_far_future_date_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    compact = _date_evidence_for_score(
        text="090637 05013",
        parsed_date=date(2037, 6, 9),
        parser_confidence=0.72,
        ocr_confidence=0.84,
    )

    assert pipeline._should_suppress_date_evidence(compact)


def test_mobile_suppresses_weak_compact_date_on_image_edge() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    candidate.detector_confidence = 0.174
    candidate.geometry_features["edge_proximity"] = 0.0
    compact = _date_evidence_for_score(
        text="02092022",
        parsed_date=date(2022, 9, 2),
        parser_confidence=0.9,
        ocr_confidence=0.59,
        candidate=candidate,
    )

    assert pipeline._should_suppress_date_evidence(compact)


def test_mobile_rejects_parseable_candidate_when_all_date_evidence_is_suppressed() -> None:
    pipeline = _pipeline_for_unit_tests()
    compact = _date_evidence_for_score(
        text="090637 05013",
        parsed_date=date(2037, 6, 9),
        parser_confidence=0.72,
        ocr_confidence=0.84,
    )
    evaluated = pipeline._evaluated_from_date_evidence(compact, date_evidence=[compact])

    assert not pipeline._has_accepted_date_evidence(evaluated)


def test_mobile_treats_all_suppressed_date_evidence_as_no_parseable_candidate() -> None:
    pipeline = _pipeline_for_unit_tests()
    compact = _date_evidence_for_score(
        text="090637 05013",
        parsed_date=date(2037, 6, 9),
        parser_confidence=0.72,
        ocr_confidence=0.84,
    )
    evaluated = [pipeline._evaluated_from_date_evidence(compact, date_evidence=[compact])]

    parseable, evidence = pipeline._accepted_parseable_items(evaluated)

    assert parseable == []
    assert evidence == []


def test_mobile_rejects_weak_single_variant_rapidocr_date_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.strict_evidence_acceptance_enabled = True
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("yolo26s_obb", "rapidocr_ppocrv5")
    rapid_candidate.detector_variant = "rapidocr_primary:PP-OCRv5_mobile:raw"
    weak = _date_evidence_for_score(
        text="20/10/2028",
        parsed_date=date(2028, 10, 20),
        parser_confidence=0.76,
        ocr_confidence=0.76,
        candidate=rapid_candidate,
    )
    evaluated = [pipeline._evaluated_from_date_evidence(weak, date_evidence=[weak])]

    parseable, evidence = pipeline._accepted_parseable_items(evaluated)

    assert parseable == []
    assert evidence == []


def test_mobile_treats_gated_rapidocr_as_risky_without_global_strict_flag() -> None:
    pipeline = _pipeline_for_unit_tests()
    core_candidate = _mobile_ranked_candidate()
    core_candidate.detector_sources = ("yolo26s_obb",)
    core_candidate.detector_variant = "yolo26s_obb"
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    core = _date_evidence_for_score(
        text="05/26",
        parsed_date=date(2026, 5, 31),
        precision="month",
        parser_confidence=0.74,
        ocr_confidence=0.99,
        candidate=core_candidate,
    )
    noisy_rapid = _date_evidence_for_score(
        text="S7TJ1/05/26",
        parsed_date=date(2026, 5, 1),
        parser_confidence=0.76,
        ocr_confidence=0.75,
        candidate=rapid_candidate,
    )
    evaluated = [
        pipeline._evaluated_from_date_evidence(core, date_evidence=[core]),
        pipeline._evaluated_from_date_evidence(noisy_rapid, date_evidence=[noisy_rapid]),
    ]

    parseable, evidence = pipeline._accepted_parseable_items(evaluated)

    assert parseable == [evaluated[0]]
    assert evidence == [core]


def test_mobile_rapidocr_evidence_cannot_override_different_core_yolo_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.strict_evidence_acceptance_enabled = True
    pipeline.rapidocr_role_constraint_enabled = True
    core_candidate = _mobile_ranked_candidate()
    core_candidate.detector_sources = ("yolo26s_obb",)
    core_candidate.detector_variant = "yolo26s_obb"
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("yolo26s_obb", "rapidocr_ppocrv5")
    rapid_candidate.detector_variant = "rapidocr_primary:PP-OCRv5_mobile:raw"
    core = _date_evidence_for_score(
        text="12/05/2027",
        parsed_date=date(2027, 5, 12),
        parser_confidence=0.92,
        ocr_confidence=0.92,
        candidate=core_candidate,
    )
    rapid = _date_evidence_for_score(
        text="20/10/2028",
        parsed_date=date(2028, 10, 20),
        parser_confidence=0.96,
        ocr_confidence=0.96,
        candidate=rapid_candidate,
    )
    evaluated = [
        pipeline._evaluated_from_date_evidence(core, date_evidence=[core]),
        pipeline._evaluated_from_date_evidence(rapid, date_evidence=[rapid]),
    ]

    parseable, evidence = pipeline._accepted_parseable_items(evaluated)

    assert parseable == [evaluated[0]]
    assert evidence == [core]


def test_mobile_accepts_multivariant_rapidocr_date_when_no_core_yolo_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.strict_evidence_acceptance_enabled = True
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("yolo26s_obb", "rapidocr_ppocrv5")
    rapid_candidate.detector_variant = "rapidocr_primary:PP-OCRv5_mobile:raw"
    evidence = [
        _date_evidence_for_score(
            text="20/10/2028",
            parsed_date=date(2028, 10, 20),
            parser_confidence=0.78,
            ocr_confidence=0.78,
            candidate=rapid_candidate,
        ),
        _date_evidence_for_score(
            text="20/10/2028",
            parsed_date=date(2028, 10, 20),
            parser_confidence=0.80,
            ocr_confidence=0.80,
            candidate=rapid_candidate,
        ),
    ]
    evidence[0].recognition_variant = "original_color_tight"
    evidence[1].recognition_variant = "clahe_gray"
    evaluated = [pipeline._evaluated_from_date_evidence(evidence[0], date_evidence=evidence)]

    parseable, accepted = pipeline._accepted_parseable_items(evaluated)

    assert parseable == evaluated
    assert accepted == evidence


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


def test_mobile_strong_yolo_expiry_skips_ppmobile_gated_rescue() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo = _YoloCandidate(
        polygon_xy=[[10.0, 20.0], [130.0, 20.0], [130.0, 45.0], [10.0, 45.0]],
        bbox_xyxy=(10, 20, 130, 45),
        confidence=0.88,
    )
    evidence = _date_evidence_for_score(
        text="EXP 12/05/2027",
        parsed_date=date(2027, 5, 12),
        parser_confidence=0.97,
        ocr_confidence=0.91,
    )
    evaluated = [pipeline._evaluated_from_date_evidence(evidence, date_evidence=[evidence])]

    assert pipeline._has_strong_expiry_answer(evaluated)
    assert not pipeline._should_run_ppmobile_proposal_rescue([yolo], evaluated)


def test_mobile_ppmobile_gated_rescue_runs_for_no_yolo_or_unresolved_yolo() -> None:
    pipeline = _pipeline_for_unit_tests()

    assert pipeline._should_run_ppmobile_proposal_rescue([], [])

    unresolved_yolo = [
        _YoloCandidate(
            polygon_xy=[[10.0, 20.0], [130.0, 20.0], [130.0, 45.0], [10.0, 45.0]],
            bbox_xyxy=(10, 20, 130, 45),
            confidence=0.55,
        )
    ]

    assert pipeline._should_run_ppmobile_proposal_rescue(unresolved_yolo, [])


def test_mobile_ppmobile_gated_rescue_runs_for_production_or_partial_yolo() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo = [
        _YoloCandidate(
            polygon_xy=[[10.0, 20.0], [130.0, 20.0], [130.0, 45.0], [10.0, 45.0]],
            bbox_xyxy=(10, 20, 130, 45),
            confidence=0.88,
        )
    ]
    production = _date_evidence_for_score(
        text="URT 01/01/2026",
        parsed_date=date(2026, 1, 1),
        parser_confidence=0.97,
        ocr_confidence=0.96,
    )
    partial = _date_evidence_for_score(
        text="12/2027",
        parsed_date=date(2027, 12, 1),
        precision="month",
        parser_confidence=0.97,
        ocr_confidence=0.96,
    )

    assert pipeline._should_run_ppmobile_proposal_rescue(
        yolo,
        [pipeline._evaluated_from_date_evidence(production, date_evidence=[production])],
    )
    assert pipeline._should_run_ppmobile_proposal_rescue(
        yolo,
        [pipeline._evaluated_from_date_evidence(partial, date_evidence=[partial])],
    )


def test_mobile_prefilters_ppmobile_gated_proposals_to_small_ranked_set() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_gated_prefilter_probe_top_k = 8
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    proposals = [
        ProposalBox(
            bbox_xyxy=(0, 0, 300, 170),
            confidence=0.99,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="full_paragraph",
            polygon_xy=None,
        ),
        ProposalBox(
            bbox_xyxy=(10, 10, 12, 12),
            confidence=0.20,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="tiny_noise",
            polygon_xy=None,
        ),
        *[
            ProposalBox(
                bbox_xyxy=(20, 20 + index * 12, 120, 30 + index * 12),
                confidence=0.90 - index * 0.01,
                source="rapidocr_ppocrv5",
                sources=("rapidocr_ppocrv5",),
                variant_name=f"date_like_{index}",
                polygon_xy=None,
            )
            for index in range(12)
        ],
    ]

    filtered = pipeline._geometry_prefilter_rapidocr_gated_proposals(image, proposals, yolo_candidates=[])

    assert len(filtered) == 8
    assert all(proposal.variant_name not in {"full_paragraph", "tiny_noise"} for proposal in filtered)


def test_mobile_rapidocr_probe_score_rejects_non_date_text() -> None:
    pipeline = _pipeline_for_unit_tests()

    assert pipeline._rapidocr_probe_text_score("AYRAN", 0.99, today=date(2026, 1, 1))[1] == "too_few_digits"
    assert pipeline._rapidocr_probe_text_score("A12", 0.99, today=date(2026, 1, 1))[1] == "too_few_digits"


def test_mobile_rapidocr_probe_score_accepts_date_like_text() -> None:
    pipeline = _pipeline_for_unit_tests()

    score, reason = pipeline._rapidocr_probe_text_score("03-05-26", 0.92, today=date(2026, 1, 1))

    assert reason is None
    assert score >= 3.0


def test_mobile_rapidocr_probe_score_accepts_date_like_fragment() -> None:
    pipeline = _pipeline_for_unit_tests()

    score, reason = pipeline._rapidocr_probe_text_score("S.11.2026", 0.82, today=date(2026, 1, 1))

    assert reason is None
    assert score >= pipeline.rapidocr_gated_min_date_likeness_score


def test_mobile_rapidocr_probe_score_rejects_dirty_extra_numeric_tail() -> None:
    pipeline = _pipeline_for_unit_tests()

    score, reason = pipeline._rapidocr_probe_text_score("2026.04.24.206", 0.92, today=date(2026, 1, 1))

    assert score == 0.0
    assert reason == "extra_numeric_date_tail"


def test_mobile_rapidocr_probe_score_accepts_expiry_date_followed_by_time() -> None:
    pipeline = _pipeline_for_unit_tests()

    score, reason = pipeline._rapidocr_probe_text_score("TETT:22.02.2611:49", 0.88, today=date(2026, 1, 1))
    noisy_score, noisy_reason = pipeline._rapidocr_probe_text_score(
        "1EY.22.02.6.1.19",
        0.75,
        today=date(2026, 1, 1),
    )
    missing_lead_score, missing_lead_reason = pipeline._rapidocr_probe_text_score(
        "ET.22.02.6.1.19",
        0.88,
        today=date(2026, 1, 1),
    )

    assert reason is None
    assert score >= 3.0
    assert noisy_reason is None
    assert noisy_score >= 3.0
    assert missing_lead_reason is None
    assert missing_lead_score >= 3.0


def test_mobile_rapidocr_geometry_prefilter_keeps_overlapping_distinct_raw_boxes() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_gated_prefilter_probe_top_k = 36
    image = np.full((4032, 3024, 3), 255, dtype=np.uint8)
    overlapping_right = ProposalBox(
        bbox_xyxy=(1371, 1343, 1693, 1785),
        confidence=0.66,
        source="rapidocr_ppocrv5",
        sources=("rapidocr_ppocrv5",),
        variant_name="rapidocr_gated:PP-OCRv5_mobile",
        polygon_xy=((1450.0, 1343.0), (1693.0, 1734.0), (1614.0, 1785.0), (1371.0, 1393.0)),
    )
    exact_left = ProposalBox(
        bbox_xyxy=(1303, 1352, 1559, 1687),
        confidence=0.54,
        source="rapidocr_ppocrv5",
        sources=("rapidocr_ppocrv5",),
        variant_name="rapidocr_gated:PP-OCRv5_mobile",
        polygon_xy=((1374.0, 1352.0), (1559.0, 1640.0), (1488.0, 1687.0), (1303.0, 1399.0)),
    )

    filtered = pipeline._geometry_prefilter_rapidocr_gated_proposals(
        image,
        [overlapping_right, exact_left],
        yolo_candidates=[],
    )

    assert overlapping_right.bbox_xyxy in {proposal.bbox_xyxy for proposal in filtered}
    assert exact_left.bbox_xyxy in {proposal.bbox_xyxy for proposal in filtered}


def test_mobile_caps_final_padding_for_rapidocr_polygon_crops() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((400, 500, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        bbox_xyxy=(100, 120, 400, 210),
        polygon_xy=((102.0, 120.0), (400.0, 126.0), (398.0, 210.0), (100.0, 204.0)),
    )
    candidate.detector_sources = ("rapidocr_ppocrv5",)
    candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"

    crops = pipeline._normalized_crops_for_candidate(image, candidate)

    assert crops
    assert crops[0].polygon_used
    assert candidate.final_crop_padding_px is not None
    assert candidate.final_crop_padding_px <= 6


def test_mobile_adds_date_tail_crop_for_rapidocr_expiry_prefix_text() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((400, 500, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        bbox_xyxy=(100, 120, 400, 210),
        polygon_xy=((100.0, 120.0), (400.0, 126.0), (398.0, 210.0), (100.0, 204.0)),
    )
    candidate.detector_sources = ("rapidocr_ppocrv5",)
    candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    candidate.rapidocr_probe_text = "S.11.26"
    candidate.rapidocr_probe_normalized_text = "S.11.26"

    crops = pipeline._normalized_crops_for_candidate(image, candidate)

    assert len(crops) > 1
    assert any(crop.bbox_xyxy[0] > candidate.bbox_xyxy[0] for crop in crops)


def test_mobile_accepts_rapidocr_date_tail_evidence_as_strong() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(
        bbox_xyxy=(100, 120, 400, 210),
        polygon_xy=((100.0, 120.0), (400.0, 126.0), (398.0, 210.0), (100.0, 204.0)),
    )
    candidate.detector_sources = ("rapidocr_ppocrv5",)
    candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    candidate.rapidocr_probe_text = "S.11.26"
    candidate.rapidocr_probe_normalized_text = "S.11.26"
    tail_crop = NormalizedTextLineCrop(
        image=np.full((90, 220, 3), 255, dtype=np.uint8),
        bbox_xyxy=(172, 120, 400, 210),
        polygon_used=True,
        crop_transform_used="polygon_warp",
        selected_orientation="original",
        original_crop_shape=[90, 220, 3],
        normalized_crop_shape=[90, 220, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="horizontal_or_square",
    )
    evidence = _date_evidence_for_score(
        text="T11.04.26",
        parsed_date=date(2026, 4, 11),
        parser_confidence=0.88,
        ocr_confidence=0.84,
        candidate=candidate,
        normalized_crop=tail_crop,
    )

    assert pipeline._is_strong_rapidocr_gated_evidence(evidence, [evidence])


def test_mobile_rejects_gated_rapidocr_alpha_completed_day_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="S.11.2026",
        parsed_date=date(2026, 11, 5),
        parser_confidence=0.82,
        ocr_confidence=0.81,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == []


def test_mobile_rejects_gated_rapidocr_dirty_extra_numeric_date_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="2026.04.4.206",
        parsed_date=date(2026, 4, 4),
        parser_confidence=0.91,
        ocr_confidence=0.93,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == []


def test_mobile_rejects_dirty_rapidocr_grouped_override_when_yolo_has_parseable_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo_candidate = _mobile_ranked_candidate()
    yolo_candidate.detector_sources = ("yolo26s_obb",)
    yolo_candidate.detector_variant = "yolo26s_obb"
    yolo_month = _date_evidence_for_score(
        text="02.2028",
        parsed_date=date(2028, 2, 29),
        precision="month",
        parser_confidence=0.75,
        ocr_confidence=0.94,
        candidate=yolo_candidate,
    )
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated_grouped:1"
    dirty_rapid = _date_evidence_for_score(
        text="128.02.20.0",
        parsed_date=date(2020, 2, 28),
        parser_confidence=0.88,
        ocr_confidence=0.88,
        candidate=rapid_candidate,
    )
    evaluated = [
        pipeline._evaluated_from_date_evidence(yolo_month, date_evidence=[yolo_month]),
        pipeline._evaluated_from_date_evidence(dirty_rapid, date_evidence=[dirty_rapid]),
    ]

    _parseable, accepted = pipeline._accepted_parseable_items(evaluated)
    selected = pipeline._select_date_evidence(accepted)

    assert dirty_rapid not in accepted
    assert selected is yolo_month


def test_mobile_accepts_clean_high_confidence_gated_rapidocr_date_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="06/05/2026",
        parsed_date=date(2026, 5, 6),
        parser_confidence=0.95,
        ocr_confidence=0.96,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_accepts_clean_space_separated_gated_rapidocr_date_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="06 05 2027",
        parsed_date=date(2027, 5, 6),
        parser_confidence=0.80,
        ocr_confidence=0.96,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_accepts_bounded_prefix_gated_rapidocr_date_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="SYT.19 07 2026",
        parsed_date=date(2026, 7, 19),
        parser_confidence=0.80,
        ocr_confidence=0.85,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_accepts_bounded_prefix_gated_rapidocr_date_when_normalized_text_differs() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="SYT.19 07 2026",
        parsed_date=date(2026, 7, 19),
        parser_confidence=0.80,
        ocr_confidence=0.85,
        candidate=rapid_candidate,
    )
    evidence.recognition.normalized_text = "5YT.19 07 2026"

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_accepts_clean_trailing_gated_rapidocr_date_as_final_evidence() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="01 TV1 24/09/2026",
        parsed_date=date(2026, 9, 24),
        parser_confidence=0.95,
        ocr_confidence=0.87,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_accepts_weak_expiry_rapidocr_date_token_inside_noisy_text() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated:PP-OCRv5_mobile"
    evidence = _date_evidence_for_score(
        text="1ET18/05/26 AR1A1EN ONCE 1U111M1A19E 121",
        parsed_date=date(2026, 5, 18),
        parser_confidence=0.93,
        ocr_confidence=0.58,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_prefilter_probe_does_not_override_conflicting_final_svtr_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    candidate.detector_sources = ("rapidocr_ppocrv5",)
    candidate.detector_variant = "rapidocr_gated_grouped:1"
    candidate.rapidocr_probe_text = "102.08.2026"
    candidate.rapidocr_probe_normalized_text = "102.08.2026"
    candidate.rapidocr_probe_confidence = 0.95
    candidate.rapidocr_probe_score = 3.2

    final_svtr = _date_evidence_for_score(
        text="107.08.2026",
        parsed_date=date(2026, 8, 7),
        parser_confidence=0.95,
        ocr_confidence=0.82,
        candidate=candidate,
    )
    evaluated = pipeline._evaluated_from_date_evidence(final_svtr, date_evidence=[final_svtr])

    evidence = pipeline._collect_rapidocr_probe_prefilter_evidence(
        candidate,
        evaluated,
        today=date(2026, 1, 1),
    )

    assert evidence == []


def test_mobile_accepts_supported_noisy_leading_digit_gated_rapidocr_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    rapid_candidate = _mobile_ranked_candidate()
    rapid_candidate.detector_sources = ("rapidocr_ppocrv5",)
    rapid_candidate.detector_variant = "rapidocr_gated_grouped:1"
    evidence = _date_evidence_for_score(
        text="107.08.2026",
        parsed_date=date(2026, 8, 7),
        parser_confidence=0.95,
        ocr_confidence=0.82,
        candidate=rapid_candidate,
    )

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_rapidocr_polygon_probe_uses_polygon_warp() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((80, 160, 3), 255, dtype=np.uint8)
    proposal = ProposalBox(
        bbox_xyxy=(20, 20, 120, 45),
        confidence=0.91,
        source="rapidocr_ppocrv5",
        sources=("rapidocr_ppocrv5",),
        variant_name="rapidocr_gated:PP-OCRv5_mobile",
        polygon_xy=((18.0, 25.0), (118.0, 18.0), (121.0, 39.0), (22.0, 48.0)),
    )

    crops = pipeline._normalized_rapidocr_polygon_probe_crops(image, proposal)

    assert crops
    assert crops[0].polygon_used is True
    assert crops[0].crop_transform_used == "polygon_warp"


def test_mobile_ocr_prefilter_keeps_date_like_ppmobile_proposal() -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_gated_max_boxes_accepted = 4
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("AYRAN", "AYRAN", 0.99, None, "original"),
            _RecognitionOutput("03-05-26", "03-05-26", 0.92, None, "original"),
        ]
    )
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    brand = ProposalBox(
        bbox_xyxy=(10, 10, 80, 28),
        confidence=0.95,
        source="rapidocr_ppocrv5",
        sources=("rapidocr_ppocrv5",),
        variant_name="rapidocr_gated:brand",
        polygon_xy=None,
    )
    date_box = ProposalBox(
        bbox_xyxy=(10, 40, 95, 58),
        confidence=0.82,
        source="rapidocr_ppocrv5",
        sources=("rapidocr_ppocrv5",),
        variant_name="rapidocr_gated:date",
        polygon_xy=None,
    )

    filtered = pipeline._ocr_prefilter_rapidocr_gated_proposals(
        image,
        [brand, date_box],
        today=date(2026, 1, 1),
    )

    assert [proposal.bbox_xyxy for proposal in filtered] == [date_box.bbox_xyxy]
    assert len(pipeline.recognizer.calls) == 2
    metadata = pipeline._rapidocr_gated_probe_metadata[date_box.bbox_xyxy]
    assert metadata["probe_text"] == "03-05-26"
    assert metadata["probe_score"] >= 3.0


def test_mobile_rapidocr_probe_metadata_bonus_is_bounded_before_final_ranking() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate()
    pipeline._rapidocr_gated_probe_metadata[candidate.bbox_xyxy] = {
        "probe_text": "03-05-26",
        "probe_normalized_text": "03-05-26",
        "probe_confidence": 0.92,
        "probe_score": 3.25,
        "probe_orientation": "original",
        "probe_polygon_used": True,
    }

    pipeline._apply_rapidocr_probe_metadata_to_ranked([candidate])

    assert candidate.score_breakdown["rapidocr_probe_date_likeness_score"] <= 2.7
    assert candidate.total_score < 5.0


def test_mobile_groups_nearby_rapidocr_same_line_proposals() -> None:
    pipeline = _pipeline_for_unit_tests()
    proposals = [
        ProposalBox(
            bbox_xyxy=(10, 20, 50, 36),
            confidence=0.91,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="rapidocr_gated:left",
            polygon_xy=((10.0, 20.0), (50.0, 20.0), (50.0, 36.0), (10.0, 36.0)),
        ),
        ProposalBox(
            bbox_xyxy=(58, 21, 110, 37),
            confidence=0.89,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="rapidocr_gated:right",
            polygon_xy=((58.0, 21.0), (110.0, 21.0), (110.0, 37.0), (58.0, 37.0)),
        ),
    ]

    grouped = pipeline._group_rapidocr_same_line_proposals(proposals)

    assert len(grouped) == 1
    assert grouped[0].bbox_xyxy == (10, 20, 110, 37)


def test_mobile_groups_rapidocr_adjacent_subgroups() -> None:
    pipeline = _pipeline_for_unit_tests()
    proposals = [
        ProposalBox(
            bbox_xyxy=(10, 20, 50, 36),
            confidence=0.91,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="rapidocr_gated:left",
            polygon_xy=((10.0, 20.0), (50.0, 20.0), (50.0, 36.0), (10.0, 36.0)),
        ),
        ProposalBox(
            bbox_xyxy=(58, 21, 110, 37),
            confidence=0.89,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="rapidocr_gated:middle",
            polygon_xy=((58.0, 21.0), (110.0, 21.0), (110.0, 37.0), (58.0, 37.0)),
        ),
        ProposalBox(
            bbox_xyxy=(118, 22, 150, 38),
            confidence=0.87,
            source="rapidocr_ppocrv5",
            sources=("rapidocr_ppocrv5",),
            variant_name="rapidocr_gated:right",
            polygon_xy=((118.0, 22.0), (150.0, 22.0), (150.0, 38.0), (118.0, 38.0)),
        ),
    ]

    grouped = pipeline._group_rapidocr_same_line_proposals(proposals)
    grouped_bboxes = {proposal.bbox_xyxy for proposal in grouped}

    assert (10, 20, 110, 37) in grouped_bboxes
    assert (58, 21, 150, 38) in grouped_bboxes
    assert (10, 20, 150, 38) in grouped_bboxes


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


def test_mobile_expands_partial_day_month_crop_when_sibling_full_date_exists(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((120, 260, 3), 255, dtype=np.uint8)
    candidate = _mobile_ranked_candidate(
        candidate_type="group",
        bbox_xyxy=(10, 20, 140, 80),
        member_bboxes=[(10, 20, 140, 40), (10, 60, 110, 80)],
    )

    upper_crop = NormalizedTextLineCrop(
        image=np.full((20, 130, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 20, 140, 40),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[20, 130, 3],
        normalized_crop_shape=[20, 130, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    clipped_lower_crop = NormalizedTextLineCrop(
        image=np.full((20, 100, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 60, 110, 80),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[20, 100, 3],
        normalized_crop_shape=[20, 100, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )

    monkeypatch.setattr(pipeline, "_normalized_crops_for_candidate", lambda _image, _candidate: [upper_crop, clipped_lower_crop])
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, allowed_names=MOBILE_RECOGNITION_VARIANTS: [
            ImageVariant("original_color_tight", crop, purpose="recognition")
        ],
    )
    monkeypatch.setattr(pipeline, "_recognition_result_needs_180_fallback", lambda _rec, _parsed: False)

    def recognize(image: np.ndarray, *, variant_name: str, orientation: str, **_kwargs) -> _RecognitionOutput:
        width = int(image.shape[1])
        if width >= 165:
            return _RecognitionOutput("14.06.26", "14.06.26", 0.96, None, orientation)
        if width == 100:
            return _RecognitionOutput("14.06", "14.06", 0.97, None, orientation)
        return _RecognitionOutput("17122025", "17122025", 0.92, None, orientation)

    monkeypatch.setattr(pipeline, "_recognize_variant", recognize)

    evaluated = pipeline._evaluate_candidate(image, candidate, today=date(2026, 5, 27))
    selected = pipeline._select_date_evidence(evaluated.date_evidence)

    assert selected.parsed.parsed_date == date(2026, 6, 14)
    assert selected.crop_policy == "partial_day_month_right_expand"
    assert selected.normalized_crop is not None
    assert selected.normalized_crop.bbox_xyxy[2] > clipped_lower_crop.bbox_xyxy[2]


def test_mobile_expands_partial_day_month_crop_across_neighbor_candidates(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    hardcase_recognizer = object()
    pipeline.hardcase_recognizer_rescue_enabled = True
    pipeline.hardcase_recognizer = hardcase_recognizer
    image = np.full((120, 260, 3), 255, dtype=np.uint8)
    today = date(2026, 5, 27)
    full_candidate = _mobile_ranked_candidate(
        candidate_id="cand_full",
        candidate_type="group",
        bbox_xyxy=(10, 20, 170, 80),
        member_bboxes=[(10, 20, 170, 40), (10, 60, 110, 80)],
    )
    partial_candidate = _mobile_ranked_candidate(
        candidate_id="cand_partial",
        candidate_type="single",
        bbox_xyxy=(10, 60, 110, 80),
        member_bboxes=[(10, 60, 110, 80)],
    )

    full_crop = NormalizedTextLineCrop(
        image=np.full((20, 160, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 20, 170, 40),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[20, 160, 3],
        normalized_crop_shape=[20, 160, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    partial_crop = NormalizedTextLineCrop(
        image=np.full((20, 100, 3), 255, dtype=np.uint8),
        bbox_xyxy=(10, 60, 110, 80),
        polygon_used=False,
        crop_transform_used="bbox",
        selected_orientation="original",
        original_crop_shape=[20, 100, 3],
        normalized_crop_shape=[20, 100, 3],
        orientation_candidates_tried=["original"],
        selected_transform_reason="test",
    )
    full_evidence = _date_evidence_for_score(
        text="17122025",
        parsed_date=date(2025, 12, 17),
        candidate=full_candidate,
    )
    full_evidence.normalized_crop = full_crop
    full_eval = _EvaluatedCandidate(
        candidate=full_candidate,
        recognition=full_evidence.recognition,
        parsed=full_evidence.parsed,
        parse_inputs_count=1,
        recognition_variant="original_color_tight",
        normalized_crop=full_crop,
        date_evidence=[full_evidence],
    )
    partial_eval = _EvaluatedCandidate(
        candidate=partial_candidate,
        recognition=_RecognitionOutput("14.06", "14.06", 0.97, None, "original"),
        parsed=ParsedDateData(
            parsed_date=None,
            date_format_detected=None,
            confidence=0.0,
            candidates=["14.06"],
            reason="missing year",
        ),
        parse_inputs_count=1,
        recognition_variant="stamp_blackhat",
        normalized_crop=partial_crop,
    )

    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda crop, allowed_names=MOBILE_RECOGNITION_VARIANTS: [
            ImageVariant("stamp_blackhat", crop, purpose="recognition")
        ],
    )

    def recognize(image: np.ndarray, *, variant_name: str, orientation: str, **kwargs) -> _RecognitionOutput:
        if int(image.shape[1]) > 150:
            if kwargs.get("recognizer") is hardcase_recognizer:
                return _RecognitionOutput("14.06.26", "14.06.26", 0.96, None, orientation)
            return _RecognitionOutput("14.06.21", "14.06.21", 0.94, None, orientation)
        if kwargs.get("recognizer") is hardcase_recognizer:
            return _RecognitionOutput("14.06.26", "14.06.26", 0.96, None, orientation)
        return _RecognitionOutput("14.06", "14.06", 0.97, None, orientation)

    monkeypatch.setattr(pipeline, "_recognize_variant", recognize)

    assert pipeline._is_day_month_year_context_evidence(full_evidence)
    assert pipeline._is_high_confidence_partial_day_month_eval(partial_eval)
    assert pipeline._partial_day_month_has_nearby_full_date_context(partial_crop.bbox_xyxy, full_evidence)
    evidence = pipeline._collect_cross_candidate_partial_day_month_right_expansion_evidence(
        image,
        [full_eval, partial_eval],
        today=today,
    )
    assert evidence
    assert pipeline._is_strong_partial_day_month_right_expansion_evidence(evidence[0])
    partial_eval.date_evidence.extend(evidence)
    _parseable, accepted = pipeline._accepted_parseable_items([full_eval, partial_eval])
    selected = pipeline._select_date_evidence(accepted)

    assert selected.parsed.parsed_date == date(2026, 6, 14)
    assert selected.crop_policy == "partial_day_month_right_expand"
    assert selected.normalized_crop is not None
    assert selected.normalized_crop.bbox_xyxy[2] > partial_crop.bbox_xyxy[2]


def test_mobile_default_recognition_variants_keep_raw_crop_as_original() -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((24, 80, 3), 127, dtype=np.uint8)

    variants = pipeline._recognition_variants_for_crop(crop)

    assert tuple(variant.name for variant in variants) == MOBILE_RECOGNITION_VARIANTS
    assert variants[0].name == "original_color_tight"
    assert np.array_equal(variants[0].image, crop)
    assert len({variant.name for variant in variants}) == len(MOBILE_RECOGNITION_VARIANTS)


def test_mobile_recognition_variants_can_be_limited_to_original_only() -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((24, 80, 3), 127, dtype=np.uint8)

    variants = pipeline._recognition_variants_for_crop(crop, allowed_names=("original_color_tight",))

    assert [variant.name for variant in variants] == ["original_color_tight"]
    assert np.array_equal(variants[0].image, crop)


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


def test_mobile_dot_matrix_rescue_proposes_lid_date_and_rejects_bottom_table_noise() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.asarray(Image.open("test64/5a6ad67b-3cf9-4c5c-9148-1fb606d5654a.jpg").convert("RGB"))

    proposals, reason = pipeline._detect_dot_matrix_rescue_proposals(image)

    assert reason is None
    assert proposals
    truth_bbox = (1021, 1778, 1777, 2118)
    assert any(pipeline._bbox_overlap(proposal.bbox_xyxy, truth_bbox)[1] >= 0.45 for proposal in proposals)
    assert any(
        proposal.polygon_xy is not None and pipeline._bbox_overlap(proposal.bbox_xyxy, truth_bbox)[1] >= 0.45
        for proposal in proposals
    )
    assert not any(proposal.bbox_xyxy[1] >= 2500 for proposal in proposals)


def test_mobile_dot_matrix_rescue_proposes_vertical_right_side_date() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = pipeline.decode_image(Path("test64/IMG_0898.JPG").read_bytes())
    assert image is not None

    proposals, reason = pipeline._detect_dot_matrix_rescue_proposals(image)

    assert reason is None
    truth_bbox = (2208, 834, 2331, 1259)
    assert any(pipeline._bbox_overlap(proposal.bbox_xyxy, truth_bbox)[1] >= 0.45 for proposal in proposals)


def test_mobile_geometry_shortlist_keeps_dot_matrix_polygon_candidate() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidates = []
    for index in range(pipeline.expiry_filter_geometry_top_n + 5):
        candidate = _mobile_ranked_candidate(candidate_id=f"regular_{index}", bbox_xyxy=(10, 10 + index * 5, 90, 35 + index * 5))
        candidate.geometry_score = 5.0 - index * 0.4
        candidates.append(candidate)
    dot_candidate = _mobile_ranked_candidate(candidate_id="vertical_dot", bbox_xyxy=(2197, 785, 2427, 1344))
    dot_candidate.geometry_score = -1.0
    dot_candidate.detector_sources = ("dot_matrix_rescue",)
    dot_candidate.detector_variant = "dot_matrix_rescue"
    dot_candidate.polygon_xy = ((2197.0, 785.0), (2427.0, 785.0), (2427.0, 1344.0), (2197.0, 1344.0))
    candidates.append(dot_candidate)
    candidates.sort(key=lambda item: item.geometry_score, reverse=True)

    selected = pipeline._apply_geometry_shortlist(candidates)

    assert dot_candidate in selected
    assert dot_candidate.selected_geometry


def test_mobile_accepts_oriented_dot_matrix_trailing_date_with_noisy_prefix() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(bbox_xyxy=(666, 1773, 1731, 2076))
    candidate.detector_confidence = 0.594
    candidate.detector_sources = ("dot_matrix_rescue",)
    candidate.detector_variant = "dot_matrix_rescue"
    candidate.polygon_xy = ((682.0, 1775.0), (1737.0, 1886.0), (1717.0, 2075.0), (663.0, 1964.0))
    evidence = _date_evidence_for_score(
        text="314-30-04-26",
        parsed_date=date(2026, 4, 30),
        parser_confidence=0.93,
        ocr_confidence=0.765,
        candidate=candidate,
    )
    evidence.parsed.date_format_detected = "TRAILING_DD/MM/YY"
    evidence.crop_policy = "dot_matrix_rescue"
    evidence.recognition_variant = "hardcase:original_padded"

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == [evidence]


def test_mobile_rejects_unoriented_dot_matrix_trailing_date_without_support() -> None:
    pipeline = _pipeline_for_unit_tests()
    candidate = _mobile_ranked_candidate(bbox_xyxy=(666, 1773, 1731, 2076))
    candidate.detector_confidence = 0.594
    candidate.detector_sources = ("dot_matrix_rescue",)
    candidate.detector_variant = "dot_matrix_rescue"
    evidence = _date_evidence_for_score(
        text="314-30-04-26",
        parsed_date=date(2026, 4, 30),
        parser_confidence=0.93,
        ocr_confidence=0.765,
        candidate=candidate,
    )
    evidence.parsed.date_format_detected = "TRAILING_DD/MM/YY"
    evidence.crop_policy = "dot_matrix_rescue"

    accepted = pipeline._accepted_date_evidence_items([evidence])

    assert accepted == []


def test_mobile_run_uses_dot_matrix_rescue_after_only_weak_edge_date(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    weak_yolo = _YoloCandidate(
        polygon_xy=[[0.0, 80.0], [90.0, 80.0], [90.0, 100.0], [0.0, 100.0]],
        bbox_xyxy=(0, 80, 90, 100),
        confidence=0.174,
    )
    weak_candidate = _mobile_ranked_candidate(
        candidate_id="weak_edge",
        bbox_xyxy=(0, 80, 90, 100),
        member_bboxes=[(0, 80, 90, 100)],
    )
    weak_candidate.detector_confidence = 0.174
    weak_candidate.geometry_features["edge_proximity"] = 0.0
    weak_evidence = _date_evidence_for_score(
        text="02092022",
        parsed_date=date(2022, 9, 2),
        parser_confidence=0.9,
        ocr_confidence=0.59,
        candidate=weak_candidate,
    )
    weak_eval = pipeline._evaluated_from_date_evidence(weak_evidence, date_evidence=[weak_evidence])

    dot_proposal = ProposalBox(
        bbox_xyxy=(20, 20, 140, 48),
        confidence=0.65,
        source="dot_matrix_rescue",
        sources=("dot_matrix_rescue",),
        variant_name="dot_matrix_rescue",
        polygon_xy=None,
    )
    dot_candidate = _mobile_ranked_candidate(
        candidate_id="dot_matrix",
        bbox_xyxy=dot_proposal.bbox_xyxy,
        member_bboxes=[dot_proposal.bbox_xyxy],
    )
    dot_candidate.detector_confidence = 0.65
    dot_candidate.detector_sources = ("dot_matrix_rescue",)
    dot_candidate.detector_variant = "dot_matrix_rescue"
    dot_candidate.final_crop_policy = "dot_matrix_rescue"
    dot_evidence = _date_evidence_for_score(
        text="30-04-26",
        parsed_date=date(2026, 4, 30),
        parser_confidence=0.95,
        ocr_confidence=0.91,
        candidate=dot_candidate,
    )
    dot_evidence.crop_policy = "dot_matrix_rescue"
    dot_eval = pipeline._evaluated_from_date_evidence(dot_evidence, date_evidence=[dot_evidence])

    monkeypatch.setattr(pipeline, "_detect_candidates", lambda _image: ([weak_yolo], None))
    monkeypatch.setattr(pipeline, "_build_ranked_candidates", lambda _image, candidates: [weak_candidate] if candidates else [])
    monkeypatch.setattr(pipeline, "_detect_variant_rescue_candidates", lambda _image: ([], "no variant rescue"))
    monkeypatch.setattr(pipeline, "_detect_dot_matrix_rescue_proposals", lambda _image: ([dot_proposal], None))
    monkeypatch.setattr(pipeline, "_build_ranked_candidates_from_proposals", lambda _image, proposals: [dot_candidate])
    monkeypatch.setattr(
        pipeline,
        "_evaluate_architecture_candidates",
        lambda _image, ranked, *, today: [dot_eval] if ranked == [dot_candidate] else [weak_eval],
    )

    result = pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert result.status == "parsed_success"
    assert result.detected_expiry_date == date(2026, 4, 30)
    assert result.final_crop_policy == "dot_matrix_rescue"


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
    assert settings.mobile_rapidocr_primary_mode == "gated"
    assert settings.mobile_rapidocr_max_candidates == 64
    assert settings.mobile_rapidocr_primary_max_rois_per_scan == 12
    assert settings.mobile_rapidocr_primary_max_boxes_accepted == 40
    assert settings.mobile_rapidocr_gated_max_boxes_accepted == 36
    assert settings.mobile_rapidocr_gated_probe_top_k == 8
    assert settings.mobile_rapidocr_gated_final_top_k == 4
    assert settings.mobile_rapidocr_gated_prefilter_probe_top_k == 36
    assert settings.mobile_rapidocr_gated_min_probe_digits == 4
    assert settings.mobile_rapidocr_gated_min_date_likeness_score == 1.0
    assert settings.mobile_product_cropper_rescue_enabled is False
    assert settings.mobile_product_cropper_model_path == Path("models/yolo20n/yolo26s/yolo26s-best.pt")
    assert settings.mobile_product_cropper_max_candidates == 3
    assert settings.mobile_legacy_wide_group_rescue_enabled is False
    assert settings.mobile_strict_evidence_acceptance_enabled is False
    assert settings.mobile_rapidocr_role_constraint_enabled is False
    assert settings.mobile_production_anchor_sibling_enabled is True
    assert settings.mobile_paired_crop_evidence_enabled is False
    assert settings.mobile_local_group_wide_crop_enabled is True


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


def test_mobile_rotation_detector_rescue_maps_90_degree_boxes_to_original() -> None:
    candidate = _YoloCandidate(
        polygon_xy=[[10.0, 20.0], [30.0, 20.0], [30.0, 60.0], [10.0, 60.0]],
        bbox_xyxy=(10, 20, 30, 60),
        confidence=0.8,
        variant_name="rot90_cw",
    )

    mapped_cw = MobileExpiryPipeline._map_rotated_yolo_candidate_to_original(
        candidate,
        orientation="rot90_cw",
        original_shape=(100, 200, 3),
    )
    mapped_ccw = MobileExpiryPipeline._map_rotated_yolo_candidate_to_original(
        candidate,
        orientation="rot90_ccw",
        original_shape=(100, 200, 3),
    )

    assert mapped_cw is not None
    assert mapped_cw.bbox_xyxy == (20, 70, 60, 90)
    assert mapped_cw.sources == ("yolo26s_obb", "rotation_rescue")
    assert mapped_cw.variant_name == "rot90_cw"
    assert mapped_ccw is not None
    assert mapped_ccw.bbox_xyxy == (140, 10, 180, 30)
    assert mapped_ccw.variant_name == "rot90_ccw"


def test_mobile_rotation_detector_rescue_runs_after_no_initial_boxes(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    calls: list[tuple[int, int, str]] = []

    def fake_detect(image: np.ndarray, *, variant_name: str):
        calls.append((int(image.shape[0]), int(image.shape[1]), variant_name))
        if variant_name == "yolo26s_obb":
            return [], "no expiry region detected"
        if variant_name == "rot90_cw":
            return [
                _YoloCandidate(
                    polygon_xy=[[10.0, 5.0], [40.0, 5.0], [40.0, 25.0], [10.0, 25.0]],
                    bbox_xyxy=(10, 5, 40, 25),
                    confidence=0.91,
                    variant_name=variant_name,
                )
            ], None
        return [], "no rotated box"

    monkeypatch.setattr(pipeline, "_detect_candidates_for_variant", fake_detect)
    candidates, reason = pipeline._detect_candidates(np.zeros((50, 80, 3), dtype=np.uint8))
    assert candidates == []
    assert reason == "no expiry region detected"

    rescue, rescue_reason = pipeline._detect_full_image_rotation_rescue_candidates(np.zeros((50, 80, 3), dtype=np.uint8))

    assert rescue_reason is None
    assert len(rescue) == 1
    assert rescue[0].bbox_xyxy == (5, 10, 25, 40)
    assert rescue[0].variant_name == "rot90_cw"
    assert calls == [
        (50, 80, "yolo26s_obb"),
        (80, 50, "rot90_cw"),
        (80, 50, "rot90_ccw"),
    ]


def test_mobile_rotation_detector_rescue_merge_uses_rotation_cap(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    pipeline.rapidocr_primary_enabled = True
    pipeline.rapidocr_primary_mode = "always"
    pipeline.full_image_rotation_detector_rescue_max_candidates = 3
    seen_counts: list[int] = []
    rotation_candidates = [
        _YoloCandidate(
            polygon_xy=[[float(index), 0.0], [float(index + 1), 0.0], [float(index + 1), 1.0], [float(index), 1.0]],
            bbox_xyxy=(index * 10, 0, index * 10 + 5, 5),
            confidence=0.9 - index * 0.01,
            variant_name="rot90_ccw",
            sources=("yolo26s_obb", "rotation_rescue"),
        )
        for index in range(5)
    ]

    monkeypatch.setattr(pipeline, "_detect_candidates", lambda image: ([], "no expiry region detected"))
    monkeypatch.setattr(
        pipeline,
        "_detect_full_image_rotation_rescue_candidates",
        lambda image: (rotation_candidates, None),
    )

    def fake_evaluate_primary(image, yolo_candidates, *, today):
        seen_counts.append(len(yolo_candidates))
        return [], None

    monkeypatch.setattr(pipeline, "_evaluate_rapidocr_primary_flow", fake_evaluate_primary)

    pipeline.run(_image_bytes(), today=date(2026, 1, 1))

    assert seen_counts == [3]
