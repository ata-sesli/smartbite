from __future__ import annotations

from datetime import date
import json

import numpy as np

from app.ai.decision import ExpiryDecisionEngine
from app.ai.crop_normalization import NormalizedTextLineCrop
from app.ai.ocr import RecognitionData, TextDetectionBox
from app.ai.preprocess import ImageVariant
from app.ai.parser import ExpiryDateParser
from app.ai.pipeline import EvaluationCandidate, ExpiryPipeline
from app.ai.types import BoundingBox, DetectionResult, OCRResultData, ParsedDateData
from app.domain.enums import FinalResultStatus


class FakeDetector:
    def __init__(self, result: DetectionResult) -> None:
        self._result = result

    def detect(self, image: np.ndarray) -> DetectionResult:
        _ = image
        return self._result


class FakePreprocessor:
    def process_variants(self, image: np.ndarray) -> list[np.ndarray]:
        _ = image
        return []

    def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
        return [ImageVariant(name="raw", image=image, purpose="detector")]

    def recognition_variants(self, crop: np.ndarray) -> list[ImageVariant]:
        return [ImageVariant(name="original_padded", image=crop, purpose="recognition")]


class VariantPreprocessor(FakePreprocessor):
    def recognition_variants(self, crop: np.ndarray) -> list[ImageVariant]:
        clahe_marker = np.zeros_like(crop)
        if clahe_marker.size:
            clahe_marker[:] = 40
        return [
            ImageVariant(name="original_padded", image=crop, purpose="recognition"),
            ImageVariant(name="clahe_gray", image=clahe_marker, purpose="recognition"),
        ]


class FakeOCRRouter:
    active_engine_name = "parseq_small"
    parseq_runtime_device = "cpu"

    def __init__(self) -> None:
        self.parseq_calls = 0
        self.probe_calls = 0

    def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
        h, w = image.shape[:2]
        if float(image.mean()) < 0.5:
            return [], "empty"
        return [TextDetectionBox(x1=0, y1=0, x2=w, y2=h, confidence=0.9)], None

    def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
        self.probe_calls += 1
        score = float(crop.mean())
        if score >= 19:
            text = "EXP 29/03/26"
        elif score >= 9:
            text = "LOT 123"
        elif score >= 1:
            text = "11"
        else:
            text = ""

        return RecognitionData(
            raw_text=text,
            normalized_text=text,
            confidence=0.9 if text else 0.0,
            reason=None if text else "empty",
        )

    def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
        self.parseq_calls += 1
        score = float(crop.mean())
        if score >= 19:
            text = "EXP 29/03/26"
        elif score >= 9:
            text = "LOT 123"
        elif score >= 1:
            text = "11"
        else:
            text = ""
        return RecognitionData(
            raw_text=text,
            normalized_text=text,
            confidence=0.9 if text else 0.0,
            reason=None if text else "empty",
        )


class GatedRescueOCR(FakeOCRRouter):
    text_detector_mode = "ensemble"

    def __init__(
        self,
        *,
        ppocr_boxes: list[TextDetectionBox] | None = None,
        craft_boxes: list[TextDetectionBox] | None = None,
        craft_reason: str | None = None,
    ) -> None:
        super().__init__()
        self.ppocr_boxes = ppocr_boxes if ppocr_boxes is not None else []
        self.craft_boxes = craft_boxes if craft_boxes is not None else []
        self.craft_reason = craft_reason
        self.ppocr_calls: list[str | None] = []
        self.craft_calls: list[tuple[str | None, int | None, float | None]] = []

    def detect_text_boxes(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        _ = image
        self.ppocr_calls.append(variant_name)
        return self.ppocr_boxes, None

    def detect_craft_text_boxes(
        self,
        image: np.ndarray,
        *,
        variant_name: str | None = None,
        canvas_size: int | None = None,
        mag_ratio: float | None = None,
    ) -> tuple[list[TextDetectionBox], str | None]:
        _ = image
        self.craft_calls.append((variant_name, canvas_size, mag_ratio))
        return self.craft_boxes, self.craft_reason

    def merge_text_boxes(
        self,
        ppocr_boxes: list[TextDetectionBox],
        craft_boxes: list[TextDetectionBox],
    ) -> list[TextDetectionBox]:
        return [*ppocr_boxes, *craft_boxes]

    def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
        if crop.size and float(crop.mean()) >= 70:
            text = "EXP 29/03/26"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.92, reason=None)
        return RecognitionData(raw_text="", normalized_text="", confidence=0.0, reason="no text")

    def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
        self.parseq_calls += 1
        return self.recognize_probe(crop)


def _image_with_regions() -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:40, 10:40] = 10  # detector box 1 -> unparseable text
    image[50:80, 50:80] = 20  # detector box 2 -> parseable date
    return image


def _role_selection_pipeline() -> ExpiryPipeline:
    return ExpiryPipeline(
        detector=FakeDetector(DetectionResult(False, [], None, None, None)),
        preprocessor=FakePreprocessor(),
        ocr_router=FakeOCRRouter(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=2,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
    )


def _evaluation_candidate(
    *,
    candidate_id: str,
    text: str,
    parsed_date: date,
    parse_confidence: float,
    candidate_total_score: float = 1.0,
    ocr_confidence: float = 0.8,
) -> EvaluationCandidate:
    return EvaluationCandidate(
        source="unit",
        variant="raw",
        variant_key="unit:raw",
        candidate_id=candidate_id,
        roi=np.zeros((8, 8, 3), dtype=np.uint8),
        detector_confidence=0.8,
        candidate_bbox=(0, 0, 8, 8),
        candidate_total_score=candidate_total_score,
        ocr=OCRResultData(
            raw_text=text,
            normalized_text=text,
            confidence=ocr_confidence,
            engine_name="svtrv2",
            runtime_device="cpu",
            reason=None,
        ),
        parsed=ParsedDateData(
            parsed_date=parsed_date,
            date_format_detected="DD/MM/YYYY",
            confidence=parse_confidence,
            candidates=[parsed_date.isoformat()],
            reason="selected_DD/MM/YYYY",
        ),
        parse_inputs_count=1,
        recognition_variant="original_padded",
    )


def test_pipeline_parseable_score_prefers_neutral_later_date_over_production_date() -> None:
    pipeline = _role_selection_pipeline()
    production = _evaluation_candidate(
        candidate_id="prod",
        text="UR.T. 12/12/2025",
        parsed_date=date(2025, 12, 12),
        parse_confidence=0.99,
        candidate_total_score=9.0,
        ocr_confidence=0.99,
    )
    expiry = _evaluation_candidate(
        candidate_id="exp",
        text="12/12/2027",
        parsed_date=date(2027, 12, 12),
        parse_confidence=0.80,
        candidate_total_score=1.0,
        ocr_confidence=0.70,
    )

    selected = max([production, expiry], key=pipeline._score_parseable_candidate)

    assert selected is expiry


def test_pipeline_parseable_score_prefers_expiry_keyword_over_higher_confidence_production_date() -> None:
    pipeline = _role_selection_pipeline()
    production = _evaluation_candidate(
        candidate_id="prod",
        text="MFG 12/12/2025",
        parsed_date=date(2025, 12, 12),
        parse_confidence=0.99,
        candidate_total_score=9.0,
        ocr_confidence=0.99,
    )
    expiry = _evaluation_candidate(
        candidate_id="exp",
        text="EXP 12/12/2027",
        parsed_date=date(2027, 12, 12),
        parse_confidence=0.70,
        candidate_total_score=1.0,
        ocr_confidence=0.60,
    )

    selected = max([production, expiry], key=pipeline._score_parseable_candidate)

    assert selected is expiry


def test_pipeline_parseable_score_prefers_later_neutral_date_over_confidence() -> None:
    pipeline = _role_selection_pipeline()
    older = _evaluation_candidate(
        candidate_id="older",
        text="12/12/2025",
        parsed_date=date(2025, 12, 12),
        parse_confidence=0.99,
        candidate_total_score=9.0,
        ocr_confidence=0.99,
    )
    later = _evaluation_candidate(
        candidate_id="later",
        text="12/12/2027",
        parsed_date=date(2027, 12, 12),
        parse_confidence=0.70,
        candidate_total_score=1.0,
        ocr_confidence=0.60,
    )

    selected = max([older, later], key=pipeline._score_parseable_candidate)

    assert selected is later


def test_pipeline_keeps_single_production_like_parseable_candidate() -> None:
    pipeline = _role_selection_pipeline()
    production = _evaluation_candidate(
        candidate_id="prod",
        text="UR.T. 12/12/2025",
        parsed_date=date(2025, 12, 12),
        parse_confidence=0.99,
    )

    selected = max([production], key=pipeline._score_parseable_candidate)

    assert selected is production


def test_pipeline_selects_best_parse_from_non_top1_roi() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[
            BoundingBox(x1=10, y1=10, x2=40, y2=40, confidence=0.95),
            BoundingBox(x1=50, y1=50, x2=80, y2=80, confidence=0.84),
        ],
        best_box=None,
        confidence=0.95,
        reason=None,
    )
    ocr = FakeOCRRouter()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=2,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        expiry_max_rois_per_scan=2,
    )

    import cv2

    encoded = cv2.imencode(".jpg", _image_with_regions())[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert output.final_status == FinalResultStatus.PARSED_SUCCESS
    assert "selected_source=detector_box_2" in output.reason
    assert ocr.parseq_calls > 0


def test_pipeline_uses_full_image_candidate_when_detector_misses() -> None:
    detection = DetectionResult(
        detected=False,
        boxes=[],
        best_box=None,
        confidence=None,
        reason="no expiry region detected",
    )

    class FullImageOCR(FakeOCRRouter):
        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            text = "EXP 07/08/26"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return self.recognize_probe(crop)

    ocr = FullImageOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=2,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
    )

    import cv2

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 8, 7)
    assert "selected_source=full_image" in output.reason
    assert ocr.parseq_calls == 1


def test_pipeline_returns_manual_review_with_best_ocr_when_no_parseable_candidate() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=10, y1=10, x2=40, y2=40, confidence=0.91)],
        best_box=None,
        confidence=0.91,
        reason=None,
    )

    class NoisyOCR(FakeOCRRouter):
        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            text = "LOT 123" if float(crop.mean()) >= 9 else "LOT"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.55, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return self.recognize_probe(crop)

    ocr = NoisyOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=True,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
    )

    import cv2

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[10:40, 10:40] = 10
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.final_status == FinalResultStatus.MANUAL_REVIEW_REQUIRED
    assert output.raw_text in {"LOT 123", "LOT"}
    assert "manual_review_no_parse" in output.reason
    assert ocr.parseq_calls > 0


def test_parseq_runs_only_for_final_top_k_candidates() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ManyBoxOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
            _ = image
            boxes = [
                TextDetectionBox(2, 2, 20, 14, 0.9),
                TextDetectionBox(24, 2, 42, 14, 0.85),
                TextDetectionBox(46, 2, 64, 14, 0.8),
                TextDetectionBox(68, 2, 86, 14, 0.75),
                TextDetectionBox(90, 2, 108, 14, 0.7),
            ]
            return boxes, None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            width = crop.shape[1]
            text = "EXP 29/03/26" if width >= 18 else "LOT"
            confidence = 0.95 if "EXP" in text else 0.2
            return RecognitionData(raw_text=text, normalized_text=text, confidence=confidence, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = ManyBoxOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=5,
        expiry_filter_final_top_k=2,
        expiry_global_final_top_k=2,
    )

    import cv2

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert ocr.parseq_calls == 2


def test_global_final_top_k_limits_parseq_across_detector_variants() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=160, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ThreeVariantPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            return [
                ImageVariant(name="raw", image=image, purpose="detector"),
                ImageVariant(name="luma_clahe", image=image.copy(), purpose="detector"),
                ImageVariant(name="unsharp", image=image.copy(), purpose="detector"),
            ]

    class ManyVariantOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(4, 4, 34, 18, 0.9),
                TextDetectionBox(40, 4, 70, 18, 0.86),
                TextDetectionBox(76, 4, 106, 18, 0.82),
                TextDetectionBox(112, 4, 150, 18, 0.78),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = ManyVariantOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=ThreeVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=4,
        expiry_global_final_top_k=3,
        expiry_global_max_per_roi=3,
        expiry_global_max_per_variant=3,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :] = 80
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert ocr.parseq_calls == 3
    assert output.debug_candidates_json_bytes is not None
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    assert payload["summary"]["selected_candidates_count"] == 3
    assert sum(1 for v in payload["variants"] for c in v["candidates"] if c["selected_final"]) == 3


def test_global_final_selection_dedupes_duplicate_candidates_across_variants() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class DuplicateVariantPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            return [
                ImageVariant(name="raw", image=image, purpose="detector"),
                ImageVariant(name="luma_clahe", image=image.copy(), purpose="detector"),
            ]

    class DuplicateOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(10, 10, 88, 32, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = DuplicateOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=DuplicateVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=4,
        expiry_global_final_top_k=6,
        expiry_detector_variants=("raw", "luma_clahe"),
        expiry_max_detector_variants=2,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = 90
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert ocr.parseq_calls == 1
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    assert payload["summary"]["global_probe_candidate_count"] == 2
    assert payload["summary"]["global_probe_deduped_candidate_count"] == 1


def test_global_final_selection_respects_per_roi_diversity_cap() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[
            BoundingBox(x1=0, y1=0, x2=100, y2=80, confidence=0.9),
            BoundingBox(x1=100, y1=0, x2=200, y2=80, confidence=0.88),
        ],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class DenseOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(4, 4, 34, 18, 0.95),
                TextDetectionBox(40, 4, 70, 18, 0.9),
                TextDetectionBox(4, 28, 34, 42, 0.85),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            return RecognitionData(raw_text="LOT 123", normalized_text="LOT 123", confidence=0.6, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="LOT 123", normalized_text="LOT 123", confidence=0.6, reason=None)

    ocr = DenseOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=2,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=4,
        expiry_global_final_top_k=6,
        expiry_global_max_per_roi=1,
        expiry_global_max_per_variant=6,
        expiry_max_rois_per_scan=2,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 200, 3), dtype=np.uint8)
    image[:, :] = 50
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert ocr.parseq_calls == 4
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected_sources = [
        variant["source"]
        for variant in payload["variants"]
        for candidate in variant["candidates"]
        if candidate["selected_final"]
    ]
    assert sorted(selected_sources) == ["detector_box_1", "detector_box_2"]


def test_global_probe_top_k_limits_mobile_probe_across_variants() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=180, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ThreeVariantPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            return [
                ImageVariant(name="raw", image=image, purpose="detector"),
                ImageVariant(name="luma_clahe", image=image.copy(), purpose="detector"),
                ImageVariant(name="unsharp", image=image.copy(), purpose="detector"),
            ]

    class ManyProbeOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(4, 4, 30, 18, 0.95),
                TextDetectionBox(36, 4, 62, 18, 0.9),
                TextDetectionBox(68, 4, 94, 18, 0.85),
                TextDetectionBox(100, 4, 126, 18, 0.8),
                TextDetectionBox(132, 4, 170, 18, 0.75),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            self.probe_calls += 1
            _ = crop
            return RecognitionData(raw_text="LOT 123", normalized_text="LOT 123", confidence=0.7, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            _ = crop
            return RecognitionData(raw_text="LOT 123", normalized_text="LOT 123", confidence=0.7, reason=None)

    ocr = ManyProbeOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=ThreeVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_global_probe_top_k=4,
        expiry_global_final_top_k=3,
        expiry_detector_variants=("raw", "luma_clahe", "unsharp"),
        expiry_max_detector_variants=3,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((120, 180, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert ocr.probe_calls == 4
    assert output.debug_summary is not None
    perf = output.debug_summary["performance"]
    assert perf["candidates_sent_to_mobile_probe"] == 4
    assert perf["mobile_probe_call_count"] == 4
    assert perf["candidates_before_global_final_cap"] == 4


def test_debug_summary_includes_detailed_timing_and_call_counts() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class SimpleOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(10, 10, 88, 32, 0.9)], None

    ocr = SimpleOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=4,
        expiry_global_final_top_k=2,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.debug_summary is not None
    perf = output.debug_summary["performance"]
    expected_keys = {
        "total_runtime_ms",
        "yolo_runtime_ms",
        "roi_count",
        "detector_variants_by_roi",
        "ppocr_detection_call_count",
        "ppocr_detection_total_ms",
        "ppocr_detection_ms_per_call",
        "craft_call_count",
        "craft_total_ms",
        "craft_ms_per_call",
        "detected_boxes_before_dedupe",
        "detected_boxes_after_dedupe",
        "candidates_before_geometry_cap",
        "candidates_sent_to_mobile_probe",
        "mobile_probe_call_count",
        "mobile_probe_total_ms",
        "mobile_probe_ms_per_candidate",
        "candidates_before_global_final_cap",
        "parseq_call_count",
        "parseq_total_ms",
        "parser_total_ms",
    }
    assert expected_keys <= set(perf)
    assert perf["roi_count"] == 1
    assert perf["ppocr_detection_call_count"] == 1


def test_probe_uses_configured_recognition_variants_and_parseq_uses_probe_winner() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ManyRecognitionVariants(FakePreprocessor):
        def recognition_variants(self, crop: np.ndarray) -> list[ImageVariant]:
            variants = []
            for index, name in enumerate(
                [
                    "original_padded",
                    "gray_upscaled",
                    "clahe_gray",
                    "unsharp_gray",
                    "adaptive_binary",
                    "adaptive_binary_inverted",
                    "stamp_blackhat",
                ],
                start=1,
            ):
                image = crop.copy()
                image[:, :] = index
                variants.append(ImageVariant(name=name, image=image, purpose="recognition"))
            return variants

    class VariantChoosingOCR(FakeOCRRouter):
        def __init__(self) -> None:
            super().__init__()
            self.probe_means: list[int] = []
            self.parseq_means: list[int] = []

        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(10, 10, 88, 32, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            self.probe_calls += 1
            marker = int(crop.mean())
            self.probe_means.append(marker)
            if marker == 5:
                return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.95, reason=None)
            return RecognitionData(raw_text="LOT", normalized_text="LOT", confidence=0.2, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            marker = int(crop.mean())
            self.parseq_means.append(marker)
            if marker == 5:
                return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.95, reason=None)
            return RecognitionData(raw_text="", normalized_text="", confidence=0.0, reason="wrong variant")

    ocr = VariantChoosingOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=ManyRecognitionVariants(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=1,
        expiry_global_final_top_k=1,
        expiry_probe_variants=("original_padded", "clahe_gray", "adaptive_binary"),
        expiry_parseq_variant_policy="best_probe",
        expiry_parseq_max_variants_per_candidate=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert ocr.probe_means == [1, 3, 5]
    assert ocr.parseq_means == [5]
    assert output.debug_summary is not None
    perf = output.debug_summary["performance"]
    assert perf["recognition_variants_generated"] == 7
    assert perf["probe_variants_per_candidate"] == 3.0
    assert perf["parseq_variants_per_candidate"] == 1.0
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for v in payload["variants"] for c in v["candidates"] if c["selected_final"])
    assert selected["recognition_variant"] == "adaptive_binary"


def test_group_candidates_are_capped_without_all_pairs() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=240, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ManyBoxOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(4 + i * 10, 10, 12 + i * 10, 22, 0.8)
                for i in range(20)
            ], None

    ocr = ManyBoxOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=4,
        max_group_candidates_per_roi=10,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((120, 240, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    assert payload["variants"][0]["candidate_count"] <= 30
    assert output.debug_summary["performance"]["candidates_before_geometry_cap"] <= 30


def test_debug_metadata_includes_grouped_candidates() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class GroupingOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
            _ = image
            return [
                TextDetectionBox(10, 10, 40, 24, 0.9, source="ppocrv5_server", sources=("ppocrv5_server", "craft")),
                TextDetectionBox(42, 10, 80, 24, 0.9, source="craft", sources=("craft",)),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = GroupingOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=2,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = 40
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.debug_candidates_json_bytes is not None
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    candidates = payload["variants"][0]["candidates"]
    assert any(c["candidate_type"] == "group" for c in candidates)
    assert payload["variants"][0]["detector_mode"] == "unknown"
    assert payload["variants"][0]["merged_box_count"] == 2
    assert payload["variants"][0]["detector_counts"]["ppocrv5_server"] == 1
    assert payload["variants"][0]["detector_counts"]["craft"] == 2
    assert all("detector_sources" in c for c in candidates)


def test_detector_variant_can_find_date_when_raw_detector_misses() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class DetectorVariantPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            enhanced = image.copy()
            enhanced[:, :, 0] = 77
            return [
                ImageVariant(name="raw", image=image, purpose="detector"),
                ImageVariant(name="luma_clahe", image=enhanced, purpose="detector"),
            ]

    class VariantAwareOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
            if int(image[0, 0, 0]) != 77:
                return [], "raw detector missed"
            return [TextDetectionBox(10, 10, 88, 32, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = VariantAwareOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=DetectorVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        expiry_detector_variants=("raw", "luma_clahe"),
        expiry_max_detector_variants=2,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert "selected_variant=luma_clahe" in output.reason


def test_recognition_variant_can_parse_when_original_crop_fails() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class RecognitionVariantPreprocessor(FakePreprocessor):
        def recognition_variants(self, crop: np.ndarray) -> list[ImageVariant]:
            enhanced = crop.copy()
            enhanced[:, :, 1] = 201
            return [
                ImageVariant(name="original_padded", image=crop, purpose="recognition"),
                ImageVariant(name="adaptive_binary", image=enhanced, purpose="recognition"),
            ]

    class RecognitionVariantOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
            _ = image
            return [TextDetectionBox(10, 10, 88, 32, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            if crop.ndim == 3 and int(crop[0, 0, 1]) == 201:
                return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)
            return RecognitionData(raw_text="", normalized_text="", confidence=0.0, reason="original unreadable")

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            if crop.ndim == 3 and int(crop[0, 0, 1]) == 201:
                return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)
            return RecognitionData(raw_text="", normalized_text="", confidence=0.0, reason="original unreadable")

    ocr = RecognitionVariantOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=RecognitionVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert output.debug_candidates_json_bytes is not None
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = payload["variants"][0]["candidates"][0]
    assert selected["recognition_variant"] == "adaptive_binary"


def test_upscaled_detector_coordinates_crop_original_roi() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=60, y2=40, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class UpscaledPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            upscaled = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
            return [ImageVariant(name="raw_upscaled", image=upscaled, scale_x=2.0, scale_y=2.0, purpose="detector")]

    class CoordinateOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray) -> tuple[list[TextDetectionBox], str | None]:
            _ = image
            return [TextDetectionBox(20, 16, 80, 40, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            if crop.shape[:2] == (12, 30) and int(crop.mean()) == 88:
                return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)
            return RecognitionData(raw_text="", normalized_text="", confidence=0.0, reason=f"bad crop {crop.shape} {crop.mean()}")

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return self.recognize_probe(crop)

    ocr = CoordinateOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=UpscaledPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        expiry_detector_variants=("raw_upscaled",),
        expiry_max_detector_variants=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
    )

    import cv2

    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[8:20, 10:40] = 88
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)


def test_craft_only_detection_can_drive_parse_success() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class CraftOnlyOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(10, 10, 88, 32, 0.7, source="craft", sources=("craft",))], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = CraftOnlyOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_trigger_min_ppocr_boxes=1,
        craft_trigger_min_best_score=0.1,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert output.debug_candidates_json_bytes is not None
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    assert payload["variants"][0]["detector_counts"]["craft"] == 1
    assert payload["variants"][0]["candidates"][0]["detector_sources"] == ["craft"]


def test_multi_source_detector_box_gets_geometry_preference() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=140, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class MultiSourceOCR(FakeOCRRouter):
        def detect_text_boxes(self, image: np.ndarray, *, variant_name: str | None = None) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(10, 10, 88, 32, 0.75, source="ensemble", sources=("ppocrv5_server", "craft")),
                TextDetectionBox(92, 10, 138, 32, 0.99, source="ppocrv5_server", sources=("ppocrv5_server",)),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return RecognitionData(raw_text="EXP 29/03/26", normalized_text="EXP 29/03/26", confidence=0.9, reason=None)

    ocr = MultiSourceOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_trigger_min_ppocr_boxes=1,
        craft_trigger_min_best_score=0.1,
    )

    import cv2

    image = np.zeros((80, 140, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.debug_candidates_json_bytes is not None
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for c in payload["variants"][0]["candidates"] if c["selected_final"])
    assert selected["detector_sources"] == ["ppocrv5_server", "craft"]


def test_craft_rescue_skipped_when_ppocr_probe_has_date() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )
    ocr = GatedRescueOCR(
        ppocr_boxes=[TextDetectionBox(10, 10, 88, 32, 0.9, source="ppocrv5_server", sources=("ppocrv5_server",))],
        craft_boxes=[TextDetectionBox(10, 40, 88, 62, 0.7, source="craft", sources=("craft",))],
    )
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_rescue_enabled=False,
        craft_trigger_min_ppocr_boxes=1,
        craft_trigger_min_best_score=0.1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:32, 10:88] = 88
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert ocr.craft_calls == []
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    variant = payload["variants"][0]
    assert variant["craft_triggered"] is False
    assert variant["craft_skipped_reason"] == "craft_rescue_disabled"


def test_craft_rescue_runs_when_ppocr_box_count_low_and_supplies_date() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )
    ocr = GatedRescueOCR(
        ppocr_boxes=[],
        craft_boxes=[TextDetectionBox(10, 10, 88, 32, 0.7, source="craft", sources=("craft",))],
    )
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_rescue_enabled=True,
        craft_rescue_variant_name="raw",
        craft_rescue_canvas_size=640,
        craft_rescue_mag_ratio=1.0,
        craft_max_rois_per_scan=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:32, 10:88] = 88
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2026, 3, 29)
    assert ocr.craft_calls == [("raw", 640, 1.0)]
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    variant = payload["variants"][0]
    assert variant["craft_triggered"] is True
    assert "ppocr_box_count_below_threshold" in variant["craft_trigger_reason"]
    assert variant["craft_box_count"] == 1
    assert variant["craft_accepted_box_count"] == 1
    assert variant["detector_counts"]["craft"] == 1


def test_craft_rescue_respects_roi_budget() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[
            BoundingBox(x1=0, y1=0, x2=80, y2=60, confidence=0.9),
            BoundingBox(x1=80, y1=0, x2=160, y2=60, confidence=0.88),
        ],
        best_box=None,
        confidence=0.9,
        reason=None,
    )
    ocr = GatedRescueOCR(ppocr_boxes=[], craft_boxes=[])
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=2,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_max_rois_per_scan=1,
        expiry_max_rois_per_scan=2,
    )

    import cv2

    image = np.zeros((60, 160, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert len(ocr.craft_calls) == 1
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    skipped = [v["craft_skipped_reason"] for v in payload["variants"]]
    assert "roi_budget_exhausted" in skipped


def test_craft_rescue_runs_only_for_raw_variant_by_default() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=120, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class TwoVariantPreprocessor(FakePreprocessor):
        def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
            enhanced = image.copy()
            enhanced[:, :, 0] = 44
            return [
                ImageVariant(name="raw", image=image, purpose="detector"),
                ImageVariant(name="luma_clahe", image=enhanced, purpose="detector"),
            ]

    ocr = GatedRescueOCR(ppocr_boxes=[], craft_boxes=[])
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=TwoVariantPreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_max_rois_per_scan=2,
        expiry_detector_variants=("raw", "luma_clahe"),
        expiry_max_detector_variants=2,
    )

    import cv2

    image = np.zeros((80, 120, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert ocr.craft_calls == [("raw", 640, 1.0)]
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    reasons_by_variant = {v["variant"]: v["craft_skipped_reason"] for v in payload["variants"]}
    assert reasons_by_variant["luma_clahe"] == "not_rescue_variant"


def test_craft_rescue_caps_accepted_boxes() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=160, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )
    craft_boxes = [
        TextDetectionBox(i % 120, i // 2, (i % 120) + 12, (i // 2) + 8, None, source="craft", sources=("craft",))
        for i in range(50)
    ]
    ocr = GatedRescueOCR(ppocr_boxes=[], craft_boxes=craft_boxes)
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_filter_geometry_top_n=12,
        expiry_filter_final_top_k=1,
        save_debug_candidates=True,
        craft_max_boxes_accepted=40,
    )

    import cv2

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    assert payload["variants"][0]["craft_box_count"] == 50
    assert payload["variants"][0]["craft_accepted_box_count"] == 40


def test_expiry_keyword_date_outranks_brand_and_production_text() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=220, y2=170, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class EvidenceOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(10, 10, 210, 70, 0.95),
                TextDetectionBox(20, 95, 150, 118, 0.75),
                TextDetectionBox(20, 130, 150, 153, 0.75),
            ], None

        def _text_for_crop(self, crop: np.ndarray) -> str:
            marker = int(round(float(crop.mean())))
            if marker >= 85:
                return "T.E.T.T: 12/12/2027"
            if marker >= 70:
                return "U.R.T: 12/12/2025"
            if marker >= 40:
                return "COMPAGNE"
            return ""

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            self.probe_calls += 1
            text = self._text_for_crop(crop)
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.92 if text else 0.0, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            text = self._text_for_crop(crop)
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.94 if text else 0.0, reason=None)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=EvidenceOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=6,
        expiry_global_final_top_k=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((170, 220, 3), dtype=np.uint8)
    image[10:70, 10:210] = 50
    image[95:118, 20:150] = 75
    image[130:153, 20:150] = 90
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2027, 12, 12)
    assert "T.E.T.T" in (output.raw_text or "")
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for v in payload["variants"] for c in v["candidates"] if c["selected_final"])
    assert selected["score_breakdown"]["expiry_keyword_score"] > 0
    assert selected["force_included_reason"] in {"expiry_keyword_date", "line_expiry_keyword_date"}


def test_date_only_does_not_outrank_expiry_keyword_context() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=240, y2=80, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class DateContextOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(10, 10, 210, 55, 0.95),
                TextDetectionBox(20, 78, 125, 100, 0.7),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            marker = int(round(float(crop.mean())))
            text = "12/12/2025" if marker < 80 else "TETT 12/12/2027"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return self.recognize_probe(crop)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=DateContextOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=4,
        expiry_global_final_top_k=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((120, 220, 3), dtype=np.uint8)
    image[10:55, 10:210] = 60
    image[78:100, 20:125] = 90
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2027, 12, 12)
    assert "TETT" in (output.raw_text or "")


def test_adjacent_context_probe_boosts_date_only_candidate_without_expanding_final_crop() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=220, y2=120, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class ContextOCR(FakeOCRRouter):
        active_engine_name = "svtrv2"
        final_recognizer_model_name = "smartbite_svtrv2_expdate_rec"
        context_recognizer_model_name = "ch_SVTRv2_rec"

        def __init__(self) -> None:
            super().__init__()
            self.context_calls = 0
            self.context_means: list[int] = []
            self.final_shapes: list[tuple[int, int]] = []

        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(10, 20, 52, 40, 0.95),
                TextDetectionBox(60, 20, 170, 40, 0.95),
                TextDetectionBox(190, 20, 232, 40, 0.95),
            ], None

        def recognize_context(self, crop: np.ndarray) -> RecognitionData:
            self.context_calls += 1
            marker = int(round(float(crop.mean())))
            self.context_means.append(marker)
            if marker == 30:
                text = "TETT"
            else:
                text = "NET"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.9, reason=None)

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            self.probe_calls += 1
            marker = int(round(float(crop.mean())))
            if marker >= 85:
                text = "13/02/2031"
            elif marker >= 75:
                text = "13/02/2025"
            else:
                text = ""
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.9 if text else 0.0, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            self.final_shapes.append(crop.shape[:2])
            marker = int(round(float(crop.mean())))
            if marker >= 85:
                text = "13/02/2031"
            elif marker >= 75:
                text = "13/02/2025"
            else:
                text = "TETT"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.95, reason=None)

    ocr = ContextOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=ocr,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=6,
        expiry_global_final_top_k=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        context_probe_enabled=True,
        context_max_boxes_per_candidate=1,
        context_max_candidates_per_scan=6,
        max_group_candidates_per_roi=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((80, 240, 3), dtype=np.uint8)
    image[20:40, 10:52] = 30
    image[20:40, 60:170] = 90
    image[20:40, 190:232] = 80
    encoded = cv2.imencode(".png", image)[1].tobytes()

    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2031, 2, 13)
    assert 30 in ocr.context_means
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for v in payload["variants"] for c in v["candidates"] if c["selected_final"])
    assert selected["bbox_xyxy"] == [60, 20, 170, 40]
    assert selected["recognition_bbox_xyxy"] == [60, 20, 170, 40]
    assert selected["context_probe_texts"] == ["TETT"]
    assert selected["adjacent_expiry_keyword"] is True
    assert selected["keyword_relation"] == "left_same_line"
    assert selected["score_breakdown"]["adjacent_expiry_keyword_bonus"] > 0
    assert ocr.final_shapes and ocr.final_shapes[0] == (20, 110)


def test_multiline_production_expiry_block_creates_line_candidates_and_prefers_expiry_line() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=220, y2=100, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class MultilineOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [
                TextDetectionBox(10, 14, 55, 32, 0.9),
                TextDetectionBox(62, 14, 150, 32, 0.9),
                TextDetectionBox(10, 50, 60, 68, 0.9),
                TextDetectionBox(66, 50, 160, 68, 0.9),
            ], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            marker = int(round(float(crop.mean())))
            if marker >= 85:
                text = "T.E.T.T: 12/12/2027"
            elif marker >= 65:
                text = "U.R.T: 12/12/2025"
            else:
                text = "COMPANY"
            return RecognitionData(raw_text=text, normalized_text=text, confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            return self.recognize_probe(crop)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=FakePreprocessor(),
        ocr_router=MultilineOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=10,
        expiry_global_final_top_k=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((100, 220, 3), dtype=np.uint8)
    image[14:32, 10:150] = 70
    image[50:68, 10:160] = 90
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2027, 12, 12)
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    candidates = [c for v in payload["variants"] for c in v["candidates"]]
    line_candidates = [c for c in candidates if c["candidate_type"] == "line"]
    assert len(line_candidates) >= 2
    assert any(c["multiline_split_source_id"] for c in line_candidates)
    selected = next(c for c in candidates if c["selected_final"])
    assert selected["candidate_type"] == "line"
    assert selected["line_index_in_group"] == 1


def test_parseq_runs_selected_variant_then_one_original_color_fallback() -> None:
    detection = DetectionResult(
        detected=True,
        boxes=[BoundingBox(x1=0, y1=0, x2=140, y2=70, confidence=0.9)],
        best_box=None,
        confidence=0.9,
        reason=None,
    )

    class FallbackOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(15, 20, 120, 45, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            marker = int(round(float(crop.mean()))) if crop.size else 0
            if marker == 40:
                return RecognitionData(raw_text="COMPANY", normalized_text="COMPANY", confidence=0.99, reason=None)
            return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.88, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            marker = int(round(float(crop.mean()))) if crop.size else 0
            if marker == 40:
                return RecognitionData(raw_text="COMPANY", normalized_text="COMPANY", confidence=0.99, reason=None)
            return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.88, reason=None)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(detection),
        preprocessor=VariantPreprocessor(),
        ocr_router=FallbackOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        top_k=1,
        roi_padding_ratio=0.0,
        high_recall_mode=False,
        expiry_global_probe_top_k=2,
        expiry_global_final_top_k=1,
        expiry_probe_variants=("original_padded", "clahe_gray"),
        expiry_parseq_max_variants_per_candidate=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )

    import cv2

    image = np.zeros((70, 140, 3), dtype=np.uint8)
    image[20:45, 15:120] = 90
    encoded = cv2.imencode(".png", image)[1].tobytes()
    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2027, 12, 12)
    assert pipeline.ocr_router.parseq_calls == 2
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for v in payload["variants"] for c in v["candidates"] if c["selected_final"])
    assert selected["selected_recognition_variant"] == "clahe_gray"
    assert selected["recognition_variant"] == "original_color"
    assert selected["original_color_fallback_used"] is True


def _normalized_crop_for_orientation_test(image: np.ndarray) -> NormalizedTextLineCrop:
    return NormalizedTextLineCrop(
        image=image,
        bbox_xyxy=(0, 0, int(image.shape[1]), int(image.shape[0])),
        polygon_used=False,
        crop_transform_used="bbox_raw",
        selected_orientation="original",
        original_crop_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
        normalized_crop_shape=[int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
        orientation_candidates_tried=["original"],
        selected_transform_reason="horizontal_or_square",
    )


def test_horizontal_final_recognition_tries_rotate_180_only_after_original_fails() -> None:
    class UpsideDownOCR(FakeOCRRouter):
        def recognize_final(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            marker = int(crop[0, 0, 0]) if crop.size else 0
            if marker == 222:
                return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.95, reason=None)
            return RecognitionData(raw_text="COMPANY", normalized_text="COMPANY", confidence=0.95, reason=None)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(DetectionResult(detected=False, boxes=[], best_box=None, confidence=None, reason=None)),
        preprocessor=FakePreprocessor(),
        ocr_router=UpsideDownOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
    )
    image = np.zeros((20, 80, 3), dtype=np.uint8)
    image[19, 79] = 222

    parseq, parsed, _, _, _, _, _, selected_crop = pipeline._best_parseq_for_crop(
        [_normalized_crop_for_orientation_test(image)],
        today=date(2026, 1, 1),
        selected_variant_name="original_padded",
    )

    assert parseq.raw_text == "EXP 12/12/2027"
    assert parsed.parsed_date == date(2027, 12, 12)
    assert pipeline.ocr_router.parseq_calls == 2
    assert selected_crop.selected_orientation == "rotate_180"
    assert selected_crop.orientation_candidates_tried == ["original", "rotate_180"]


def test_horizontal_final_recognition_skips_rotate_180_when_original_parses() -> None:
    class UprightOCR(FakeOCRRouter):
        def recognize_final(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            _ = crop
            return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.95, reason=None)

    pipeline = ExpiryPipeline(
        detector=FakeDetector(DetectionResult(detected=False, boxes=[], best_box=None, confidence=None, reason=None)),
        preprocessor=FakePreprocessor(),
        ocr_router=UprightOCR(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
    )
    image = np.zeros((20, 80, 3), dtype=np.uint8)

    _, parsed, _, _, _, _, _, selected_crop = pipeline._best_parseq_for_crop(
        [_normalized_crop_for_orientation_test(image)],
        today=date(2026, 1, 1),
        selected_variant_name="original_padded",
    )

    assert parsed.parsed_date == date(2027, 12, 12)
    assert pipeline.ocr_router.parseq_calls == 1
    assert selected_crop.selected_orientation == "original"


def test_final_crop_padding_is_bounded_and_recorded() -> None:
    pipeline = ExpiryPipeline(
        detector=FakeDetector(DetectionResult(detected=False, boxes=[], best_box=None, confidence=None, reason=None)),
        preprocessor=FakePreprocessor(),
        ocr_router=FakeOCRRouter(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        expiry_final_crop_padding_ratio=0.10,
        expiry_final_crop_min_padding_px=3,
        expiry_final_crop_max_padding_px=8,
    )
    candidate = pipeline._candidate_from_bbox(
        bbox=(20, 20, 40, 30),
        member_indices=[],
        candidate_type="single",
        image_shape=(80, 100),
        boxes=[],
        neighbor_counts={},
    )
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    final_crop = pipeline._final_parseq_crop_for_candidate(candidate, image)

    assert final_crop.bbox == (17, 17, 43, 33)
    assert final_crop.padding_px == 3
    assert final_crop.policy == "single_tight"
    assert final_crop.image.shape[:2] == (16, 26)


def test_final_crop_prefers_expiry_line_from_multiline_group() -> None:
    pipeline = ExpiryPipeline(
        detector=FakeDetector(DetectionResult(detected=False, boxes=[], best_box=None, confidence=None, reason=None)),
        preprocessor=FakePreprocessor(),
        ocr_router=FakeOCRRouter(),
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
    )
    boxes = [
        TextDetectionBox(10, 12, 70, 30, 0.9),
        TextDetectionBox(12, 50, 88, 70, 0.9),
    ]
    candidate = pipeline._candidate_from_bbox(
        bbox=(10, 12, 88, 70),
        member_indices=[0, 1],
        candidate_type="group",
        image_shape=(100, 120),
        boxes=boxes,
        neighbor_counts={},
    )
    candidate.probe_text = "U.R.T: 12/12/2025\nT.E.T.T: 12/12/2027"
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    final_crop = pipeline._final_parseq_crop_for_candidate(candidate, image)

    assert final_crop.bbox == (12, 50, 88, 70)
    assert final_crop.policy == "line_from_multiline_group"


def test_vertical_final_crop_tries_bounded_rotations_and_records_selected_transform() -> None:
    class VerticalOCR(FakeOCRRouter):
        def detect_text_boxes(
            self,
            image: np.ndarray,
            *,
            variant_name: str | None = None,
        ) -> tuple[list[TextDetectionBox], str | None]:
            _ = image, variant_name
            return [TextDetectionBox(30, 10, 48, 100, 0.9)], None

        def recognize_probe(self, crop: np.ndarray) -> RecognitionData:
            _ = crop
            self.probe_calls += 1
            return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.9, reason=None)

        def recognize_parseq(self, crop: np.ndarray) -> RecognitionData:
            self.parseq_calls += 1
            if crop.shape[1] > crop.shape[0]:
                return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.92, reason=None)
            return RecognitionData(raw_text="T", normalized_text="T", confidence=0.3, reason=None)

    router = VerticalOCR()
    pipeline = ExpiryPipeline(
        detector=FakeDetector(DetectionResult(detected=False, boxes=[], best_box=None, confidence=None, reason=None)),
        preprocessor=FakePreprocessor(),
        ocr_router=router,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        decision=ExpiryDecisionEngine(alert_threshold_days=3),
        roi_padding_ratio=0.0,
        expiry_global_probe_top_k=1,
        expiry_global_final_top_k=1,
        expiry_final_crop_padding_ratio=0.0,
        expiry_final_crop_min_padding_px=0,
        expiry_final_crop_max_padding_px=0,
        save_debug_candidates=True,
    )
    import cv2

    image = np.full((120, 80, 3), 40, dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()

    output = pipeline.run(encoded, today=date(2026, 1, 1))

    assert output.parsed_date == date(2027, 12, 12)
    assert 2 <= router.parseq_calls <= 4
    payload = json.loads(output.debug_candidates_json_bytes.decode("utf-8"))
    selected = next(c for v in payload["variants"] for c in v["candidates"] if c["selected_final"])
    assert selected["orientation_candidates_tried"] == ["original", "rotate_90_cw", "rotate_90_ccw", "rotate_180"]
    assert selected["selected_orientation"] in {"rotate_90_cw", "rotate_90_ccw"}
    assert selected["crop_transform_used"] in {"rotate_90_cw", "rotate_90_ccw"}
    assert selected["original_crop_shape"] == [90, 18, 3]
    assert selected["normalized_crop_shape"][0] == 18
