from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter

import cv2
import numpy as np

from app.ai.decision import ExpiryDecisionEngine
from app.ai.detector import ExpiryRegionDetector
from app.ai.ocr import OCRRouter
from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ROIImagePreprocessor
from app.ai.types import DecisionResult, DetectionResult, OCRResultData, ParsedDateData
from app.domain.enums import ExpiryClassification, FinalResultStatus


@dataclass(slots=True)
class PipelineRunOutput:
    detected: bool
    detector_confidence: float | None
    raw_text: str | None
    normalized_text: str | None
    ocr_confidence: float | None
    ocr_engine: str | None
    ocr_runtime_device: str | None
    parsed_date: date | None
    date_format_detected: str | None
    parse_confidence: float | None
    expiry_classification: str
    days_remaining: int | None
    alert_required: bool
    needs_review: bool
    final_status: str
    reason: str
    stage_timings_ms: dict[str, int]
    roi_png_bytes: bytes | None


class ExpiryPipeline:
    def __init__(
        self,
        detector: ExpiryRegionDetector,
        preprocessor: ROIImagePreprocessor,
        ocr_router: OCRRouter,
        parser: ExpiryDateParser,
        decision: ExpiryDecisionEngine,
    ) -> None:
        self.detector = detector
        self.preprocessor = preprocessor
        self.ocr_router = ocr_router
        self.parser = parser
        self.decision = decision

    def run(self, image_bytes: bytes, *, today: date) -> PipelineRunOutput:
        timings: dict[str, int] = {}

        t0 = perf_counter()
        image = self._decode_image(image_bytes)
        timings["decode"] = int((perf_counter() - t0) * 1000)
        if image is None:
            return self._failed(
                final_status=FinalResultStatus.DETECTOR_FAILED,
                reason="image unreadable",
                timings=timings,
            )

        t1 = perf_counter()
        detection: DetectionResult = self.detector.detect(image)
        timings["detect"] = int((perf_counter() - t1) * 1000)
        if not detection.detected or detection.best_box is None:
            fallback = self._run_full_image_fallback(image=image, today=today, timings=timings, reason_prefix="detector_fallback")
            if fallback is not None:
                return fallback
            return self._failed(
                final_status=FinalResultStatus.DETECTOR_FAILED,
                reason=detection.reason or "no expiry region detected",
                timings=timings,
                detector_confidence=detection.confidence,
            )

        roi = self._crop_roi(image, detection)
        if roi is None:
            return self._failed(
                final_status=FinalResultStatus.DETECTOR_FAILED,
                reason="detected roi invalid",
                timings=timings,
                detector_confidence=detection.confidence,
            )

        t2 = perf_counter()
        preprocess_variants = self.preprocessor.process_variants(roi)
        timings["preprocess"] = int((perf_counter() - t2) * 1000)

        t3 = perf_counter()
        ocr_candidates: list[OCRResultData] = [self.ocr_router.run(roi)]
        # Preprocessed variants are important for dot-matrix and low-resolution text.
        for variant in preprocess_variants:
            ocr_candidates.append(self.ocr_router.run(variant))
        ocr_data = max(ocr_candidates, key=self._score_ocr_candidate)
        timings["ocr"] = int((perf_counter() - t3) * 1000)

        if ocr_data.reason and not ocr_data.raw_text:
            return self._failed(
                final_status=FinalResultStatus.OCR_FAILED,
                reason=ocr_data.reason,
                timings=timings,
                detected=True,
                detector_confidence=detection.confidence,
                ocr=ocr_data,
                roi=roi,
            )

        t4 = perf_counter()
        parsed: ParsedDateData = self.parser.parse(ocr_data.normalized_text, reference_date=today)
        timings["parse"] = int((perf_counter() - t4) * 1000)

        if parsed.parsed_date is None:
            fallback = self._run_full_image_fallback(
                image=image,
                today=today,
                timings=timings,
                reason_prefix="roi_parse_failed_fallback",
                detector_confidence=detection.confidence,
            )
            if fallback is not None:
                return fallback
            return self._failed(
                final_status=FinalResultStatus.PARSER_FAILED,
                reason=parsed.reason,
                timings=timings,
                detected=True,
                detector_confidence=detection.confidence,
                ocr=ocr_data,
                parsed=parsed,
                roi=roi,
            )

        t5 = perf_counter()
        decision: DecisionResult = self.decision.decide(parsed.parsed_date, today=today, parse_confidence=parsed.confidence)
        timings["decision"] = int((perf_counter() - t5) * 1000)

        roi_png = self._encode_png(roi)

        return PipelineRunOutput(
            detected=True,
            detector_confidence=detection.confidence,
            raw_text=ocr_data.raw_text,
            normalized_text=ocr_data.normalized_text,
            ocr_confidence=ocr_data.confidence,
            ocr_engine=ocr_data.engine_name,
            ocr_runtime_device=ocr_data.runtime_device,
            parsed_date=parsed.parsed_date,
            date_format_detected=parsed.date_format_detected,
            parse_confidence=parsed.confidence,
            expiry_classification=decision.expiry_classification,
            days_remaining=decision.days_remaining,
            alert_required=decision.alert_required,
            needs_review=decision.needs_review,
            final_status=decision.final_status,
            reason=f"{parsed.reason};{decision.reason}",
            stage_timings_ms=timings,
            roi_png_bytes=roi_png,
        )

    def _run_full_image_fallback(
        self,
        *,
        image: np.ndarray,
        today: date,
        timings: dict[str, int],
        reason_prefix: str,
        detector_confidence: float | None = None,
    ) -> PipelineRunOutput | None:
        t_ocr = perf_counter()
        ocr_data: OCRResultData = self.ocr_router.run(image)
        timings["ocr_full_image_fallback"] = int((perf_counter() - t_ocr) * 1000)
        if not ocr_data.raw_text:
            return None

        t_parse = perf_counter()
        parsed: ParsedDateData = self.parser.parse(ocr_data.normalized_text, reference_date=today)
        timings["parse_full_image_fallback"] = int((perf_counter() - t_parse) * 1000)
        if parsed.parsed_date is None:
            return None

        t_decision = perf_counter()
        decision: DecisionResult = self.decision.decide(parsed.parsed_date, today=today, parse_confidence=parsed.confidence)
        timings["decision"] = int((perf_counter() - t_decision) * 1000)

        return PipelineRunOutput(
            detected=detector_confidence is not None,
            detector_confidence=detector_confidence,
            raw_text=ocr_data.raw_text,
            normalized_text=ocr_data.normalized_text,
            ocr_confidence=ocr_data.confidence,
            ocr_engine=ocr_data.engine_name,
            ocr_runtime_device=ocr_data.runtime_device,
            parsed_date=parsed.parsed_date,
            date_format_detected=parsed.date_format_detected,
            parse_confidence=parsed.confidence,
            expiry_classification=decision.expiry_classification,
            days_remaining=decision.days_remaining,
            alert_required=decision.alert_required,
            needs_review=decision.needs_review,
            final_status=decision.final_status,
            reason=f"{reason_prefix};{parsed.reason};{decision.reason}",
            stage_timings_ms=timings,
            roi_png_bytes=self._encode_png(image),
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray | None:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        if array.size == 0:
            return None
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    @staticmethod
    def _score_ocr_candidate(ocr: OCRResultData) -> float:
        if not ocr.raw_text:
            return -1.0
        score = (ocr.confidence or 0.0) * 10.0
        text = ocr.raw_text
        if any(sep in text for sep in ("/", "-", ".")):
            score += 1.0
        if any(char.isdigit() for char in text):
            score += 0.5
        return score

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return None
        return encoded.tobytes()

    @staticmethod
    def _crop_roi(image: np.ndarray, detection: DetectionResult) -> np.ndarray | None:
        box = detection.best_box
        if box is None:
            return None

        h, w = image.shape[:2]
        x1 = max(0, min(box.x1, w - 1))
        y1 = max(0, min(box.y1, h - 1))
        x2 = max(0, min(box.x2, w))
        y2 = max(0, min(box.y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def _failed(
        self,
        *,
        final_status: str,
        reason: str,
        timings: dict[str, int],
        detected: bool = False,
        detector_confidence: float | None = None,
        ocr: OCRResultData | None = None,
        parsed: ParsedDateData | None = None,
        roi: np.ndarray | None = None,
    ) -> PipelineRunOutput:
        roi_png = self._encode_png(roi) if roi is not None else None

        return PipelineRunOutput(
            detected=detected,
            detector_confidence=detector_confidence,
            raw_text=ocr.raw_text if ocr else None,
            normalized_text=ocr.normalized_text if ocr else None,
            ocr_confidence=ocr.confidence if ocr else None,
            ocr_engine=ocr.engine_name if ocr else None,
            ocr_runtime_device=ocr.runtime_device if ocr else None,
            parsed_date=parsed.parsed_date if parsed else None,
            date_format_detected=parsed.date_format_detected if parsed else None,
            parse_confidence=parsed.confidence if parsed else None,
            expiry_classification=ExpiryClassification.MANUAL_REVIEW_REQUIRED,
            days_remaining=None,
            alert_required=True,
            needs_review=True,
            final_status=final_status,
            reason=reason,
            stage_timings_ms=timings,
            roi_png_bytes=roi_png,
        )
