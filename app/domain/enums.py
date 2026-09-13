from __future__ import annotations

from enum import StrEnum


class ScanStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class FinalResultStatus(StrEnum):
    PARSED_SUCCESS = "parsed_success"
    PARSED_WITH_LOW_CONFIDENCE = "parsed_with_low_confidence"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    DETECTOR_FAILED = "detector_failed"
    OCR_FAILED = "ocr_failed"
    PARSER_FAILED = "parser_failed"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ExpiryClassification(StrEnum):
    SAFE = "safe"
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class AlertType(StrEnum):
    EXPIRING_SOON = "expiring_soon"
    EXPIRED = "expired"
    MANUAL_REVIEW = "manual_review"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class ReviewVerdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    SKIP = "skip"
