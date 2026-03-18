from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(slots=True)
class DetectionResult:
    detected: bool
    boxes: list[BoundingBox]
    best_box: BoundingBox | None
    confidence: float | None
    reason: str | None


@dataclass(slots=True)
class OCRResultData:
    raw_text: str
    normalized_text: str
    confidence: float | None
    engine_name: str
    runtime_device: str
    reason: str | None


@dataclass(slots=True)
class ParsedDateData:
    parsed_date: date | None
    date_format_detected: str | None
    confidence: float
    candidates: list[str]
    reason: str


@dataclass(slots=True)
class DecisionResult:
    final_status: str
    expiry_classification: str
    days_remaining: int | None
    alert_required: bool
    needs_review: bool
    reason: str
