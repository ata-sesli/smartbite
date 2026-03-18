from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ExpiryClassification, FinalResultStatus, ScanStatus


class ScanCreateMetadata(BaseModel):
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    extra: dict[str, Any] | None = None


class ScanCreateResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus


class ScanResultPayload(BaseModel):
    final_status: FinalResultStatus | None = None
    detected: bool | None = None
    detector_confidence: float | None = None
    raw_text: str | None = None
    parsed_date: date | None = None
    date_format_detected: str | None = None
    parse_confidence: float | None = None
    expiry_classification: ExpiryClassification | None = None
    days_remaining: int | None = None
    alert_required: bool | None = None
    needs_review: bool | None = None
    reason: str | None = None
    ocr_engine: str | None = None
    ocr_runtime_device: str | None = None


class ScanGetResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus
    result: ScanResultPayload | None = None


class ManualCorrectionRequest(BaseModel):
    parsed_date: date | None = None
    reason: str = Field(min_length=3)


class ManualCorrectionResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus
    final_status: FinalResultStatus
    expiry_classification: ExpiryClassification
    days_remaining: int | None


class ExpiryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    scan_id: UUID
    user_id: str
    product_id: UUID
    parsed_date: date | None
    expiry_classification: ExpiryClassification
    days_remaining: int | None
    needs_review: bool


class ExpiryListResponse(BaseModel):
    items: list[ExpiryItemResponse]


class AlertsProcessResponse(BaseModel):
    created: int
    skipped_existing: int


class HealthResponse(BaseModel):
    status: str
    app: str
    timestamp: datetime


class AdminMetricsResponse(BaseModel):
    total_scans: int
    detector_failures: int
    ocr_failures: int
    parser_failures: int
    manual_review_count: int
    manual_review_rate: float
    average_processing_latency_ms: float
    alert_events: int
