from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ExpiryClassification, FinalResultStatus, ReviewVerdict, ScanStatus


class ScanCreateMetadata(BaseModel):
    name: str | None = None
    brand: str | None = None
    category: str | None = None
    extra: dict[str, Any] | None = None


class ScanCreateResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus


class QueueTestImagesResponse(BaseModel):
    queued: int
    skipped: int
    total_candidates: int
    scan_ids: list[UUID]
    errors: list[str] = []


class Test64ExpectedDateItem(BaseModel):
    filename: str
    image_url: str
    expected_day: int | None = None
    expected_month: int | None = None
    expected_year: int | None = None
    updated_at: datetime | None = None


class Test64ExpectedDateListResponse(BaseModel):
    items: list[Test64ExpectedDateItem]
    total_candidates: int
    labeled_count: int


class Test64ExpectedDateUpsertRequest(BaseModel):
    filename: str
    day: int | None = None
    month: int | None = None
    year: int | None = None


class Test64ExpectedDateUpsertResponse(BaseModel):
    item: Test64ExpectedDateItem


class Test64TruthBBoxItem(BaseModel):
    filename: str
    image_url: str
    expected_day: int | None = None
    expected_month: int | None = None
    expected_year: int | None = None
    image_width: int
    image_height: int
    true_bbox_xyxy: list[float] | None = None
    true_polygon_xy: list[list[float]] | None = None
    annotation_rotation_degrees: float | None = None
    updated_at: datetime | None = None


class Test64TruthBBoxManifestResponse(BaseModel):
    version: int
    coordinate_space: str
    audit_set: list[str]
    items: dict[str, Test64TruthBBoxItem]
    annotated_count: int
    total_count: int
    manifest_path: str


class Test64TruthBBoxUpsertRequest(BaseModel):
    true_bbox_xyxy: list[float] | None = None
    true_polygon_xy: list[list[float]] | None = None
    annotation_rotation_degrees: float | None = None


class Test64TruthBBoxUpsertResponse(BaseModel):
    item: Test64TruthBBoxItem
    annotated_count: int
    total_count: int
    manifest_path: str


class MobileExpiryScanResponse(BaseModel):
    id: UUID
    status: str
    expiry_date: date | None = None
    detected_expiry_date: date | None = None
    corrected_expiry_date: date | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    recognition_confidence: float | None = None
    detector_confidence: float | None = None
    reason: str | None = None
    created_at: datetime


class MobileExpiryCorrectionRequest(BaseModel):
    corrected_expiry_date: date
    reason: str | None = None


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


class ScanReviewPayload(BaseModel):
    scan_id: UUID | None = None
    verdict: ReviewVerdict | None = None
    accepted_for_training: bool = False
    final_text: str | None = None
    final_parsed_date: date | None = None
    bbox_xyxy: list[float] | None = None
    bbox_source: str = "original_image"
    reviewer_id: str = "console"
    notes: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ScanReviewUpsertRequest(BaseModel):
    verdict: ReviewVerdict
    accepted_for_training: bool | None = None
    final_text: str | None = None
    final_parsed_date: date | None = None
    bbox_xyxy: list[float] | None = None
    bbox_source: str = "original_image"
    reviewer_id: str = "console"
    notes: str | None = None


class ScanReviewResponse(BaseModel):
    scan_id: UUID
    review: ScanReviewPayload | None = None


class ScanGetResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus
    result: ScanResultPayload | None = None


class ScanListItemResponse(BaseModel):
    scan_id: UUID
    created_at: datetime
    processed_at: datetime | None = None
    status: ScanStatus
    original_filename: str | None = None
    final_status: FinalResultStatus | None = None
    detected: bool | None = None
    detector_confidence: float | None = None
    raw_text: str | None = None
    parsed_date: date | None = None
    parse_confidence: float | None = None
    expiry_classification: ExpiryClassification | None = None
    days_remaining: int | None = None
    needs_review: bool | None = None
    reason: str | None = None
    ocr_engine: str | None = None
    ocr_runtime_device: str | None = None
    image_url: str
    roi_url: str | None = None
    review_verdict: ReviewVerdict | None = None
    accepted_for_training: bool = False
    review: ScanReviewPayload | None = None
    result: ScanResultPayload | None = None


class ScanListResponse(BaseModel):
    items: list[ScanListItemResponse]


class OneShotAnalyzeResponse(BaseModel):
    scan_id: UUID
    status: ScanStatus
    result: ScanResultPayload
    stage_timings_ms: dict[str, int] | None = None


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


class TrainingDataExportRequest(BaseModel):
    output_dir: str | None = None
    include_detector: bool = True
    include_recognition: bool = True


class TrainingDataExportResponse(BaseModel):
    output_dir: str
    accepted_reviews: int
    exported_detector_labels: int
    exported_recognition_crops: int
    skipped: int
    skipped_by_reason: dict[str, int]


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
