from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Date, DateTime, Enum, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from sqlalchemy.types import Uuid

from app.domain.enums import AlertType, DeliveryStatus, ExpiryClassification, FinalResultStatus, ReviewVerdict, ScanStatus


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    qr_code: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    product_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("products.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    image_path: Mapped[str] = mapped_column(String(500))
    roi_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[ScanStatus] = mapped_column(Enum(ScanStatus, native_enum=False), index=True)
    final_result_status: Mapped[FinalResultStatus | None] = mapped_column(
        Enum(FinalResultStatus, native_enum=False), nullable=True, index=True
    )
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    product: Mapped[Product] = relationship()


class OCRResult(Base):
    __tablename__ = "ocr_results"
    __table_args__ = (UniqueConstraint("scan_id", name="uq_ocr_results_scan_id"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    engine_name: Mapped[str] = mapped_column(String(64))
    runtime_device: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProductTextResult(Base):
    __tablename__ = "product_text_results"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), nullable=True, index=True)
    product_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    engine_name: Mapped[str] = mapped_column(String(64))
    runtime_device: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_endpoint: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MobileExpiryScan(Base):
    __tablename__ = "mobile_expiry_scans"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    image_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    image_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    image_content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    corrected_expiry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    recognition_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    detection_polygon_json: Mapped[list[list[float]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Test64ExpectedDate(Base):
    __tablename__ = "test64_expected_dates"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    expected_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_month: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_year: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ParsedDateResult(Base):
    __tablename__ = "parsed_date_results"
    __table_args__ = (UniqueConstraint("scan_id", name="uq_parsed_date_results_scan_id"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    parsed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    date_format_detected: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parse_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    parser_reason: Mapped[str] = mapped_column(Text)
    candidate_dates_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExpiryStatus(Base):
    __tablename__ = "expiry_statuses"
    __table_args__ = (UniqueConstraint("scan_id", name="uq_expiry_statuses_scan_id"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    status: Mapped[ExpiryClassification] = mapped_column(Enum(ExpiryClassification, native_enum=False), index=True)
    days_remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    alert_required: Mapped[bool] = mapped_column(Boolean, default=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AlertEvent(Base):
    __tablename__ = "alert_events"
    __table_args__ = (UniqueConstraint("scan_id", "alert_type", name="uq_alert_events_scan_type"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType, native_enum=False), index=True)
    delivery_status: Mapped[DeliveryStatus] = mapped_column(
        Enum(DeliveryStatus, native_enum=False), default=DeliveryStatus.PENDING, index=True
    )
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    stage: Mapped[str] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[str] = mapped_column(Text)
    elapsed_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScanReview(Base):
    __tablename__ = "scan_reviews"
    __table_args__ = (UniqueConstraint("scan_id", name="uq_scan_reviews_scan_id"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    scan_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("scans.id"), index=True)
    verdict: Mapped[ReviewVerdict] = mapped_column(Enum(ReviewVerdict, native_enum=False), index=True)
    accepted_for_training: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    final_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_parsed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    bbox_xyxy: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    bbox_source: Mapped[str] = mapped_column(String(64), default="original_image")
    reviewer_id: Mapped[str] = mapped_column(String(128), default="console", index=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
