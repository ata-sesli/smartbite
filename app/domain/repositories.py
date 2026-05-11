from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import AlertType, DeliveryStatus, ExpiryClassification, FinalResultStatus, ReviewVerdict, ScanStatus
from app.domain.models import (
    AlertEvent,
    ExpiryStatus,
    MobileExpiryScan,
    OCRResult,
    ParsedDateResult,
    ProcessingLog,
    Product,
    ProductTextResult,
    Scan,
    ScanReview,
    Test64ExpectedDate,
)
from app.infra.clock import utcnow


@dataclass(slots=True)
class ScanAggregate:
    scan: Scan
    ocr: OCRResult | None
    parsed: ParsedDateResult | None
    expiry: ExpiryStatus | None


@dataclass(slots=True)
class ReviewExportItem:
    scan: Scan
    review: ScanReview
    ocr: OCRResult | None
    parsed: ParsedDateResult | None
    expiry: ExpiryStatus | None


@dataclass(slots=True)
class ExpiryListItem:
    scan_id: UUID
    user_id: str
    product_id: UUID
    parsed_date: date | None
    expiry_classification: ExpiryClassification
    days_remaining: int | None
    needs_review: bool


@dataclass(slots=True)
class ScanListItem:
    scan_id: UUID
    created_at: datetime
    processed_at: datetime | None
    status: ScanStatus
    original_filename: str | None
    final_status: FinalResultStatus | None
    detected: bool | None
    detector_confidence: float | None
    raw_text: str | None
    parsed_date: date | None
    date_format_detected: str | None
    parse_confidence: float | None
    expiry_classification: ExpiryClassification | None
    days_remaining: int | None
    alert_required: bool | None
    needs_review: bool | None
    reason: str | None
    ocr_engine: str | None
    ocr_runtime_device: str | None
    has_roi: bool
    review: ScanReview | None


@dataclass(slots=True)
class ProductTextCreateData:
    scan_id: UUID | None
    product_name: str | None
    brand: str | None
    full_text: str
    confidence: float | None
    engine_name: str
    runtime_device: str | None
    reason: str | None
    source_endpoint: str


@dataclass(slots=True)
class MobileExpiryScanCreateData:
    message: str | None
    metadata_json: dict | None
    image_blob: bytes
    image_filename: str
    image_content_type: str
    detected_expiry_date: date | None
    raw_text: str | None
    normalized_text: str | None
    recognition_confidence: float | None
    detector_confidence: float | None
    status: str
    reason: str | None
    detection_polygon_json: list[list[float]] | None


@dataclass(slots=True)
class Test64ExpectedDateUpsertData:
    filename: str
    expected_day: int | None
    expected_month: int
    expected_year: int


class DomainRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create_product(
        self,
        qr_code: str,
        *,
        name: str | None = None,
        brand: str | None = None,
        category: str | None = None,
    ) -> Product:
        product = await self.session.scalar(select(Product).where(Product.qr_code == qr_code))
        if product:
            return product
        product = Product(qr_code=qr_code, name=name, brand=brand, category=category)
        self.session.add(product)
        await self.session.flush()
        return product

    async def create_scan(
        self,
        *,
        product_id: UUID,
        user_id: str,
        image_path: str,
        metadata_json: dict | None,
    ) -> Scan:
        scan = Scan(
            product_id=product_id,
            user_id=user_id,
            image_path=image_path,
            status=ScanStatus.QUEUED,
            metadata_json=metadata_json,
        )
        self.session.add(scan)
        await self.session.flush()
        return scan

    async def get_scan(self, scan_id: UUID) -> Scan | None:
        return await self.session.get(Scan, scan_id)

    async def get_scan_aggregate(self, scan_id: UUID) -> ScanAggregate | None:
        scan = await self.get_scan(scan_id)
        if scan is None:
            return None
        ocr = await self.session.scalar(select(OCRResult).where(OCRResult.scan_id == scan_id))
        parsed = await self.session.scalar(select(ParsedDateResult).where(ParsedDateResult.scan_id == scan_id))
        expiry = await self.session.scalar(select(ExpiryStatus).where(ExpiryStatus.scan_id == scan_id))
        return ScanAggregate(scan=scan, ocr=ocr, parsed=parsed, expiry=expiry)

    async def get_scan_review(self, scan_id: UUID) -> ScanReview | None:
        return await self.session.scalar(select(ScanReview).where(ScanReview.scan_id == scan_id))

    async def upsert_scan_review(
        self,
        *,
        scan_id: UUID,
        verdict: ReviewVerdict,
        accepted_for_training: bool,
        final_text: str | None,
        final_parsed_date: date | None,
        bbox_xyxy: list[float] | None,
        bbox_source: str,
        reviewer_id: str,
        notes: str | None,
    ) -> ScanReview:
        existing = await self.get_scan_review(scan_id)
        if existing is None:
            existing = ScanReview(
                scan_id=scan_id,
                verdict=verdict,
                accepted_for_training=accepted_for_training,
                final_text=final_text,
                final_parsed_date=final_parsed_date,
                bbox_xyxy=bbox_xyxy,
                bbox_source=bbox_source,
                reviewer_id=reviewer_id,
                notes=notes,
            )
            self.session.add(existing)
        else:
            existing.verdict = verdict
            existing.accepted_for_training = accepted_for_training
            existing.final_text = final_text
            existing.final_parsed_date = final_parsed_date
            existing.bbox_xyxy = bbox_xyxy
            existing.bbox_source = bbox_source
            existing.reviewer_id = reviewer_id
            existing.notes = notes
            existing.updated_at = utcnow()
        await self.session.flush()
        return existing

    async def list_scans(
        self,
        *,
        limit: int,
        status: ScanStatus | None = None,
        final_status: FinalResultStatus | None = None,
        user_id: str | None = None,
    ) -> list[ScanListItem]:
        stmt = (
            select(
                Scan.id,
                Scan.created_at,
                Scan.processed_at,
                Scan.status,
                Scan.metadata_json,
                Scan.final_result_status,
                Scan.detected,
                Scan.detector_confidence,
                Scan.failure_reason,
                Scan.roi_path,
                OCRResult.raw_text,
                OCRResult.ocr_confidence,
                OCRResult.engine_name,
                OCRResult.runtime_device,
                ParsedDateResult.parsed_date,
                ParsedDateResult.date_format_detected,
                ParsedDateResult.parse_confidence,
                ExpiryStatus.status,
                ExpiryStatus.days_remaining,
                ExpiryStatus.alert_required,
                ExpiryStatus.needs_review,
                ExpiryStatus.reason,
                ScanReview.id,
            )
            .select_from(Scan)
            .join(OCRResult, OCRResult.scan_id == Scan.id, isouter=True)
            .join(ParsedDateResult, ParsedDateResult.scan_id == Scan.id, isouter=True)
            .join(ExpiryStatus, ExpiryStatus.scan_id == Scan.id, isouter=True)
            .join(ScanReview, ScanReview.scan_id == Scan.id, isouter=True)
            .order_by(Scan.created_at.desc(), Scan.id.desc())
            .limit(limit)
        )
        if status is not None:
            stmt = stmt.where(Scan.status == status)
        if final_status is not None:
            stmt = stmt.where(Scan.final_result_status == final_status)
        if user_id:
            stmt = stmt.where(Scan.user_id == user_id)

        rows = (await self.session.execute(stmt)).all()
        items: list[ScanListItem] = []
        for row in rows:
            review = await self.get_scan_review(row[0]) if row[22] is not None else None
            items.append(
                ScanListItem(
                    scan_id=row[0],
                    created_at=row[1],
                    processed_at=row[2],
                    status=row[3],
                    original_filename=(row[4] or {}).get("original_filename") if isinstance(row[4], dict) else None,
                    final_status=row[5],
                    detected=row[6],
                    detector_confidence=row[7],
                    reason=row[21] or row[8],
                    has_roi=bool(row[9]),
                    raw_text=row[10],
                    parsed_date=row[14],
                    date_format_detected=row[15],
                    parse_confidence=row[16],
                    expiry_classification=row[17],
                    days_remaining=row[18],
                    alert_required=row[19],
                    needs_review=row[20],
                    ocr_engine=row[12],
                    ocr_runtime_device=row[13],
                    review=review,
                )
            )
        return items

    async def accepted_reviews_for_export(self) -> list[ReviewExportItem]:
        rows = (
            await self.session.execute(
                select(Scan, ScanReview, OCRResult, ParsedDateResult, ExpiryStatus)
                .join(ScanReview, ScanReview.scan_id == Scan.id)
                .join(OCRResult, OCRResult.scan_id == Scan.id, isouter=True)
                .join(ParsedDateResult, ParsedDateResult.scan_id == Scan.id, isouter=True)
                .join(ExpiryStatus, ExpiryStatus.scan_id == Scan.id, isouter=True)
                .where(ScanReview.accepted_for_training.is_(True))
                .order_by(ScanReview.updated_at.desc(), ScanReview.created_at.desc())
            )
        ).all()
        return [ReviewExportItem(scan=row[0], review=row[1], ocr=row[2], parsed=row[3], expiry=row[4]) for row in rows]

    async def set_scan_processing(self, scan: Scan) -> None:
        scan.status = ScanStatus.PROCESSING
        await self.session.flush()

    async def finalize_scan(
        self,
        scan: Scan,
        *,
        final_status: FinalResultStatus,
        detected: bool | None,
        detector_confidence: float | None,
        failure_reason: str | None,
        roi_path: str | None = None,
    ) -> None:
        scan.status = ScanStatus.DONE
        scan.final_result_status = final_status
        scan.detected = detected
        scan.detector_confidence = detector_confidence
        scan.failure_reason = failure_reason
        scan.roi_path = roi_path
        scan.processed_at = utcnow()
        await self.session.flush()

    async def upsert_ocr_result(
        self,
        *,
        scan_id: UUID,
        raw_text: str,
        normalized_text: str,
        ocr_confidence: float | None,
        engine_name: str,
        runtime_device: str | None,
        reason: str | None,
    ) -> OCRResult:
        existing = await self.session.scalar(select(OCRResult).where(OCRResult.scan_id == scan_id))
        if existing is None:
            existing = OCRResult(
                scan_id=scan_id,
                raw_text=raw_text,
                normalized_text=normalized_text,
                ocr_confidence=ocr_confidence,
                engine_name=engine_name,
                runtime_device=runtime_device,
                reason=reason,
            )
            self.session.add(existing)
        else:
            existing.raw_text = raw_text
            existing.normalized_text = normalized_text
            existing.ocr_confidence = ocr_confidence
            existing.engine_name = engine_name
            existing.runtime_device = runtime_device
            existing.reason = reason
        await self.session.flush()
        return existing

    async def create_product_text_result(self, payload: ProductTextCreateData) -> ProductTextResult:
        row = ProductTextResult(
            scan_id=payload.scan_id,
            product_name=payload.product_name,
            brand=payload.brand,
            full_text=payload.full_text,
            confidence=payload.confidence,
            engine_name=payload.engine_name,
            runtime_device=payload.runtime_device,
            reason=payload.reason,
            source_endpoint=payload.source_endpoint,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def create_mobile_expiry_scan(self, payload: MobileExpiryScanCreateData) -> MobileExpiryScan:
        row = MobileExpiryScan(
            message=payload.message,
            metadata_json=payload.metadata_json,
            image_blob=payload.image_blob,
            image_filename=payload.image_filename,
            image_content_type=payload.image_content_type,
            detected_expiry_date=payload.detected_expiry_date,
            raw_text=payload.raw_text,
            normalized_text=payload.normalized_text,
            recognition_confidence=payload.recognition_confidence,
            detector_confidence=payload.detector_confidence,
            status=payload.status,
            reason=payload.reason,
            detection_polygon_json=payload.detection_polygon_json,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_mobile_expiry_scan(self, scan_id: UUID) -> MobileExpiryScan | None:
        return await self.session.get(MobileExpiryScan, scan_id)

    async def correct_mobile_expiry_scan(
        self,
        scan: MobileExpiryScan,
        *,
        corrected_expiry_date: date,
        reason: str | None,
    ) -> MobileExpiryScan:
        now = utcnow()
        scan.corrected_expiry_date = corrected_expiry_date
        scan.corrected_at = now
        scan.updated_at = now
        scan.status = "corrected"
        scan.reason = f"manual_correction:{reason}" if reason else "manual_correction"
        await self.session.flush()
        return scan

    async def list_test64_expected_dates(self) -> list[Test64ExpectedDate]:
        rows = await self.session.scalars(select(Test64ExpectedDate).order_by(Test64ExpectedDate.filename.asc()))
        return list(rows)

    async def get_test64_expected_date(self, filename: str) -> Test64ExpectedDate | None:
        return await self.session.scalar(select(Test64ExpectedDate).where(Test64ExpectedDate.filename == filename))

    async def upsert_test64_expected_date(self, payload: Test64ExpectedDateUpsertData) -> Test64ExpectedDate:
        existing = await self.get_test64_expected_date(payload.filename)
        if existing is None:
            existing = Test64ExpectedDate(
                filename=payload.filename,
                expected_day=payload.expected_day,
                expected_month=payload.expected_month,
                expected_year=payload.expected_year,
            )
            self.session.add(existing)
        else:
            existing.expected_day = payload.expected_day
            existing.expected_month = payload.expected_month
            existing.expected_year = payload.expected_year
            existing.updated_at = utcnow()
        await self.session.flush()
        return existing

    async def upsert_parsed_result(
        self,
        *,
        scan_id: UUID,
        parsed_date: date | None,
        date_format_detected: str | None,
        parse_confidence: float | None,
        parser_reason: str,
        candidates: list[str],
    ) -> ParsedDateResult:
        existing = await self.session.scalar(select(ParsedDateResult).where(ParsedDateResult.scan_id == scan_id))
        if existing is None:
            existing = ParsedDateResult(
                scan_id=scan_id,
                parsed_date=parsed_date,
                date_format_detected=date_format_detected,
                parse_confidence=parse_confidence,
                parser_reason=parser_reason,
                candidate_dates_json=candidates,
            )
            self.session.add(existing)
        else:
            existing.parsed_date = parsed_date
            existing.date_format_detected = date_format_detected
            existing.parse_confidence = parse_confidence
            existing.parser_reason = parser_reason
            existing.candidate_dates_json = candidates
        await self.session.flush()
        return existing

    async def upsert_expiry_status(
        self,
        *,
        scan_id: UUID,
        status: ExpiryClassification,
        days_remaining: int | None,
        alert_required: bool,
        needs_review: bool,
        reason: str,
    ) -> ExpiryStatus:
        existing = await self.session.scalar(select(ExpiryStatus).where(ExpiryStatus.scan_id == scan_id))
        if existing is None:
            existing = ExpiryStatus(
                scan_id=scan_id,
                status=status,
                days_remaining=days_remaining,
                alert_required=alert_required,
                needs_review=needs_review,
                reason=reason,
            )
            self.session.add(existing)
        else:
            existing.status = status
            existing.days_remaining = days_remaining
            existing.alert_required = alert_required
            existing.needs_review = needs_review
            existing.reason = reason
        await self.session.flush()
        return existing

    async def create_alert_if_missing(
        self,
        *,
        scan_id: UUID,
        user_id: str,
        alert_type: AlertType,
        payload: dict,
    ) -> bool:
        existing = await self.session.scalar(
            select(AlertEvent).where(and_(AlertEvent.scan_id == scan_id, AlertEvent.alert_type == alert_type))
        )
        if existing:
            return False
        event = AlertEvent(
            scan_id=scan_id,
            user_id=user_id,
            alert_type=alert_type,
            delivery_status=DeliveryStatus.PENDING,
            payload_json=payload,
        )
        self.session.add(event)
        await self.session.flush()
        return True

    async def mark_alert_delivery(self, event_id: UUID, success: bool) -> None:
        event = await self.session.get(AlertEvent, event_id)
        if event is None:
            return
        event.delivery_status = DeliveryStatus.SENT if success else DeliveryStatus.FAILED
        if success:
            event.sent_at = utcnow()
        await self.session.flush()

    async def list_pending_alert_events(self) -> list[AlertEvent]:
        rows = await self.session.scalars(select(AlertEvent).where(AlertEvent.delivery_status == DeliveryStatus.PENDING))
        return list(rows)

    async def list_expiry(
        self,
        *,
        status: ExpiryClassification | None = None,
        user_id: str | None = None,
        product_id: UUID | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[ExpiryListItem]:
        stmt = (
            select(Scan.id, Scan.user_id, Scan.product_id, ParsedDateResult.parsed_date, ExpiryStatus.status,
                   ExpiryStatus.days_remaining, ExpiryStatus.needs_review)
            .join(ExpiryStatus, ExpiryStatus.scan_id == Scan.id)
            .join(ParsedDateResult, ParsedDateResult.scan_id == Scan.id, isouter=True)
            .where(Scan.status == ScanStatus.DONE)
        )
        if status:
            stmt = stmt.where(ExpiryStatus.status == status)
        if user_id:
            stmt = stmt.where(Scan.user_id == user_id)
        if product_id:
            stmt = stmt.where(Scan.product_id == product_id)
        if date_from:
            stmt = stmt.where(ParsedDateResult.parsed_date >= date_from)
        if date_to:
            stmt = stmt.where(ParsedDateResult.parsed_date <= date_to)

        rows = (await self.session.execute(stmt)).all()
        return [
            ExpiryListItem(
                scan_id=row[0],
                user_id=row[1],
                product_id=row[2],
                parsed_date=row[3],
                expiry_classification=row[4],
                days_remaining=row[5],
                needs_review=row[6],
            )
            for row in rows
        ]

    async def add_log(
        self,
        *,
        scan_id: UUID,
        stage: str,
        message: str,
        level: str = "info",
        elapsed_ms: int | None = None,
        payload: dict | None = None,
    ) -> None:
        log = ProcessingLog(
            scan_id=scan_id,
            stage=stage,
            level=level,
            message=message,
            elapsed_ms=elapsed_ms,
            payload_json=payload,
        )
        self.session.add(log)
        await self.session.flush()

    async def metrics(self) -> dict[str, float | int]:
        total_scans = int((await self.session.scalar(select(func.count()).select_from(Scan))) or 0)

        detector_failures = int(
            (
                await self.session.scalar(
                    select(func.count()).select_from(Scan).where(Scan.final_result_status == FinalResultStatus.DETECTOR_FAILED)
                )
            )
            or 0
        )
        ocr_failures = int(
            (
                await self.session.scalar(
                    select(func.count()).select_from(Scan).where(Scan.final_result_status == FinalResultStatus.OCR_FAILED)
                )
            )
            or 0
        )
        parser_failures = int(
            (
                await self.session.scalar(
                    select(func.count()).select_from(Scan).where(Scan.final_result_status == FinalResultStatus.PARSER_FAILED)
                )
            )
            or 0
        )
        manual_review_count = int(
            (
                await self.session.scalar(
                    select(func.count()).select_from(ExpiryStatus).where(ExpiryStatus.needs_review.is_(True))
                )
            )
            or 0
        )
        alert_events = int((await self.session.scalar(select(func.count()).select_from(AlertEvent))) or 0)

        latency_rows = (
            await self.session.execute(select(Scan.created_at, Scan.processed_at).where(Scan.processed_at.is_not(None)))
        ).all()
        latency_ms_values: list[float] = []
        for created_at, processed_at in latency_rows:
            if created_at is None or processed_at is None:
                continue
            latency_ms_values.append((processed_at - created_at).total_seconds() * 1000.0)
        avg_latency_ms = sum(latency_ms_values) / len(latency_ms_values) if latency_ms_values else 0.0
        manual_review_rate = (manual_review_count / total_scans) if total_scans else 0.0

        return {
            "total_scans": total_scans,
            "detector_failures": detector_failures,
            "ocr_failures": ocr_failures,
            "parser_failures": parser_failures,
            "manual_review_count": manual_review_count,
            "manual_review_rate": round(manual_review_rate, 6),
            "average_processing_latency_ms": float(avg_latency_ms),
            "alert_events": alert_events,
        }

    async def scans_requiring_alerts(self) -> list[ScanAggregate]:
        stmt = (
            select(Scan)
            .join(ExpiryStatus, ExpiryStatus.scan_id == Scan.id)
            .where(Scan.status == ScanStatus.DONE)
            .where(ExpiryStatus.alert_required.is_(True))
        )
        scans = list(await self.session.scalars(stmt))
        results: list[ScanAggregate] = []
        for scan in scans:
            results.append((await self.get_scan_aggregate(scan.id)))
        return [x for x in results if x is not None]
