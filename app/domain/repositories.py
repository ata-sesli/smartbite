from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import AlertType, DeliveryStatus, ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.models import AlertEvent, ExpiryStatus, OCRResult, ParsedDateResult, ProcessingLog, Product, Scan
from app.infra.clock import utcnow


@dataclass(slots=True)
class ScanAggregate:
    scan: Scan
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
