from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from arq import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.decision import ExpiryDecisionEngine
from app.domain.enums import AlertType, ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.repositories import DomainRepository, ScanAggregate
from app.domain.schemas import ScanResultPayload
from app.infra.queue import SCAN_JOB_NAME
from app.infra.storage import LocalStorage

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    pass


class ScanService:
    def __init__(
        self,
        session: AsyncSession,
        storage: LocalStorage,
        redis: ArqRedis | None,
        *,
        max_upload_size_bytes: int,
        accepted_file_types: tuple[str, ...],
        accepted_file_extensions: tuple[str, ...],
        alert_threshold_days: int,
    ) -> None:
        self.session = session
        self.repo = DomainRepository(session)
        self.storage = storage
        self.redis = redis
        self.max_upload_size_bytes = max_upload_size_bytes
        self.accepted_file_types = accepted_file_types
        self.accepted_file_extensions = accepted_file_extensions
        self.decision_engine = ExpiryDecisionEngine(alert_threshold_days)

    async def create_scan(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        qr_code: str,
        user_id: str,
        metadata: dict[str, Any] | None,
    ) -> UUID:
        self._validate_file(image_bytes=image_bytes, filename=filename, content_type=content_type)
        sanitized_metadata = self._sanitize_metadata(metadata)

        product = await self.repo.get_or_create_product(
            qr_code=qr_code,
            name=sanitized_metadata.get("name") if sanitized_metadata else None,
            brand=sanitized_metadata.get("brand") if sanitized_metadata else None,
            category=sanitized_metadata.get("category") if sanitized_metadata else None,
        )
        scan = await self.repo.create_scan(
            product_id=product.id,
            user_id=user_id,
            image_path="pending",
            metadata_json=sanitized_metadata,
        )

        extension = Path(filename).suffix.lower()
        relative_path = self.storage.raw_relative_path(scan.id, extension)
        self.storage.save_bytes(relative_path, image_bytes)
        scan.image_path = relative_path

        await self.session.commit()

        if self.redis is not None:
            try:
                await self.redis.enqueue_job(SCAN_JOB_NAME, str(scan.id), _job_id=f"scan:{scan.id}")
            except Exception as exc:  # pragma: no cover - depends on external queue
                logger.warning("failed to enqueue scan job", extra={"scan_id": str(scan.id), "reason": str(exc)})

        return scan.id

    async def get_scan_payload(self, scan_id: UUID) -> tuple[ScanStatus, ScanResultPayload | None]:
        aggregate = await self.repo.get_scan_aggregate(scan_id)
        if aggregate is None:
            raise KeyError("scan not found")

        scan = aggregate.scan
        if scan.status != ScanStatus.DONE:
            return scan.status, None

        result = ScanResultPayload(
            final_status=scan.final_result_status,
            detected=scan.detected,
            detector_confidence=scan.detector_confidence,
            raw_text=aggregate.ocr.raw_text if aggregate.ocr else None,
            parsed_date=aggregate.parsed.parsed_date if aggregate.parsed else None,
            date_format_detected=aggregate.parsed.date_format_detected if aggregate.parsed else None,
            parse_confidence=aggregate.parsed.parse_confidence if aggregate.parsed else None,
            expiry_classification=aggregate.expiry.status if aggregate.expiry else None,
            days_remaining=aggregate.expiry.days_remaining if aggregate.expiry else None,
            alert_required=aggregate.expiry.alert_required if aggregate.expiry else None,
            needs_review=aggregate.expiry.needs_review if aggregate.expiry else None,
            reason=scan.failure_reason or (aggregate.expiry.reason if aggregate.expiry else None),
            ocr_engine=aggregate.ocr.engine_name if aggregate.ocr else None,
            ocr_runtime_device=aggregate.ocr.runtime_device if aggregate.ocr else None,
        )
        return scan.status, result

    async def manual_correct_scan(self, scan_id: UUID, parsed_date: date | None, reason: str) -> tuple:
        aggregate = await self.repo.get_scan_aggregate(scan_id)
        if aggregate is None:
            raise KeyError("scan not found")

        scan = aggregate.scan
        decision = self.decision_engine.decide(parsed_date, today=date.today(), parse_confidence=1.0)

        await self.repo.upsert_parsed_result(
            scan_id=scan_id,
            parsed_date=parsed_date,
            date_format_detected="manual",
            parse_confidence=1.0,
            parser_reason=f"manual_override:{reason}",
            candidates=[parsed_date.isoformat()] if parsed_date else [],
        )
        await self.repo.upsert_expiry_status(
            scan_id=scan_id,
            status=ExpiryClassification(decision.expiry_classification),
            days_remaining=decision.days_remaining,
            alert_required=decision.alert_required,
            needs_review=decision.needs_review,
            reason=f"manual_override:{reason}",
        )
        await self.repo.finalize_scan(
            scan,
            final_status=FinalResultStatus.PARSED_SUCCESS if parsed_date else FinalResultStatus.MANUAL_REVIEW_REQUIRED,
            detected=scan.detected,
            detector_confidence=scan.detector_confidence,
            failure_reason=f"manual_override:{reason}",
            roi_path=scan.roi_path,
        )
        await self.session.commit()

        final_status = FinalResultStatus.PARSED_SUCCESS if parsed_date else FinalResultStatus.MANUAL_REVIEW_REQUIRED
        expiry = ExpiryClassification(decision.expiry_classification)
        return scan.status, final_status, expiry, decision.days_remaining

    async def list_expiry(
        self,
        *,
        status: ExpiryClassification | None,
        user_id: str | None,
        product_id: UUID | None,
        date_from: date | None,
        date_to: date | None,
    ):
        return await self.repo.list_expiry(
            status=status,
            user_id=user_id,
            product_id=product_id,
            date_from=date_from,
            date_to=date_to,
        )

    def _validate_file(self, *, image_bytes: bytes, filename: str, content_type: str) -> None:
        if len(image_bytes) > self.max_upload_size_bytes:
            raise ValidationError("file too large")
        if content_type not in self.accepted_file_types:
            raise ValidationError("unsupported content type")
        extension = Path(filename).suffix.lower()
        if extension not in self.accepted_file_extensions:
            raise ValidationError("unsupported file extension")

    def _sanitize_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValidationError("metadata must be a JSON object")

        allowed = (str, int, float, bool)

        def sanitize(value: Any) -> Any:
            if isinstance(value, allowed):
                if isinstance(value, str):
                    return value.strip()[:500]
                return value
            if isinstance(value, list):
                return [sanitize(v) for v in value[:20]]
            if isinstance(value, dict):
                out: dict[str, Any] = {}
                for key, inner in list(value.items())[:20]:
                    out[str(key)[:100]] = sanitize(inner)
                return out
            return str(value)[:200]

        sanitized = sanitize(metadata)
        if not isinstance(sanitized, dict):
            raise ValidationError("metadata must be a JSON object")
        return sanitized


class AlertService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repo = DomainRepository(session)

    async def process_alerts(self) -> tuple[int, int]:
        created = 0
        skipped = 0

        aggregates = await self.repo.scans_requiring_alerts()
        for aggregate in aggregates:
            alert_type = self._alert_type_for(aggregate)
            if alert_type is None:
                continue
            created_now = await self.repo.create_alert_if_missing(
                scan_id=aggregate.scan.id,
                user_id=aggregate.scan.user_id,
                alert_type=alert_type,
                payload={
                    "scan_id": str(aggregate.scan.id),
                    "user_id": aggregate.scan.user_id,
                    "classification": aggregate.expiry.status if aggregate.expiry else None,
                    "days_remaining": aggregate.expiry.days_remaining if aggregate.expiry else None,
                },
            )
            if created_now:
                created += 1
            else:
                skipped += 1

        await self.session.commit()
        return created, skipped

    def _alert_type_for(self, aggregate: ScanAggregate) -> AlertType | None:
        if aggregate.expiry is None:
            return None

        if aggregate.expiry.needs_review:
            return AlertType.MANUAL_REVIEW
        if aggregate.expiry.status == ExpiryClassification.EXPIRING_SOON:
            return AlertType.EXPIRING_SOON
        if aggregate.expiry.status == ExpiryClassification.EXPIRED:
            return AlertType.EXPIRED
        return None
