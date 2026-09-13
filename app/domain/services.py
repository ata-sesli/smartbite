from __future__ import annotations

import logging
import json
import mimetypes
import os
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID
from uuid import uuid4

from PIL import Image, ImageOps
from arq import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.decision import ExpiryDecisionEngine
from app.domain.enums import AlertType, ExpiryClassification, FinalResultStatus, ReviewVerdict, ScanStatus
from app.domain.repositories import DomainRepository, ScanAggregate, Test64ExpectedDateUpsertData
from app.domain.schemas import (
    QueueTestImagesResponse,
    ScanListItemResponse,
    ScanResultPayload,
    ScanReviewPayload,
    Test64ExpectedDateItem,
    Test64ExpectedDateListResponse,
    Test64ExpectedDateUpsertResponse,
    Test64TruthBBoxItem,
    Test64TruthBBoxManifestResponse,
    Test64TruthBBoxUpsertResponse,
    TrainingDataExportResponse,
)
from app.infra.queue import SCAN_JOB_NAME
from app.infra.settings import PROJECT_ROOT
from app.infra.storage import LocalStorage

logger = logging.getLogger(__name__)

CROP_TRUTH_AUDIT_FILENAMES = (
    "d603c295-cce7-4b11-8a0c-8b4c9d4c72e1.jpg",
    "IMG_0905.JPG",
    "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg",
    "sarelle-chocolate.jpeg",
    "IMG_0900.JPG",
    "67d17a14-978f-4a71-8d12-476bf0fa64af.jpg",
    "IMG_0886.JPG",
    "IMG_0908.JPG",
    "e80af774-e166-4a2d-b6b7-83ab8a750fb1.jpg",
    "5a6ad67b-3cf9-4c5c-9148-1fb606d5654a.jpg",
)
CROP_TRUTH_ANNOTATIONS_PATH = Path.cwd() / "artifacts" / "forensics" / "crop_truth_annotations" / "test64_truth_bboxes.json"
CROP_TRUTH_COORDINATE_SPACE = "original_image_xyxy"
MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR = Path.cwd() / "artifacts" / "forensics"
TEST64_DETECTION_REVIEW_ARTIFACTS_DIR = Path.cwd() / "artifacts" / "forensics"
TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR = Path.cwd() / "artifacts" / "onnx_parity"


class ValidationError(ValueError):
    pass


def _review_payload(review) -> ScanReviewPayload | None:
    if review is None:
        return None
    return ScanReviewPayload(
        scan_id=review.scan_id,
        verdict=review.verdict,
        accepted_for_training=review.accepted_for_training,
        final_text=review.final_text,
        final_parsed_date=review.final_parsed_date,
        bbox_xyxy=review.bbox_xyxy,
        bbox_source=review.bbox_source,
        reviewer_id=review.reviewer_id,
        notes=review.notes,
        created_at=review.created_at,
        updated_at=review.updated_at,
    )


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
        scan_job_expires_seconds: int,
        general_text_ocr_timeout_seconds: float = 45.0,
    ) -> None:
        self.session = session
        self.repo = DomainRepository(session)
        self.storage = storage
        self.redis = redis
        self.max_upload_size_bytes = max_upload_size_bytes
        self.accepted_file_types = accepted_file_types
        self.accepted_file_extensions = accepted_file_extensions
        self.decision_engine = ExpiryDecisionEngine(alert_threshold_days)
        self.scan_job_expires_seconds = max(60, int(scan_job_expires_seconds))
        self.general_text_ocr_timeout_seconds = max(1.0, float(general_text_ocr_timeout_seconds))

    @staticmethod
    def _resolve_test_images_dir() -> Path | None:
        candidates = (
            PROJECT_ROOT / "test64",
            Path.cwd() / "test64",
            Path("/app/test64"),
            PROJECT_ROOT / "test-images",
            Path.cwd() / "test-images",
            Path("/app/test-images"),
        )
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _resolve_test64_dir() -> Path | None:
        candidates = (
            PROJECT_ROOT / "test64",
            Path.cwd() / "test64",
            Path("/app/test64"),
        )
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        if not path.is_file():
            return False
        content_type = mimetypes.guess_type(path.name)[0] or ""
        return content_type.startswith("image/")

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
        sanitized_metadata = self._sanitize_metadata(metadata) or {}
        sanitized_metadata["original_filename"] = Path(filename).name

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
                await self.redis.enqueue_job(
                    SCAN_JOB_NAME,
                    str(scan.id),
                    _job_id=f"scan:{scan.id}",
                    _expires=self.scan_job_expires_seconds,
                )
            except Exception as exc:  # pragma: no cover - depends on external queue
                logger.warning("failed to enqueue scan job", extra={"scan_id": str(scan.id), "reason": str(exc)})

        return scan.id

    async def queue_test_images(self, *, user_id: str = "console_test_images") -> QueueTestImagesResponse:
        test_images_dir = self._resolve_test_images_dir()
        if test_images_dir is None:
            raise ValidationError(
                f"test-image directories not found: {PROJECT_ROOT / 'test64'} or {PROJECT_ROOT / 'test-images'}"
            )

        candidates = sorted(path for path in test_images_dir.iterdir() if self._is_image_file(path))
        queued_scan_ids: list[UUID] = []
        errors: list[str] = []
        skipped = 0

        for path in candidates:
            try:
                image_bytes = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                scan_id = await self.create_scan(
                    image_bytes=image_bytes,
                    filename=path.name,
                    content_type=content_type,
                    qr_code=f"{test_images_dir.name}-{path.stem}-{uuid4().hex[:8]}",
                    user_id=user_id,
                    metadata={"source": f"{test_images_dir.name}-bulk-queue"},
                )
                queued_scan_ids.append(scan_id)
            except ValidationError as exc:
                skipped += 1
                errors.append(f"{path.name}: {exc}")
            except Exception as exc:  # pragma: no cover - defensive runtime path
                skipped += 1
                errors.append(f"{path.name}: unexpected error ({exc})")

        return QueueTestImagesResponse(
            queued=len(queued_scan_ids),
            skipped=skipped,
            total_candidates=len(candidates),
            scan_ids=queued_scan_ids,
            errors=errors,
        )

    async def list_test64_expected_dates(self) -> Test64ExpectedDateListResponse:
        test64_dir = self._resolve_test64_dir()
        if test64_dir is None:
            raise ValidationError(f"test64 directory not found: {PROJECT_ROOT / 'test64'}")

        files = sorted(path for path in test64_dir.iterdir() if self._is_image_file(path))
        try:
            labels = await self.repo.list_test64_expected_dates()
        except Exception as exc:  # pragma: no cover - dev database may not be migrated for labels
            logger.warning("test64 expected-date labels unavailable for crop truth manifest: %s", exc)
            labels = []
        labels_by_name = {row.filename: row for row in labels}

        items: list[Test64ExpectedDateItem] = []
        labeled_count = 0
        for file_path in files:
            label = labels_by_name.get(file_path.name)
            if label is not None:
                labeled_count += 1
            items.append(
                Test64ExpectedDateItem(
                    filename=file_path.name,
                    image_url=f"/test64/images/{file_path.name}",
                    expected_day=label.expected_day if label is not None else None,
                    expected_month=label.expected_month if label is not None else None,
                    expected_year=label.expected_year if label is not None else None,
                    updated_at=label.updated_at if label is not None else None,
                )
            )

        return Test64ExpectedDateListResponse(
            items=items,
            total_candidates=len(files),
            labeled_count=labeled_count,
        )

    async def upsert_test64_expected_date(
        self,
        *,
        filename: str,
        day: int | None,
        month: int,
        year: int,
    ) -> Test64ExpectedDateUpsertResponse:
        test64_dir = self._resolve_test64_dir()
        if test64_dir is None:
            raise ValidationError(f"test64 directory not found: {PROJECT_ROOT / 'test64'}")

        normalized_filename = Path(filename).name
        if normalized_filename != filename:
            raise ValidationError("filename must not include directory components")

        target = test64_dir / normalized_filename
        if not target.exists() or not target.is_file():
            raise ValidationError(f"test64 image not found: {normalized_filename}")

        row = await self.repo.upsert_test64_expected_date(
            Test64ExpectedDateUpsertData(
                filename=normalized_filename,
                expected_day=day,
                expected_month=month,
                expected_year=year,
            )
        )
        await self.session.commit()

        return Test64ExpectedDateUpsertResponse(
            item=Test64ExpectedDateItem(
                filename=row.filename,
                image_url=f"/test64/images/{row.filename}",
                expected_day=row.expected_day,
                expected_month=row.expected_month,
                expected_year=row.expected_year,
                updated_at=row.updated_at,
            )
        )

    async def get_test64_image_path(self, filename: str) -> str:
        test64_dir = self._resolve_test64_dir()
        if test64_dir is None:
            raise ValidationError(f"test64 directory not found: {PROJECT_ROOT / 'test64'}")

        normalized_filename = Path(filename).name
        if normalized_filename != filename:
            raise ValidationError("filename must not include directory components")

        target = test64_dir / normalized_filename
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"test64 image not found: {normalized_filename}")
        if not self._is_image_file(target):
            raise ValidationError(f"unsupported test64 file type: {normalized_filename}")
        return str(target)

    async def get_test64_truth_bboxes(self) -> Test64TruthBBoxManifestResponse:
        return await self._load_or_create_truth_bbox_manifest()

    async def get_manual_crop_recognition_review(self, *, include_success: bool = False) -> dict[str, Any]:
        report_path = self._latest_manual_crop_recognition_report_path()
        report = self._read_manual_crop_recognition_report(report_path)
        report_root = self._manual_crop_report_root(report, report_path)
        run_id = report_root.name
        items = report.get("items") if isinstance(report.get("items"), list) else []
        filtered_items = [
            self._manual_crop_review_item(item, run_id=run_id, report_root=report_root)
            for item in items
            if isinstance(item, dict) and (include_success or not self._manual_crop_item_is_success(item))
        ]
        return {
            "run_id": run_id,
            "generated_at": report.get("generated_at"),
            "report_path": str(report_path),
            "report_root": str(report_root),
            "recognizer_config": report.get("recognizer_config") if isinstance(report.get("recognizer_config"), dict) else {},
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "total_items": len(items),
            "filtered_count": len(filtered_items),
            "include_success": include_success,
            "items": filtered_items,
        }

    async def get_test64_detection_review(self) -> dict[str, Any]:
        detector_audit = self._latest_test64_detector_audit_payload()
        items = detector_audit.get("items") if isinstance(detector_audit.get("items"), list) else []
        truth_manifest = self._read_truth_bbox_manifest_payload()
        truth_items = truth_manifest.get("items") if isinstance(truth_manifest.get("items"), dict) else {}
        return {
            "run_id": detector_audit.get("run_id"),
            "report_path": detector_audit.get("report_path"),
            "generated_at": detector_audit.get("generated_at"),
            "source": detector_audit.get("source"),
            "summary": detector_audit.get("summary") if isinstance(detector_audit.get("summary"), dict) else {},
            "configs": detector_audit.get("configs") if isinstance(detector_audit.get("configs"), list) else [],
            "total_items": len(items),
            "filtered_count": len(items),
            "items": [
                self._test64_detector_audit_review_item(item, truth_items=truth_items)
                for item in items
                if isinstance(item, dict)
            ],
        }

    async def get_test64_product_cropper_review(self) -> dict[str, Any]:
        detector_audit = self._latest_test64_product_cropper_audit_payload()
        items = detector_audit.get("items") if isinstance(detector_audit.get("items"), list) else []
        truth_manifest = self._read_truth_bbox_manifest_payload()
        truth_items = truth_manifest.get("items") if isinstance(truth_manifest.get("items"), dict) else {}
        return {
            "run_id": detector_audit.get("run_id"),
            "report_path": detector_audit.get("report_path"),
            "generated_at": detector_audit.get("generated_at"),
            "source": detector_audit.get("source"),
            "summary": detector_audit.get("summary") if isinstance(detector_audit.get("summary"), dict) else {},
            "configs": detector_audit.get("configs") if isinstance(detector_audit.get("configs"), list) else [],
            "total_items": len(items),
            "filtered_count": len(items),
            "items": [
                self._test64_detector_audit_review_item(item, truth_items=truth_items)
                for item in items
                if isinstance(item, dict)
            ],
        }

    async def get_test64_full_pipeline_review(self) -> dict[str, Any]:
        report_path = self._latest_test64_detection_review_report_path()
        report = self._read_test64_detection_review_report(report_path)
        evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), dict) else {}
        per_scan = evaluation.get("per_scan") if isinstance(evaluation.get("per_scan"), list) else []
        detector_stats = report.get("per_scan_detector_stats") if isinstance(report.get("per_scan_detector_stats"), list) else []
        detector_by_name = {
            stat.get("filename"): stat
            for stat in detector_stats
            if isinstance(stat, dict) and isinstance(stat.get("filename"), str)
        }
        items = [
            self._test64_detection_review_item(item, detector_by_name.get(item.get("filename")))
            for item in per_scan
            if isinstance(item, dict)
        ]
        return {
            "run_id": report_path.parent.name,
            "report_path": str(report_path),
            "source": report.get("source"),
            "text_detector_mode": report.get("text_detector_mode"),
            "craft_model_path": report.get("craft_model_path"),
            "detector_config": report.get("detector_config") if isinstance(report.get("detector_config"), dict) else {},
            "detector_totals": report.get("detector_totals") if isinstance(report.get("detector_totals"), dict) else {},
            "summary": evaluation.get("summary") if isinstance(evaluation.get("summary"), dict) else {},
            "parseable_candidates_total": evaluation.get("parseable_candidates_total"),
            "average_parseq_candidates_per_scan": evaluation.get("average_parseq_candidates_per_scan"),
            "average_runtime_ms_per_scan": evaluation.get("average_runtime_ms_per_scan"),
            "total_items": len(per_scan),
            "filtered_count": len(items),
            "items": items,
        }

    async def get_test64_full_pipeline_results(self) -> dict[str, Any]:
        report_path = self._latest_test64_mobile_pipeline_report_path()
        report = self._read_test64_mobile_upload_report(report_path)
        rows = report.get("rows") if isinstance(report.get("rows"), list) else []
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        source = "mobile-direct-benchmark" if "direct" in report_path.name else "mobile-upload-benchmark"
        return {
            "run_id": report_path.parent.name,
            "report_path": str(report_path),
            "source": source,
            "base_url": report.get("base_url"),
            "endpoint": report.get("endpoint"),
            "images_dir": report.get("images_dir"),
            "created_at": report.get("created_at"),
            "summary": summary,
            "latency": summary.get("latency") if isinstance(summary.get("latency"), dict) else {},
            "total_items": len(rows),
            "filtered_count": len(rows),
            "wrong_dates": report.get("wrong_dates") if isinstance(report.get("wrong_dates"), list) else [],
            "manual_review": report.get("manual_review") if isinstance(report.get("manual_review"), list) else [],
            "items": [
                self._test64_mobile_full_pipeline_item(item)
                for item in rows
                if isinstance(item, dict)
            ],
        }

    async def get_test64_mobile_oracle_forensic_audit(self) -> dict[str, Any]:
        report_path = self._latest_mobile_oracle_forensic_audit_report_path()
        report = self._read_mobile_oracle_forensic_audit_report(report_path)
        run_id = report_path.parent.name
        rows = report.get("rows") if isinstance(report.get("rows"), list) else []
        return {
            "run_id": run_id,
            "report_path": str(report_path),
            "stable_report": report.get("stable_report"),
            "truth_manifest": report.get("truth_manifest"),
            "images_dir": report.get("images_dir"),
            "created_at": report.get("created_at"),
            "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
            "total_items": len(rows),
            "filtered_count": len(rows),
            "items": [
                self._mobile_oracle_forensic_review_item(item, run_id=run_id, report_root=report_path.parent)
                for item in rows
                if isinstance(item, dict)
            ],
        }

    def get_manual_crop_recognition_asset_path(self, *, run_id: str, relative_path: str) -> Path:
        run_dir = self._manual_crop_run_dir(run_id)
        if not relative_path or Path(relative_path).is_absolute():
            raise ValidationError("invalid artifact path")
        candidate = (run_dir / relative_path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValidationError("invalid artifact path") from exc
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"manual crop recognition artifact not found: {relative_path}")
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValidationError("unsupported manual crop recognition artifact type")
        return candidate

    def get_mobile_oracle_forensic_asset_path(self, *, run_id: str, relative_path: str) -> Path:
        run_dir = self._mobile_oracle_forensic_run_dir(run_id)
        if not relative_path or Path(relative_path).is_absolute():
            raise ValidationError("invalid artifact path")
        candidate = (run_dir / relative_path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValidationError("invalid artifact path") from exc
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"mobile oracle forensic artifact not found: {relative_path}")
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValidationError("unsupported mobile oracle forensic artifact type")
        return candidate

    async def upsert_test64_truth_bbox(
        self,
        *,
        filename: str,
        bbox_xyxy: list[float] | None,
        polygon_xy: list[list[float]] | None = None,
        rotation_degrees: float | None = None,
    ) -> Test64TruthBBoxUpsertResponse:
        normalized_filename = self._normalize_crop_truth_filename(filename)
        manifest = await self._load_or_create_truth_bbox_manifest()
        item = manifest.items[normalized_filename]
        cleaned_bbox = self._validate_truth_bbox(
            bbox_xyxy,
            image_width=item.image_width,
            image_height=item.image_height,
        )
        cleaned_polygon = self._validate_truth_polygon(
            polygon_xy if cleaned_bbox is not None else None,
            image_width=item.image_width,
            image_height=item.image_height,
        )
        cleaned_rotation = self._validate_truth_rotation(rotation_degrees) if cleaned_bbox is not None else None
        updated = item.model_copy(
            update={
                "true_bbox_xyxy": cleaned_bbox,
                "true_polygon_xy": cleaned_polygon,
                "annotation_rotation_degrees": cleaned_rotation,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        items = dict(manifest.items)
        items[normalized_filename] = updated
        payload = {
            "version": manifest.version,
            "coordinate_space": manifest.coordinate_space,
            "audit_set": manifest.audit_set,
            "items": {name: value.model_dump(mode="json") for name, value in items.items()},
        }
        self._write_truth_bbox_manifest_payload(payload)
        annotated_count = sum(1 for value in items.values() if value.true_bbox_xyxy is not None)
        return Test64TruthBBoxUpsertResponse(
            item=updated,
            annotated_count=annotated_count,
            total_count=len(items),
            manifest_path=str(CROP_TRUTH_ANNOTATIONS_PATH),
        )

    @staticmethod
    def _latest_manual_crop_recognition_report_path() -> Path:
        if not MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"manual crop recognition artifacts not found: {MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR}"
            )
        reports = sorted(
            MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR.glob(
                "manual_crop_recognition_*/manual_crop_recognition_report.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            raise FileNotFoundError(
                f"manual crop recognition report not found under {MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR}"
            )
        return reports[0]

    @staticmethod
    def _latest_test64_detection_review_report_path() -> Path:
        if not TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"test64 detection artifacts not found: {TEST64_DETECTION_REVIEW_ARTIFACTS_DIR}"
            )
        reports = sorted(
            TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.glob("test64_plain_*/plain_test64_report.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            raise FileNotFoundError(
                f"test64 detection review report not found under {TEST64_DETECTION_REVIEW_ARTIFACTS_DIR}"
            )
        return reports[0]

    @staticmethod
    def _latest_test64_mobile_upload_report_path() -> Path:
        if not TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"test64 mobile upload artifacts not found: {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        reports = sorted(
            TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.glob("test64_mobile*upload_*/mobile_test64_upload_report.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            raise FileNotFoundError(
                f"test64 mobile upload report not found under {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        return reports[0]

    @staticmethod
    def _latest_test64_mobile_direct_report_path() -> Path:
        if not TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"test64 mobile artifacts not found: {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        reports = sorted(
            TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.glob("test64_mobile_direct_*/mobile_test64_direct_report.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            raise FileNotFoundError(
                f"test64 mobile direct report not found under {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        return reports[0]

    @classmethod
    def _latest_test64_mobile_pipeline_report_path(cls) -> Path:
        try:
            return cls._latest_test64_mobile_direct_report_path()
        except FileNotFoundError:
            return cls._latest_test64_mobile_upload_report_path()

    @staticmethod
    def _latest_mobile_oracle_forensic_audit_report_path() -> Path:
        if not TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.exists():
            raise FileNotFoundError(
                f"mobile oracle forensic artifacts not found: {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        reports = sorted(
            TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.glob(
                "mobile_oracle_forensic_audit_*/mobile_oracle_forensic_audit_report.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            raise FileNotFoundError(
                f"mobile oracle forensic report not found under {TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR}"
            )
        return reports[0]

    @staticmethod
    def _latest_test64_detector_audit_payload() -> dict[str, Any]:
        if not TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.exists():
            return {}
        reports = sorted(
            [
                *TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.glob(
                    "test64_plain_yolo_obb_audit_*/plain_yolo_obb_test64_report.json"
                ),
                *TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.glob(
                    "test64_detector_audit_*/detector_truth_ablation_report.json"
                ),
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            return {}
        report_path = reports[0]
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        if report_path.name == "plain_yolo_obb_test64_report.json":
            items = payload.get("items") if isinstance(payload.get("items"), list) else []
            config_name = "plain_yolo_obb"
            for item in items:
                if not isinstance(item, dict):
                    continue
                config = item.get("config")
                if isinstance(config, dict) and isinstance(config.get("config_name"), str):
                    config_name = config["config_name"]
                    break
            adapted_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                config = item.get("config")
                adapted = dict(item)
                adapted["configs"] = [config] if isinstance(config, dict) else []
                adapted_items.append(adapted)
            summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
            settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
            return {
                "run_id": report_path.parent.name,
                "report_path": str(report_path),
                "generated_at": payload.get("generated_at"),
                "source": payload.get("source") or payload.get("backend"),
                "summary": {"configs": {config_name: summary}},
                "configs": [{"name": config_name, "settings": settings}],
                "items": adapted_items,
            }
        return {
            "run_id": report_path.parent.name,
            "report_path": str(report_path),
            "generated_at": payload.get("generated_at"),
            "source": payload.get("source"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "configs": payload.get("configs") if isinstance(payload.get("configs"), list) else [],
            "items": payload.get("items") if isinstance(payload.get("items"), list) else [],
        }

    @staticmethod
    def _latest_test64_product_cropper_audit_payload() -> dict[str, Any]:
        if not TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.exists():
            return {}
        reports = sorted(
            TEST64_DETECTION_REVIEW_ARTIFACTS_DIR.glob(
                "test64_product_roi_yolo_audit_*/product_roi_yolo_test64_report.json"
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            return {}
        report_path = reports[0]
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            "run_id": report_path.parent.name,
            "report_path": str(report_path),
            "generated_at": payload.get("generated_at"),
            "source": payload.get("source") or payload.get("backend"),
            "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
            "configs": payload.get("configs") if isinstance(payload.get("configs"), list) else [],
            "items": payload.get("items") if isinstance(payload.get("items"), list) else [],
        }

    @staticmethod
    def _read_manual_crop_recognition_report(report_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"manual crop recognition report is invalid JSON: {report_path}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("manual crop recognition report must be a JSON object")
        return payload

    @staticmethod
    def _read_test64_detection_review_report(report_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"test64 detection review report is invalid JSON: {report_path}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("test64 detection review report must be a JSON object")
        return payload

    @staticmethod
    def _read_test64_mobile_upload_report(report_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"test64 mobile upload report is invalid JSON: {report_path}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("test64 mobile upload report must be a JSON object")
        return payload

    @staticmethod
    def _read_mobile_oracle_forensic_audit_report(report_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"mobile oracle forensic report is invalid JSON: {report_path}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("mobile oracle forensic report must be a JSON object")
        return payload

    @staticmethod
    def _test64_mobile_full_pipeline_item(row: dict[str, Any]) -> dict[str, Any]:
        filename = row.get("filename")
        response_status = row.get("response_status")
        detected_date = row.get("detected_expiry_date")
        if row.get("exact_match") is True:
            verdict = "correct_match"
        elif response_status == "parsed_success" and detected_date:
            verdict = "wrong_date"
        elif response_status == "manual_review_required":
            verdict = "manual_review"
        elif row.get("http_status") != 201:
            verdict = "http_error"
        else:
            verdict = "failed"
        return {
            "filename": filename,
            "image_url": f"/test64/images/{quote(str(filename))}" if filename else None,
            "verdict": verdict,
            "expected_date": row.get("expected_date"),
            "expected_precision": row.get("expected_precision"),
            "expected_day": row.get("expected_day"),
            "expected_month": row.get("expected_month"),
            "expected_year": row.get("expected_year"),
            "predicted_date": detected_date,
            "detected_expiry_date": detected_date,
            "exact_match": row.get("exact_match"),
            "http_status": row.get("http_status"),
            "final_status": response_status,
            "response_status": response_status,
            "raw_text": row.get("raw_text"),
            "normalized_text": row.get("normalized_text"),
            "recognition_confidence": row.get("recognition_confidence"),
            "detector_confidence": row.get("detector_confidence"),
            "final_recognition_bbox_xyxy": row.get("final_recognition_bbox_xyxy"),
            "final_recognition_polygon_json": row.get("final_recognition_polygon_json"),
            "final_crop_policy": row.get("final_crop_policy"),
            "final_crop_padding_px": row.get("final_crop_padding_px"),
            "reason": row.get("reason"),
            "runtime_ms": row.get("latency_ms"),
            "latency_ms": row.get("latency_ms"),
            "raw": row,
        }

    @classmethod
    def _mobile_oracle_forensic_review_item(
        cls,
        row: dict[str, Any],
        *,
        run_id: str,
        report_root: Path,
    ) -> dict[str, Any]:
        filename = row.get("filename")
        enriched = cls._mobile_oracle_forensic_attach_asset_urls(row, run_id=run_id, report_root=report_root)
        return {
            **enriched,
            "filename": filename,
            "image_url": f"/test64/images/{quote(str(filename))}" if filename else None,
        }

    @classmethod
    def _mobile_oracle_forensic_attach_asset_urls(
        cls,
        value: Any,
        *,
        run_id: str,
        report_root: Path,
    ) -> Any:
        if isinstance(value, list):
            return [
                cls._mobile_oracle_forensic_attach_asset_urls(item, run_id=run_id, report_root=report_root)
                for item in value
            ]
        if not isinstance(value, dict):
            return value
        out: dict[str, Any] = {
            key: cls._mobile_oracle_forensic_attach_asset_urls(item, run_id=run_id, report_root=report_root)
            for key, item in value.items()
        }
        for key in ("crop_path", "manual_true_crop_path", "manual_true_crop_normalized_path"):
            if key in value:
                out[f"{key.removesuffix('path')}url"] = cls._mobile_oracle_forensic_asset_url(
                    value.get(key),
                    run_id=run_id,
                    report_root=report_root,
                )
        return out

    @staticmethod
    def _mobile_oracle_forensic_asset_url(path_value: Any, *, run_id: str, report_root: Path) -> str | None:
        if not isinstance(path_value, str) or not path_value:
            return None
        path = Path(path_value)
        try:
            relative = path.resolve().relative_to(report_root.resolve())
        except ValueError:
            run_name = report_root.name
            parts = path.parts
            if run_name not in parts:
                return None
            relative = Path(*parts[parts.index(run_name) + 1 :])
        return (
            "/test64/mobile-oracle-forensic-audit/assets"
            f"?run_id={quote(run_id)}&path={quote(relative.as_posix())}"
        )

    @staticmethod
    def _test64_detection_review_item(scan: dict[str, Any], detector: dict[str, Any] | None) -> dict[str, Any]:
        detector = detector if isinstance(detector, dict) else {}
        return {
            "filename": scan.get("filename"),
            "image_url": f"/test64/images/{quote(str(scan.get('filename')))}" if scan.get("filename") else None,
            "verdict": scan.get("verdict"),
            "predicted_date": scan.get("predicted_date"),
            "final_status": scan.get("final_status"),
            "reason": scan.get("reason"),
            "parseable_candidates": scan.get("parseable_candidates"),
            "selected_candidates": scan.get("selected_candidates"),
            "runtime_ms": scan.get("runtime_ms"),
            "detector_counts": {
                "variants": detector.get("variants"),
                "ppocrv5_server_boxes": detector.get("ppocrv5_server_boxes"),
                "craft_boxes": detector.get("craft_boxes"),
                "ensemble_boxes": detector.get("ensemble_boxes"),
                "detector_modes": detector.get("detector_modes") if isinstance(detector.get("detector_modes"), dict) else {},
                "craft_unavailable_reasons": (
                    detector.get("craft_unavailable_reasons")
                    if isinstance(detector.get("craft_unavailable_reasons"), list)
                    else []
                ),
            },
            "detector_performance": detector.get("performance") if isinstance(detector.get("performance"), dict) else {},
            "detector_boxes": detector.get("detector_boxes") if isinstance(detector.get("detector_boxes"), list) else [],
            "overlay_available": False,
        }

    @staticmethod
    def _test64_detector_audit_review_item(
        item: dict[str, Any],
        *,
        truth_items: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filename = item.get("filename")
        truth_item = truth_items.get(str(filename)) if truth_items is not None and filename else None
        truth_item = truth_item if isinstance(truth_item, dict) else {}
        return {
            "filename": filename,
            "image_url": f"/test64/images/{quote(str(filename))}" if filename else None,
            "truth_bbox_xyxy": item.get("truth_bbox_xyxy"),
            "truth_polygon_xy": item.get("truth_polygon_xy")
            if isinstance(item.get("truth_polygon_xy"), list)
            else truth_item.get("true_polygon_xy"),
            "truth_annotation_rotation_degrees": truth_item.get("annotation_rotation_degrees"),
            "configs": item.get("configs") if isinstance(item.get("configs"), list) else [],
        }

    @staticmethod
    def _manual_crop_report_root(report: dict[str, Any], report_path: Path) -> Path:
        raw_root = report.get("report_root")
        if isinstance(raw_root, str) and raw_root:
            report_root = Path(raw_root)
            if not report_root.is_absolute():
                report_root = (report_path.parent / report_root).resolve()
            if report_root.exists():
                return report_root
        return report_path.parent

    @staticmethod
    def _manual_crop_item_is_success(item: dict[str, Any]) -> bool:
        winner = item.get("winner") if isinstance(item.get("winner"), dict) else {}
        return bool(winner.get("exact_match")) or item.get("outcome") == "success_without_preprocessing_requirement"

    @classmethod
    def _manual_crop_review_item(cls, item: dict[str, Any], *, run_id: str, report_root: Path) -> dict[str, Any]:
        variants = item.get("variants") if isinstance(item.get("variants"), list) else []
        winner = item.get("winner") if isinstance(item.get("winner"), dict) else None
        return {
            "filename": item.get("filename"),
            "expected_date": item.get("expected_date"),
            "outcome": item.get("outcome") or item.get("classification"),
            "true_bbox_xyxy": item.get("true_bbox_xyxy"),
            "true_polygon_xy": item.get("true_polygon_xy"),
            "manual_true_crop_source": item.get("manual_true_crop_source"),
            "manual_true_crop_url": cls._manual_crop_asset_url(
                item.get("manual_true_crop_path"),
                run_id=run_id,
                report_root=report_root,
            ),
            "manual_true_crop_normalized_url": cls._manual_crop_asset_url(
                item.get("manual_true_crop_normalized_path"),
                run_id=run_id,
                report_root=report_root,
            ),
            "winner": cls._manual_crop_review_variant(winner, run_id=run_id, report_root=report_root)
            if winner is not None
            else None,
            "variants": [
                cls._manual_crop_review_variant(variant, run_id=run_id, report_root=report_root)
                for variant in variants
                if isinstance(variant, dict)
            ],
            "special_checks": item.get("special_checks") if isinstance(item.get("special_checks"), dict) else {},
        }

    @classmethod
    def _manual_crop_review_variant(cls, variant: dict[str, Any], *, run_id: str, report_root: Path) -> dict[str, Any]:
        return {
            "variant_name": variant.get("variant_name"),
            "crop_url": cls._manual_crop_asset_url(
                variant.get("crop_path"),
                run_id=run_id,
                report_root=report_root,
            ),
            "raw_text": variant.get("parseq_raw_output"),
            "normalized_text": variant.get("parseq_normalized_output"),
            "confidence": variant.get("parseq_confidence"),
            "runtime_ms": variant.get("parseq_runtime_ms"),
            "parser_parsed_date": variant.get("parser_parsed_date"),
            "parser_date_precision": variant.get("parser_date_precision"),
            "parser_parsed_day": variant.get("parser_parsed_day"),
            "parser_parsed_month": variant.get("parser_parsed_month"),
            "parser_parsed_year": variant.get("parser_parsed_year"),
            "parser_confidence": variant.get("parser_confidence"),
            "parser_reason": variant.get("parser_reason"),
            "exact_match": variant.get("exact_match"),
            "malformed_but_potentially_recoverable": variant.get("malformed_but_potentially_recoverable"),
            "generic_text_output": variant.get("generic_text_output"),
            "crop_transform_used": variant.get("crop_transform_used"),
            "selected_orientation": variant.get("selected_orientation"),
            "orientation_candidates_tried": (
                variant.get("orientation_candidates_tried")
                if isinstance(variant.get("orientation_candidates_tried"), list)
                else []
            ),
            "original_crop_shape": variant.get("original_crop_shape") if isinstance(variant.get("original_crop_shape"), list) else None,
            "normalized_crop_shape": variant.get("normalized_crop_shape")
            if isinstance(variant.get("normalized_crop_shape"), list)
            else None,
            "selected_transform_reason": variant.get("selected_transform_reason"),
        }

    @staticmethod
    def _manual_crop_asset_url(path_value: Any, *, run_id: str, report_root: Path) -> str | None:
        if not isinstance(path_value, str) or not path_value:
            return None
        path = Path(path_value)
        try:
            relative = path.resolve().relative_to(report_root.resolve())
        except ValueError:
            run_name = report_root.name
            parts = path.parts
            if run_name not in parts:
                return None
            relative = Path(*parts[parts.index(run_name) + 1 :])
        return (
            "/test64/manual-crop-recognition-review/assets"
            f"?run_id={quote(run_id)}&path={quote(relative.as_posix())}"
        )

    @staticmethod
    def _manual_crop_run_dir(run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id or not run_id.startswith("manual_crop_recognition_"):
            raise ValidationError("invalid manual crop recognition run_id")
        artifacts_root = MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR.resolve()
        run_dir = (MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR / run_id).resolve()
        try:
            run_dir.relative_to(artifacts_root)
        except ValueError as exc:
            raise ValidationError("invalid manual crop recognition run_id") from exc
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"manual crop recognition run not found: {run_id}")
        return run_dir

    @staticmethod
    def _mobile_oracle_forensic_run_dir(run_id: str) -> Path:
        if not run_id or Path(run_id).name != run_id or not run_id.startswith("mobile_oracle_forensic_audit_"):
            raise ValidationError("invalid mobile oracle forensic run_id")
        artifacts_root = TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR.resolve()
        run_dir = (TEST64_MOBILE_UPLOAD_ARTIFACTS_DIR / run_id).resolve()
        try:
            run_dir.relative_to(artifacts_root)
        except ValueError as exc:
            raise ValidationError("invalid mobile oracle forensic run_id") from exc
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"mobile oracle forensic run not found: {run_id}")
        return run_dir

    def _normalize_crop_truth_filename(self, filename: str) -> str:
        normalized_filename = Path(filename).name
        if normalized_filename != filename:
            raise ValidationError("filename must not include directory components")
        test64_dir = self._resolve_test64_dir()
        if test64_dir is None:
            raise ValidationError(f"test64 directory not found: {PROJECT_ROOT / 'test64'}")
        target = test64_dir / normalized_filename
        if not target.exists() or not target.is_file():
            raise ValidationError(f"test64 image not found: {normalized_filename}")
        if not self._is_image_file(target):
            raise ValidationError(f"unsupported test64 file type: {normalized_filename}")
        return normalized_filename

    def _crop_truth_filenames(self, test64_dir: Path) -> list[str]:
        supported = sorted(path.name for path in test64_dir.iterdir() if path.is_file() and self._is_image_file(path))
        supported_set = set(supported)
        audit_first = [filename for filename in CROP_TRUTH_AUDIT_FILENAMES if filename in supported_set]
        remaining = [filename for filename in supported if filename not in CROP_TRUTH_AUDIT_FILENAMES]
        return audit_first + remaining

    async def _load_or_create_truth_bbox_manifest(self) -> Test64TruthBBoxManifestResponse:
        test64_dir = self._resolve_test64_dir()
        if test64_dir is None:
            raise ValidationError(f"test64 directory not found: {PROJECT_ROOT / 'test64'}")

        try:
            labels = await self.repo.list_test64_expected_dates()
        except Exception as exc:  # pragma: no cover - dev database may be offline for crop truth annotation
            logger.warning("test64 expected-date labels unavailable for crop truth manifest: %s", exc)
            labels = []
        labels_by_name = {row.filename: row for row in labels}
        existing = self._read_truth_bbox_manifest_payload()
        existing_items = existing.get("items") if isinstance(existing.get("items"), dict) else {}
        filenames = self._crop_truth_filenames(test64_dir)
        items: dict[str, Test64TruthBBoxItem] = {}
        for filename in filenames:
            path = test64_dir / filename
            if not path.exists() or not path.is_file():
                raise ValidationError(f"test64 image not found: {filename}")
            if not self._is_image_file(path):
                raise ValidationError(f"unsupported test64 file type: {filename}")
            with Image.open(path) as image:
                width, height = ImageOps.exif_transpose(image).size
            label = labels_by_name.get(filename)
            previous = existing_items.get(filename) if isinstance(existing_items, dict) else None
            bbox = previous.get("true_bbox_xyxy") if isinstance(previous, dict) else None
            polygon = previous.get("true_polygon_xy") if isinstance(previous, dict) else None
            rotation = previous.get("annotation_rotation_degrees") if isinstance(previous, dict) else None
            updated_at_raw = previous.get("updated_at") if isinstance(previous, dict) else None
            updated_at = None
            if isinstance(updated_at_raw, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at_raw.replace("Z", "+00:00"))
                except ValueError:
                    updated_at = None
            cleaned_bbox = self._validate_truth_bbox(bbox, image_width=width, image_height=height)
            cleaned_polygon = self._validate_truth_polygon(
                polygon if cleaned_bbox is not None else None,
                image_width=width,
                image_height=height,
            )
            cleaned_rotation = self._validate_truth_rotation(rotation) if cleaned_bbox is not None else None
            items[filename] = Test64TruthBBoxItem(
                filename=filename,
                image_url=f"/test64/images/{filename}",
                expected_day=label.expected_day if label is not None else None,
                expected_month=label.expected_month if label is not None else None,
                expected_year=label.expected_year if label is not None else None,
                image_width=width,
                image_height=height,
                true_bbox_xyxy=cleaned_bbox,
                true_polygon_xy=cleaned_polygon,
                annotation_rotation_degrees=cleaned_rotation,
                updated_at=updated_at,
            )

        payload = {
            "version": 1,
            "coordinate_space": CROP_TRUTH_COORDINATE_SPACE,
            "audit_set": filenames,
            "items": {name: value.model_dump(mode="json") for name, value in items.items()},
        }
        self._write_truth_bbox_manifest_payload(payload)
        annotated_count = sum(1 for item in items.values() if item.true_bbox_xyxy is not None)
        return Test64TruthBBoxManifestResponse(
            version=1,
            coordinate_space=CROP_TRUTH_COORDINATE_SPACE,
            audit_set=filenames,
            items=items,
            annotated_count=annotated_count,
            total_count=len(items),
            manifest_path=str(CROP_TRUTH_ANNOTATIONS_PATH),
        )

    @staticmethod
    def _read_truth_bbox_manifest_payload() -> dict[str, Any]:
        if not CROP_TRUTH_ANNOTATIONS_PATH.exists():
            return {}
        try:
            payload = json.loads(CROP_TRUTH_ANNOTATIONS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"crop truth bbox manifest is invalid JSON: {CROP_TRUTH_ANNOTATIONS_PATH}") from exc
        if not isinstance(payload, dict):
            raise ValidationError("crop truth bbox manifest must be a JSON object")
        return payload

    @staticmethod
    def _write_truth_bbox_manifest_payload(payload: dict[str, Any]) -> None:
        CROP_TRUTH_ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = CROP_TRUTH_ANNOTATIONS_PATH.with_suffix(f"{CROP_TRUTH_ANNOTATIONS_PATH.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_path, CROP_TRUTH_ANNOTATIONS_PATH)

    @staticmethod
    def _validate_truth_bbox(
        bbox_xyxy: list[float] | None,
        *,
        image_width: int,
        image_height: int,
    ) -> list[float] | None:
        if bbox_xyxy is None:
            return None
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
            raise ValidationError("true_bbox_xyxy must be [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        except (TypeError, ValueError) as exc:
            raise ValidationError("true_bbox_xyxy values must be numeric") from exc
        if x2 <= x1 or y2 <= y1:
            raise ValidationError("true_bbox_xyxy must describe a positive box")
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            raise ValidationError("true_bbox_xyxy must be within image bounds")
        return [x1, y1, x2, y2]

    @staticmethod
    def _validate_truth_polygon(
        polygon_xy: list[list[float]] | None,
        *,
        image_width: int,
        image_height: int,
    ) -> list[list[float]] | None:
        if polygon_xy is None:
            return None
        if not isinstance(polygon_xy, list) or len(polygon_xy) != 4:
            raise ValidationError("true_polygon_xy must contain four [x, y] points")

        cleaned: list[list[float]] = []
        for point in polygon_xy:
            if not isinstance(point, list) or len(point) != 2:
                raise ValidationError("true_polygon_xy must contain four [x, y] points")
            try:
                x, y = [float(value) for value in point]
            except (TypeError, ValueError) as exc:
                raise ValidationError("true_polygon_xy values must be numeric") from exc
            if x < 0 or y < 0 or x > image_width or y > image_height:
                raise ValidationError("true_polygon_xy must be within image bounds")
            cleaned.append([x, y])
        return cleaned

    @staticmethod
    def _validate_truth_rotation(rotation_degrees: float | None) -> float | None:
        if rotation_degrees is None:
            return None
        try:
            cleaned = float(rotation_degrees)
        except (TypeError, ValueError) as exc:
            raise ValidationError("annotation_rotation_degrees must be numeric") from exc
        if cleaned < -180 or cleaned > 180:
            raise ValidationError("annotation_rotation_degrees must be between -180 and 180")
        return cleaned

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

    async def list_scan_summaries(
        self,
        *,
        limit: int,
        status: ScanStatus | None,
        final_status: FinalResultStatus | None,
        user_id: str | None,
    ) -> list[ScanListItemResponse]:
        rows = await self.repo.list_scans(limit=limit, status=status, final_status=final_status, user_id=user_id)
        return [
            ScanListItemResponse(
                scan_id=item.scan_id,
                created_at=item.created_at,
                processed_at=item.processed_at,
                status=item.status,
                original_filename=item.original_filename,
                final_status=item.final_status,
                detected=item.detected,
                detector_confidence=item.detector_confidence,
                raw_text=item.raw_text,
                parsed_date=item.parsed_date,
                parse_confidence=item.parse_confidence,
                expiry_classification=item.expiry_classification,
                days_remaining=item.days_remaining,
                needs_review=item.needs_review,
                reason=item.reason,
                ocr_engine=item.ocr_engine,
                ocr_runtime_device=item.ocr_runtime_device,
                image_url=f"/scans/{item.scan_id}/image",
                roi_url=f"/scans/{item.scan_id}/roi" if item.has_roi else None,
                review_verdict=item.review.verdict if item.review else None,
                accepted_for_training=item.review.accepted_for_training if item.review else False,
                review=_review_payload(item.review),
                result=ScanResultPayload(
                    final_status=item.final_status,
                    detected=item.detected,
                    detector_confidence=item.detector_confidence,
                    raw_text=item.raw_text,
                    parsed_date=item.parsed_date,
                    date_format_detected=item.date_format_detected,
                    parse_confidence=item.parse_confidence,
                    expiry_classification=item.expiry_classification,
                    days_remaining=item.days_remaining,
                    alert_required=item.alert_required,
                    needs_review=item.needs_review,
                    reason=item.reason,
                    ocr_engine=item.ocr_engine,
                    ocr_runtime_device=item.ocr_runtime_device,
                )
                if item.final_status is not None or item.raw_text is not None or item.parsed_date is not None
                else None,
            )
            for item in rows
        ]

    async def get_scan_review_payload(self, scan_id: UUID) -> ScanReviewPayload | None:
        if await self.repo.get_scan(scan_id) is None:
            raise KeyError("scan not found")
        return _review_payload(await self.repo.get_scan_review(scan_id))

    async def upsert_scan_review(
        self,
        *,
        scan_id: UUID,
        verdict: ReviewVerdict,
        accepted_for_training: bool | None,
        final_text: str | None,
        final_parsed_date: date | None,
        bbox_xyxy: list[float] | None,
        bbox_source: str,
        reviewer_id: str,
        notes: str | None,
    ) -> ScanReviewPayload:
        scan = await self.repo.get_scan(scan_id)
        if scan is None:
            raise KeyError("scan not found")

        cleaned_text = final_text.strip() if isinstance(final_text, str) and final_text.strip() else None
        cleaned_notes = notes.strip()[:2000] if isinstance(notes, str) and notes.strip() else None
        cleaned_source = bbox_source.strip()[:64] if bbox_source.strip() else "original_image"
        cleaned_reviewer = reviewer_id.strip()[:128] if reviewer_id.strip() else "console"
        cleaned_bbox = self._validate_bbox(bbox_xyxy)
        if verdict == ReviewVerdict.INCORRECT and final_parsed_date is None:
            raise ValidationError("final_parsed_date is required when verdict is incorrect")
        if verdict == ReviewVerdict.INCORRECT and not cleaned_notes:
            raise ValidationError("notes is required when verdict is incorrect")
        if accepted_for_training is None:
            accepted_for_training = verdict in {ReviewVerdict.CORRECT, ReviewVerdict.INCORRECT}
        if accepted_for_training and not cleaned_text and final_parsed_date is None:
            raise ValidationError("final_text is required when accepted_for_training is true")

        review = await self.repo.upsert_scan_review(
            scan_id=scan.id,
            verdict=verdict,
            accepted_for_training=accepted_for_training,
            final_text=cleaned_text,
            final_parsed_date=final_parsed_date,
            bbox_xyxy=cleaned_bbox,
            bbox_source=cleaned_source,
            reviewer_id=cleaned_reviewer,
            notes=cleaned_notes,
        )
        await self.session.commit()
        payload = _review_payload(review)
        assert payload is not None
        return payload

    async def export_training_data(
        self,
        *,
        output_dir: str | None,
        include_detector: bool,
        include_recognition: bool,
    ) -> TrainingDataExportResponse:
        root = Path(output_dir).expanduser() if output_dir else self.storage.root / "training_exports" / "review_export"
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root.mkdir(parents=True, exist_ok=True)

        detector_images = root / "detector" / "images"
        detector_labels = root / "detector" / "labels"
        rec_images = root / "recognition" / "train_images"
        for directory in (detector_images, detector_labels, rec_images):
            directory.mkdir(parents=True, exist_ok=True)
        rec_label_path = root / "recognition" / "train_label.txt"
        metadata_path = root / "metadata.jsonl"

        accepted = await self.repo.accepted_reviews_for_export()
        skipped_by_reason: dict[str, int] = {}
        exported_detector = 0
        exported_recognition = 0
        rec_label_lines: list[str] = []

        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            for item in accepted:
                reason = None
                bbox = self._validate_bbox(item.review.bbox_xyxy)
                if bbox is None:
                    reason = "missing_bbox"
                elif not item.review.final_text:
                    reason = "missing_final_text"

                source_path = self._scan_image_absolute_path(item.scan)
                if reason is None and not source_path.exists():
                    reason = "missing_source_image"

                if reason is not None:
                    skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
                    continue

                assert bbox is not None
                with Image.open(source_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = image.size
                    x1, y1, x2, y2 = self._clamp_bbox(bbox, width, height)
                    if x2 <= x1 or y2 <= y1:
                        skipped_by_reason["invalid_clamped_bbox"] = skipped_by_reason.get("invalid_clamped_bbox", 0) + 1
                        continue

                    stem = str(item.scan.id)
                    image_ext = source_path.suffix.lower() or ".jpg"
                    detector_image_name = f"{stem}{image_ext}"
                    if include_detector:
                        detector_image_path = detector_images / detector_image_name
                        shutil.copyfile(source_path, detector_image_path)
                        cx = ((x1 + x2) / 2) / width
                        cy = ((y1 + y2) / 2) / height
                        bw = (x2 - x1) / width
                        bh = (y2 - y1) / height
                        (detector_labels / f"{stem}.txt").write_text(
                            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
                        )
                        exported_detector += 1

                    if include_recognition:
                        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                        rec_name = f"{stem}.jpg"
                        crop.save(rec_images / rec_name, format="JPEG", quality=95)
                        rec_label_lines.append(f"train_images/{rec_name}\t{item.review.final_text}")
                        exported_recognition += 1

                    metadata_file.write(
                        json.dumps(
                            {
                                "scan_id": str(item.scan.id),
                                "source_image_path": str(source_path),
                                "bbox_xyxy": [x1, y1, x2, y2],
                                "bbox_source": item.review.bbox_source,
                                "verdict": item.review.verdict,
                                "final_text": item.review.final_text,
                                "final_parsed_date": item.review.final_parsed_date.isoformat()
                                if item.review.final_parsed_date
                                else None,
                                "raw_text": item.ocr.raw_text if item.ocr else None,
                                "parsed_date": item.parsed.parsed_date.isoformat() if item.parsed and item.parsed.parsed_date else None,
                                "expiry_classification": item.expiry.status if item.expiry else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        if include_recognition:
            rec_label_path.write_text("\n".join(rec_label_lines) + ("\n" if rec_label_lines else ""), encoding="utf-8")

        return TrainingDataExportResponse(
            output_dir=str(root),
            accepted_reviews=len(accepted),
            exported_detector_labels=exported_detector,
            exported_recognition_crops=exported_recognition,
            skipped=sum(skipped_by_reason.values()),
            skipped_by_reason=skipped_by_reason,
        )

    async def get_scan_image_path(self, scan_id: UUID) -> str:
        scan = await self.repo.get_scan(scan_id)
        if scan is None:
            raise KeyError("scan not found")
        original_filename = None
        if isinstance(scan.metadata_json, dict):
            raw_name = scan.metadata_json.get("original_filename")
            if isinstance(raw_name, str) and raw_name.strip():
                original_filename = Path(raw_name).name
        if original_filename is None and scan.image_path and scan.image_path != "pending":
            original_filename = Path(scan.image_path).name
        if original_filename:
            test_images_dir = self._resolve_test_images_dir()
            if test_images_dir is not None:
                candidate = test_images_dir / original_filename
                if candidate.exists() and candidate.is_file():
                    return str(candidate)
        return scan.image_path

    def _scan_image_absolute_path(self, scan) -> Path:
        original_filename = None
        if isinstance(scan.metadata_json, dict):
            raw_name = scan.metadata_json.get("original_filename")
            if isinstance(raw_name, str) and raw_name.strip():
                original_filename = Path(raw_name).name
        if original_filename:
            test_images_dir = self._resolve_test_images_dir()
            if test_images_dir is not None:
                candidate = test_images_dir / original_filename
                if candidate.exists() and candidate.is_file():
                    return candidate
        return self.storage.absolute_path(scan.image_path)

    @staticmethod
    def _validate_bbox(bbox_xyxy: list[float] | None) -> list[float] | None:
        if bbox_xyxy is None:
            return None
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
            raise ValidationError("bbox_xyxy must be [x1, y1, x2, y2]")
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        except (TypeError, ValueError) as exc:
            raise ValidationError("bbox_xyxy values must be numeric") from exc
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValidationError("bbox_xyxy must describe a positive box in original-image coordinates")
        return [x1, y1, x2, y2]

    @staticmethod
    def _clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [
            max(0.0, min(x1, float(width - 1))),
            max(0.0, min(y1, float(height - 1))),
            max(0.0, min(x2, float(width))),
            max(0.0, min(y2, float(height))),
        ]

    async def get_scan_roi_path(self, scan_id: UUID) -> str:
        scan = await self.repo.get_scan(scan_id)
        if scan is None:
            raise KeyError("scan not found")
        if not scan.roi_path:
            raise FileNotFoundError("roi not available")
        return scan.roi_path

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

    def validate_upload(self, *, image_bytes: bytes, filename: str, content_type: str) -> None:
        self._validate_file(image_bytes=image_bytes, filename=filename, content_type=content_type)

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

    def sanitize_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any] | None:
        return self._sanitize_metadata(metadata)


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
