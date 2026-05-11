from __future__ import annotations

import asyncio
from datetime import date
from io import BytesIO
import json
from pathlib import Path
from uuid import UUID, uuid4

from litestar.testing import TestClient
from sqlalchemy import select

from app.ai.mobile_expiry_pipeline import MobileExpiryPipelineResult
from app.application import create_app
from app.domain.enums import ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.models import MobileExpiryScan
from app.domain.schemas import ScanResultPayload
from app.domain.services import CROP_TRUTH_AUDIT_FILENAMES, ScanService
from app.infra.db import get_session_factory


def _jpeg_bytes(width: int = 64, height: int = 40) -> bytes:
    from PIL import Image

    image = Image.new("RGB", (width, height), color=(240, 240, 240))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


class _FakeMobileExpiryPipeline:
    def __init__(self, result: MobileExpiryPipelineResult) -> None:
        self.result = result
        self.calls: list[bytes] = []

    def run(self, image_bytes: bytes, *, today: date) -> MobileExpiryPipelineResult:
        assert isinstance(today, date)
        self.calls.append(image_bytes)
        return self.result


async def _get_mobile_scan_row(scan_id: UUID) -> MobileExpiryScan | None:
    async with get_session_factory()() as session:
        return await session.scalar(select(MobileExpiryScan).where(MobileExpiryScan.id == scan_id))


def test_mobile_expiry_scan_success_persists_image_and_detected_date(monkeypatch) -> None:
    image_bytes = _jpeg_bytes()
    fake = _FakeMobileExpiryPipeline(
        MobileExpiryPipelineResult(
            status="parsed_success",
            detected_expiry_date=date(2027, 7, 23),
            raw_text="23.07.2027",
            normalized_text="23.07.2027",
            recognition_confidence=0.91,
            detector_confidence=0.82,
            reason="DMY;rotation=original",
            detection_polygon_json=[[1.0, 2.0], [30.0, 2.0], [30.0, 12.0], [1.0, 12.0]],
            runtime_ms=12,
        )
    )

    async def fake_get_pipeline(request):
        return fake

    monkeypatch.setattr("app.api.mobile_expiry._get_mobile_pipeline", fake_get_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/mobile/expiry-scans",
            data={"message": "front label", "metadata": json.dumps({"sku": "A-1"})},
            files={"image": ("mobile.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "parsed_success"
    assert payload["expiry_date"] == "2027-07-23"
    assert payload["detected_expiry_date"] == "2027-07-23"
    assert payload["corrected_expiry_date"] is None
    assert payload["raw_text"] == "23.07.2027"
    assert payload["recognition_confidence"] == 0.91
    assert fake.calls == [image_bytes]

    row = asyncio.run(_get_mobile_scan_row(UUID(payload["id"])))
    assert row is not None
    assert row.image_blob == image_bytes
    assert row.image_filename == "mobile.jpg"
    assert row.image_content_type == "image/jpeg"
    assert row.metadata_json == {"sku": "A-1"}
    assert row.message == "front label"


def test_mobile_expiry_scan_no_date_still_persists_for_review(monkeypatch) -> None:
    image_bytes = _jpeg_bytes()
    fake = _FakeMobileExpiryPipeline(
        MobileExpiryPipelineResult(
            status="manual_review_required",
            detected_expiry_date=None,
            raw_text=None,
            normalized_text=None,
            recognition_confidence=None,
            detector_confidence=None,
            reason="no expiry region detected",
            detection_polygon_json=None,
            runtime_ms=3,
        )
    )

    async def fake_get_pipeline(request):
        return fake

    monkeypatch.setattr("app.api.mobile_expiry._get_mobile_pipeline", fake_get_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/mobile/expiry-scans",
            files={"image": ("review.jpeg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "manual_review_required"
    assert payload["expiry_date"] is None
    assert payload["reason"] == "no expiry region detected"
    row = asyncio.run(_get_mobile_scan_row(UUID(payload["id"])))
    assert row is not None
    assert row.image_blob == image_bytes


def test_mobile_expiry_scan_rejects_invalid_upload() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/mobile/expiry-scans",
            files={"image": ("mobile.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"]


def test_mobile_expiry_scan_patch_correction_updates_effective_expiry_date(monkeypatch) -> None:
    fake = _FakeMobileExpiryPipeline(
        MobileExpiryPipelineResult(
            status="parsed_success",
            detected_expiry_date=date(2027, 7, 23),
            raw_text="23.07.2027",
            normalized_text="23.07.2027",
            recognition_confidence=0.91,
            detector_confidence=0.82,
            reason=None,
            detection_polygon_json=None,
            runtime_ms=4,
        )
    )

    async def fake_get_pipeline(request):
        return fake

    monkeypatch.setattr("app.api.mobile_expiry._get_mobile_pipeline", fake_get_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        created = client.post(
            "/mobile/expiry-scans",
            files={"image": ("mobile.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        scan_id = created.json()["id"]
        response = client.patch(
            f"/mobile/expiry-scans/{scan_id}",
            json={"corrected_expiry_date": "2028-01-02", "reason": "user checked package"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detected_expiry_date"] == "2027-07-23"
    assert payload["corrected_expiry_date"] == "2028-01-02"
    assert payload["expiry_date"] == "2028-01-02"
    assert payload["status"] == "corrected"
    assert payload["reason"] == "manual_correction:user checked package"


def test_mobile_expiry_scan_patch_missing_id_returns_404() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        response = client.patch(
            f"/mobile/expiry-scans/{uuid4()}",
            json={"corrected_expiry_date": "2028-01-02"},
        )

    assert response.status_code == 404


def test_create_scan_rejects_unsupported_extension() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/scans",
            data={"qr_code": "qr-1", "user_id": "user-1"},
            files={"image": ("scan.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"]


def test_create_scan_accepts_valid_upload() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/scans",
            data={"qr_code": "qr-2", "user_id": "user-2"},
            files={"image": ("scan.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["scan_id"]


def test_list_scans_returns_recent_scan_summaries_and_filters_by_user() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        first = client.post(
            "/scans",
            data={"qr_code": "qr-list-1", "user_id": "history-user-a"},
            files={"image": ("first.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )
        second = client.post(
            "/scans",
            data={"qr_code": "qr-list-2", "user_id": "history-user-b"},
            files={"image": ("second.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

        assert first.status_code == 201
        assert second.status_code == 201

        response = client.get("/history/scans?limit=10&user_id=history-user-b")

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["scan_id"] == second.json()["scan_id"]
    assert items[0]["status"] == "queued"
    assert items[0]["original_filename"] == "second.jpg"
    assert items[0]["image_url"] == f"/scans/{second.json()['scan_id']}/image"
    assert items[0]["roi_url"] is None
    assert items[0]["result"] is None


def test_scan_image_and_roi_preview_endpoints() -> None:
    image_bytes = _jpeg_bytes()
    app = create_app()
    with TestClient(app=app) as client:
        created = client.post(
            "/scans",
            data={"qr_code": "qr-preview", "user_id": "preview-user"},
            files={"image": ("ketchup.jpeg", image_bytes, "image/jpeg")},
        )
        scan_id = created.json()["scan_id"]

        image_response = client.get(f"/scans/{scan_id}/image")
        roi_response = client.get(f"/scans/{scan_id}/roi")

    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.content.startswith(b"\xff\xd8")
    assert roi_response.status_code == 404


def test_create_one_shot_scan_accepts_valid_upload_without_qr_or_user(monkeypatch) -> None:
    expected_scan_id: UUID = uuid4()

    class FakeRedis:
        async def close(self, close_connection_pool: bool = True) -> None:
            return None

    async def fake_create_redis_pool() -> FakeRedis:
        return FakeRedis()

    async def fake_create_scan(
        self: ScanService,
        *,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        qr_code: str,
        user_id: str,
        metadata: dict | None,
    ) -> UUID:
        assert image_bytes == b"\xff\xd8\xff\xd9"
        assert filename == "scan.jpg"
        assert content_type == "image/jpeg"
        assert qr_code.startswith("oneshot-")
        assert user_id == "oneshot"
        assert metadata is None
        return expected_scan_id

    async def fake_get_scan_payload(
        self: ScanService, scan_id: UUID
    ) -> tuple[ScanStatus, ScanResultPayload | None]:
        assert scan_id == expected_scan_id
        return (
            ScanStatus.DONE,
            ScanResultPayload(
                final_status=FinalResultStatus.PARSED_SUCCESS,
                detected=True,
                detector_confidence=0.99,
                raw_text="12/12/2026",
                parsed_date=date(2026, 12, 12),
                date_format_detected="dd/mm/yyyy",
                parse_confidence=0.92,
                expiry_classification=ExpiryClassification.SAFE,
                days_remaining=100,
                alert_required=False,
                needs_review=False,
                reason="test_ok",
                ocr_engine="ppocrv5_main",
                ocr_runtime_device="cpu",
            ),
        )

    monkeypatch.setattr(ScanService, "create_scan", fake_create_scan)
    monkeypatch.setattr(ScanService, "get_scan_payload", fake_get_scan_payload)
    monkeypatch.setattr("app.application.create_redis_pool", fake_create_redis_pool)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/scans/oneshot",
            files={"image": ("scan.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_id"] == str(expected_scan_id)
    assert payload["status"] == "done"
    assert payload["result"]["final_status"] == "parsed_success"
    assert payload["result"]["raw_text"] == "12/12/2026"


def test_scan_review_create_update_and_history_summary() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        created = client.post(
            "/scans",
            data={"qr_code": "qr-review", "user_id": "review-user"},
            files={"image": ("review.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        scan_id = created.json()["scan_id"]

        empty = client.get(f"/scans/{scan_id}/review")
        saved = client.put(
            f"/scans/{scan_id}/review",
            json={
                "verdict": "incorrect",
                "accepted_for_training": True,
                "final_text": "23.07.2026",
                "final_parsed_date": "2026-07-23",
                "bbox_xyxy": [10, 8, 40, 24],
                "notes": "corrected from console",
            },
        )
        history = client.get("/history/scans?limit=10&user_id=review-user")

    assert empty.status_code == 200
    assert empty.json()["review"] is None
    assert saved.status_code == 200
    assert saved.json()["review"]["verdict"] == "incorrect"
    assert saved.json()["review"]["accepted_for_training"] is True
    item = history.json()["items"][0]
    assert item["review_verdict"] == "incorrect"
    assert item["accepted_for_training"] is True
    assert item["review"]["final_text"] == "23.07.2026"


def test_scan_review_rejects_invalid_bbox() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        created = client.post(
            "/scans",
            data={"qr_code": "qr-review-invalid", "user_id": "review-user"},
            files={"image": ("review-invalid.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        scan_id = created.json()["scan_id"]
        response = client.put(
            f"/scans/{scan_id}/review",
            json={
                "verdict": "incorrect",
                "accepted_for_training": True,
                "final_text": "23.07.2026",
                "final_parsed_date": "2026-07-23",
                "bbox_xyxy": [40, 8, 10, 24],
                "notes": "bbox is intentionally invalid",
            },
        )

    assert response.status_code == 400
    assert "bbox" in response.json()["detail"]


def test_test64_truth_bboxes_initializes_manifest(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/truth-bboxes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == 1
    assert payload["coordinate_space"] == "original_image_xyxy"
    test64_image_count = sum(1 for path in Path("test64").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    assert len(payload["audit_set"]) == test64_image_count
    assert payload["annotated_count"] == 0
    assert payload["total_count"] == test64_image_count
    assert "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg" in payload["items"]
    assert "451fb754-ba88-47fa-854a-ca0dba7b14aa.jpg" in payload["items"]
    item = payload["items"]["65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"]
    assert item["true_bbox_xyxy"] is None
    assert item["image_width"] > 0
    assert item["image_height"] > 0
    oriented_item = payload["items"]["IMG_0897.JPG"]
    assert oriented_item["image_width"] == 3024
    assert oriented_item["image_height"] == 4032
    assert manifest_path.exists()


def test_test64_truth_bboxes_loads_when_expected_date_labels_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    async def fail_list_expected_dates(self):
        raise OSError("database unavailable")

    monkeypatch.setattr(
        "app.domain.repositories.DomainRepository.list_test64_expected_dates",
        fail_list_expected_dates,
    )

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/truth-bboxes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] > 0
    assert payload["annotated_count"] == 0
    assert manifest_path.exists()


def test_test64_truth_bboxes_saves_and_reloads_bbox(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    filename = "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"
    with TestClient(app=app) as client:
        saved = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": [10, 20, 110, 55]})
        reloaded = client.get("/test64/truth-bboxes")

    assert saved.status_code == 200
    assert saved.json()["item"]["true_bbox_xyxy"] == [10.0, 20.0, 110.0, 55.0]
    assert saved.json()["annotated_count"] == 1
    assert reloaded.json()["items"][filename]["true_bbox_xyxy"] == [10.0, 20.0, 110.0, 55.0]
    assert reloaded.json()["annotated_count"] == 1


def test_test64_truth_bboxes_saves_and_reloads_rotated_polygon(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    filename = "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"
    polygon = [[10, 20], [120, 28], [115, 62], [8, 54]]
    with TestClient(app=app) as client:
        saved = client.put(
            f"/test64/truth-bboxes/{filename}",
            json={
                "true_bbox_xyxy": [8, 20, 120, 62],
                "true_polygon_xy": polygon,
                "annotation_rotation_degrees": -7.5,
            },
        )
        reloaded = client.get("/test64/truth-bboxes")

    assert saved.status_code == 200
    assert saved.json()["item"]["true_bbox_xyxy"] == [8.0, 20.0, 120.0, 62.0]
    assert saved.json()["item"]["true_polygon_xy"] == [[10.0, 20.0], [120.0, 28.0], [115.0, 62.0], [8.0, 54.0]]
    assert saved.json()["item"]["annotation_rotation_degrees"] == -7.5
    assert reloaded.json()["items"][filename]["true_polygon_xy"] == [[10.0, 20.0], [120.0, 28.0], [115.0, 62.0], [8.0, 54.0]]
    assert reloaded.json()["annotated_count"] == 1


def test_test64_truth_bboxes_preserves_existing_audit_labels_when_expanding_to_all_test64(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    filename = "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinate_space": "original_image_xyxy",
                "audit_set": list(CROP_TRUTH_AUDIT_FILENAMES),
                "items": {
                    filename: {
                        "filename": filename,
                        "image_url": f"/test64/images/{filename}",
                        "expected_day": 12,
                        "expected_month": 12,
                        "expected_year": 2027,
                        "image_width": 3024,
                        "image_height": 4032,
                        "true_bbox_xyxy": [10, 20, 110, 55],
                        "updated_at": "2026-05-06T15:20:00Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/truth-bboxes")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][filename]["true_bbox_xyxy"] == [10.0, 20.0, 110.0, 55.0]
    assert payload["annotated_count"] == 1
    assert "451fb754-ba88-47fa-854a-ca0dba7b14aa.jpg" in payload["items"]
    assert payload["total_count"] > len(CROP_TRUTH_AUDIT_FILENAMES)


def test_test64_truth_bboxes_saves_non_audit_test64_image(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)
    filename = "451fb754-ba88-47fa-854a-ca0dba7b14aa.jpg"
    assert filename not in CROP_TRUTH_AUDIT_FILENAMES

    app = create_app()
    with TestClient(app=app) as client:
        saved = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": [10, 20, 110, 55]})
        reloaded = client.get("/test64/truth-bboxes")

    assert saved.status_code == 200
    assert saved.json()["item"]["true_bbox_xyxy"] == [10.0, 20.0, 110.0, 55.0]
    assert reloaded.json()["items"][filename]["true_bbox_xyxy"] == [10.0, 20.0, 110.0, 55.0]


def test_test64_truth_bboxes_clears_bbox(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    filename = "IMG_0905.JPG"
    with TestClient(app=app) as client:
        saved = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": [5, 6, 90, 40]})
        cleared = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": None})

    assert saved.status_code == 200
    assert cleared.status_code == 200
    assert cleared.json()["item"]["true_bbox_xyxy"] is None
    assert cleared.json()["annotated_count"] == 0


def test_test64_truth_bboxes_rejects_invalid_bbox(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    filename = "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"
    with TestClient(app=app) as client:
        inverted = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": [50, 20, 10, 55]})
        out_of_bounds = client.put(f"/test64/truth-bboxes/{filename}", json={"true_bbox_xyxy": [-1, 20, 110, 55]})
        unsupported = client.put("/test64/truth-bboxes/not-in-audit.jpg", json={"true_bbox_xyxy": [1, 2, 3, 4]})

    assert inverted.status_code == 400
    assert "positive" in inverted.json()["detail"]
    assert out_of_bounds.status_code == 400
    assert "within image bounds" in out_of_bounds.json()["detail"]
    assert unsupported.status_code == 400
    assert "test64 image not found" in unsupported.json()["detail"]


def test_test64_truth_bboxes_rejects_invalid_polygon(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "truth" / "test64_truth_bboxes.json"
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", manifest_path)

    app = create_app()
    filename = "65940c3a-9719-4b05-8c3e-3de89f8b300c.jpg"
    with TestClient(app=app) as client:
        wrong_point_count = client.put(
            f"/test64/truth-bboxes/{filename}",
            json={"true_bbox_xyxy": [10, 20, 110, 55], "true_polygon_xy": [[10, 20], [110, 20], [110, 55]]},
        )
        out_of_bounds = client.put(
            f"/test64/truth-bboxes/{filename}",
            json={"true_bbox_xyxy": [10, 20, 110, 55], "true_polygon_xy": [[10, 20], [110, 20], [110, 55], [-1, 55]]},
        )

    assert wrong_point_count.status_code == 400
    assert "true_polygon_xy" in wrong_point_count.json()["detail"]
    assert out_of_bounds.status_code == 400
    assert "within image bounds" in out_of_bounds.json()["detail"]


def test_manual_crop_recognition_review_loads_latest_failure_report(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    run_dir = root / "manual_crop_recognition_svtrv2_20260509T102021Z"
    crop_dir = run_dir / "images" / "sample" / "variants"
    crop_dir.mkdir(parents=True)
    (run_dir / "images" / "sample" / "manual_true_crop.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (crop_dir / "original_color_tight.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (crop_dir / "adaptive_binary.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    report = {
        "generated_at": "2026-05-09T10:21:38Z",
        "report_root": str(run_dir),
        "recognizer_config": {"recognizer": "svtrv2"},
        "summary": {
            "total_crops": 2,
            "exact_match_crops": 1,
            "outcome_counts": {
                "success_without_preprocessing_requirement": 1,
                "wrong_date_from_crop": 1,
            },
        },
        "items": [
            {
                "filename": "ok.jpg",
                "expected_date": "2026-01-01",
                "outcome": "success_without_preprocessing_requirement",
                "manual_true_crop_path": str(run_dir / "images" / "sample" / "manual_true_crop.png"),
                "winner": {"exact_match": True, "variant_name": "original_color_tight"},
                "variants": [],
            },
            {
                "filename": "bad.jpg",
                "expected_date": "2026-02-02",
                "outcome": "wrong_date_from_crop",
                "manual_true_crop_path": str(run_dir / "images" / "sample" / "manual_true_crop.png"),
                "winner": {
                    "exact_match": False,
                    "variant_name": "adaptive_binary",
                    "parseq_raw_output": "02/02/2025",
                    "parser_parsed_date": "2025-02-02",
                    "selected_orientation": "rotate_90_cw",
                    "crop_transform_used": "rotate_90_cw",
                    "orientation_candidates_tried": ["original", "rotate_90_cw", "rotate_90_ccw", "rotate_180"],
                },
                "variants": [
                    {
                        "variant_name": "original_color_tight",
                        "crop_path": str(crop_dir / "original_color_tight.png"),
                        "parseq_raw_output": "TEXT",
                        "parser_parsed_date": None,
                        "exact_match": False,
                    },
                    {
                        "variant_name": "adaptive_binary",
                        "crop_path": str(crop_dir / "adaptive_binary.png"),
                        "parseq_raw_output": "02/02/2025",
                        "parser_parsed_date": "2025-02-02",
                        "exact_match": False,
                    },
                ],
            },
        ],
    }
    (run_dir / "manual_crop_recognition_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("app.domain.services.MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/manual-crop-recognition-review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_dir.name
    assert payload["summary"]["exact_match_crops"] == 1
    assert payload["filtered_count"] == 1
    assert payload["items"][0]["filename"] == "bad.jpg"
    assert payload["items"][0]["manual_true_crop_url"].startswith("/test64/manual-crop-recognition-review/assets")
    assert payload["items"][0]["variants"][0]["crop_url"].startswith("/test64/manual-crop-recognition-review/assets")
    assert payload["items"][0]["winner"]["selected_orientation"] == "rotate_90_cw"
    assert payload["items"][0]["winner"]["orientation_candidates_tried"] == [
        "original",
        "rotate_90_cw",
        "rotate_90_ccw",
        "rotate_180",
    ]


def test_manual_crop_recognition_review_falls_back_when_report_root_is_host_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "forensics"
    run_dir = root / "manual_crop_recognition_svtrv2_20260509T125233Z"
    crop_dir = run_dir / "images" / "sample" / "variants" / "rotate_90_cw"
    crop_dir.mkdir(parents=True)
    crop_path = crop_dir / "original_color_tight.png"
    crop_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    manual_path = run_dir / "images" / "sample" / "manual_true_crop.png"
    manual_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    host_root = Path("/Volumes/wipesides/VSCode/smartbite/artifacts/forensics") / run_dir.name
    report = {
        "generated_at": "2026-05-09T12:52:33Z",
        "report_root": str(host_root),
        "summary": {"total_crops": 1, "exact_match_crops": 1},
        "items": [
            {
                "filename": "rotated.jpg",
                "expected_date": "2026-01-01",
                "outcome": "success_without_preprocessing_requirement",
                "manual_true_crop_path": str(host_root / "images" / "sample" / "manual_true_crop.png"),
                "winner": {
                    "variant_name": "rotate_90_cw/original_color_tight",
                    "crop_path": str(host_root / "images" / "sample" / "variants" / "rotate_90_cw" / "original_color_tight.png"),
                    "exact_match": True,
                    "selected_orientation": "rotate_90_cw",
                },
                "variants": [],
            }
        ],
    }
    (run_dir / "manual_crop_recognition_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("app.domain.services.MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/manual-crop-recognition-review?include_success=true")

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["manual_true_crop_url"].endswith("manual_true_crop.png")
    assert item["winner"]["crop_url"].endswith("rotate_90_cw/original_color_tight.png")


def test_manual_crop_recognition_review_can_include_successes(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    run_dir = root / "manual_crop_recognition_svtrv2_20260509T102021Z"
    run_dir.mkdir(parents=True)
    report = {
        "generated_at": "2026-05-09T10:21:38Z",
        "report_root": str(run_dir),
        "summary": {"total_crops": 1, "exact_match_crops": 1, "outcome_counts": {"success_without_preprocessing_requirement": 1}},
        "items": [{"filename": "ok.jpg", "outcome": "success_without_preprocessing_requirement", "winner": {"exact_match": True}, "variants": []}],
    }
    (run_dir / "manual_crop_recognition_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("app.domain.services.MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/manual-crop-recognition-review?include_success=true")

    assert response.status_code == 200
    assert response.json()["filtered_count"] == 1
    assert response.json()["items"][0]["filename"] == "ok.jpg"


def test_manual_crop_recognition_review_asset_rejects_path_traversal(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    run_dir = root / "manual_crop_recognition_svtrv2_20260509T102021Z"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr("app.domain.services.MANUAL_CROP_RECOGNITION_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get(f"/test64/manual-crop-recognition-review/assets?run_id={run_dir.name}&path=../secret.png")

    assert response.status_code == 400
    assert "invalid artifact path" in response.json()["detail"]


def test_test64_full_pipeline_review_loads_latest_plain_report_with_detector_details(
    monkeypatch, tmp_path: Path
) -> None:
    root = tmp_path / "forensics"
    older_dir = root / "test64_plain_20260509T082438Z"
    run_dir = root / "test64_plain_20260510T082438Z"
    older_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (older_dir / "plain_test64_report.json").write_text(
        json.dumps({"evaluation": {"summary": {"total": 0}, "per_scan": []}}),
        encoding="utf-8",
    )
    report = {
        "source": "test64-bulk-queue",
        "text_detector_mode": "ensemble",
        "detector_config": {
            "text_detector_mode": "ppocrv5_server",
            "ppocrv5_effective_text_det_model_dir": "models/pp-ocrv5-text-detection/best_model_inference",
            "craft_enabled": False,
        },
        "detector_totals": {
            "ppocrv5_server_boxes": 3,
            "craft_boxes": 5,
            "ensemble_boxes": 2,
            "variants": 2,
            "performance": {"detected_boxes_after_dedupe": 8, "total_runtime_ms": 4321},
        },
        "evaluation": {
            "summary": {"total": 2, "correct_guess": 1, "recognition_fail": 1},
            "parseable_candidates_total": 1,
            "average_runtime_ms_per_scan": 1234.5,
            "per_scan": [
                {
                    "filename": "ok.jpg",
                    "verdict": "correct_guess",
                    "predicted_date": "2026-01-01",
                    "final_status": "parsed_success",
                    "reason": "selected_source=detector_box_1",
                    "parseable_candidates": 1,
                    "selected_candidates": 3,
                    "runtime_ms": 1000,
                },
                {
                    "filename": "fail.jpg",
                    "verdict": "recognition_fail",
                    "predicted_date": None,
                    "final_status": "manual_review_required",
                    "reason": "manual_review_no_parse",
                    "parseable_candidates": 0,
                    "selected_candidates": 1,
                    "runtime_ms": 1500,
                },
            ],
        },
        "per_scan_detector_stats": [
            {
                "filename": "ok.jpg",
                "ppocrv5_server_boxes": 2,
                "craft_boxes": 4,
                "ensemble_boxes": 1,
                "variants": 1,
                "detector_modes": {"ensemble": 1},
                "performance": {"detected_boxes_after_dedupe": 5, "context_probe_call_count": 2},
                "detector_boxes": [
                    {
                        "bbox_xyxy": [10, 20, 110, 50],
                        "polygon_xy": [[10, 20], [110, 20], [110, 50], [10, 50]],
                        "confidence": 0.91,
                        "source": "ppocrv5_server",
                        "sources": ["ppocrv5_server"],
                        "variant_name": "original",
                    }
                ],
            },
            {
                "filename": "fail.jpg",
                "ppocrv5_server_boxes": 1,
                "craft_boxes": 1,
                "ensemble_boxes": 1,
                "variants": 1,
                "detector_modes": {"ensemble": 1},
                "performance": {"detected_boxes_after_dedupe": 3, "context_probe_call_count": 0},
            },
        ],
    }
    (run_dir / "plain_test64_report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("app.domain.services.TEST64_DETECTION_REVIEW_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/full-pipeline-review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_dir.name
    assert payload["summary"]["correct_guess"] == 1
    assert payload["detector_config"]["craft_enabled"] is False
    assert payload["detector_totals"]["ppocrv5_server_boxes"] == 3
    assert payload["items"][0]["filename"] == "ok.jpg"
    assert payload["items"][0]["image_url"] == "/test64/images/ok.jpg"
    assert payload["items"][0]["detector_counts"]["ppocrv5_server_boxes"] == 2
    assert payload["items"][0]["detector_performance"]["detected_boxes_after_dedupe"] == 5
    assert payload["items"][0]["detector_boxes"] == [
        {
            "bbox_xyxy": [10, 20, 110, 50],
            "polygon_xy": [[10, 20], [110, 20], [110, 50], [10, 50]],
            "confidence": 0.91,
            "source": "ppocrv5_server",
            "sources": ["ppocrv5_server"],
            "variant_name": "original",
        }
    ]


def test_test64_detection_review_includes_latest_detector_audit(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    truth_manifest_path = tmp_path / "test64_truth_bboxes.json"
    plain_dir = root / "test64_plain_20260510T082438Z"
    audit_dir = root / "test64_detector_audit_20260510T092438Z"
    plain_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    truth_manifest_path.write_text(
        json.dumps(
            {
                "items": {
                    "ok.jpg": {
                        "true_polygon_xy": [[10, 20], [110, 21], [108, 50], [9, 49]],
                        "annotation_rotation_degrees": 12.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (plain_dir / "plain_test64_report.json").write_text(
        json.dumps(
            {
                "source": "test64-bulk-queue",
                "text_detector_mode": "ppocrv5_server",
                "evaluation": {
                    "summary": {"total": 1},
                    "per_scan": [{"filename": "ok.jpg", "verdict": "recognition_fail"}],
                },
                "per_scan_detector_stats": [{"filename": "ok.jpg", "detector_boxes": []}],
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "detector_truth_ablation_report.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-10T09:24:38Z",
                "summary": {"configs": {"recall_1216": {"covered": 1}}},
                "configs": [{"name": "recall_1216", "settings": {"ppocrv5_det_limit_side_len": 1216}}],
                "items": [
                    {
                        "filename": "ok.jpg",
                        "truth_bbox_xyxy": [10, 20, 110, 50],
                        "configs": [
                            {
                                "config_name": "recall_1216",
                                "verdict": "covered",
                                "box_count": 1,
                                "best_iou": 0.9,
                                "truth_coverage_ratio": 1.0,
                                "best_box": {"bbox_xyxy": [10, 20, 110, 50]},
                                "detector_boxes": [{"bbox_xyxy": [10, 20, 110, 50]}],
                                "runtime_ms": 123.0,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.domain.services.TEST64_DETECTION_REVIEW_ARTIFACTS_DIR", root)
    monkeypatch.setattr("app.domain.services.CROP_TRUTH_ANNOTATIONS_PATH", truth_manifest_path)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/detection-review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == audit_dir.name
    assert payload["items"][0]["truth_bbox_xyxy"] == [10, 20, 110, 50]
    assert payload["items"][0]["truth_polygon_xy"] == [[10, 20], [110, 21], [108, 50], [9, 49]]
    assert payload["items"][0]["truth_annotation_rotation_degrees"] == 12.5
    assert payload["items"][0]["configs"][0]["verdict"] == "covered"


def test_test64_detection_review_is_detector_audit_only(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    plain_dir = root / "test64_plain_20260510T082438Z"
    audit_dir = root / "test64_detector_audit_20260510T092438Z"
    plain_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    (plain_dir / "plain_test64_report.json").write_text(
        json.dumps(
            {
                "evaluation": {
                    "summary": {"recognition_fail": 1},
                    "per_scan": [{"filename": "ok.jpg", "verdict": "recognition_fail"}],
                },
                "per_scan_detector_stats": [{"filename": "ok.jpg", "detector_boxes": []}],
            }
        ),
        encoding="utf-8",
    )
    (audit_dir / "detector_truth_ablation_report.json").write_text(
        json.dumps(
            {
                "summary": {"configs": {"recall_1216_unclip18": {"covered": 1}}},
                "configs": [{"name": "recall_1216_unclip18", "settings": {"ppocrv5_det_limit_side_len": 1216}}],
                "items": [
                    {
                        "filename": "ok.jpg",
                        "truth_bbox_xyxy": [10, 20, 110, 50],
                        "configs": [{"config_name": "recall_1216_unclip18", "verdict": "covered"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.domain.services.TEST64_DETECTION_REVIEW_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/detection-review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == audit_dir.name
    assert payload["items"][0]["filename"] == "ok.jpg"
    assert payload["items"][0]["image_url"] == "/test64/images/ok.jpg"
    assert "detector_audit" not in payload
    assert "detector_config" not in payload
    assert "recognition_fail" not in json.dumps(payload)


def test_test64_full_pipeline_review_loads_latest_plain_report(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "forensics"
    run_dir = root / "test64_plain_20260510T082438Z"
    run_dir.mkdir(parents=True)
    (run_dir / "plain_test64_report.json").write_text(
        json.dumps(
            {
                "source": "test64-bulk-queue",
                "text_detector_mode": "ppocrv5_server",
                "detector_config": {"craft_enabled": False},
                "detector_totals": {"ppocrv5_server_boxes": 2},
                "evaluation": {
                    "summary": {"total": 1, "recognition_fail": 1},
                    "parseable_candidates_total": 0,
                    "average_runtime_ms_per_scan": 1234.5,
                    "per_scan": [
                        {
                            "filename": "ok.jpg",
                            "verdict": "recognition_fail",
                            "predicted_date": None,
                            "final_status": "manual_review_required",
                        }
                    ],
                },
                "per_scan_detector_stats": [
                    {
                        "filename": "ok.jpg",
                        "ppocrv5_server_boxes": 2,
                        "craft_boxes": 0,
                        "ensemble_boxes": 0,
                        "variants": 1,
                        "performance": {"detected_boxes_after_dedupe": 2},
                        "detector_boxes": [{"bbox_xyxy": [10, 20, 110, 50]}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.domain.services.TEST64_DETECTION_REVIEW_ARTIFACTS_DIR", root)

    app = create_app()
    with TestClient(app=app) as client:
        response = client.get("/test64/full-pipeline-review")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_dir.name
    assert payload["summary"]["recognition_fail"] == 1
    assert payload["detector_config"]["craft_enabled"] is False
    assert payload["items"][0]["verdict"] == "recognition_fail"
    assert payload["items"][0]["detector_counts"]["ppocrv5_server_boxes"] == 2


def test_general_text_endpoint_is_not_registered() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        response = client.post(
            "/ocr/general-text",
            files={"image": ("ocr.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )

    assert response.status_code == 404


def test_scan_review_incorrect_requires_date_and_feedback() -> None:
    app = create_app()
    with TestClient(app=app) as client:
        created = client.post(
            "/scans",
            data={"qr_code": "qr-review-rules", "user_id": "review-user"},
            files={"image": ("review-rules.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        scan_id = created.json()["scan_id"]
        missing_date = client.put(
            f"/scans/{scan_id}/review",
            json={
                "verdict": "incorrect",
                "accepted_for_training": False,
                "notes": "text was wrong",
            },
        )
        missing_notes = client.put(
            f"/scans/{scan_id}/review",
            json={
                "verdict": "incorrect",
                "accepted_for_training": False,
                "final_parsed_date": "2026-07-23",
            },
        )

    assert missing_date.status_code == 400
    assert "final_parsed_date" in missing_date.json()["detail"]
    assert missing_notes.status_code == 400
    assert "notes" in missing_notes.json()["detail"]


def test_training_data_export_creates_detector_and_recognition_artifacts(tmp_path: Path) -> None:
    app = create_app()
    output_dir = tmp_path / "export"
    with TestClient(app=app) as client:
        created = client.post(
            "/scans",
            data={"qr_code": "qr-export", "user_id": "export-user"},
            files={"image": ("export-source.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        scan_id = created.json()["scan_id"]
        review = client.put(
            f"/scans/{scan_id}/review",
            json={
                "verdict": "correct",
                "accepted_for_training": True,
                "final_text": "23.07.2026",
                "bbox_xyxy": [4, 5, 44, 25],
            },
        )
        response = client.post(
            "/training-data/export",
            json={"output_dir": str(output_dir), "include_detector": True, "include_recognition": True},
        )

    assert review.status_code == 200
    assert response.status_code == 201
    payload = response.json()
    assert payload["accepted_reviews"] >= 1
    assert payload["exported_detector_labels"] >= 1
    assert payload["exported_recognition_crops"] >= 1
    assert (output_dir / "detector" / "labels" / f"{scan_id}.txt").exists()
    assert (output_dir / "recognition" / "train_images" / f"{scan_id}.jpg").exists()
    assert f"train_images/{scan_id}.jpg\t23.07.2026" in (
        output_dir / "recognition" / "train_label.txt"
    ).read_text()
    assert str(scan_id) in (output_dir / "metadata.jsonl").read_text()
