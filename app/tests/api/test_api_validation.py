from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

from litestar.testing import TestClient

from app.application import create_app
from app.domain.enums import ExpiryClassification, FinalResultStatus, ScanStatus
from app.domain.schemas import ScanResultPayload
from app.domain.services import ScanService


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


def test_create_one_shot_scan_accepts_valid_upload_without_qr_or_user(monkeypatch) -> None:
    expected_scan_id: UUID = uuid4()

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
