from __future__ import annotations

from litestar.testing import TestClient

from app.application import create_app


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
