from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.infra.storage import LocalStorage


def test_storage_path_generation(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    scan_id = UUID("12345678-1234-5678-1234-567812345678")

    raw = storage.raw_relative_path(scan_id, ".JPG")
    roi = storage.roi_relative_path(scan_id)

    assert raw == "raw/12345678-1234-5678-1234-567812345678.jpg"
    assert roi == "roi/12345678-1234-5678-1234-567812345678.png"
