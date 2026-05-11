from __future__ import annotations

from pathlib import Path
from uuid import UUID


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def raw_relative_path(self, scan_id: UUID, extension: str) -> str:
        ext = extension if extension.startswith(".") else f".{extension}"
        return f"raw/{scan_id}{ext.lower()}"

    def roi_relative_path(self, scan_id: UUID) -> str:
        return f"roi/{scan_id}.png"

    def debug_candidates_relative_path(self, scan_id: UUID) -> str:
        return f"debug/{scan_id}/candidates.json"

    def debug_overlay_relative_path(self, scan_id: UUID) -> str:
        return f"debug/{scan_id}/overlay.png"

    def absolute_path(self, relative_path: str) -> Path:
        return self.root / relative_path

    def save_bytes(self, relative_path: str, payload: bytes) -> str:
        target = self.absolute_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return relative_path

    def read_bytes(self, relative_path: str) -> bytes:
        return self.absolute_path(relative_path).read_bytes()
