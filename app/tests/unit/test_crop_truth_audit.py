from __future__ import annotations

import json

from app.domain.services import CROP_TRUTH_AUDIT_FILENAMES
from app.scripts.crop_truth_audit import load_truth_manifest, validate_truth_manifest


def test_crop_truth_audit_accepts_complete_website_manifest(tmp_path) -> None:
    path = tmp_path / "test64_truth_bboxes.json"
    payload = {
        "version": 1,
        "coordinate_space": "original_image_xyxy",
        "audit_set": list(CROP_TRUTH_AUDIT_FILENAMES),
        "items": {
            filename: {
                "filename": filename,
                "true_bbox_xyxy": [10, 20, 110, 55],
            }
            for filename in CROP_TRUTH_AUDIT_FILENAMES
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_truth_manifest(path)

    assert validate_truth_manifest(loaded) == []


def test_crop_truth_audit_reports_missing_bboxes(tmp_path) -> None:
    path = tmp_path / "test64_truth_bboxes.json"
    missing_filename = CROP_TRUTH_AUDIT_FILENAMES[0]
    payload = {
        "version": 1,
        "coordinate_space": "original_image_xyxy",
        "audit_set": list(CROP_TRUTH_AUDIT_FILENAMES),
        "items": {
            filename: {
                "filename": filename,
                "true_bbox_xyxy": None if filename == missing_filename else [10, 20, 110, 55],
            }
            for filename in CROP_TRUTH_AUDIT_FILENAMES
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_truth_manifest(path)

    assert validate_truth_manifest(loaded) == [missing_filename]
