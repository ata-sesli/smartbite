from __future__ import annotations

import json
from pathlib import Path
import asyncio

from argparse import Namespace

from app.scripts.mobile_test64_direct_benchmark import (
    _blind_expected_label,
    _expected_labels_from_report,
    _filenames_from_non_exact_report,
    _labels_for_benchmark,
    _run_prefix,
    _settings_for_benchmark,
)


def test_direct_benchmark_filters_non_exact_rows_from_prior_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {"filename": "correct.jpg", "exact_match": True},
                    {"filename": "wrong.jpg", "exact_match": False},
                    {"filename": "manual.jpg", "exact_match": False, "response_status": "manual_review_required"},
                    {"filename": "", "exact_match": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _filenames_from_non_exact_report(report) == {"wrong.jpg", "manual.jpg"}


def test_direct_benchmark_can_load_expected_labels_from_prior_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "filename": "month.jpg",
                        "expected_day": None,
                        "expected_month": 10,
                        "expected_year": 2026,
                    },
                    {
                        "filename": "day.jpg",
                        "expected_day": 3,
                        "expected_month": 5,
                        "expected_year": 2027,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    labels = _expected_labels_from_report(report)

    assert labels["month.jpg"]["expected_day"] is None
    assert labels["month.jpg"]["expected_month"] == 10
    assert labels["day.jpg"]["expected_day"] == 3


def test_direct_benchmark_can_prefer_expected_labels_from_report(monkeypatch, tmp_path: Path) -> None:
    from app.scripts import mobile_test64_direct_benchmark as benchmark

    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "filename": "day.jpg",
                        "expected_day": 12,
                        "expected_month": 5,
                        "expected_year": 2026,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    async def fake_load_expected_labels():
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(benchmark, "_load_expected_labels", fake_load_expected_labels)

    labels = asyncio.run(
        _labels_for_benchmark(
            Namespace(blind=False, expected_from_report=str(report)),
            report,
        )
    )

    assert labels["day.jpg"]["expected_day"] == 12
    assert calls == 0


def test_direct_benchmark_supports_blind_expected_labels() -> None:
    expected = _blind_expected_label()

    assert expected["label"] is None
    assert expected["precision"] is None
    assert expected["day"] is None
    assert expected["month"] is None
    assert expected["year"] is None


def test_direct_benchmark_uses_test_images_run_prefix() -> None:
    assert _run_prefix(Path("test-images")) == "test_images_direct"
    assert _run_prefix(Path("test64")) == "test64_mobile_direct"


def test_direct_benchmark_can_enable_product_cropper_rescue_without_changing_default() -> None:
    settings = _settings_for_benchmark(
        Namespace(
            enable_product_cropper_rescue=True,
            enable_legacy_wide_group_rescue=False,
            enable_strict_evidence_acceptance=False,
            enable_rapidocr_role_constraint=False,
        )
    )

    assert settings.mobile_product_cropper_rescue_enabled is True
    assert settings.mobile_strict_evidence_acceptance_enabled is True


def test_direct_benchmark_can_set_mobile_rapidocr_primary_mode() -> None:
    settings = _settings_for_benchmark(
        Namespace(
            enable_product_cropper_rescue=False,
            enable_legacy_wide_group_rescue=False,
            enable_strict_evidence_acceptance=False,
            enable_rapidocr_role_constraint=False,
            enable_production_anchor_sibling=False,
            enable_paired_crop_evidence=False,
            enable_local_group_wide_crop=False,
            enable_rotation_wide_crop=False,
            enable_hardcase_recognizer_rescue=False,
            hardcase_rec_model_dir=None,
            rotation_rescue_max_candidates=None,
            rapidocr_primary_mode="always",
        )
    )

    assert settings.mobile_rapidocr_primary_mode == "always"
