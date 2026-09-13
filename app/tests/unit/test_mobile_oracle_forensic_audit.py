from __future__ import annotations

from datetime import date

from app.scripts.mobile_oracle_forensic_audit import (
    _bbox_overlap_metrics,
    _classify_row,
    _classify_truth_overlap,
    _expected_date_matches,
)


def test_expected_date_matches_day_exact_only() -> None:
    expected = {"precision": "day", "year": 2026, "month": 5, "day": 19}

    assert _expected_date_matches(date(2026, 5, 19), expected)
    assert not _expected_date_matches(date(2026, 5, 31), expected)


def test_expected_date_matches_month_by_year_and_month() -> None:
    expected = {"precision": "month", "year": 2026, "month": 10, "day": None}

    assert _expected_date_matches(date(2026, 10, 1), expected)
    assert _expected_date_matches(date(2026, 10, 31), expected)
    assert not _expected_date_matches(date(2026, 11, 1), expected)


def test_truth_overlap_classifies_covered_partial_and_missed() -> None:
    truth = (100, 100, 200, 140)

    assert _classify_truth_overlap(_bbox_overlap_metrics((90, 90, 210, 150), truth)) == "covered"
    assert _classify_truth_overlap(_bbox_overlap_metrics((180, 100, 280, 140), truth)) == "partial"
    assert _classify_truth_overlap(_bbox_overlap_metrics((250, 100, 320, 140), truth)) == "missed"


def test_row_classification_prefers_ambiguous_month_before_selector() -> None:
    row = {
        "expected": {"precision": "month", "year": 2029, "month": 3, "day": None},
        "truth": {"valid": True},
        "selected_date": "2026-03-07",
        "expected_in_accepted": True,
        "expected_before_filter": True,
        "truth_overlap": {"best_verdict": "covered"},
        "truth_crop_oracle": {"reliable": True},
    }

    assert _classify_row(row) == "E_ambiguous_month_only"


def test_row_classification_distinguishes_selector_filter_recognition_and_recall() -> None:
    base = {
        "expected": {"precision": "day", "year": 2027, "month": 5, "day": 19},
        "truth": {"valid": True},
        "selected_date": "2025-11-19",
        "expected_in_accepted": False,
        "expected_before_filter": False,
        "truth_overlap": {"best_verdict": "covered"},
        "truth_crop_oracle": {"reliable": True},
    }

    assert _classify_row({**base, "expected_in_accepted": True}) == "A_selector"
    assert _classify_row({**base, "expected_before_filter": True}) == "B_filter"
    assert _classify_row(base) == "C_crop_recognition"
    assert _classify_row({**base, "truth_overlap": {"best_verdict": "missed"}}) == "D_proposal_recall"
    assert _classify_row({**base, "truth": {"valid": False}}) == "F_truth_unreliable"
