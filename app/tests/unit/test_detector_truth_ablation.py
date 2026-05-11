from __future__ import annotations

from app.scripts.detector_truth_ablation import (
    _best_overlap,
    _classify_truth_overlap,
    _summarize_config_results,
)


def test_detector_truth_ablation_marks_no_box() -> None:
    best = _best_overlap([], (100, 100, 200, 140))

    assert best["verdict"] == "no_box"
    assert best["box_count"] == 0
    assert best["best_box"] is None
    assert best["truth_coverage_ratio"] == 0.0


def test_detector_truth_ablation_classifies_covered_truth() -> None:
    result = _best_overlap([(95, 95, 205, 145)], (100, 100, 200, 140))

    assert result["verdict"] == "covered"
    assert result["covers_truth"] is True
    assert result["truth_coverage_ratio"] == 1.0
    assert result["best_iou"] > 0.7


def test_detector_truth_ablation_classifies_tight_partial_and_missed_boxes() -> None:
    assert _classify_truth_overlap(
        box_count=1,
        best_iou=0.30,
        truth_coverage_ratio=0.62,
        candidate_coverage_ratio=0.95,
    ) == "tight"
    assert _classify_truth_overlap(
        box_count=1,
        best_iou=0.08,
        truth_coverage_ratio=0.32,
        candidate_coverage_ratio=0.60,
    ) == "partial"
    assert _classify_truth_overlap(
        box_count=1,
        best_iou=0.01,
        truth_coverage_ratio=0.05,
        candidate_coverage_ratio=0.08,
    ) == "missed_truth"


def test_detector_truth_ablation_summary_counts_config_verdicts() -> None:
    summary = _summarize_config_results(
        [
            {"verdict": "covered", "box_count": 2, "runtime_ms": 100.0, "truth_coverage_ratio": 0.9},
            {"verdict": "tight", "box_count": 3, "runtime_ms": 200.0, "truth_coverage_ratio": 0.6},
            {"verdict": "no_box", "box_count": 0, "runtime_ms": 300.0, "truth_coverage_ratio": 0.0},
        ]
    )

    assert summary["images"] == 3
    assert summary["covered"] == 1
    assert summary["tight"] == 1
    assert summary["no_box"] == 1
    assert summary["average_box_count"] == 5 / 3
    assert summary["average_runtime_ms"] == 200.0
    assert summary["mean_truth_coverage_ratio"] == 0.5
