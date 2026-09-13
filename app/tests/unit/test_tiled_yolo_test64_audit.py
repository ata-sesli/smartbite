from __future__ import annotations

from app.scripts.tiled_yolo_test64_audit import _summary_with_success, _tile_windows


def test_tile_windows_cover_image_edges() -> None:
    windows = _tile_windows((3000, 4000, 3), tile_size=1280, overlap=0.35, include_full_image=False)

    assert windows[0].x1 == 0
    assert windows[0].y1 == 0
    assert max(window.x2 for window in windows) == 4000
    assert max(window.y2 for window in windows) == 3000
    assert any(window.x1 == 4000 - 1280 for window in windows)
    assert any(window.y1 == 3000 - 1280 for window in windows)


def test_tile_windows_include_full_image_when_requested() -> None:
    windows = _tile_windows((720, 960, 3), tile_size=1280, overlap=0.35, include_full_image=True)

    assert windows == [windows[0]]
    assert windows[0].source == "full_image"
    assert (windows[0].x1, windows[0].y1, windows[0].x2, windows[0].y2) == (0, 0, 960, 720)


def test_summary_with_success_counts_truth_coverage_threshold() -> None:
    summary = _summary_with_success(
        [
            {"verdict": "partial", "box_count": 1, "runtime_ms": 10.0, "truth_coverage_ratio": 0.21},
            {"verdict": "missed_truth", "box_count": 1, "runtime_ms": 20.0, "truth_coverage_ratio": 0.20},
            {"verdict": "no_box", "box_count": 0, "runtime_ms": 30.0, "truth_coverage_ratio": 0.0},
        ],
        success_threshold=0.20,
    )

    assert summary["success_gt_threshold"] == 1
    assert summary["success_rate"] == 1 / 3
