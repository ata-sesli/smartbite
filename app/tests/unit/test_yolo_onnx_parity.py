from __future__ import annotations

from app.scripts.yolo_onnx_parity import summarize_detector_parity


def test_summarize_detector_parity_compares_top_boxes() -> None:
    ultralytics_rows = [
        {
            "filename": "a.jpg",
            "detector_boxes": [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.90}],
        },
        {
            "filename": "b.jpg",
            "detector_boxes": [{"bbox_xyxy": [20, 20, 40, 40], "confidence": 0.70}],
        },
    ]
    onnx_rows = [
        {
            "filename": "a.jpg",
            "detector_boxes": [{"bbox_xyxy": [1, 1, 11, 11], "confidence": 0.85}],
        },
        {
            "filename": "b.jpg",
            "detector_boxes": [],
        },
    ]

    summary = summarize_detector_parity(ultralytics_rows, onnx_rows)

    assert summary["total"] == 2
    assert summary["both_detected"] == 1
    assert summary["missing_in_onnx"] == 1
    assert summary["mean_top_box_iou"] == 0.680672
    assert [item["filename"] for item in summary["mismatches"]] == ["b.jpg"]


def test_summarize_detector_parity_reads_plain_yolo_nested_report_items() -> None:
    ultralytics_report = {
        "items": [
            {
                "filename": "a.jpg",
                "config": {"detector_boxes": [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.90}]},
            }
        ]
    }
    onnx_report = {
        "items": [
            {
                "filename": "a.jpg",
                "config": {"detector_boxes": [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.80}]},
            }
        ]
    }

    summary = summarize_detector_parity(ultralytics_report, onnx_report)

    assert summary["total"] == 1
    assert summary["both_detected"] == 1
    assert summary["mean_top_box_iou"] == 1.0
