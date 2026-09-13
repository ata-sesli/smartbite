from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, area_a + area_b - inter)


def _rows_by_filename(report_or_rows: dict[str, Any] | list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if isinstance(report_or_rows, dict):
        rows = report_or_rows.get("results") or report_or_rows.get("items") or []
    else:
        rows = report_or_rows
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("filename"), str):
            config = row.get("config")
            if isinstance(config, dict):
                out[str(row["filename"])] = {**config, "filename": row["filename"]}
            else:
                out[str(row["filename"])] = row
    return out


def _top_box(row: dict[str, Any]) -> dict[str, Any] | None:
    boxes = row.get("detector_boxes")
    if not isinstance(boxes, list) or not boxes:
        return None
    valid = [box for box in boxes if isinstance(box, dict) and _bbox(box.get("bbox_xyxy")) is not None]
    if not valid:
        return None
    return max(valid, key=lambda box: float(box.get("confidence") or 0.0))


def summarize_detector_parity(
    ultralytics_report_or_rows: dict[str, Any] | list[dict[str, Any]],
    onnx_report_or_rows: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    ultralytics = _rows_by_filename(ultralytics_report_or_rows)
    onnx = _rows_by_filename(onnx_report_or_rows)
    filenames = sorted(set(ultralytics) | set(onnx))
    ious: list[float] = []
    mismatches: list[dict[str, Any]] = []
    both_detected = 0
    missing_in_onnx = 0
    missing_in_ultralytics = 0

    for filename in filenames:
        ul_box = _top_box(ultralytics.get(filename, {}))
        onnx_box = _top_box(onnx.get(filename, {}))
        if ul_box is None and onnx_box is None:
            continue
        if ul_box is None:
            missing_in_ultralytics += 1
            mismatches.append({"filename": filename, "reason": "missing_in_ultralytics"})
            continue
        if onnx_box is None:
            missing_in_onnx += 1
            mismatches.append({"filename": filename, "reason": "missing_in_onnx"})
            continue
        both_detected += 1
        iou = _iou(_bbox(ul_box["bbox_xyxy"]), _bbox(onnx_box["bbox_xyxy"]))  # type: ignore[arg-type]
        ious.append(iou)
        if iou < 0.5:
            mismatches.append(
                {
                    "filename": filename,
                    "reason": "low_iou",
                    "top_box_iou": round(iou, 6),
                    "ultralytics_bbox_xyxy": ul_box.get("bbox_xyxy"),
                    "onnx_bbox_xyxy": onnx_box.get("bbox_xyxy"),
                }
            )

    return {
        "total": len(filenames),
        "both_detected": both_detected,
        "missing_in_onnx": missing_in_onnx,
        "missing_in_ultralytics": missing_in_ultralytics,
        "mean_top_box_iou": round(sum(ious) / len(ious), 6) if ious else None,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Ultralytics YOLO OBB and ONNX detector audit reports")
    parser.add_argument("--ultralytics-report", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ultralytics_report = json.loads(args.ultralytics_report.read_text(encoding="utf-8"))
    onnx_report = json.loads(args.onnx_report.read_text(encoding="utf-8"))
    summary = summarize_detector_parity(ultralytics_report, onnx_report)
    payload = json.dumps(summary, ensure_ascii=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
