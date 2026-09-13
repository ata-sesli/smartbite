from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ai.rapidocr_text_detector import RapidOCRTextProposalDetector
from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.scripts.detector_truth_ablation import (
    YoloObbDetector,
    _best_overlap,
    _clean_bbox,
    _ensure_dir,
    _load_truth_items,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_YOLO_MODEL_PATH = Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt")


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _selected_filenames(raw_names: str | None) -> set[str] | None:
    if not raw_names:
        return None
    return {Path(name.strip()).name for name in raw_names.split(",") if name.strip()}


def _decode_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image)
    except Exception:
        return None


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _proposal_to_dict(proposal: Any, *, detector_name: str) -> dict[str, Any]:
    polygon = getattr(proposal, "polygon_xy", None)
    return {
        "bbox_xyxy": [int(value) for value in proposal.bbox_xyxy],
        "polygon_xy": [[float(x), float(y)] for x, y in polygon] if polygon else None,
        "confidence": float(proposal.confidence or 0.0),
        "source": str(proposal.source),
        "sources": list(proposal.sources),
        "variant_name": str(proposal.variant_name),
        "detector_name": detector_name,
    }


def _box_center_inside(box: tuple[int, int, int, int], truth_bbox: tuple[int, int, int, int]) -> bool:
    cx = (truth_bbox[0] + truth_bbox[2]) / 2.0
    cy = (truth_bbox[1] + truth_bbox[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


class DetectorAdapter:
    name: str

    def detect(self, image_rgb: np.ndarray) -> tuple[list[dict[str, Any]], str | None]:
        raise NotImplementedError


class YoloAdapter(DetectorAdapter):
    def __init__(self, *, name: str, model_path: Path, confidence: float, max_candidates: int) -> None:
        self.name = name
        self.model_path = model_path
        self.max_candidates = max_candidates
        self.detector = YoloObbDetector(model_path, confidence_threshold=confidence)

    def detect(self, image_rgb: np.ndarray) -> tuple[list[dict[str, Any]], str | None]:
        boxes, meta = self.detector.detect(image_rgb)
        boxes = boxes[: self.max_candidates]
        for box in boxes:
            box["detector_name"] = self.name
        return boxes, meta.get("yolo_reason")


class RapidOcrAdapter(DetectorAdapter):
    def __init__(self, *, name: str, detector: RapidOCRTextProposalDetector) -> None:
        self.name = name
        self.detector = detector

    def detect(self, image_rgb: np.ndarray) -> tuple[list[dict[str, Any]], str | None]:
        proposals, reason = self.detector.detect(_to_bgr(image_rgb))
        return [_proposal_to_dict(proposal, detector_name=self.name) for proposal in proposals], reason


def _audit_detector(
    detector: DetectorAdapter,
    image_rgb: np.ndarray,
    truth_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    started = perf_counter()
    boxes, reason = detector.detect(image_rgb)
    runtime_ms = (perf_counter() - started) * 1000.0
    bbox_tuples = [
        tuple(int(round(float(value))) for value in box["bbox_xyxy"])
        for box in boxes
        if isinstance(box.get("bbox_xyxy"), list) and len(box["bbox_xyxy"]) == 4
    ]
    overlap = _best_overlap(bbox_tuples, truth_bbox)
    center_hits = [idx for idx, box in enumerate(bbox_tuples) if _box_center_inside(box, truth_bbox)]
    best_box_payload = None
    if overlap["best_box"] is not None:
        best_tuple = tuple(int(value) for value in overlap["best_box"])
        for box in boxes:
            if tuple(int(round(float(value))) for value in box["bbox_xyxy"]) == best_tuple:
                best_box_payload = box
                break
    return {
        "config_name": detector.name,
        **overlap,
        "runtime_ms": runtime_ms,
        "reason": reason,
        "best_box": best_box_payload,
        "center_hit": bool(center_hits),
        "center_hit_indices": center_hits,
        "detector_boxes": boxes,
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    images = len(results)
    summary: dict[str, Any] = {
        "images": images,
        "covered": 0,
        "tight": 0,
        "partial": 0,
        "missed_truth": 0,
        "no_box": 0,
        "center_hit": 0,
        "found_any_box": 0,
        "average_box_count": 0.0,
        "average_runtime_ms": 0.0,
        "p50_runtime_ms": 0.0,
        "p90_runtime_ms": 0.0,
        "mean_truth_coverage_ratio": 0.0,
        "mean_best_iou": 0.0,
    }
    if not results:
        return summary
    runtimes = sorted(float(result.get("runtime_ms") or 0.0) for result in results)
    for result in results:
        verdict = str(result.get("verdict") or "missed_truth")
        if verdict in summary:
            summary[verdict] += 1
        if bool(result.get("center_hit")):
            summary["center_hit"] += 1
        if int(result.get("box_count") or 0) > 0:
            summary["found_any_box"] += 1
    summary["average_box_count"] = sum(float(result.get("box_count") or 0) for result in results) / images
    summary["average_runtime_ms"] = sum(runtimes) / images
    summary["p50_runtime_ms"] = runtimes[min(len(runtimes) - 1, int(round((len(runtimes) - 1) * 0.50)))]
    summary["p90_runtime_ms"] = runtimes[min(len(runtimes) - 1, int(round((len(runtimes) - 1) * 0.90)))]
    summary["mean_truth_coverage_ratio"] = (
        sum(float(result.get("truth_coverage_ratio") or 0.0) for result in results) / images
    )
    summary["mean_best_iou"] = sum(float(result.get("best_iou") or 0.0) for result in results) / images
    return summary


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Detector Comparison Test64",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Images: `{payload['summary']['images']}`",
        f"Candidate cap per detector/image: `{payload['candidate_cap']}`",
        "",
        "## Summary",
        "",
        "| Detector | Covered | Tight | Partial | Missed | No box | Any box | Center hit | Avg boxes | Avg ms/image | P90 ms | Mean truth coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in payload["configs"]:
        name = config["name"]
        summary = payload["summary"]["configs"][name]
        lines.append(
            f"| {name} | {summary['covered']} | {summary['tight']} | {summary['partial']} | "
            f"{summary['missed_truth']} | {summary['no_box']} | {summary['found_any_box']} | "
            f"{summary['center_hit']} | {summary['average_box_count']:.2f} | "
            f"{summary['average_runtime_ms']:.1f} | {summary['p90_runtime_ms']:.1f} | "
            f"{summary['mean_truth_coverage_ratio']:.3f} |"
        )
    lines.extend(["", "## Per-Image", ""])
    detector_headers = [str(config["name"]) for config in payload["configs"]]
    lines.append("| Filename | " + " | ".join(detector_headers) + " |")
    lines.append("|---|" + "|".join("---" for _ in detector_headers) + "|")
    for item in payload["items"]:
        by_name = {result["config_name"]: result for result in item["configs"]}
        cells = []
        for config in payload["configs"]:
            result = by_name.get(config["name"], {})
            cells.append(
                f"{result.get('verdict', '-')} "
                f"({int(result.get('box_count') or 0)} boxes, "
                f"{float(result.get('truth_coverage_ratio') or 0.0):.2f} truth, "
                f"{float(result.get('runtime_ms') or 0.0):.0f} ms)"
            )
        lines.append(f"| {item['filename']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_detectors(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[DetectorAdapter]]:
    configs = [
        {
            "name": "yolo26s_obb_best",
            "kind": "yolo_obb",
            "settings": {
                "model_path": str(Path(args.yolo_model_path)),
                "confidence_threshold": args.yolo_conf,
                "max_candidates": args.candidate_cap,
            },
            "color": "#f97316",
        },
        {
            "name": "ppocrv5_mobile_pretrained",
            "kind": "rapidocr_ppocrv5",
            "settings": {
                "engine_type": "onnxruntime",
                "model_type": "mobile",
                "ocr_version": "PP-OCRv5",
                "limit_side_len": args.rapidocr_limit_side_len,
                "limit_type": "max",
                "max_candidates": args.candidate_cap,
                "max_boxes": args.candidate_cap,
                "min_confidence": args.rapidocr_min_confidence,
            },
            "color": "#2563eb",
        },
    ]
    detectors: list[DetectorAdapter] = [
        YoloAdapter(
            name=configs[0]["name"],
            model_path=Path(args.yolo_model_path),
            confidence=args.yolo_conf,
            max_candidates=args.candidate_cap,
        ),
        RapidOcrAdapter(
            name=configs[1]["name"],
            detector=RapidOCRTextProposalDetector(
                ocr_version="PP-OCRv5",
                model_type="mobile",
                lang_type="ch",
                engine_type="onnxruntime",
                limit_side_len=args.rapidocr_limit_side_len,
                limit_type="max",
                max_candidates=args.candidate_cap,
                max_boxes=args.candidate_cap,
                min_confidence=args.rapidocr_min_confidence,
                min_box_area=args.min_box_area,
            ),
        ),
    ]
    return configs, detectors


def run(args: argparse.Namespace) -> int:
    truth_items = _load_truth_items(Path(args.truth_manifest))
    selected = _selected_filenames(args.filenames)
    rows: list[dict[str, Any]] = []
    for image_path in sorted(Path(args.images_dir).iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if selected is not None and image_path.name not in selected:
            continue
        truth_item = truth_items.get(image_path.name)
        if not isinstance(truth_item, dict):
            continue
        truth_bbox = _clean_bbox(truth_item.get("true_bbox_xyxy"))
        if truth_bbox is None:
            continue
        rows.append({"filename": image_path.name, "image_path": image_path, "truth_item": truth_item, "truth_bbox": truth_bbox})
    if not rows:
        print("No test64 images with truth bboxes found.", flush=True)
        return 1

    configs, detectors = _build_detectors(args)
    results_by_config: dict[str, list[dict[str, Any]]] = {config["name"]: [] for config in configs}
    items: list[dict[str, Any]] = []
    print(
        f"Starting detector comparison: images={len(rows)} detectors={len(detectors)} candidate_cap={args.candidate_cap}",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        image_rgb = _decode_rgb(Path(row["image_path"]))
        if image_rgb is None:
            print(f"[{index}/{len(rows)}] skipped unreadable {row['filename']}", flush=True)
            continue
        item = {
            "filename": row["filename"],
            "image_url": f"/test64/images/{row['filename']}",
            "truth_bbox_xyxy": list(row["truth_bbox"]),
            "truth_polygon_xy": row["truth_item"].get("true_polygon_xy"),
            "expected": {
                "day": row["truth_item"].get("expected_day"),
                "month": row["truth_item"].get("expected_month"),
                "year": row["truth_item"].get("expected_year"),
            },
            "configs": [],
        }
        for detector in detectors:
            result = _audit_detector(detector, image_rgb, row["truth_bbox"])
            item["configs"].append(result)
            results_by_config[detector.name].append(result)
            print(
                f"[{index}/{len(rows)}] {row['filename']} {detector.name} -> "
                f"{result['verdict']} boxes={result['box_count']} "
                f"truth={result['truth_coverage_ratio']:.3f} "
                f"time={result['runtime_ms']:.1f}ms",
                flush=True,
            )
        items.append(item)

    report_root = Path(args.output_dir).resolve() / f"test64_detector_comparison_{_utc_stamp()}"
    _ensure_dir(report_root)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "test64-files",
        "images_dir": str(Path(args.images_dir).resolve()),
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "candidate_cap": args.candidate_cap,
        "configs": configs,
        "summary": {
            "images": len(items),
            "configs": {name: _summary(results) for name, results in results_by_config.items()},
        },
        "items": items,
    }
    report_json = report_root / "detector_comparison_report.json"
    report_md = report_root / "detector_comparison_summary.md"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(report_md, payload)
    print("Summary:", json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Report root: {report_root}", flush=True)
    print(report_json, flush=True)
    print(report_md, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO and pretrained PP-OCRv5 mobile detectors on test64.")
    parser.add_argument("--images-dir", type=Path, default=Path("test64"))
    parser.add_argument("--truth-manifest", type=Path, default=Path(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics/detector_comparison"))
    parser.add_argument("--filenames", default=None)
    parser.add_argument("--candidate-cap", type=int, default=64)
    parser.add_argument("--yolo-model-path", type=Path, default=DEFAULT_YOLO_MODEL_PATH)
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--rapidocr-limit-side-len", type=int, default=512)
    parser.add_argument("--rapidocr-min-confidence", type=float, default=0.08)
    parser.add_argument("--min-box-area", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
