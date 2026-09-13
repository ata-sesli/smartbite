from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.ai.onnx_inference import YoloObbOnnxDetector
from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.scripts.detector_truth_ablation import (
    DEFAULT_YOLO_OBB_MODEL_PATH,
    YoloObbDetector,
    _best_overlap,
    _clean_bbox,
    _ensure_dir,
    _load_truth_items,
    _xyxy_from_polygon,
)
from app.scripts.tiled_yolo_test64_audit import _dedupe_boxes_by_bbox


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_YOLO_OBB_ONNX_PATH = Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.onnx")


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _load_image(path: Path, *, apply_exif_orientation: bool) -> np.ndarray:
    with Image.open(path) as image:
        if apply_exif_orientation:
            image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def _rotate_image(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "rot90_cw":
        return np.rot90(image, k=3)
    if orientation == "rot90_ccw":
        return np.rot90(image, k=1)
    if orientation == "rot180":
        return np.rot90(image, k=2)
    raise ValueError(f"unsupported fallback orientation: {orientation}")


def _map_point_to_original(x: float, y: float, *, orientation: str, width: int, height: int) -> list[float]:
    if orientation == "rot90_cw":
        return [y, height - x]
    if orientation == "rot90_ccw":
        return [width - y, x]
    if orientation == "rot180":
        return [width - x, height - y]
    raise ValueError(f"unsupported fallback orientation: {orientation}")


def _map_rotated_box_to_original(
    box: dict[str, Any],
    *,
    orientation: str,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    polygon = box.get("polygon_xy")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    mapped_polygon: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        mapped_polygon.append(
            _map_point_to_original(
                float(point[0]),
                float(point[1]),
                orientation=orientation,
                width=width,
                height=height,
            )
        )
    mapped = dict(box)
    mapped["bbox_xyxy"] = _xyxy_from_polygon(mapped_polygon)
    mapped["polygon_xy"] = mapped_polygon
    mapped["source"] = "yolo26s_obb_rotation_fallback"
    mapped["sources"] = ["yolo26s_obb", "rotation_fallback"]
    mapped["variant_name"] = orientation
    mapped["roi_source"] = orientation
    mapped["fallback_orientation"] = orientation
    return mapped


def _summary(results: list[dict[str, Any]], *, success_threshold: float) -> dict[str, Any]:
    images = len(results)
    successes = sum(float(item["truth_coverage_ratio"]) > success_threshold for item in results)
    summary: dict[str, Any] = {
        "images": images,
        "success_threshold": success_threshold,
        "success_gt_threshold": successes,
        "success_rate": successes / images if images else 0.0,
        "average_box_count": 0.0,
        "mean_truth_coverage_ratio": 0.0,
        "covered": 0,
        "tight": 0,
        "partial": 0,
        "missed_truth": 0,
        "no_box": 0,
    }
    if not results:
        return summary
    for item in results:
        verdict = str(item["verdict"])
        if verdict in summary:
            summary[verdict] += 1
    summary["average_box_count"] = sum(float(item["box_count"]) for item in results) / images
    summary["mean_truth_coverage_ratio"] = (
        sum(float(item["truth_coverage_ratio"]) for item in results) / images
    )
    return summary


def _audit_image(
    detector: YoloObbDetector,
    image: np.ndarray,
    truth_bbox: tuple[int, int, int, int],
    *,
    max_candidates: int,
    rotation_fallback_on_no_box: bool,
    rotation_fallback_orientations: tuple[str, ...],
) -> dict[str, Any]:
    started = perf_counter()
    detector_boxes, meta = detector.detect(image)
    fallback_orientations_tried: list[str] = []
    fallback_boxes_added = 0
    fallback_triggered = rotation_fallback_on_no_box and not detector_boxes
    if fallback_triggered:
        height, width = image.shape[:2]
        for orientation in rotation_fallback_orientations:
            fallback_orientations_tried.append(orientation)
            rotated = _rotate_image(image, orientation)
            rotated_boxes, _ = detector.detect(rotated)
            for box in rotated_boxes:
                mapped = _map_rotated_box_to_original(
                    box,
                    orientation=orientation,
                    width=width,
                    height=height,
                )
                if mapped is not None:
                    detector_boxes.append(mapped)
                    fallback_boxes_added += 1
        detector_boxes = _dedupe_boxes_by_bbox(detector_boxes)
        detector_boxes.sort(key=lambda box: float(box.get("confidence") or 0.0), reverse=True)
        meta = {
            **meta,
            "yolo_detected": bool(detector_boxes),
            "yolo_reason": None if detector_boxes else "no candidate passed confidence threshold",
        }
    if max_candidates > 0:
        detector_boxes = detector_boxes[:max_candidates]
    runtime_ms = (perf_counter() - started) * 1000
    bbox_tuples = [
        tuple(int(round(float(value))) for value in box["bbox_xyxy"])
        for box in detector_boxes
        if isinstance(box.get("bbox_xyxy"), list) and len(box["bbox_xyxy"]) == 4
    ]
    overlap = _best_overlap(bbox_tuples, truth_bbox)
    best_box_payload = None
    if overlap["best_box"] is not None:
        best_tuple = tuple(overlap["best_box"])
        for box in detector_boxes:
            if tuple(int(round(float(value))) for value in box["bbox_xyxy"]) == best_tuple:
                best_box_payload = box
                break
    return {
        **overlap,
        "runtime_ms": runtime_ms,
        "best_box": best_box_payload,
        "detector_boxes": detector_boxes,
        "rotation_fallback_triggered": fallback_triggered,
        "rotation_fallback_orientations_tried": fallback_orientations_tried,
        "rotation_fallback_boxes_added": fallback_boxes_added,
        **meta,
    }


class OnnxYoloObbAuditDetector:
    def __init__(self, model_path: Path, *, confidence_threshold: float, imgsz: int, max_candidates: int) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._detector = YoloObbOnnxDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            imgsz=imgsz,
            max_candidates=max_candidates,
        )

    def detect(self, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        detections, reason = self._detector.detect(image)
        boxes = [
            {
                "bbox_xyxy": list(item.bbox_xyxy),
                "polygon_xy": item.polygon_xy,
                "confidence": item.confidence,
                "source": "yolo26s_obb_onnx",
                "sources": ["yolo26s_obb_onnx"],
                "variant_name": "original",
                "roi_source": "full_image",
                "model_path": str(self.model_path),
            }
            for item in detections
        ]
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": reason,
            "roi_count": 1,
            "detector_unavailable_reasons": [reason] if reason and not boxes else [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure YOLO-OBB truth-overlap audit for local test64 files")
    parser.add_argument("--images-dir", type=Path, default=Path("test64"))
    parser.add_argument("--truth-manifest", type=Path, default=Path(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--backend", choices=("ultralytics", "onnx"), default="ultralytics")
    parser.add_argument("--yolo-model-path", type=Path, default=DEFAULT_YOLO_OBB_MODEL_PATH)
    parser.add_argument("--yolo-onnx-model-path", type=Path, default=DEFAULT_YOLO_OBB_ONNX_PATH)
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.20)
    parser.add_argument(
        "--ignore-exif-orientation",
        action="store_true",
        help="Decode raw pixels without applying EXIF orientation. Useful for reproducing old runs.",
    )
    parser.add_argument(
        "--rotation-fallback-on-no-box",
        action="store_true",
        help="If the normal pass produces zero boxes, run rotation fallback passes and map boxes back.",
    )
    parser.add_argument(
        "--rotation-fallback-orientations",
        default="rot90_cw,rot90_ccw",
        help="Comma-separated fallback orientations. Supported: rot90_cw, rot90_ccw, rot180.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics"))
    parser.add_argument("--config-name", default="plain_yolo_obb")
    args = parser.parse_args()

    truth_items = _load_truth_items(args.truth_manifest)
    if args.backend == "onnx":
        detector = OnnxYoloObbAuditDetector(
            args.yolo_onnx_model_path,
            confidence_threshold=args.yolo_conf,
            imgsz=args.imgsz,
            max_candidates=args.max_candidates,
        )
    else:
        detector = YoloObbDetector(args.yolo_model_path, confidence_threshold=args.yolo_conf)

    rows: list[tuple[str, Path, tuple[int, int, int, int]]] = []
    for filename, item in sorted(truth_items.items()):
        image_path = args.images_dir / Path(filename).name
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        truth_bbox = _clean_bbox(item.get("true_bbox_xyxy"))
        if truth_bbox is None:
            continue
        rows.append((Path(filename).name, image_path, truth_bbox))

    if not rows:
        raise SystemExit(f"No truth-bbox images found under {args.images_dir}")

    # Keep the Ultralytics detector pure, but make imgsz explicit by setting
    # the model override used by the parity run.
    original_detect = detector.detect

    def detect_with_imgsz(image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not isinstance(detector, YoloObbDetector):
            return detector.detect(image)
        model = detector._ensure_model()
        results = model.predict(image, conf=detector.confidence_threshold, imgsz=args.imgsz, verbose=False)
        if not results:
            return [], {"yolo_detected": False, "yolo_reason": "detector returned no result"}
        result = results[0]
        obb = getattr(result, "obb", None)
        if obb is None or len(obb) == 0:
            return [], {"yolo_detected": False, "yolo_reason": "no expiry region detected"}
        polys_raw = getattr(obb, "xyxyxyxy", None)
        conf_raw = getattr(obb, "conf", None)
        if polys_raw is None or conf_raw is None:
            return [], {"yolo_detected": False, "yolo_reason": "obb output missing polygons or confidences"}
        from app.scripts.detector_truth_ablation import _yolo_obb_box_dict

        polygons = polys_raw.cpu().numpy().tolist()
        confidences = conf_raw.cpu().numpy().tolist()
        boxes = [
            _yolo_obb_box_dict(
                polygon=[[float(x), float(y)] for x, y in polygon],
                confidence=float(confidence),
                model_path=args.yolo_model_path,
            )
            for polygon, confidence in zip(polygons, confidences, strict=False)
            if float(confidence) >= detector.confidence_threshold
        ]
        boxes.sort(key=lambda box: float(box.get("confidence") or 0.0), reverse=True)
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": None if boxes else "no candidate passed confidence threshold",
            "roi_count": 1,
            "detector_unavailable_reasons": [],
        }

    if isinstance(detector, YoloObbDetector):
        detector.detect = detect_with_imgsz  # type: ignore[method-assign]
    rotation_fallback_orientations = tuple(
        orientation.strip()
        for orientation in args.rotation_fallback_orientations.split(",")
        if orientation.strip()
    )
    invalid_orientations = sorted(set(rotation_fallback_orientations) - {"rot90_cw", "rot90_ccw", "rot180"})
    if invalid_orientations:
        raise SystemExit(f"Unsupported rotation fallback orientation(s): {', '.join(invalid_orientations)}")

    item_results = []
    print(
        f"Starting pure YOLO audit: items={len(rows)} backend={args.backend} "
        f"model={args.yolo_model_path if args.backend == 'ultralytics' else args.yolo_onnx_model_path} "
        f"conf={args.yolo_conf} imgsz={args.imgsz} max_candidates={args.max_candidates} "
        f"apply_exif_orientation={not args.ignore_exif_orientation} "
        f"rotation_fallback_on_no_box={args.rotation_fallback_on_no_box}",
        flush=True,
    )
    try:
        for index, (filename, image_path, truth_bbox) in enumerate(rows, start=1):
            image = _load_image(image_path, apply_exif_orientation=not args.ignore_exif_orientation)
            result = _audit_image(
                detector,
                image,
                truth_bbox,
                max_candidates=args.max_candidates,
                rotation_fallback_on_no_box=args.rotation_fallback_on_no_box,
                rotation_fallback_orientations=rotation_fallback_orientations,
            )
            item_result = {
                "filename": filename,
                "truth_bbox_xyxy": list(truth_bbox),
                "config": {"config_name": args.config_name, **result},
            }
            item_results.append(item_result)
            success = float(result["truth_coverage_ratio"]) > args.success_threshold
            print(
                f"[{args.config_name} {index}/{len(rows)}] {filename} -> "
                f"{result['verdict']} boxes={result['box_count']} "
                f"truth={result['truth_coverage_ratio']:.3f} success={success} "
                f"rotation_fallback={result['rotation_fallback_triggered']}",
                flush=True,
            )
    finally:
        if isinstance(detector, YoloObbDetector):
            detector.detect = original_detect  # type: ignore[method-assign]

    flat_results = [item["config"] for item in item_results]
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "backend": f"pure_yolo_obb_{args.backend}",
        "images_dir": str(args.images_dir.resolve()),
        "truth_manifest": str(args.truth_manifest.resolve()),
        "yolo_model_path": str(args.yolo_model_path),
        "yolo_onnx_model_path": str(args.yolo_onnx_model_path),
        "settings": {
            "backend": args.backend,
            "confidence_threshold": args.yolo_conf,
            "imgsz": args.imgsz,
            "max_candidates": args.max_candidates,
            "success_threshold": args.success_threshold,
            "apply_exif_orientation": not args.ignore_exif_orientation,
            "rotation_fallback_on_no_box": args.rotation_fallback_on_no_box,
            "rotation_fallback_orientations": list(rotation_fallback_orientations),
        },
        "summary": _summary(flat_results, success_threshold=args.success_threshold),
        "items": item_results,
    }

    report_root = args.output_dir.resolve() / f"test64_plain_yolo_obb_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    report_json = report_root / "plain_yolo_obb_test64_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Summary:", json.dumps(payload["summary"], indent=2), flush=True)
    print("Report root:", report_root, flush=True)
    print(report_json, flush=True)


if __name__ == "__main__":
    main()
