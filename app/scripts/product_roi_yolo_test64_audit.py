from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.infra.settings import Settings
from app.scripts.detector_truth_ablation import (
    DEFAULT_YOLO_OBB_MODEL_PATH,
    _best_overlap,
    _clean_bbox,
    _decode_image_bytes,
    _ensure_dir,
    _load_truth_items,
    _selected_filenames,
    _xyxy_from_polygon,
)
from app.scripts.tiled_yolo_test64_audit import _dedupe_boxes_by_bbox, _summary_with_success


DEFAULT_PRODUCT_MODEL_PATH = Settings().detector_model_path


@dataclass(frozen=True)
class ProductRoi:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    source: str


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _offset_polygon(polygon: list[list[float]], *, offset_x: int, offset_y: int) -> list[list[float]]:
    return [[float(x) + offset_x, float(y) + offset_y] for x, y in polygon]


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width, x2 + pad_x),
        min(image_height, y2 + pad_y),
    )


class ProductThenExpiryYoloDetector:
    def __init__(
        self,
        *,
        product_model_path: Path,
        product_conf: float,
        product_top_k: int,
        product_padding_ratio: float,
        product_imgsz: int,
        expiry_model_path: Path,
        expiry_conf: float,
        expiry_imgsz: int,
        max_candidates: int,
    ) -> None:
        self.product_model_path = product_model_path
        self.product_conf = product_conf
        self.product_top_k = product_top_k
        self.product_padding_ratio = product_padding_ratio
        self.product_imgsz = product_imgsz
        self.expiry_model_path = expiry_model_path
        self.expiry_conf = expiry_conf
        self.expiry_imgsz = expiry_imgsz
        self.max_candidates = max_candidates
        self._product_model: Any | None = None
        self._expiry_model: Any | None = None

    def _ensure_models(self) -> tuple[Any, Any]:
        if self._product_model is None or self._expiry_model is None:
            from ultralytics import YOLO

            if self._product_model is None:
                if not self.product_model_path.exists():
                    raise ValueError(f"product YOLO model not found: {self.product_model_path}")
                self._product_model = YOLO(str(self.product_model_path))
            if self._expiry_model is None:
                if not self.expiry_model_path.exists():
                    raise ValueError(f"expiry YOLO OBB model not found: {self.expiry_model_path}")
                self._expiry_model = YOLO(str(self.expiry_model_path))
        return self._product_model, self._expiry_model

    def _detect_products(self, image: np.ndarray) -> list[ProductRoi]:
        product_model, _ = self._ensure_models()
        results = product_model.predict(image, conf=self.product_conf, imgsz=self.product_imgsz, verbose=False)
        if not results:
            return []
        boxes_raw = getattr(results[0], "boxes", None)
        if boxes_raw is None or len(boxes_raw) == 0:
            return []
        height, width = image.shape[:2]
        rois: list[ProductRoi] = []
        for row in boxes_raw:
            confidence = float(row.conf.item())
            if confidence < self.product_conf:
                continue
            xyxy = row.xyxy[0].tolist()
            raw_bbox = (
                int(round(float(xyxy[0]))),
                int(round(float(xyxy[1]))),
                int(round(float(xyxy[2]))),
                int(round(float(xyxy[3]))),
            )
            padded_bbox = _pad_bbox(
                raw_bbox,
                image_width=width,
                image_height=height,
                padding_ratio=self.product_padding_ratio,
            )
            if padded_bbox[2] <= padded_bbox[0] or padded_bbox[3] <= padded_bbox[1]:
                continue
            rois.append(ProductRoi(padded_bbox, confidence, "product_yolo"))
        rois.sort(key=lambda roi: roi.confidence, reverse=True)
        return rois[: self.product_top_k]

    def _detect_expiry_in_roi(self, image: np.ndarray, roi: ProductRoi) -> list[dict[str, Any]]:
        _, expiry_model = self._ensure_models()
        x1, y1, x2, y2 = roi.bbox_xyxy
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        results = expiry_model.predict(crop, conf=self.expiry_conf, imgsz=self.expiry_imgsz, verbose=False)
        if not results:
            return []
        obb = getattr(results[0], "obb", None)
        if obb is None or len(obb) == 0:
            return []
        polys_raw = getattr(obb, "xyxyxyxy", None)
        conf_raw = getattr(obb, "conf", None)
        if polys_raw is None or conf_raw is None:
            return []

        polygons = polys_raw.cpu().numpy().tolist()
        confidences = conf_raw.cpu().numpy().tolist()
        boxes: list[dict[str, Any]] = []
        for polygon, confidence in zip(polygons, confidences, strict=False):
            confidence_float = float(confidence)
            if confidence_float < self.expiry_conf:
                continue
            mapped_polygon = _offset_polygon(
                [[float(x), float(y)] for x, y in polygon],
                offset_x=x1,
                offset_y=y1,
            )
            boxes.append(
                {
                    "bbox_xyxy": _xyxy_from_polygon(mapped_polygon),
                    "polygon_xy": mapped_polygon,
                    "confidence": confidence_float,
                    "source": "product_roi_yolo26s_obb",
                    "sources": ["product_yolo", "yolo26s_obb"],
                    "variant_name": "product_roi",
                    "roi_source": "product_yolo",
                    "product_bbox_xyxy": list(roi.bbox_xyxy),
                    "product_confidence": roi.confidence,
                    "model_path": str(self.expiry_model_path),
                }
            )
        return boxes

    def detect(self, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        product_rois = self._detect_products(image)
        boxes: list[dict[str, Any]] = []
        for roi in product_rois:
            boxes.extend(self._detect_expiry_in_roi(image, roi))
        boxes = _dedupe_boxes_by_bbox(boxes)
        if self.max_candidates > 0:
            boxes = boxes[: self.max_candidates]
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": None if boxes else "no expiry region detected inside product rois",
            "product_roi_count": len(product_rois),
            "product_rois": [
                {"bbox_xyxy": list(roi.bbox_xyxy), "confidence": roi.confidence, "source": roi.source}
                for roi in product_rois
            ],
            "detector_unavailable_reasons": [],
        }


def _audit_product_roi_yolo_image(
    detector: ProductThenExpiryYoloDetector,
    image: np.ndarray,
    truth_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    started = perf_counter()
    detector_boxes, meta = detector.detect(image)
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
        **meta,
    }


def _load_test64_rows(args: argparse.Namespace, truth_items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected_names = _selected_filenames(args.filenames)
    images_dir = Path(args.images_dir)
    rows = []
    for filename, item in sorted(truth_items.items()):
        basename = Path(filename).name
        if selected_names is not None and basename not in selected_names:
            continue
        image_path = images_dir / basename
        if not image_path.exists():
            continue
        if _clean_bbox(item.get("true_bbox_xyxy")) is None:
            continue
        rows.append({"filename": basename, "image_path": str(image_path)})
    return rows


async def main_async(args: argparse.Namespace) -> int:
    truth_items = _load_truth_items(Path(args.truth_manifest))
    rows = _load_test64_rows(args, truth_items)
    if not rows:
        print("No rows with manual truth bboxes found.", flush=True)
        return 1

    detector = ProductThenExpiryYoloDetector(
        product_model_path=Path(args.product_model_path),
        product_conf=args.product_conf,
        product_top_k=args.product_top_k,
        product_padding_ratio=args.product_padding_ratio,
        product_imgsz=args.product_imgsz,
        expiry_model_path=Path(args.expiry_model_path),
        expiry_conf=args.expiry_conf,
        expiry_imgsz=args.expiry_imgsz,
        max_candidates=args.max_candidates,
    )
    config_name = args.config_name
    results: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    print(
        f"Starting product-ROI YOLO audit: items={len(rows)} product_conf={args.product_conf} "
        f"product_top_k={args.product_top_k} expiry_conf={args.expiry_conf} "
        f"max_candidates={args.max_candidates}",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        filename = Path(str(row["filename"])).name
        item_started = perf_counter()
        print(f"[{config_name} {index}/{len(rows)}] starting {filename}", flush=True)
        image = _decode_image_bytes(Path(str(row["image_path"])).read_bytes())
        truth_bbox = _clean_bbox(truth_items[filename].get("true_bbox_xyxy"))
        if image is None or truth_bbox is None:
            print(f"[{config_name} {index}/{len(rows)}] skipped {filename}: invalid image or truth bbox", flush=True)
            continue
        result = _audit_product_roi_yolo_image(detector, image, truth_bbox)
        config_result = {"config_name": config_name, **result}
        results.append(config_result)
        items.append(
            {
                "filename": filename,
                "truth_bbox_xyxy": truth_items[filename].get("true_bbox_xyxy"),
                "configs": [config_result],
            }
        )
        success_mark = "success" if result["truth_coverage_ratio"] > args.success_threshold else "miss"
        print(
            f"[{config_name} {index}/{len(rows)}] {filename} -> {result['verdict']} "
            f"{success_mark} product_rois={result['product_roi_count']} boxes={result['box_count']} "
            f"truth={result['truth_coverage_ratio']:.3f} runtime_ms={result['runtime_ms']:.1f} "
            f"elapsed_ms={(perf_counter() - item_started) * 1000:.1f}",
            flush=True,
        )

    report_root = Path(args.output_dir).resolve() / f"test64_product_roi_yolo_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "test64-files",
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "backend": "product_roi_yolo_obb",
        "configs": [
            {
                "name": config_name,
                "settings": {
                    "product_model_path": str(Path(args.product_model_path)),
                    "product_confidence_threshold": args.product_conf,
                    "product_top_k": args.product_top_k,
                    "product_padding_ratio": args.product_padding_ratio,
                    "product_imgsz": args.product_imgsz,
                    "expiry_model_path": str(Path(args.expiry_model_path)),
                    "expiry_confidence_threshold": args.expiry_conf,
                    "expiry_imgsz": args.expiry_imgsz,
                    "max_candidates": args.max_candidates,
                    "success_threshold": args.success_threshold,
                    "apply_exif_orientation": True,
                },
            }
        ],
        "summary": {
            "configs": {
                config_name: _summary_with_success(results, success_threshold=args.success_threshold),
            }
        },
        "items": items,
    }
    report_json = report_root / "product_roi_yolo_test64_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload["summary"]["configs"][config_name]
    print("Summary:", json.dumps(summary, indent=2), flush=True)
    print("Report root:", report_root, flush=True)
    print(report_json, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Product-ROI YOLO -> expiry YOLO-OBB audit for test64")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--truth-manifest", default=str(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--product-model-path", default=str(DEFAULT_PRODUCT_MODEL_PATH))
    parser.add_argument("--product-conf", type=float, default=0.15)
    parser.add_argument("--product-top-k", type=int, default=4)
    parser.add_argument("--product-padding-ratio", type=float, default=0.12)
    parser.add_argument("--product-imgsz", type=int, default=1024)
    parser.add_argument("--expiry-model-path", default=str(DEFAULT_YOLO_OBB_MODEL_PATH))
    parser.add_argument("--expiry-conf", type=float, default=0.01)
    parser.add_argument("--expiry-imgsz", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.20)
    parser.add_argument("--config-name", default="product_roi_yolo26s_expdate_conf001")
    parser.add_argument("--filenames", default=None, help="Comma-separated filename subset for a quick run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
