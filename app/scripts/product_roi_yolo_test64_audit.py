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


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = _bbox_area((x1, y1, x2, y2))
    if intersection <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_gap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    scale = max(1, max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]))
    return float((gap_x**2 + gap_y**2) ** 0.5) / scale


def _bbox_centrality(bbox: tuple[int, int, int, int], *, image_width: int, image_height: int) -> float:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    image_cx = image_width / 2.0
    image_cy = image_height / 2.0
    max_distance = max(1.0, (image_cx**2 + image_cy**2) ** 0.5)
    distance = ((cx - image_cx) ** 2 + (cy - image_cy) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / max_distance)


def _should_merge_rois(
    a: ProductRoi,
    b: ProductRoi,
    *,
    iou_threshold: float,
    nearby_gap_ratio: float,
) -> bool:
    return _bbox_iou(a.bbox_xyxy, b.bbox_xyxy) >= iou_threshold or _bbox_gap_ratio(
        a.bbox_xyxy,
        b.bbox_xyxy,
    ) <= nearby_gap_ratio


def _merge_product_rois(
    rois: list[ProductRoi],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float,
    nearby_gap_ratio: float,
) -> tuple[list[ProductRoi], dict[str, Any]]:
    clusters: list[list[ProductRoi]] = [[roi] for roi in rois]
    changed = True
    while changed:
        changed = False
        for left_index in range(len(clusters)):
            if changed:
                break
            for right_index in range(left_index + 1, len(clusters)):
                left = clusters[left_index]
                right = clusters[right_index]
                if any(
                    _should_merge_rois(
                        left_roi,
                        right_roi,
                        iou_threshold=iou_threshold,
                        nearby_gap_ratio=nearby_gap_ratio,
                    )
                    for left_roi in left
                    for right_roi in right
                ):
                    clusters[left_index] = [*left, *right]
                    del clusters[right_index]
                    changed = True
                    break

    image_area = max(1, image_width * image_height)
    scored_clusters: list[tuple[float, ProductRoi, list[ProductRoi]]] = []
    cluster_payloads: list[dict[str, Any]] = []
    for cluster in clusters:
        union_bbox = _bbox_union([roi.bbox_xyxy for roi in cluster])
        area_ratio = _bbox_area(union_bbox) / image_area
        centrality = _bbox_centrality(union_bbox, image_width=image_width, image_height=image_height)
        confidence = max(roi.confidence for roi in cluster)
        score = (area_ratio * 0.65) + (centrality * 0.25) + (confidence * 0.10)
        merged_roi = ProductRoi(union_bbox, confidence, "product_yolo_obb_merged")
        scored_clusters.append((score, merged_roi, cluster))
        cluster_payloads.append(
            {
                "bbox_xyxy": list(union_bbox),
                "score": score,
                "area_ratio": area_ratio,
                "centrality": centrality,
                "confidence": confidence,
                "member_count": len(cluster),
                "members": [
                    {"bbox_xyxy": list(roi.bbox_xyxy), "confidence": roi.confidence, "source": roi.source}
                    for roi in cluster
                ],
            }
        )

    scored_clusters.sort(key=lambda item: item[0], reverse=True)
    selected = [scored_clusters[0][1]] if scored_clusters else []
    return selected, {
        "product_roi_merge_enabled": True,
        "raw_product_roi_count": len(rois),
        "product_roi_cluster_count": len(clusters),
        "product_roi_clusters": sorted(cluster_payloads, key=lambda item: float(item["score"]), reverse=True),
    }


class ProductThenExpiryYoloDetector:
    def __init__(
        self,
        *,
        product_model_path: Path,
        product_conf: float,
        product_top_k: int,
        product_padding_ratio: float,
        product_imgsz: int,
        merge_product_rois: bool,
        product_merge_iou_threshold: float,
        product_merge_nearby_gap_ratio: float,
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
        self.merge_product_rois = merge_product_rois
        self.product_merge_iou_threshold = product_merge_iou_threshold
        self.product_merge_nearby_gap_ratio = product_merge_nearby_gap_ratio
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

    def _select_product_rois(self, rois: list[ProductRoi], image: np.ndarray) -> tuple[list[ProductRoi], dict[str, Any]]:
        if not self.merge_product_rois:
            return rois, {
                "product_roi_merge_enabled": False,
                "raw_product_roi_count": len(rois),
                "product_roi_cluster_count": None,
                "product_roi_clusters": [],
            }
        height, width = image.shape[:2]
        return _merge_product_rois(
            rois,
            image_width=width,
            image_height=height,
            iou_threshold=self.product_merge_iou_threshold,
            nearby_gap_ratio=self.product_merge_nearby_gap_ratio,
        )

    def _detect_products(self, image: np.ndarray) -> tuple[list[ProductRoi], dict[str, Any]]:
        product_model, _ = self._ensure_models()
        results = product_model.predict(image, conf=self.product_conf, imgsz=self.product_imgsz, verbose=False)
        if not results:
            return [], {
                "product_roi_merge_enabled": self.merge_product_rois,
                "raw_product_roi_count": 0,
                "product_roi_cluster_count": 0 if self.merge_product_rois else None,
                "product_roi_clusters": [],
            }
        height, width = image.shape[:2]
        rois: list[ProductRoi] = []

        obb_raw = getattr(results[0], "obb", None)
        if obb_raw is not None and len(obb_raw) > 0:
            polys_raw = getattr(obb_raw, "xyxyxyxy", None)
            conf_raw = getattr(obb_raw, "conf", None)
            if polys_raw is not None and conf_raw is not None:
                polygons = polys_raw.cpu().numpy().tolist()
                confidences = conf_raw.cpu().numpy().tolist()
                for polygon, confidence in zip(polygons, confidences, strict=False):
                    confidence_float = float(confidence)
                    if confidence_float < self.product_conf:
                        continue
                    raw_bbox_list = _xyxy_from_polygon([[float(x), float(y)] for x, y in polygon])
                    clipped_bbox = (
                        max(0, min(width, int(raw_bbox_list[0]))),
                        max(0, min(height, int(raw_bbox_list[1]))),
                        max(0, min(width, int(raw_bbox_list[2]))),
                        max(0, min(height, int(raw_bbox_list[3]))),
                    )
                    padded_bbox = _pad_bbox(
                        clipped_bbox,
                        image_width=width,
                        image_height=height,
                        padding_ratio=self.product_padding_ratio,
                    )
                    if padded_bbox[2] <= padded_bbox[0] or padded_bbox[3] <= padded_bbox[1]:
                        continue
                    rois.append(ProductRoi(padded_bbox, confidence_float, "product_yolo_obb"))
                rois.sort(key=lambda roi: roi.confidence, reverse=True)
                return self._select_product_rois(rois[: self.product_top_k], image)

        boxes_raw = getattr(results[0], "boxes", None)
        if boxes_raw is None or len(boxes_raw) == 0:
            return [], {
                "product_roi_merge_enabled": self.merge_product_rois,
                "raw_product_roi_count": 0,
                "product_roi_cluster_count": 0 if self.merge_product_rois else None,
                "product_roi_clusters": [],
            }
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
        return self._select_product_rois(rois[: self.product_top_k], image)

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
        product_rois, product_meta = self._detect_products(image)
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
            **product_meta,
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
        merge_product_rois=args.merge_product_rois,
        product_merge_iou_threshold=args.product_merge_iou_threshold,
        product_merge_nearby_gap_ratio=args.product_merge_nearby_gap_ratio,
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
                    "merge_product_rois": args.merge_product_rois,
                    "product_merge_iou_threshold": args.product_merge_iou_threshold,
                    "product_merge_nearby_gap_ratio": args.product_merge_nearby_gap_ratio,
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
    parser.add_argument(
        "--merge-product-rois",
        action="store_true",
        help="Merge nearby/overlapping top product ROIs and keep one large central product crop.",
    )
    parser.add_argument("--product-merge-iou-threshold", type=float, default=0.01)
    parser.add_argument("--product-merge-nearby-gap-ratio", type=float, default=0.25)
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
