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
from app.infra.settings import get_settings
from app.infra.storage import LocalStorage
from app.scripts.detector_truth_ablation import (
    DEFAULT_YOLO_OBB_MODEL_PATH,
    YoloObbDetector,
    _best_overlap,
    _clean_bbox,
    _decode_image_bytes,
    _ensure_dir,
    _load_truth_items,
    _selected_filenames,
    _summarize_config_results,
    _xyxy_from_polygon,
)
from app.scripts.forensic_test64_topk import _load_latest_batch


@dataclass(frozen=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int
    source: str


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _tile_windows(
    image_shape: tuple[int, int, int],
    *,
    tile_size: int,
    overlap: float,
    include_full_image: bool,
) -> list[TileWindow]:
    height, width = image_shape[:2]
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    windows: list[TileWindow] = []
    if include_full_image:
        windows.append(TileWindow(0, 0, width, height, "full_image"))
    for y1 in _tile_starts(height, tile_size, stride):
        for x1 in _tile_starts(width, tile_size, stride):
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            if include_full_image and x1 == 0 and y1 == 0 and x2 == width and y2 == height:
                continue
            windows.append(TileWindow(x1, y1, x2, y2, "tile"))
    return windows


def _offset_polygon(polygon: list[list[float]], *, offset_x: int, offset_y: int) -> list[list[float]]:
    return [[float(x) + offset_x, float(y) + offset_y] for x, y in polygon]


def _dedupe_boxes_by_bbox(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_bbox: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for box in boxes:
        bbox = box.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        key = tuple(int(round(float(value))) for value in bbox)
        current = best_by_bbox.get(key)
        if current is None or float(box.get("confidence") or 0.0) > float(current.get("confidence") or 0.0):
            best_by_bbox[key] = box
    return sorted(best_by_bbox.values(), key=lambda item: float(item.get("confidence") or 0.0), reverse=True)


def _summary_with_success(results: list[dict[str, Any]], *, success_threshold: float) -> dict[str, Any]:
    summary = _summarize_config_results(results)
    successes = sum(float(result.get("truth_coverage_ratio") or 0.0) > success_threshold for result in results)
    images = int(summary["images"])
    summary["success_threshold"] = success_threshold
    summary["success_gt_threshold"] = successes
    summary["success_rate"] = successes / images if images else 0.0
    return summary


class TiledYoloObbDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        tile_size: int,
        overlap: float,
        include_full_image: bool,
        imgsz: int,
        max_candidates: int,
    ) -> None:
        self.base_detector = YoloObbDetector(model_path, confidence_threshold=confidence_threshold)
        self.tile_size = tile_size
        self.overlap = overlap
        self.include_full_image = include_full_image
        self.imgsz = imgsz
        self.max_candidates = max_candidates

    def _detect_window(self, image: np.ndarray, window: TileWindow) -> list[dict[str, Any]]:
        model = self.base_detector._ensure_model()
        crop = image[window.y1 : window.y2, window.x1 : window.x2]
        results = model.predict(crop, conf=self.base_detector.confidence_threshold, imgsz=self.imgsz, verbose=False)
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
            if confidence_float < self.base_detector.confidence_threshold:
                continue
            mapped_polygon = _offset_polygon(
                [[float(x), float(y)] for x, y in polygon],
                offset_x=window.x1,
                offset_y=window.y1,
            )
            boxes.append(
                {
                    "bbox_xyxy": _xyxy_from_polygon(mapped_polygon),
                    "polygon_xy": mapped_polygon,
                    "confidence": confidence_float,
                    "source": "yolo26s_obb_tiled",
                    "sources": ["yolo26s_obb_tiled"],
                    "variant_name": window.source,
                    "roi_source": window.source,
                    "tile_xyxy": [window.x1, window.y1, window.x2, window.y2],
                    "model_path": str(self.base_detector.model_path),
                }
            )
        return boxes

    def detect(self, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        windows = _tile_windows(
            image.shape,
            tile_size=self.tile_size,
            overlap=self.overlap,
            include_full_image=self.include_full_image,
        )
        boxes: list[dict[str, Any]] = []
        for window in windows:
            boxes.extend(self._detect_window(image, window))
        boxes = _dedupe_boxes_by_bbox(boxes)
        if self.max_candidates > 0:
            boxes = boxes[: self.max_candidates]
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": None if boxes else "no tiled expiry region detected",
            "tile_count": len(windows),
            "detector_unavailable_reasons": [],
        }


def _audit_tiled_yolo_image(
    detector: TiledYoloObbDetector,
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


async def _load_rows(args: argparse.Namespace, truth_items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected_names = _selected_filenames(args.filenames)
    if args.source == "test64-files":
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
            rows.append({"filename": basename, "image_path": str(image_path), "image_source": "filesystem"})
        return rows

    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    rows = [row for row in rows if isinstance(row.get("filename"), str)]
    if selected_names is not None:
        rows = [row for row in rows if Path(str(row["filename"])).name in selected_names]
    return [
        row
        for row in rows
        if (Path(str(row["filename"])).name in truth_items)
        and _clean_bbox(truth_items[Path(str(row["filename"])).name].get("true_bbox_xyxy")) is not None
    ]


def _read_row_image_bytes(row: dict[str, Any], storage: LocalStorage | None) -> bytes:
    if row.get("image_source") == "filesystem":
        return Path(str(row["image_path"])).read_bytes()
    if storage is None:
        raise ValueError("storage is required for non-filesystem rows")
    return storage.read_bytes(str(row["image_path"]))


async def main_async(args: argparse.Namespace) -> int:
    truth_items = _load_truth_items(Path(args.truth_manifest))
    storage = None if args.source == "test64-files" else LocalStorage(get_settings().storage_root)
    rows = await _load_rows(args, truth_items)
    if not rows:
        print("No rows with manual truth bboxes found.")
        return 1

    detector = TiledYoloObbDetector(
        Path(args.yolo_model_path),
        confidence_threshold=args.yolo_conf,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        include_full_image=not args.no_full_image,
        imgsz=args.imgsz,
        max_candidates=args.max_candidates,
    )
    config_name = args.config_name
    results: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    print(
        f"Starting tiled YOLO audit: items={len(rows)} conf={args.yolo_conf} "
        f"tile_size={args.tile_size} overlap={args.tile_overlap} "
        f"imgsz={args.imgsz} max_candidates={args.max_candidates}",
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        filename = Path(str(row["filename"])).name
        item_started = perf_counter()
        print(f"[{config_name} {index}/{len(rows)}] starting {filename}", flush=True)
        image_bytes = _read_row_image_bytes(row, storage)
        image = _decode_image_bytes(image_bytes)
        truth_bbox = _clean_bbox(truth_items[filename].get("true_bbox_xyxy"))
        if image is None or truth_bbox is None:
            print(f"[{config_name} {index}/{len(rows)}] skipped {filename}: invalid image or truth bbox", flush=True)
            continue
        result = _audit_tiled_yolo_image(detector, image, truth_bbox)
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
            f"{success_mark} boxes={result['box_count']} tiles={result['tile_count']} "
            f"truth={result['truth_coverage_ratio']:.3f} "
            f"runtime_ms={result['runtime_ms']:.1f} elapsed_ms={(perf_counter() - item_started) * 1000:.1f}",
            flush=True,
        )

    report_root = Path(args.output_dir).resolve() / f"test64_tiled_yolo_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": args.source,
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "backend": "yolo_obb_tiled",
        "yolo_model_path": str(Path(args.yolo_model_path)),
        "configs": [
            {
                "name": config_name,
                "settings": {
                    "model_path": str(Path(args.yolo_model_path)),
                    "confidence_threshold": args.yolo_conf,
                    "tile_size": args.tile_size,
                    "tile_overlap": args.tile_overlap,
                    "include_full_image": not args.no_full_image,
                    "imgsz": args.imgsz,
                    "max_candidates": args.max_candidates,
                    "success_threshold": args.success_threshold,
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
    report_json = report_root / "tiled_yolo_test64_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload["summary"]["configs"][config_name]
    print("Summary:", json.dumps(summary, indent=2), flush=True)
    print("Report root:", report_root, flush=True)
    print(report_json, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiled YOLO-OBB candidate audit for test64")
    parser.add_argument("--source", default="test64-files", help="Use test64-files for local files, or a queued batch source.")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--truth-manifest", default=str(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--yolo-model-path", default=str(DEFAULT_YOLO_OBB_MODEL_PATH))
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--config-name", default="yolo26s_obb_tiled_conf001")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.20)
    parser.add_argument("--no-full-image", action="store_true")
    parser.add_argument("--filenames", default=None, help="Comma-separated filename subset for a quick run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
