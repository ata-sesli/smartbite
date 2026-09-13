from __future__ import annotations

import argparse
import asyncio
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
    _best_overlap,
    _clean_bbox,
    _decode_image_bytes,
    _ensure_dir,
    _load_truth_items,
    _selected_filenames,
    _xyxy_from_polygon,
)
from app.scripts.forensic_test64_topk import _load_latest_batch
from app.scripts.tiled_yolo_test64_audit import _summary_with_success


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _polygon_from_sahi_prediction(prediction: Any) -> list[list[float]]:
    mask = getattr(prediction, "mask", None)
    segmentation = getattr(mask, "segmentation", None)
    if isinstance(segmentation, list) and segmentation:
        first_segment = segmentation[0]
        if isinstance(first_segment, list) and len(first_segment) >= 8:
            return [
                [float(first_segment[index]), float(first_segment[index + 1])]
                for index in range(0, min(len(first_segment), 8), 2)
            ]

    x1, y1, x2, y2 = [float(value) for value in prediction.bbox.to_xyxy()]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _sahi_prediction_to_box(prediction: Any, *, model_path: Path) -> dict[str, Any]:
    polygon = _polygon_from_sahi_prediction(prediction)
    return {
        "bbox_xyxy": _xyxy_from_polygon(polygon),
        "polygon_xy": polygon,
        "confidence": float(prediction.score.value),
        "source": "sahi_yolo26s_obb",
        "sources": ["sahi", "yolo26s_obb"],
        "variant_name": "sahi_sliced",
        "roi_source": "sahi_sliced",
        "category_name": str(prediction.category.name),
        "model_path": str(model_path),
    }


def _sort_and_cap_boxes(boxes: list[dict[str, Any]], *, max_candidates: int) -> list[dict[str, Any]]:
    boxes = sorted(boxes, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    if max_candidates > 0:
        return boxes[:max_candidates]
    return boxes


class SahiYoloObbDetector:
    def __init__(
        self,
        model_path: Path,
        *,
        confidence_threshold: float,
        slice_size: int,
        overlap: float,
        include_full_image: bool,
        imgsz: int,
        max_candidates: int,
        device: str,
        postprocess_type: str,
        postprocess_match_metric: str,
        postprocess_match_threshold: float,
        postprocess_class_agnostic: bool,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.slice_size = slice_size
        self.overlap = overlap
        self.include_full_image = include_full_image
        self.imgsz = imgsz
        self.max_candidates = max_candidates
        self.device = device
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold
        self.postprocess_class_agnostic = postprocess_class_agnostic
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise ValueError(f"YOLO OBB model not found: {self.model_path}")
            from sahi import AutoDetectionModel

            self._model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(self.model_path),
                confidence_threshold=self.confidence_threshold,
                device=self.device,
                image_size=self.imgsz,
            )
        return self._model

    def detect(self, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import get_slice_bboxes

        model = self._ensure_model()
        slice_bboxes = get_slice_bboxes(
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            auto_slice_resolution=False,
        )
        result = get_sliced_prediction(
            image,
            model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            perform_standard_pred=self.include_full_image,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
            postprocess_class_agnostic=self.postprocess_class_agnostic,
            auto_slice_resolution=False,
            verbose=0,
        )
        boxes = [
            _sahi_prediction_to_box(prediction, model_path=self.model_path)
            for prediction in result.object_prediction_list
            if float(prediction.score.value) >= self.confidence_threshold
        ]
        boxes = _sort_and_cap_boxes(boxes, max_candidates=self.max_candidates)
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": None if boxes else "no SAHI sliced expiry region detected",
            "slice_count": len(slice_bboxes) + (1 if self.include_full_image else 0),
            "detector_unavailable_reasons": [],
        }


def _audit_sahi_yolo_image(
    detector: SahiYoloObbDetector,
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

    detector = SahiYoloObbDetector(
        Path(args.yolo_model_path),
        confidence_threshold=args.yolo_conf,
        slice_size=args.slice_size,
        overlap=args.slice_overlap,
        include_full_image=not args.no_full_image,
        imgsz=args.imgsz,
        max_candidates=args.max_candidates,
        device=args.device,
        postprocess_type=args.postprocess_type,
        postprocess_match_metric=args.postprocess_match_metric,
        postprocess_match_threshold=args.postprocess_match_threshold,
        postprocess_class_agnostic=args.postprocess_class_agnostic,
    )
    config_name = args.config_name
    results: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    print(
        f"Starting SAHI YOLO audit: items={len(rows)} conf={args.yolo_conf} "
        f"slice_size={args.slice_size} overlap={args.slice_overlap} imgsz={args.imgsz} "
        f"max_candidates={args.max_candidates} postprocess={args.postprocess_type}/"
        f"{args.postprocess_match_metric}@{args.postprocess_match_threshold}",
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
        result = _audit_sahi_yolo_image(detector, image, truth_bbox)
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
            f"{success_mark} boxes={result['box_count']} slices={result['slice_count']} "
            f"truth={result['truth_coverage_ratio']:.3f} "
            f"runtime_ms={result['runtime_ms']:.1f} elapsed_ms={(perf_counter() - item_started) * 1000:.1f}",
            flush=True,
        )

    report_root = Path(args.output_dir).resolve() / f"test64_sahi_yolo_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": args.source,
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "backend": "sahi_yolo_obb",
        "yolo_model_path": str(Path(args.yolo_model_path)),
        "configs": [
            {
                "name": config_name,
                "settings": {
                    "model_path": str(Path(args.yolo_model_path)),
                    "confidence_threshold": args.yolo_conf,
                    "slice_size": args.slice_size,
                    "slice_overlap": args.slice_overlap,
                    "include_full_image": not args.no_full_image,
                    "imgsz": args.imgsz,
                    "max_candidates": args.max_candidates,
                    "device": args.device,
                    "postprocess_type": args.postprocess_type,
                    "postprocess_match_metric": args.postprocess_match_metric,
                    "postprocess_match_threshold": args.postprocess_match_threshold,
                    "postprocess_class_agnostic": args.postprocess_class_agnostic,
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
    report_json = report_root / "sahi_yolo_test64_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload["summary"]["configs"][config_name]
    print("Summary:", json.dumps(summary, indent=2), flush=True)
    print("Report root:", report_root, flush=True)
    print(report_json, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAHI sliced YOLO-OBB candidate audit for test64")
    parser.add_argument("--source", default="test64-files", help="Use test64-files for local files, or a queued batch source.")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--truth-manifest", default=str(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--yolo-model-path", default=str(DEFAULT_YOLO_OBB_MODEL_PATH))
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--config-name", default="yolo26s_obb_sahi_conf001")
    parser.add_argument("--slice-size", type=int, default=1280)
    parser.add_argument("--slice-overlap", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--success-threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--postprocess-type", default="GREEDYNMM")
    parser.add_argument("--postprocess-match-metric", default="IOS")
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess-class-agnostic", action="store_true")
    parser.add_argument("--no-full-image", action="store_true")
    parser.add_argument("--filenames", default=None, help="Comma-separated filename subset for a quick run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
