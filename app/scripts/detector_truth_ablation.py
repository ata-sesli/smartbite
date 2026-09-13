from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.infra.settings import Settings, get_settings
from app.infra.storage import LocalStorage
from app.scripts.forensic_test64_topk import _load_latest_batch
from app.workers.scan_jobs import build_pipeline


DEFAULT_CUSTOM_DET_MODEL_DIR = Path("models/pp-ocrv5-text-detection/best_model_inference")
DEFAULT_YOLO_OBB_MODEL_PATH = Path("models/yolo26s_obb_expdate_det/weights/best.pt")


@dataclass(frozen=True)
class DetectorAblationConfig:
    name: str
    det_db_thresh: float
    det_db_box_thresh: float
    det_limit_side_len: int | None = None
    det_limit_type: str | None = None
    det_db_unclip_ratio: float | None = None

    def settings_payload(self) -> dict[str, Any]:
        return {
            "ppocrv5_det_db_thresh": self.det_db_thresh,
            "ppocrv5_det_db_box_thresh": self.det_db_box_thresh,
            "ppocrv5_det_limit_side_len": self.det_limit_side_len,
            "ppocrv5_det_limit_type": self.det_limit_type,
            "ppocrv5_det_db_unclip_ratio": self.det_db_unclip_ratio,
        }


ABLATION_CONFIGS: tuple[DetectorAblationConfig, ...] = (
    DetectorAblationConfig("baseline_custom", det_db_thresh=0.30, det_db_box_thresh=0.50),
    DetectorAblationConfig(
        "scale_1216",
        det_db_thresh=0.30,
        det_db_box_thresh=0.50,
        det_limit_side_len=1216,
        det_limit_type="max",
    ),
    DetectorAblationConfig(
        "recall_1216",
        det_db_thresh=0.25,
        det_db_box_thresh=0.35,
        det_limit_side_len=1216,
        det_limit_type="max",
    ),
    DetectorAblationConfig(
        "recall_1216_unclip18",
        det_db_thresh=0.25,
        det_db_box_thresh=0.35,
        det_limit_side_len=1216,
        det_limit_type="max",
        det_db_unclip_ratio=1.8,
    ),
    DetectorAblationConfig(
        "scale_1536_recall",
        det_db_thresh=0.25,
        det_db_box_thresh=0.35,
        det_limit_side_len=1536,
        det_limit_type="max",
        det_db_unclip_ratio=1.8,
    ),
    DetectorAblationConfig(
        "min960_diagnostic",
        det_db_thresh=0.25,
        det_db_box_thresh=0.35,
        det_limit_side_len=960,
        det_limit_type="min",
        det_db_unclip_ratio=1.8,
    ),
)


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_truth_items(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError(f"truth manifest must contain an items object: {path}")
    return {name: item for name, item in items.items() if isinstance(item, dict)}


def _clean_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_overlap(
    candidate: tuple[int, int, int, int],
    truth: tuple[int, int, int, int],
) -> dict[str, float]:
    ix1 = max(candidate[0], truth[0])
    iy1 = max(candidate[1], truth[1])
    ix2 = min(candidate[2], truth[2])
    iy2 = min(candidate[3], truth[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
    truth_area = max(1, (truth[2] - truth[0]) * (truth[3] - truth[1]))
    union = max(1, candidate_area + truth_area - inter)
    return {
        "iou": inter / float(union),
        "truth_coverage_ratio": inter / float(truth_area),
        "candidate_coverage_ratio": inter / float(candidate_area),
    }


def _classify_truth_overlap(
    *,
    box_count: int,
    best_iou: float,
    truth_coverage_ratio: float,
    candidate_coverage_ratio: float,
) -> str:
    if box_count <= 0:
        return "no_box"
    if truth_coverage_ratio >= 0.75 or best_iou >= 0.50:
        return "covered"
    if truth_coverage_ratio >= 0.45 and candidate_coverage_ratio >= 0.55:
        return "tight"
    if truth_coverage_ratio >= 0.20 or best_iou >= 0.08:
        return "partial"
    return "missed_truth"


def _best_overlap(
    boxes: list[tuple[int, int, int, int]],
    truth_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    if not boxes:
        return {
            "verdict": "no_box",
            "box_count": 0,
            "best_box": None,
            "best_iou": 0.0,
            "truth_coverage_ratio": 0.0,
            "candidate_coverage_ratio": 0.0,
            "covers_truth": False,
        }
    scored = [(box, _bbox_overlap(box, truth_bbox)) for box in boxes]
    best_box, best = max(scored, key=lambda item: (item[1]["truth_coverage_ratio"], item[1]["iou"]))
    verdict = _classify_truth_overlap(
        box_count=len(boxes),
        best_iou=best["iou"],
        truth_coverage_ratio=best["truth_coverage_ratio"],
        candidate_coverage_ratio=best["candidate_coverage_ratio"],
    )
    return {
        "verdict": verdict,
        "box_count": len(boxes),
        "best_box": list(best_box),
        "best_iou": best["iou"],
        "truth_coverage_ratio": best["truth_coverage_ratio"],
        "candidate_coverage_ratio": best["candidate_coverage_ratio"],
        "covers_truth": verdict == "covered",
    }


def _summarize_config_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    images = len(results)
    summary: dict[str, Any] = {
        "images": images,
        "covered": 0,
        "tight": 0,
        "partial": 0,
        "missed_truth": 0,
        "no_box": 0,
        "average_box_count": 0.0,
        "average_runtime_ms": 0.0,
        "mean_truth_coverage_ratio": 0.0,
    }
    if not results:
        return summary
    for result in results:
        verdict = str(result.get("verdict") or "missed_truth")
        if verdict in summary:
            summary[verdict] += 1
    summary["average_box_count"] = sum(float(result.get("box_count") or 0) for result in results) / images
    summary["average_runtime_ms"] = sum(float(result.get("runtime_ms") or 0) for result in results) / images
    summary["mean_truth_coverage_ratio"] = (
        sum(float(result.get("truth_coverage_ratio") or 0) for result in results) / images
    )
    return summary


def _settings_for_config(base: Settings, config: DetectorAblationConfig, *, model_dir: Path) -> Settings:
    return base.model_copy(
        update={
            "text_detector_mode": "ppocrv5_server",
            "ocr_ppocrv5_use_custom_text_det_model": True,
            "ocr_ppocrv5_text_det_model_dir": model_dir,
            "craft_enabled": False,
            "ocr_ppocrv5_det_db_thresh": config.det_db_thresh,
            "ocr_ppocrv5_det_db_box_thresh": config.det_db_box_thresh,
            "ocr_ppocrv5_det_limit_side_len": config.det_limit_side_len,
            "ocr_ppocrv5_det_limit_type": config.det_limit_type,
            "ocr_ppocrv5_det_db_unclip_ratio": config.det_db_unclip_ratio,
        }
    )


def _selected_configs(raw_names: str | None) -> list[DetectorAblationConfig]:
    if not raw_names:
        return list(ABLATION_CONFIGS)
    wanted = {name.strip() for name in raw_names.split(",") if name.strip()}
    configs = [config for config in ABLATION_CONFIGS if config.name in wanted]
    missing = wanted - {config.name for config in configs}
    if missing:
        raise ValueError(f"unknown config names: {', '.join(sorted(missing))}")
    return configs


def _selected_filenames(raw_names: str | None) -> set[str] | None:
    if not raw_names:
        return None
    return {Path(name.strip()).name for name in raw_names.split(",") if name.strip()}


def _scan_box_dict(box: Any, *, offset_x: int, offset_y: int, roi_source: str) -> dict[str, Any]:
    bbox = [box.x1 + offset_x, box.y1 + offset_y, box.x2 + offset_x, box.y2 + offset_y]
    polygon = None
    if box.polygon_xy is not None:
        polygon = [[float(x) + offset_x, float(y) + offset_y] for x, y in box.polygon_xy]
    return {
        "bbox_xyxy": bbox,
        "polygon_xy": polygon,
        "confidence": box.confidence,
        "source": box.source,
        "sources": list(box.sources),
        "variant_name": box.variant_name,
        "roi_source": roi_source,
    }


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    try:
        with Image.open(BytesIO(image_bytes)) as image_raw:
            image = ImageOps.exif_transpose(image_raw).convert("RGB")
            return np.asarray(image)
    except Exception:
        return None


def _xyxy_from_polygon(polygon: list[list[float]]) -> list[int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    ]


def _yolo_obb_box_dict(*, polygon: list[list[float]], confidence: float, model_path: Path) -> dict[str, Any]:
    return {
        "bbox_xyxy": _xyxy_from_polygon(polygon),
        "polygon_xy": polygon,
        "confidence": confidence,
        "source": "yolo26s_obb",
        "sources": ["yolo26s_obb"],
        "variant_name": "original",
        "roi_source": "full_image",
        "model_path": str(model_path),
    }


class YoloObbDetector:
    def __init__(self, model_path: Path, *, confidence_threshold: float) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            if not self.model_path.exists():
                raise ValueError(f"YOLO OBB model not found: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        model = self._ensure_model()
        results = model.predict(image, conf=self.confidence_threshold, verbose=False)
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

        polygons = polys_raw.cpu().numpy().tolist()
        confidences = conf_raw.cpu().numpy().tolist()
        boxes = [
            _yolo_obb_box_dict(
                polygon=[[float(x), float(y)] for x, y in polygon],
                confidence=float(confidence),
                model_path=self.model_path,
            )
            for polygon, confidence in zip(polygons, confidences, strict=False)
            if float(confidence) >= self.confidence_threshold
        ]
        boxes.sort(key=lambda box: float(box.get("confidence") or 0.0), reverse=True)
        return boxes, {
            "yolo_detected": bool(boxes),
            "yolo_reason": None if boxes else "no candidate passed confidence threshold",
            "roi_count": 1,
            "detector_unavailable_reasons": [],
        }


def _detect_boxes_for_image(pipeline: Any, image: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detection = pipeline.detector.detect(image)
    roi_candidates = pipeline._collect_roi_candidates(image=image, detection=detection)
    detector_boxes: list[dict[str, Any]] = []
    reasons: list[str] = []
    for roi_candidate in roi_candidates:
        for variant in pipeline._variants_for_roi(roi_candidate.image):
            raw_boxes, reason = pipeline._detect_text_boxes(variant.image, variant_name=variant.name)
            if reason:
                reasons.append(reason)
            mapped = pipeline._map_text_boxes_to_original(
                boxes=raw_boxes,
                variant=variant,
                original_shape=roi_candidate.image.shape,
            )
            for box in mapped:
                detector_boxes.append(
                    _scan_box_dict(
                        box,
                        offset_x=roi_candidate.offset_x,
                        offset_y=roi_candidate.offset_y,
                        roi_source=roi_candidate.source,
                    )
                )
    meta = {
        "yolo_detected": detection.detected,
        "yolo_reason": detection.reason,
        "roi_count": len(roi_candidates),
        "detector_unavailable_reasons": sorted(set(reasons)),
    }
    return detector_boxes, meta


def _audit_yolo_obb_image(detector: YoloObbDetector, image: np.ndarray, truth_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
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


def _audit_single_image(pipeline: Any, image: np.ndarray, truth_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    started = perf_counter()
    detector_boxes, meta = _detect_boxes_for_image(pipeline, image)
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


async def main_async(args: argparse.Namespace) -> int:
    truth_items = _load_truth_items(Path(args.truth_manifest))
    selected_names = _selected_filenames(args.filenames)
    configs = _selected_configs(args.configs) if args.backend == "ppocrv5" else []
    settings = get_settings()
    storage = LocalStorage(settings.storage_root)
    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    rows = [row for row in rows if isinstance(row.get("filename"), str)]
    if selected_names is not None:
        rows = [row for row in rows if Path(str(row["filename"])).name in selected_names]
    rows = [
        row
        for row in rows
        if (Path(str(row["filename"])).name in truth_items)
        and _clean_bbox(truth_items[Path(str(row["filename"])).name].get("true_bbox_xyxy")) is not None
    ]
    if not rows:
        print("No rows with manual truth bboxes found.")
        return 1

    report_root = Path(args.output_dir).resolve() / f"test64_detector_audit_{_utc_stamp()}"
    _ensure_dir(report_root)
    item_results: dict[str, dict[str, Any]] = {
        Path(str(row["filename"])).name: {
            "filename": Path(str(row["filename"])).name,
            "truth_bbox_xyxy": truth_items[Path(str(row["filename"])).name].get("true_bbox_xyxy"),
            "configs": [],
        }
        for row in rows
    }
    config_names = [config.name for config in configs] if args.backend == "ppocrv5" else [args.yolo_config_name]
    results_by_config: dict[str, list[dict[str, Any]]] = {name: [] for name in config_names}

    if args.backend == "yolo_obb":
        detector = YoloObbDetector(Path(args.yolo_model_path), confidence_threshold=args.yolo_conf)
        for index, row in enumerate(rows, start=1):
            filename = Path(str(row["filename"])).name
            image_bytes = storage.read_bytes(str(row["image_path"]))
            image = _decode_image_bytes(image_bytes)
            if image is None:
                continue
            truth_bbox = _clean_bbox(truth_items[filename].get("true_bbox_xyxy"))
            if truth_bbox is None:
                continue
            result = _audit_yolo_obb_image(detector, image, truth_bbox)
            config_result = {"config_name": args.yolo_config_name, **result}
            item_results[filename]["configs"].append(config_result)
            results_by_config[args.yolo_config_name].append(config_result)
            print(
                f"[{args.yolo_config_name} {index}/{len(rows)}] {filename} -> "
                f"{result['verdict']} boxes={result['box_count']} truth={result['truth_coverage_ratio']:.3f}"
            )

    for config in configs:
        config_settings = _settings_for_config(settings, config, model_dir=Path(args.ppocrv5_text_det_model_dir))
        pipeline = build_pipeline(config_settings)
        for index, row in enumerate(rows, start=1):
            filename = Path(str(row["filename"])).name
            image_bytes = storage.read_bytes(str(row["image_path"]))
            image = pipeline._decode_image(image_bytes)
            if image is None:
                continue
            truth_bbox = _clean_bbox(truth_items[filename].get("true_bbox_xyxy"))
            if truth_bbox is None:
                continue
            result = _audit_single_image(pipeline, image, truth_bbox)
            config_result = {"config_name": config.name, **result}
            item_results[filename]["configs"].append(config_result)
            results_by_config[config.name].append(config_result)
            print(
                f"[{config.name} {index}/{len(rows)}] {filename} -> "
                f"{result['verdict']} boxes={result['box_count']} truth={result['truth_coverage_ratio']:.3f}"
            )

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": args.source,
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "backend": args.backend,
        "ppocrv5_text_det_model_dir": str(Path(args.ppocrv5_text_det_model_dir)),
        "yolo_model_path": str(Path(args.yolo_model_path)) if args.backend == "yolo_obb" else None,
        "configs": (
            [{"name": config.name, "settings": config.settings_payload()} for config in configs]
            if args.backend == "ppocrv5"
            else [
                {
                    "name": args.yolo_config_name,
                    "settings": {
                        "model_path": str(Path(args.yolo_model_path)),
                        "confidence_threshold": args.yolo_conf,
                    },
                }
            ]
        ),
        "summary": {
            "configs": {
                config_name: _summarize_config_results(results)
                for config_name, results in results_by_config.items()
            }
        },
        "items": list(item_results.values()),
    }
    report_json = report_root / "detector_truth_ablation_report.json"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Report root:", report_root)
    print(report_json)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-only truth-overlap ablation for test64")
    parser.add_argument("--backend", choices=("ppocrv5", "yolo_obb"), default="ppocrv5")
    parser.add_argument("--source", default="test64-bulk-queue")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--truth-manifest", default=str(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--ppocrv5-text-det-model-dir", default=str(DEFAULT_CUSTOM_DET_MODEL_DIR))
    parser.add_argument("--yolo-model-path", default=str(DEFAULT_YOLO_OBB_MODEL_PATH))
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-config-name", default="yolo26s_obb_best")
    parser.add_argument("--configs", default=None, help="Comma-separated config names. Defaults to all ablation configs.")
    parser.add_argument("--filenames", default=None, help="Comma-separated filename subset for a quick bad-case run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
