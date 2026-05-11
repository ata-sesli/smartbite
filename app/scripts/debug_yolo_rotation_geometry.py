from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.scripts.detector_truth_ablation import _load_truth_items, _xyxy_from_polygon


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _load_exif_normalized(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _rotate_image(image: np.ndarray, source_pass: str) -> np.ndarray:
    if source_pass == "normal":
        return image
    if source_pass == "rotate90":
        return np.rot90(image, k=3)
    if source_pass == "rotate180":
        return np.rot90(image, k=2)
    if source_pass == "rotate270":
        return np.rot90(image, k=1)
    raise ValueError(f"unsupported source_pass: {source_pass}")


def _inverse_point(x: float, y: float, *, source_pass: str, width: int, height: int) -> list[float]:
    if source_pass == "normal":
        return [x, y]
    if source_pass == "rotate90":
        return [y, height - x]
    if source_pass == "rotate180":
        return [width - x, height - y]
    if source_pass == "rotate270":
        return [width - y, x]
    raise ValueError(f"unsupported source_pass: {source_pass}")


def _forward_point(x: float, y: float, *, source_pass: str, width: int, height: int) -> list[float]:
    if source_pass == "normal":
        return [x, y]
    if source_pass == "rotate90":
        return [height - y, x]
    if source_pass == "rotate180":
        return [width - x, height - y]
    if source_pass == "rotate270":
        return [y, width - x]
    raise ValueError(f"unsupported source_pass: {source_pass}")


def _letterbox_meta(shape: tuple[int, int], imgsz: int) -> dict[str, Any]:
    height, width = shape
    ratio = min(imgsz / height, imgsz / width)
    new_width = round(width * ratio)
    new_height = round(height * ratio)
    pad_x = (imgsz - new_width) / 2
    pad_y = (imgsz - new_height) / 2
    return {
        "input_hw": [height, width],
        "model_input_hw": [imgsz, imgsz],
        "scale": ratio,
        "pad_xy": [pad_x, pad_y],
        "resized_wh": [new_width, new_height],
    }


def _to_letterbox_polygon(polygon: list[list[float]], meta: dict[str, Any]) -> list[list[float]]:
    scale = float(meta["scale"])
    pad_x, pad_y = [float(v) for v in meta["pad_xy"]]
    return [[x * scale + pad_x, y * scale + pad_y] for x, y in polygon]


def _from_letterbox_polygon(polygon: list[list[float]], meta: dict[str, Any]) -> list[list[float]]:
    scale = float(meta["scale"])
    pad_x, pad_y = [float(v) for v in meta["pad_xy"]]
    return [[(x - pad_x) / scale, (y - pad_y) / scale] for x, y in polygon]


def _polygon_area(polygon: list[list[float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    pts = np.asarray(polygon, dtype=np.float32)
    return float(abs(cv2.contourArea(pts)))


def _polygon_iou(a: list[list[float]], b: list[list[float]]) -> dict[str, float]:
    if len(a) < 3 or len(b) < 3:
        return {"iou": 0.0, "coverage": 0.0}
    poly_a = cv2.convexHull(np.asarray(a, dtype=np.float32))
    poly_b = cv2.convexHull(np.asarray(b, dtype=np.float32))
    area_a = max(float(cv2.contourArea(poly_a)), 1.0)
    area_b = max(float(cv2.contourArea(poly_b)), 1.0)
    inter_area, _ = cv2.intersectConvexConvex(poly_a, poly_b)
    inter = max(float(inter_area), 0.0)
    return {
        "iou": inter / max(area_a + area_b - inter, 1.0),
        "coverage": inter / area_a,
    }


def _draw_overlay(
    image: np.ndarray,
    *,
    truth_polygon: list[list[float]] | None,
    candidate_polygons: list[list[list[float]]],
    best_index: int | None = None,
) -> Image.Image:
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    for index, polygon in enumerate(candidate_polygons):
        color = "lime" if best_index is not None and index == best_index else "orange"
        draw.line([tuple(point) for point in polygon + [polygon[0]]], fill=color, width=5)
    if truth_polygon:
        draw.line([tuple(point) for point in truth_polygon + [truth_polygon[0]]], fill="dodgerblue", width=5)
    return canvas


def _crop(image: np.ndarray, bbox: list[int]) -> Image.Image:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (1, 1), "black")
    return Image.fromarray(image[y1:y2, x1:x2])


def _box_dict(
    *,
    source_pass: str,
    source_image_shape: tuple[int, int, int],
    model_input_shape: tuple[int, int, int],
    tile_origin: list[int] | None,
    raw_obb_xywhr: list[float],
    raw_polygon_in_model_space: list[list[float]],
    polygon_after_letterbox_unscale: list[list[float]],
    polygon_after_tile_offset: list[list[float]],
    polygon_after_rotation_inverse: list[list[float]],
    final_polygon_drawn: list[list[float]],
    confidence: float,
) -> dict[str, Any]:
    return {
        "source_pass": source_pass,
        "source_image_shape": list(source_image_shape),
        "model_input_shape": list(model_input_shape),
        "tile_origin": tile_origin,
        "raw_obb_xywhr": raw_obb_xywhr,
        "raw_polygon_in_model_space": raw_polygon_in_model_space,
        "polygon_after_letterbox_unscale": polygon_after_letterbox_unscale,
        "polygon_after_tile_offset": polygon_after_tile_offset,
        "polygon_after_rotation_inverse": polygon_after_rotation_inverse,
        "final_polygon_drawn": final_polygon_drawn,
        "confidence": confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug YOLO rotation fallback coordinate geometry for one test64 image")
    parser.add_argument("--filename", required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("test64"))
    parser.add_argument("--truth-manifest", type=Path, default=Path(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--model-path", type=Path, default=Path("models/yolo26s_obb_brazil_expdate_ft_date_due/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics"))
    args = parser.parse_args()

    image_path = args.images_dir / Path(args.filename).name
    truth_items = _load_truth_items(args.truth_manifest)
    truth_item = truth_items[Path(args.filename).name]
    truth_polygon = truth_item.get("true_polygon_xy")
    if not isinstance(truth_polygon, list):
        bbox = truth_item["true_bbox_xyxy"]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        truth_polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    truth_polygon = [[float(point[0]), float(point[1])] for point in truth_polygon]

    image = _load_exif_normalized(image_path)
    height, width = image.shape[:2]
    run_root = args.output_dir.resolve() / f"rotation_geometry_debug_{Path(args.filename).stem}_{_utc_stamp()}"
    crops_root = run_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image).save(run_root / "01_exif_normalized_input.jpg", quality=95)

    model = YOLO(str(args.model_path))
    trace: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "filename": Path(args.filename).name,
        "image_path": str(image_path),
        "model_path": str(args.model_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "exif_normalized_shape": list(image.shape),
        "truth_polygon": truth_polygon,
        "passes": [],
        "round_trip_tests": [],
        "notes": [
            "Ultralytics high-level Results.obb polygons are exposed in source-image pixel coordinates after letterbox unscale.",
            "raw_polygon_in_model_space is reconstructed by applying the same letterbox transform to the exposed source-space polygon.",
            "No tile pass is used by the Direct Brazil audit; tile_origin is null for every candidate.",
            "SVTR is not invoked by this detector audit; crop_used_for_SVTR is saved as the final inverse-mapped candidate crop proxy.",
        ],
    }

    final_polygons: list[list[list[float]]] = []
    crop_records: list[dict[str, Any]] = []
    best_index: int | None = None
    best_iou = -1.0

    for source_pass in ("normal", "rotate90", "rotate180", "rotate270"):
        pass_image = _rotate_image(image, source_pass)
        if source_pass != "normal":
            Image.fromarray(pass_image).save(run_root / f"02_{source_pass}_fallback_image.jpg", quality=95)

        letterbox = _letterbox_meta(pass_image.shape[:2], args.imgsz)
        model_input_shape = (args.imgsz, args.imgsz, 3)
        results = model.predict(pass_image, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0] if results else None
        pass_record: dict[str, Any] = {
            "source_pass": source_pass,
            "source_image_shape": list(pass_image.shape),
            "model_input_shape": list(model_input_shape),
            "letterbox": letterbox,
            "tile_origin": None,
            "candidates": [],
        }

        raw_polygons: list[list[list[float]]] = []
        if result is not None and getattr(result, "obb", None) is not None and len(result.obb):
            xywhr = result.obb.xywhr.cpu().numpy().tolist()
            polygons = result.obb.xyxyxyxy.cpu().numpy().tolist()
            confidences = result.obb.conf.cpu().numpy().tolist()
            for candidate_index, (raw_xywhr, polygon, confidence) in enumerate(
                zip(xywhr, polygons, confidences, strict=False),
                start=1,
            ):
                polygon_after_letterbox_unscale = [[float(x), float(y)] for x, y in polygon]
                raw_polygon_in_model_space = _to_letterbox_polygon(polygon_after_letterbox_unscale, letterbox)
                polygon_after_tile_offset = polygon_after_letterbox_unscale
                polygon_after_rotation_inverse = [
                    _inverse_point(
                        float(x),
                        float(y),
                        source_pass=source_pass,
                        width=width,
                        height=height,
                    )
                    for x, y in polygon_after_tile_offset
                ]
                final_polygon_drawn = polygon_after_rotation_inverse
                iou = _polygon_iou(truth_polygon, final_polygon_drawn)
                if iou["coverage"] > best_iou:
                    best_iou = iou["coverage"]
                    best_index = len(final_polygons)
                final_polygons.append(final_polygon_drawn)
                raw_polygons.append(raw_polygon_in_model_space)
                candidate = _box_dict(
                    source_pass=source_pass,
                    source_image_shape=pass_image.shape,
                    model_input_shape=model_input_shape,
                    tile_origin=None,
                    raw_obb_xywhr=[float(v) for v in raw_xywhr],
                    raw_polygon_in_model_space=raw_polygon_in_model_space,
                    polygon_after_letterbox_unscale=polygon_after_letterbox_unscale,
                    polygon_after_tile_offset=polygon_after_tile_offset,
                    polygon_after_rotation_inverse=polygon_after_rotation_inverse,
                    final_polygon_drawn=final_polygon_drawn,
                    confidence=float(confidence),
                )
                candidate["truth_polygon_iou"] = iou["iou"]
                candidate["truth_polygon_coverage"] = iou["coverage"]
                pass_record["candidates"].append(candidate)

                before_bbox = _xyxy_from_polygon(polygon_after_letterbox_unscale)
                after_bbox = _xyxy_from_polygon(final_polygon_drawn)
                stem = f"{source_pass}_{candidate_index:02d}"
                before_path = crops_root / f"{stem}_crop_before_inverse_mapping.jpg"
                after_path = crops_root / f"{stem}_crop_after_inverse_mapping.jpg"
                svtr_path = crops_root / f"{stem}_crop_used_for_SVTR.jpg"
                _crop(pass_image, before_bbox).save(before_path, quality=95)
                after_crop = _crop(image, after_bbox)
                after_crop.save(after_path, quality=95)
                after_crop.save(svtr_path, quality=95)
                crop_records.append(
                    {
                        "source_pass": source_pass,
                        "candidate_index": candidate_index,
                        "crop_before_inverse_mapping": str(before_path.relative_to(run_root)),
                        "crop_after_inverse_mapping": str(after_path.relative_to(run_root)),
                        "crop_used_for_SVTR": str(svtr_path.relative_to(run_root)),
                    }
                )

        letterbox_image = cv2.resize(
            pass_image,
            tuple(int(v) for v in letterbox["resized_wh"]),
            interpolation=cv2.INTER_LINEAR,
        )
        model_canvas = np.full((args.imgsz, args.imgsz, 3), 114, dtype=np.uint8)
        pad_x, pad_y = [int(round(float(v))) for v in letterbox["pad_xy"]]
        resized_w, resized_h = [int(v) for v in letterbox["resized_wh"]]
        model_canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = letterbox_image
        _draw_overlay(
            model_canvas,
            truth_polygon=None,
            candidate_polygons=raw_polygons,
        ).save(run_root / f"03_{source_pass}_model_letterbox_raw_detections.jpg", quality=95)
        trace["passes"].append(pass_record)

    for source_pass in ("normal", "rotate90", "rotate180", "rotate270"):
        meta = _letterbox_meta(_rotate_image(image, source_pass).shape[:2], args.imgsz)
        forward = [
            _forward_point(x, y, source_pass=source_pass, width=width, height=height)
            for x, y in truth_polygon
        ]
        letterboxed = _to_letterbox_polygon(forward, meta)
        unscaled = _from_letterbox_polygon(letterboxed, meta)
        inverse = [
            _inverse_point(x, y, source_pass=source_pass, width=width, height=height)
            for x, y in unscaled
        ]
        rt = _polygon_iou(truth_polygon, inverse)
        trace["round_trip_tests"].append(
            {
                "source_pass": source_pass,
                "forward_truth_polygon": forward,
                "letterboxed_truth_polygon": letterboxed,
                "inverse_truth_polygon": inverse,
                "iou": rt["iou"],
                "coverage": rt["coverage"],
                "passed": rt["iou"] > 0.999 and rt["coverage"] > 0.999,
            }
        )

    _draw_overlay(
        image,
        truth_polygon=truth_polygon,
        candidate_polygons=final_polygons,
        best_index=best_index,
    ).save(run_root / "04_full_image_overlay_after_inverse_mapping.jpg", quality=95)
    _draw_overlay(
        image,
        truth_polygon=truth_polygon,
        candidate_polygons=final_polygons,
        best_index=best_index,
    ).save(run_root / "05_final_web_visualization_overlay.jpg", quality=95)

    trace["candidate_crops"] = crop_records
    trace["final_candidate_count"] = len(final_polygons)
    trace["best_truth_coverage"] = best_iou
    trace_path = run_root / "geometry_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(trace_path)


if __name__ == "__main__":
    main()
