from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


BBox = tuple[int, int, int, int]
Polygon = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TextLineGeometry:
    bbox_xyxy: BBox
    polygon_xy: Polygon | None = None


@dataclass(frozen=True, slots=True)
class TextLineCropConfig:
    padding_px: int = 0
    vertical_aspect_threshold: float = 1.5


@dataclass(frozen=True, slots=True)
class NormalizedTextLineCrop:
    image: np.ndarray
    bbox_xyxy: BBox
    polygon_used: bool
    crop_transform_used: str
    selected_orientation: str
    original_crop_shape: list[int]
    normalized_crop_shape: list[int]
    orientation_candidates_tried: list[str]
    selected_transform_reason: str


def normalize_textline_crop(
    image: np.ndarray,
    geometry: TextLineGeometry,
    config: TextLineCropConfig | None = None,
) -> NormalizedTextLineCrop:
    return normalize_textline_crops(image, geometry, config)[0]


def normalize_textline_crops(
    image: np.ndarray,
    geometry: TextLineGeometry,
    config: TextLineCropConfig | None = None,
) -> list[NormalizedTextLineCrop]:
    config = config or TextLineCropConfig()
    base, transform, polygon_used = _base_crop(image, geometry, config)
    if base.size == 0:
        base = np.zeros((1, 1, 3), dtype=np.uint8)
    original_shape = _shape(base)
    orientations = ["original"]
    h, w = base.shape[:2]
    if w > 0 and h / float(w) >= config.vertical_aspect_threshold:
        orientations.extend(["rotate_90_cw", "rotate_90_ccw", "rotate_180"])

    out: list[NormalizedTextLineCrop] = []
    for orientation in orientations:
        normalized = _orient(base, orientation)
        crop_transform = transform if orientation == "original" else orientation
        reason = "vertical_aspect" if len(orientations) > 1 else "horizontal_or_square"
        out.append(
            NormalizedTextLineCrop(
                image=normalized,
                bbox_xyxy=geometry.bbox_xyxy,
                polygon_used=polygon_used,
                crop_transform_used=crop_transform,
                selected_orientation=orientation,
                original_crop_shape=original_shape,
                normalized_crop_shape=_shape(normalized),
                orientation_candidates_tried=orientations,
                selected_transform_reason=reason,
            )
        )
    return out


def _base_crop(
    image: np.ndarray,
    geometry: TextLineGeometry,
    config: TextLineCropConfig,
) -> tuple[np.ndarray, str, bool]:
    polygon = _valid_polygon(geometry.polygon_xy)
    if polygon is not None:
        warped = _warp_polygon(image, polygon)
        if config.padding_px > 0:
            warped = _pad_image(warped, config.padding_px)
        return warped, "polygon_warp", True
    return _crop_bbox(image, geometry.bbox_xyxy, config.padding_px), "bbox_raw", False


def _valid_polygon(polygon: Polygon | None) -> np.ndarray | None:
    if polygon is None or len(polygon) != 4:
        return None
    arr = np.asarray(polygon, dtype=np.float32)
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        return None
    if cv2.contourArea(arr) <= 1.0:
        return None
    return arr


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_polygon(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    rect = _order_points(polygon)
    tl, tr, br, bl = rect
    width = max(_distance(br, bl), _distance(tr, tl))
    height = max(_distance(tr, br), _distance(tl, bl))
    out_w = max(1, int(round(width)))
    out_h = max(1, int(round(height)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _crop_bbox(image: np.ndarray, bbox: BBox, padding_px: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = max(0, int(padding_px))
    x1 = max(0, min(w, int(math.floor(x1)) - pad))
    y1 = max(0, min(h, int(math.floor(y1)) - pad))
    x2 = max(0, min(w, int(math.ceil(x2)) + pad))
    y2 = max(0, min(h, int(math.ceil(y2)) + pad))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0].copy()
    return image[y1:y2, x1:x2].copy()


def _pad_image(image: np.ndarray, padding_px: int) -> np.ndarray:
    pad = max(0, int(padding_px))
    if pad <= 0 or image.size == 0:
        return image
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def _orient(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "rotate_90_cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "rotate_90_ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if orientation == "rotate_180":
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _shape(image: np.ndarray) -> list[int]:
    if image.ndim == 2:
        return [int(image.shape[0]), int(image.shape[1]), 1]
    return [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]
