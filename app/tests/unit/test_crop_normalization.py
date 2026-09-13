from __future__ import annotations

import numpy as np

from app.ai.crop_normalization import TextLineCropConfig, TextLineGeometry, normalize_textline_crops
from app.ai.ocr import TextDetectionBox
from app.ai.pipeline import ExpiryPipeline
from app.ai.preprocess import ImageVariant


def test_bbox_horizontal_crop_returns_only_original_orientation() -> None:
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    geometry = TextLineGeometry(bbox_xyxy=(20, 30, 120, 50))

    crops = normalize_textline_crops(image, geometry, TextLineCropConfig(padding_px=0))

    assert [crop.crop_transform_used for crop in crops] == ["bbox_raw"]
    assert [crop.selected_orientation for crop in crops] == ["original"]
    assert crops[0].image.shape[:2] == (20, 100)


def test_tall_bbox_crop_returns_bounded_rotation_candidates() -> None:
    image = np.zeros((200, 80, 3), dtype=np.uint8)
    geometry = TextLineGeometry(bbox_xyxy=(20, 10, 40, 150))

    crops = normalize_textline_crops(image, geometry, TextLineCropConfig(padding_px=0, vertical_aspect_threshold=1.5))

    assert [crop.selected_orientation for crop in crops] == [
        "original",
        "rotate_90_cw",
        "rotate_90_ccw",
        "rotate_180",
    ]
    assert len(crops) == 4
    assert crops[1].image.shape[:2] == (20, 140)


def test_polygon_crop_is_perspective_warped_and_records_transform() -> None:
    image = np.zeros((80, 140, 3), dtype=np.uint8)
    polygon = ((20.0, 20.0), (100.0, 24.0), (96.0, 44.0), (18.0, 40.0))
    geometry = TextLineGeometry(bbox_xyxy=(18, 20, 100, 44), polygon_xy=polygon)

    crops = normalize_textline_crops(image, geometry, TextLineCropConfig(padding_px=0))

    assert crops[0].crop_transform_used == "polygon_warp"
    assert crops[0].polygon_used is True
    assert crops[0].selected_orientation == "original"
    assert crops[0].image.shape[1] > crops[0].image.shape[0]


def test_detector_variant_mapping_preserves_polygon_points() -> None:
    box = TextDetectionBox(
        20,
        10,
        80,
        40,
        0.9,
        polygon_xy=((20.0, 10.0), (80.0, 10.0), (80.0, 40.0), (20.0, 40.0)),
    )
    variant = ImageVariant("raw_upscaled", np.zeros((100, 120, 3), dtype=np.uint8), scale_x=2.0, scale_y=2.0, purpose="detector")

    mapped = ExpiryPipeline._map_text_boxes_to_original(
        boxes=[box],
        variant=variant,
        original_shape=(80, 80, 3),
    )

    assert mapped[0].bbox_xyxy == (10, 5, 40, 20)
    assert mapped[0].polygon_xy == ((10.0, 5.0), (40.0, 5.0), (40.0, 20.0), (10.0, 20.0))
