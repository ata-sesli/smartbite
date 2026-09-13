from __future__ import annotations

import cv2
import numpy as np

from app.ai.preprocess import ImageVariant, PreprocessConfig, ROIImagePreprocessor


def _low_contrast_text_image() -> np.ndarray:
    image = np.full((48, 160, 3), 118, dtype=np.uint8)
    cv2.putText(image, "EXP 12/12/27", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (132, 132, 132), 1, cv2.LINE_AA)
    return image


def test_detector_variants_are_natural_three_channel_images() -> None:
    preprocessor = ROIImagePreprocessor(PreprocessConfig(detector_min_height=96))
    variants = preprocessor.detector_variants(_low_contrast_text_image())

    names = {variant.name for variant in variants}
    assert {"raw", "raw_upscaled", "luma_clahe", "illumination_normalized", "unsharp"} <= names
    assert all(isinstance(variant, ImageVariant) for variant in variants)

    for variant in variants:
        assert variant.purpose == "detector"
        assert variant.image.dtype == np.uint8
        assert variant.image.ndim == 3
        assert variant.image.shape[2] == 3
        assert len(np.unique(variant.image.reshape(-1, 3), axis=0)) > 2


def test_recognition_variants_include_aggressive_text_enhancements() -> None:
    preprocessor = ROIImagePreprocessor(PreprocessConfig(min_ocr_height=96))
    variants = preprocessor.recognition_variants(_low_contrast_text_image())

    names = {variant.name for variant in variants}
    assert {
        "original_padded",
        "gray_upscaled",
        "clahe_gray",
        "unsharp_gray",
        "adaptive_binary",
        "adaptive_binary_inverted",
        "stamp_blackhat",
    } <= names
    assert all(variant.purpose == "recognition" for variant in variants)
    assert any(variant.image.shape[0] >= 96 for variant in variants)
    assert any(len(np.unique(variant.image)) <= 2 for variant in variants if variant.image.ndim == 2)


def test_contrast_variants_increase_low_contrast_text_signal() -> None:
    image = _low_contrast_text_image()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessor = ROIImagePreprocessor(PreprocessConfig())

    variants = {variant.name: variant.image for variant in preprocessor.recognition_variants(image)}
    clahe = variants["clahe_gray"]
    stamp = variants["stamp_blackhat"]

    assert float(clahe.std()) > float(gray.std())
    assert float(stamp.std()) > 0.0
