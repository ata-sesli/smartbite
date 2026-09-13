from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    resize_width: int = 640
    detector_min_height: int = 192
    min_ocr_height: int = 96
    recognition_padding_ratio: float = 0.12
    denoise_strength: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    morph_kernel_size: int = 2


@dataclass(slots=True)
class ImageVariant:
    name: str
    image: np.ndarray
    scale_x: float = 1.0
    scale_y: float = 1.0
    purpose: Literal["detector", "recognition"] = "detector"

    def __iter__(self) -> Iterator[object]:
        yield self.name
        yield self.image


class ROIImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.dtype != np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _base_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(self._ensure_bgr(image), cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self.config.resize_width:
            ratio = self.config.resize_width / float(w)
            gray = cv2.resize(gray, (self.config.resize_width, int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        return gray

    def _enhanced_binary(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size),
        )
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        thresholded = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        denoised = cv2.fastNlMeansDenoising(dilated, None, self.config.denoise_strength, 7, 21)
        return denoised

    def _clahe_gray(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size),
        )
        return clahe.apply(gray)

    @staticmethod
    def _unsharp(image: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.55, blur, -0.55, 0)

    @staticmethod
    def _illumination_normalized(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        background = cv2.GaussianBlur(l, (0, 0), 15)
        normalized = cv2.divide(l, background, scale=180)
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((normalized, a, b)), cv2.COLOR_LAB2BGR)

    def _maybe_upscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h >= self.config.min_ocr_height:
            return image
        scale = self.config.min_ocr_height / float(max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _maybe_upscale_detector(self, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0 or h >= self.config.detector_min_height:
            return image, 1.0, 1.0
        scale = self.config.detector_min_height / float(h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale, scale

    def _pad_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image
        pad_x = max(1, int(round(w * self.config.recognition_padding_ratio)))
        pad_y = max(1, int(round(h * self.config.recognition_padding_ratio)))
        return cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

    def detector_variants(self, image: np.ndarray) -> list[ImageVariant]:
        bgr = self._ensure_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe_gray = self._clahe_gray(gray)
        luma_clahe = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
        normalized = self._illumination_normalized(bgr)
        unsharp = self._unsharp(bgr)
        upscaled, scale_x, scale_y = self._maybe_upscale_detector(bgr)

        return [
            ImageVariant("raw", bgr, purpose="detector"),
            ImageVariant("raw_upscaled", upscaled, scale_x=scale_x, scale_y=scale_y, purpose="detector"),
            ImageVariant("luma_clahe", luma_clahe, purpose="detector"),
            ImageVariant("illumination_normalized", normalized, purpose="detector"),
            ImageVariant("unsharp", unsharp, purpose="detector"),
        ]

    def recognition_variants(
        self,
        crop: np.ndarray,
        *,
        allowed_names: tuple[str, ...] | None = None,
    ) -> list[ImageVariant]:
        allowed = set(allowed_names or ())

        def wants(name: str) -> bool:
            return not allowed or name in allowed

        variants: list[ImageVariant] = []
        padded = self._pad_crop(self._ensure_bgr(crop))
        if wants("original_padded"):
            variants.append(ImageVariant("original_padded", padded, purpose="recognition"))
        needs_gray = any(
            wants(name)
            for name in (
                "gray_upscaled",
                "clahe_gray",
                "unsharp_gray",
                "adaptive_binary",
                "adaptive_binary_inverted",
                "stamp_blackhat",
            )
        )
        if not needs_gray:
            return variants

        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        upscaled_gray = self._maybe_upscale(gray)
        if wants("gray_upscaled"):
            variants.append(ImageVariant("gray_upscaled", upscaled_gray, purpose="recognition"))

        needs_clahe = any(wants(name) for name in ("clahe_gray", "adaptive_binary", "adaptive_binary_inverted"))
        clahe_gray = self._clahe_gray(upscaled_gray) if needs_clahe else None
        if wants("clahe_gray") and clahe_gray is not None:
            variants.append(ImageVariant("clahe_gray", clahe_gray, purpose="recognition"))
        if wants("unsharp_gray"):
            variants.append(ImageVariant("unsharp_gray", self._unsharp(upscaled_gray), purpose="recognition"))

        needs_adaptive = any(wants(name) for name in ("adaptive_binary", "adaptive_binary_inverted"))
        if needs_adaptive and clahe_gray is not None:
            blur = cv2.GaussianBlur(clahe_gray, (3, 3), 0)
            adaptive = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                8,
            )
            if wants("adaptive_binary"):
                variants.append(ImageVariant("adaptive_binary", adaptive, purpose="recognition"))
            if wants("adaptive_binary_inverted"):
                variants.append(ImageVariant("adaptive_binary_inverted", cv2.bitwise_not(adaptive), purpose="recognition"))

        if wants("stamp_blackhat"):
            kernel_w = max(3, min(21, (upscaled_gray.shape[1] // 18) | 1))
            kernel_h = max(3, min(9, (upscaled_gray.shape[0] // 8) | 1))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
            blackhat = cv2.morphologyEx(upscaled_gray, cv2.MORPH_BLACKHAT, kernel)
            tophat = cv2.morphologyEx(upscaled_gray, cv2.MORPH_TOPHAT, kernel)
            stamp = cv2.max(blackhat, tophat)
            stamp = cv2.normalize(stamp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            variants.append(ImageVariant("stamp_blackhat", stamp, purpose="recognition"))

        return variants

    def process(self, image: np.ndarray) -> np.ndarray:
        gray = self._base_gray(image)
        return self._enhanced_binary(gray)

    def process_variants(self, image: np.ndarray) -> list[np.ndarray]:
        return [variant.image for variant in self.recognition_variants(image)]
