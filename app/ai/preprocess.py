from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    resize_width: int = 640
    min_ocr_height: int = 96
    denoise_strength: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    morph_kernel_size: int = 2


class ROIImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def _base_gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    def _maybe_upscale(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h >= self.config.min_ocr_height:
            return image
        scale = self.config.min_ocr_height / float(max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def process(self, image: np.ndarray) -> np.ndarray:
        gray = self._base_gray(image)
        return self._enhanced_binary(gray)

    def process_variants(self, image: np.ndarray) -> list[np.ndarray]:
        gray = self._base_gray(image)
        binary = self._enhanced_binary(gray)
        variants: list[np.ndarray] = [binary]

        upscaled_binary = self._maybe_upscale(binary)
        if upscaled_binary.shape != binary.shape:
            variants.append(upscaled_binary)

        upscaled_gray = self._maybe_upscale(gray)
        if upscaled_gray.shape != gray.shape:
            variants.append(upscaled_gray)

        return variants
