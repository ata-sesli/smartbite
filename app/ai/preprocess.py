from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class PreprocessConfig:
    resize_width: int = 640
    threshold: int = 0
    denoise_strength: int = 3
    deskew: bool = False


class ROIImagePreprocessor:
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def process(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self.config.resize_width:
            ratio = self.config.resize_width / float(w)
            gray = cv2.resize(gray, (self.config.resize_width, int(h * ratio)), interpolation=cv2.INTER_LINEAR)

        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresholded = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        denoised = cv2.fastNlMeansDenoising(thresholded, None, self.config.denoise_strength, 7, 21)
        return denoised
