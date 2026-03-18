from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from app.ai.decision import ExpiryDecisionEngine
from app.ai.detector import ExpiryRegionDetector
from app.ai.ocr import OCRConfig, OCRRouter
from app.ai.parser import ExpiryDateParser
from app.ai.pipeline import ExpiryPipeline
from app.ai.preprocess import ROIImagePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmartBite AI pipeline on a single image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector-model", type=Path, default=Path("models/detector.pt"))
    parser.add_argument("--onnx-model", type=Path, default=Path("models/ocr_lite.onnx"))
    parser.add_argument("--device-mode", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    pipeline = ExpiryPipeline(
        detector=ExpiryRegionDetector(args.detector_model),
        preprocessor=ROIImagePreprocessor(),
        ocr_router=OCRRouter(
            OCRConfig(
                onnx_model_path=args.onnx_model,
                onnx_charset_path=None,
                paddle_lang="en",
                device_mode=args.device_mode,
            )
        ),
        parser=ExpiryDateParser(),
        decision=ExpiryDecisionEngine(),
    )

    output = pipeline.run(args.image.read_bytes(), today=date.today())
    print(json.dumps(output.__dict__, default=str, indent=2))


if __name__ == "__main__":
    main()
