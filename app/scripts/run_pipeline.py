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
    parser.add_argument("--detector-model", type=Path, default=Path("yolo26n.pt"))
    parser.add_argument("--ppocrv5-model-dir", type=Path, default=Path("models/ppocrv5/main"))
    parser.add_argument("--ppocrv5-char-dict", type=Path, default=Path("models/ppocrv5/char_dict.txt"))
    parser.add_argument("--substitute-config", type=Path, default=Path("app/ai/ocr_substitute_config.json"))
    parser.add_argument("--enable-substitute", action="store_true")
    parser.add_argument("--device-mode", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    pipeline = ExpiryPipeline(
        detector=ExpiryRegionDetector(args.detector_model),
        preprocessor=ROIImagePreprocessor(),
        ocr_router=OCRRouter(
            OCRConfig(
                ppocrv5_main_model_dir=args.ppocrv5_model_dir,
                ppocrv5_main_char_dict_path=args.ppocrv5_char_dict,
                paddle_lang="en",
                device_mode=args.device_mode,
                ppocrv5_use_angle_cls=True,
                ppocrv5_det_db_thresh=0.3,
                substitute_config_path=args.substitute_config,
                enable_substitute_model=args.enable_substitute,
            )
        ),
        parser=ExpiryDateParser(),
        decision=ExpiryDecisionEngine(),
    )

    output = pipeline.run(args.image.read_bytes(), today=date.today())
    print(json.dumps(output.__dict__, default=str, indent=2))


if __name__ == "__main__":
    main()
