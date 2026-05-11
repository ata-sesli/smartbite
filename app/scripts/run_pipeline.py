from __future__ import annotations

import argparse
from dataclasses import asdict
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
    parser.add_argument("--detector-model", type=Path, default=Path("models/yolo20n/yolo26s/yolo26s-best.pt"))
    parser.add_argument(
        "--ppocrv5-det-model-dir",
        type=Path,
        default=Path("models/fine-tuned-models/ppocr-detection-best-model-inference"),
    )
    parser.add_argument("--use-custom-text-detector", dest="use_custom_text_detector", action="store_true")
    parser.add_argument("--use-default-text-detector", dest="use_custom_text_detector", action="store_false")
    parser.set_defaults(use_custom_text_detector=False)
    parser.add_argument("--parseq-model-dir", type=Path, default=Path("models/parseq-small"))
    parser.add_argument("--probe-mobile-rec-model-dir", type=Path, default=Path("models/ppocrv5/mobile_rec"))
    parser.add_argument("--probe-mobile-rec-model-name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--device-mode", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--parseq-device-mode", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--detector-confidence-threshold", type=float, default=0.15)
    parser.add_argument("--detector-top-k", type=int, default=4)
    parser.add_argument("--detector-box-padding-ratio", type=float, default=0.12)
    parser.add_argument("--parser-min-candidate-confidence", type=float, default=0.40)
    parser.add_argument("--expiry-filter-geometry-top-n", type=int, default=12)
    parser.add_argument("--expiry-filter-final-top-k", type=int, default=4)
    parser.add_argument("--high-recall-mode", dest="high_recall_mode", action="store_true")
    parser.add_argument("--low-recall-mode", dest="high_recall_mode", action="store_false")
    parser.set_defaults(high_recall_mode=True)
    args = parser.parse_args()

    pipeline = ExpiryPipeline(
        detector=ExpiryRegionDetector(
            args.detector_model,
            confidence_threshold=args.detector_confidence_threshold,
        ),
        preprocessor=ROIImagePreprocessor(),
        ocr_router=OCRRouter(
            OCRConfig(
                parseq_model_dir=args.parseq_model_dir,
                ppocrv5_text_det_model_dir=args.ppocrv5_det_model_dir if args.use_custom_text_detector else None,
                ppocrv5_text_det_model_name="PP-OCRv5_server_det",
                paddle_lang="en",
                device_mode=args.device_mode,
                parseq_device_mode=args.parseq_device_mode,
                ppocrv5_use_angle_cls=True,
                ppocrv5_det_db_thresh=0.3,
                expiry_probe_rec_model_name=args.probe_mobile_rec_model_name,
                expiry_probe_mobile_rec_model_dir=args.probe_mobile_rec_model_dir,
            )
        ),
        parser=ExpiryDateParser(min_candidate_confidence=args.parser_min_candidate_confidence),
        decision=ExpiryDecisionEngine(),
        top_k=args.detector_top_k,
        roi_padding_ratio=args.detector_box_padding_ratio,
        high_recall_mode=args.high_recall_mode,
        expiry_filter_geometry_top_n=args.expiry_filter_geometry_top_n,
        expiry_filter_final_top_k=args.expiry_filter_final_top_k,
    )

    output = pipeline.run(args.image.read_bytes(), today=date.today())
    print(json.dumps(asdict(output), default=str, indent=2))


if __name__ == "__main__":
    main()
