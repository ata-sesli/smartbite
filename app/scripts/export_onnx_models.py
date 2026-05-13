from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_YOLO_MODEL = Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt")
DEFAULT_SVTR_MODEL_DIR = Path("models/svtrv2/smartbite_svtrv2_expdate_rec")
DEFAULT_SVTR_ONNX = Path("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx")


def build_yolo_export_kwargs(*, model_path: Path, imgsz: int) -> dict[str, Any]:
    _ = model_path
    return {
        "format": "onnx",
        "imgsz": int(imgsz),
        "opset": 17,
        "simplify": True,
        "dynamic": False,
        "nms": False,
    }


def export_yolo_obb(*, model_path: Path, imgsz: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    exported = model.export(**build_yolo_export_kwargs(model_path=model_path, imgsz=imgsz))
    return Path(str(exported))


def build_svtr_export_command(*, model_dir: Path, save_file: Path) -> list[str]:
    return [
        "paddle2onnx",
        "--model_dir",
        str(model_dir),
        "--model_filename",
        "inference.json",
        "--params_filename",
        "inference.pdiparams",
        "--save_file",
        str(save_file),
        "--opset_version",
        "17",
        "--enable_onnx_checker",
        "True",
    ]


def export_svtr(*, model_dir: Path, save_file: Path) -> None:
    save_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_svtr_export_command(model_dir=model_dir, save_file=save_file), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SmartBite detector/recognizer models to ONNX")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-svtr", action="store_true")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-imgsz", type=int, default=1024)
    parser.add_argument("--svtr-model-dir", type=Path, default=DEFAULT_SVTR_MODEL_DIR)
    parser.add_argument("--svtr-save-file", type=Path, default=DEFAULT_SVTR_ONNX)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_yolo:
        kwargs = build_yolo_export_kwargs(model_path=args.yolo_model, imgsz=args.yolo_imgsz)
        if args.dry_run:
            print(f"YOLO export: YOLO({args.yolo_model!s}).export({kwargs})")
        else:
            print(f"YOLO ONNX exported to: {export_yolo_obb(model_path=args.yolo_model, imgsz=args.yolo_imgsz)}")
    if not args.skip_svtr:
        command = build_svtr_export_command(model_dir=args.svtr_model_dir, save_file=args.svtr_save_file)
        if args.dry_run:
            print("SVTR export:", " ".join(command))
        else:
            export_svtr(model_dir=args.svtr_model_dir, save_file=args.svtr_save_file)
            print(f"SVTR ONNX exported to: {args.svtr_save_file}")


if __name__ == "__main__":
    main()
