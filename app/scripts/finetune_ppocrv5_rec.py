from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DataLayout:
    data_root: Path
    train_images_dir: Path
    val_images_dir: Path
    train_label_file: Path
    val_label_file: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate data and fine-tune PP-OCRv5 recognition model with PaddleOCR-native dataset format"
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing train/val images and label files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where fine-tuned model checkpoints are saved")
    parser.add_argument("--pretrained-model", type=Path, required=True, help="Pretrained PP-OCRv5 recognition model path")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/rec/PP-OCRv5/rec_ppocr_v5_train.yml"),
        help="PaddleOCR recognition training config path",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config and print launch command only")
    return parser.parse_args(argv)


def detect_runtime_device(mode: str) -> str:
    explicit = mode.lower()
    if explicit in {"cpu", "mps", "cuda"}:
        return explicit

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_training_device(mode: str) -> str:
    detected = detect_runtime_device(mode)
    # Paddle OCR training supports CUDA or CPU. MPS request keeps ppocrv5 path
    # but falls back to CPU training execution.
    if detected == "mps":
        return "cpu"
    return detected


def require_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise ValueError(f"missing {kind}: {path}")


def validate_label_file(label_file: Path, data_root: Path, split_name: str) -> int:
    lines = label_file.read_text(encoding="utf-8").splitlines()
    valid_count = 0

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"{split_name} label line {idx} must contain tab separator: {line!r}")

        rel_path, text = line.split("\t", 1)
        rel_path = rel_path.strip()
        text = text.strip()

        if not rel_path:
            raise ValueError(f"{split_name} label line {idx} has empty image path")
        if not text:
            raise ValueError(f"{split_name} label line {idx} has empty text")

        image_path = data_root / rel_path
        if not image_path.exists():
            raise ValueError(f"{split_name} label line {idx} references missing image: {image_path}")

        valid_count += 1

    if valid_count == 0:
        raise ValueError(f"{split_name} label file has no valid samples: {label_file}")

    return valid_count


def validate_dataset_layout(data_root: Path) -> tuple[DataLayout, dict[str, int]]:
    layout = DataLayout(
        data_root=data_root,
        train_images_dir=data_root / "train_images",
        val_images_dir=data_root / "val_images",
        train_label_file=data_root / "train_label.txt",
        val_label_file=data_root / "val_label.txt",
    )

    require_exists(layout.data_root, "data root")
    require_exists(layout.train_images_dir, "train_images directory")
    require_exists(layout.val_images_dir, "val_images directory")
    require_exists(layout.train_label_file, "train_label.txt")
    require_exists(layout.val_label_file, "val_label.txt")

    train_count = validate_label_file(layout.train_label_file, layout.data_root, "train")
    val_count = validate_label_file(layout.val_label_file, layout.data_root, "val")

    return layout, {"train_samples": train_count, "val_samples": val_count}


def build_training_command(args: argparse.Namespace, layout: DataLayout, training_device: str) -> list[str]:
    use_gpu = training_device == "cuda"

    overrides = [
        f"Global.pretrained_model={args.pretrained_model}",
        f"Global.save_model_dir={args.output_dir}",
        f"Global.epoch_num={args.epochs}",
        f"Global.use_gpu={str(use_gpu)}",
        f"Optimizer.lr.learning_rate={args.learning_rate}",
        f"Train.loader.batch_size_per_card={args.batch_size}",
        f"Eval.loader.batch_size_per_card={args.batch_size}",
        f"Train.dataset.data_dir={layout.data_root}",
        f"Train.dataset.label_file_list=['{layout.train_label_file}']",
        f"Eval.dataset.data_dir={layout.data_root}",
        f"Eval.dataset.label_file_list=['{layout.val_label_file}']",
    ]

    command = [
        sys.executable,
        "-m",
        "paddleocr.tools.train",
        "-c",
        str(args.base_config),
        "-o",
        *overrides,
    ]
    return command


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        layout, stats = validate_dataset_layout(args.data_root)
        require_exists(args.pretrained_model, "pretrained model")
        require_exists(args.base_config, "base config")
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 2

    training_device = resolve_training_device(args.device)
    command = build_training_command(args, layout, training_device)

    print("PP-OCRv5 recognition fine-tuning setup")
    print(f"- data_root: {args.data_root}")
    print(f"- train_samples: {stats['train_samples']}")
    print(f"- val_samples: {stats['val_samples']}")
    print(f"- device_requested: {args.device}")
    print(f"- device_used: {training_device}")
    print("- command:")
    print("  " + shlex.join(command))

    if args.dry_run:
        print("Dry run successful. Launch skipped.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
