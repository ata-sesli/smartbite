#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Freiburg Groceries VOC annotations to YOLO format and split train/val/test."
    )
    p.add_argument(
        "--source-dir",
        default="data/freiburg-groceries",
        help="Source Freiburg dataset root containing images/ and annotations/.",
    )
    p.add_argument(
        "--output-dir",
        default="data/freiburg-groceries-yolo",
        help="Output YOLO dataset directory.",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete output dir before writing.",
    )
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def voc_to_yolo_line(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    cls_id: int,
) -> str:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def main() -> int:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: pyyaml. Install with `pip install pyyaml`."
        ) from exc

    args = parse_args()
    source_root = Path(args.source_dir).resolve()
    images_root = source_root / "images"
    ann_root = source_root / "annotations"

    if not images_root.exists() or not ann_root.exists():
        raise FileNotFoundError(f"Expected images/ and annotations/ under: {source_root}")

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    output_root = Path(args.output_dir).resolve()
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Match each xml to a same-stem image in its class folder.
    samples: list[tuple[Path, Path, str, str]] = []
    for xml_path in sorted(ann_root.rglob("*.xml")):
        class_dir = xml_path.parent.name
        stem = xml_path.stem

        image_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = images_root / class_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        samples.append((image_path, xml_path, class_dir, stem))

    if not samples:
        raise RuntimeError("No matched image/xml samples found.")

    class_names: set[str] = set()
    for _, xml_path, _, _ in samples:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name:
                class_names.add(name.strip())

    classes = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    rnd = random.Random(args.seed)
    rnd.shuffle(samples)

    n = len(samples)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    split_map = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    skipped = 0
    for split, items in split_map.items():
        for img_path, xml_path, _, stem in items:
            dst_img = output_root / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            if size is None:
                raise ValueError(f"Missing size node in {xml_path}")
            width = int(float(size.findtext("width", "0")))
            height = int(float(size.findtext("height", "0")))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size in {xml_path}")

            yolo_lines: list[str] = []
            for obj in root.findall("object"):
                cls_name = (obj.findtext("name") or "").strip()
                if cls_name not in class_to_id:
                    continue

                box = obj.find("bndbox")
                if box is None:
                    continue

                xmin = float(box.findtext("xmin", "0"))
                ymin = float(box.findtext("ymin", "0"))
                xmax = float(box.findtext("xmax", "0"))
                ymax = float(box.findtext("ymax", "0"))

                xmin = clamp(xmin, 0.0, float(width))
                xmax = clamp(xmax, 0.0, float(width))
                ymin = clamp(ymin, 0.0, float(height))
                ymax = clamp(ymax, 0.0, float(height))

                if xmax <= xmin or ymax <= ymin:
                    skipped += 1
                    continue

                yolo_lines.append(
                    voc_to_yolo_line(
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        width,
                        height,
                        class_to_id[cls_name],
                    )
                )

            (output_root / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines))

    dataset_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(classes)},
    }
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))

    print("Done.")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print("Split sizes:", {k: len(v) for k, v in split_map.items()})
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Skipped degenerate boxes: {skipped}")
    print(f"Dataset YAML: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
