from __future__ import annotations

import json
import zipfile
from pathlib import Path

from scripts.prepare_grocery_products_yolo import run


def _write_case(
    root: Path,
    store: str,
    stem: str,
    rows: list[str],
    image_bytes: bytes = b"fake-jpg",
) -> None:
    image_dir = root / "Testing" / store / "images"
    ann_dir = root / "Testing" / store / "annotation"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(image_bytes)
    (ann_dir / f"{stem}.csv").write_text(
        "class,label,left_x,right_x,top_y,bottom_y\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_prepare_grocery_products_yolo_exports_single_class_dataset(tmp_path: Path) -> None:
    source = tmp_path / "Grocery_products"
    _write_case(
        source,
        "store1",
        "14",
        [
            "32,17,0.100000,0.300000,0.200000,0.600000",
            "32,-1,0.500000,0.800000,0.100000,0.400000",
            "32,99,0.900000,0.800000,0.100000,0.400000",
        ],
    )
    _write_case(source, "store2", "7", ["10,1,0.200000,0.400000,0.300000,0.500000"])
    _write_case(source, "store5", "1", ["9,1,0.100000,0.200000,0.300000,0.700000"])
    _write_case(source, "store5", "2", ["9,2,0.300000,0.500000,0.100000,0.300000"])
    _write_case(source, "store5", "3", ["9,3,0.600000,0.700000,0.200000,0.400000"])

    output = tmp_path / "grocery-products-yolo"
    zip_out = tmp_path / "grocery-products-yolo.zip"
    code = run(
        [
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--zip-out",
            str(zip_out),
            "--force",
            "--seed",
            "42",
        ]
    )

    assert code == 0
    assert (output / "dataset.yaml").read_text(encoding="utf-8") == (
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: product\n"
    )

    train_label = (output / "labels" / "train" / "store1_14.txt").read_text(encoding="utf-8")
    assert train_label.splitlines() == [
        "0 0.200000 0.400000 0.200000 0.400000",
        "0 0.650000 0.250000 0.300000 0.300000",
    ]

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["total_usable_images"] == 5
    assert manifest["counts"]["total_usable_boxes"] == 6
    assert manifest["counts"]["invalid_bbox_rows_skipped"] == 1
    assert manifest["counts"]["splits"]["train"] == {"images": 2, "boxes": 3}
    assert manifest["counts"]["splits"]["val"] == {"images": 1, "boxes": 1}
    assert manifest["counts"]["splits"]["test"] == {"images": 2, "boxes": 2}
    assert manifest["names"] == {"0": "product"}

    assert zip_out.exists()
    with zipfile.ZipFile(zip_out) as zf:
        names = set(zf.namelist())
    assert "dataset.yaml" in names
    assert "manifest.json" in names
    assert "images/train/store1_14.jpg" in names
    assert "labels/train/store1_14.txt" in names


def test_prepare_grocery_products_yolo_check_only_validates_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "grocery-products-yolo"
    (output / "images" / "train").mkdir(parents=True)
    (output / "images" / "val").mkdir(parents=True)
    (output / "images" / "test").mkdir(parents=True)
    (output / "labels" / "train").mkdir(parents=True)
    (output / "labels" / "val").mkdir(parents=True)
    (output / "labels" / "test").mkdir(parents=True)
    (output / "labels" / "train" / "bad.txt").write_text(
        "1 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    code = run(["--output-dir", str(output), "--check-only"])

    assert code == 2
