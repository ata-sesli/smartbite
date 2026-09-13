from __future__ import annotations

from pathlib import Path

from app.scripts.prepare_expiry_date_detection_obb_join import DatasetItem, normalize_obb_label, split_items


def test_normalize_obb_label_maps_any_source_class_to_expiry_date(tmp_path: Path) -> None:
    label = tmp_path / "sample.txt"
    label.write_text("3 0 0.1 0.2 0.1 0.2 0.3 0 0.3\n", encoding="utf-8")

    rows, empty = normalize_obb_label(label)

    assert empty == 0
    assert rows == ["0 0.000000 0.100000 0.200000 0.100000 0.200000 0.300000 0.000000 0.300000"]


def test_split_items_uses_requested_ratios() -> None:
    items = [
        DatasetItem("src", Path(f"image_{idx:03d}.jpg"), Path(f"image_{idx:03d}.txt"))
        for idx in range(20)
    ]

    splits = split_items(items, seed=7, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)

    assert {split: len(values) for split, values in splits.items()} == {
        "train": 16,
        "valid": 2,
        "test": 2,
    }
