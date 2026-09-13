from __future__ import annotations

import json
import zipfile
from pathlib import Path

from PIL import Image

from app.scripts.prepare_svtrv2_hardcase_expdate_rec_dataset import run


def _write_image(path: Path, size: tuple[int, int] = (140, 90), color: tuple[int, int, int] = (235, 235, 225)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fixture_dataset(tmp_path: Path) -> tuple[Path, Path]:
    crop_truth = tmp_path / "crop_truth"
    _write_image(crop_truth / "val_images" / "truth_a.png", color=(250, 245, 160))
    _write_image(crop_truth / "val_images" / "truth_b.png", color=(235, 235, 235))
    _write_json(
        crop_truth / "manifest.json",
        {
            "items": [
                {"output_path": "val_images/truth_a.png", "label": "01.06.2026", "source_name": "truth_a"},
                {"output_path": "val_images/truth_b.png", "label": "23.07.2025", "source_name": "truth_b"},
            ]
        },
    )

    expdate_root = tmp_path / "SMARTBITE-DATASET"
    products_train = {
        "product_a.jpg": {
            "ann": [
                {"cls": "date", "bbox": [10, 10, 90, 35], "transcription": "2026/01/02"},
                {"cls": "code", "bbox": [10, 45, 90, 65], "transcription": "LOT999"},
            ]
        },
        "product_b.jpg": {
            "ann": [
                {"cls": "date", "bbox": [15, 12, 95, 38], "transcription": "2027/03/04"},
            ]
        },
    }
    products_eval = {
        "eval_a.jpg": {
            "ann": [
                {"cls": "exp", "bbox": [12, 10, 92, 34], "transcription": "2028/05/06"},
                {"cls": "due", "bbox": [5, 50, 60, 70]},
            ]
        },
    }
    for image_name in products_train:
        _write_image(expdate_root / "Products-Real" / "train" / "images" / image_name)
    for image_name in products_eval:
        _write_image(expdate_root / "Products-Real" / "evaluation" / "images" / image_name)
    _write_json(expdate_root / "Products-Real" / "train" / "annotations.json", products_train)
    _write_json(expdate_root / "Products-Real" / "evaluation" / "annotations.json", products_eval)
    return crop_truth, expdate_root


def test_builds_hardcase_dataset_from_crop_truth_and_expdate_products(tmp_path: Path) -> None:
    crop_truth, expdate_root = _fixture_dataset(tmp_path)
    out = tmp_path / "out"
    zip_path = tmp_path / "hardcase.zip"

    code = run(
        [
            "--crop-truth-dir",
            str(crop_truth),
            "--expdate-root",
            str(expdate_root),
            "--output-dir",
            str(out),
            "--zip-output",
            str(zip_path),
            "--crop-truth-variants-per-crop",
            "2",
            "--expdate-train-count",
            "2",
            "--expdate-val-count",
            "1",
            "--seed",
            "7",
        ]
    )

    assert code == 0
    assert zip_path.exists()

    train_lines = _read_lines(out / "train_label.txt")
    val_lines = _read_lines(out / "val_label.txt")
    test_lines = _read_lines(out / "test_label.txt")
    records = _metadata(out / "metadata.jsonl")

    assert len([r for r in records if r["source"] == "crop_truth_aug" and r["split"] == "train"]) == 4
    assert len([r for r in records if r["source"] == "expdate_products_real" and r["split"] == "train"]) == 2
    assert len([r for r in records if r["source"] == "expdate_products_real" and r["split"] == "val"]) == 1
    assert len([r for r in records if r["source"] == "crop_truth_original" and r["split"] == "test"]) == 2

    assert len(train_lines) == 6
    assert len(val_lines) == 1
    assert len(test_lines) == 2
    assert all("\tLOT999" not in line for line in train_lines + val_lines + test_lines)
    assert {line.split("\t", 1)[1] for line in test_lines} == {"01.06.2026", "23.07.2025"}

    for line in train_lines + val_lines + test_lines:
        rel_path, label = line.split("\t", 1)
        assert label
        assert (out / rel_path).exists()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["crop_truth_items"] == 2
    assert summary["counts"]["exported_train"] == 6
    assert summary["counts"]["exported_val"] == 1
    assert summary["counts"]["exported_test"] == 2

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "train_label.txt" in names
    assert "val_label.txt" in names
    assert "test_label.txt" in names
    assert "summary.json" in names
