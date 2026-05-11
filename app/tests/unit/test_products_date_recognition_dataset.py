from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from app.scripts.build_products_date_recognition_dataset import run


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
    yield


def _write_image(path: Path, size: tuple[int, int] = (140, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 230, 230)).save(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path]:
    real = tmp_path / "Products-Real"
    synth = tmp_path / "Products-Synth"

    real_train = {
        "real_a.jpg": {
            "ann": [
                {"cls": "date", "bbox": [10, 10, 70, 30], "transcription": "2026/01/01"},
                {"cls": "code", "bbox": [10, 40, 70, 55], "transcription": "LOT999"},
            ]
        },
        "real_b.jpg": {
            "ann": [
                {"cls": "date", "bbox": [12, 10, 80, 35], "transcription": "2026/02/02"},
            ]
        },
        "real_c.jpg": {
            "ann": [
                {"cls": "date", "bbox": [15, 12, 90, 38], "transcription": "2026/03/03"},
            ]
        },
        "real_d.jpg": {"ann": [{"cls": "due", "bbox": [5, 5, 40, 20]}]},
    }
    real_eval = {
        "eval_a.jpg": {
            "ann": [
                {"cls": "exp", "bbox": [10, 10, 80, 30], "transcription": "2027/04/04"},
                {"cls": "due", "bbox": [10, 40, 50, 55]},
            ]
        },
        "eval_b.jpg": {
            "ann": [
                {"cls": "date", "bbox": [12, 12, 82, 32], "transcription": "2027/05/05"},
            ]
        },
    }
    synth_ann = {
        f"synth_{idx}.jpg": {
            "ann": [
                {"cls": "date", "bbox": [10, 10, 80, 35], "transcription": f"2028/01/{idx:02d}"},
                {"cls": "code", "bbox": [10, 45, 60, 60], "transcription": "NOPE"},
            ]
        }
        for idx in range(1, 8)
    }

    for name in real_train:
        _write_image(real / "train" / "images" / name)
    for name in real_eval:
        _write_image(real / "evaluation" / "images" / name)
    for name in synth_ann:
        _write_image(synth / "images" / name)

    _write_json(real / "train" / "annotations.json", real_train)
    _write_json(real / "evaluation" / "annotations.json", real_eval)
    _write_json(synth / "annotations.json", synth_ann)
    return real, synth


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metadata(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_builds_balanced_recognition_dataset_without_real_repetition(tmp_path: Path) -> None:
    real, synth = _fixture_roots(tmp_path)
    out = tmp_path / "out"
    zip_path = tmp_path / "products_date_recognition_balanced.zip"

    code = run(
        [
            "--products-real-root",
            str(real),
            "--products-synth-root",
            str(synth),
            "--output-dir",
            str(out),
            "--real-train-ratio",
            "0.75",
            "--seed",
            "11",
            "--zip-output",
            str(zip_path),
        ]
    )

    assert code == 0
    assert zip_path.exists()

    train_lines = _read_lines(out / "train_label.txt")
    val_lines = _read_lines(out / "val_label.txt")
    test_lines = _read_lines(out / "test_label.txt")
    records = _metadata(out / "metadata.jsonl")

    real_train_records = [r for r in records if r["split"] == "train" and r["source"] == "real_train"]
    synth_train_records = [r for r in records if r["split"] == "train" and r["source"] == "synth"]
    val_records = [r for r in records if r["split"] == "val"]
    test_records = [r for r in records if r["split"] == "test"]

    assert len(synth_train_records) == len(real_train_records)
    assert len(real_train_records) == len({(r["source_image"], r["annotation_index"]) for r in real_train_records})
    assert all(r["source"] == "real_train" for r in val_records)
    assert all(r["source"] == "real_test" for r in test_records)
    assert len(test_records) == 2
    assert len(train_lines) == len(real_train_records) + len(synth_train_records)
    assert len(val_lines) == len(val_records)
    assert len(test_lines) == len(test_records)

    train_real_sources = {r["source_image"] for r in real_train_records}
    val_sources = {r["source_image"] for r in val_records}
    assert train_real_sources.isdisjoint(val_sources)

    assert all("\tLOT999" not in line and "\tNOPE" not in line for line in train_lines + val_lines + test_lines)
    for line in train_lines + val_lines + test_lines:
        rel_path, text = line.split("\t", 1)
        assert text
        assert (out / rel_path).exists()

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["synth_train_crops_sampled"] == summary["counts"]["real_train_crops"]
    assert summary["counts"]["real_test_crops"] == 2

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "train_label.txt" in names
    assert "val_label.txt" in names
    assert "test_label.txt" in names
    assert "summary.json" in names


def test_recognition_dry_run_writes_no_outputs(tmp_path: Path) -> None:
    real, synth = _fixture_roots(tmp_path)
    out = tmp_path / "out"

    code = run(
        [
            "--products-real-root",
            str(real),
            "--products-synth-root",
            str(synth),
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )

    assert code == 0
    assert not out.exists()
