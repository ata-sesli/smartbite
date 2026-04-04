from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.scripts.build_ppocrv5_rec_dataset import (
    assign_crop_level_splits,
    assign_image_level_splits,
    build_main_crop_filename,
    clamp_bbox,
    normalize_transcription,
    parse_bbox,
    run,
)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
    # Override global DB fixture: this module only tests filesystem/CLI behavior.
    yield


def _write_image(path: Path, size: tuple[int, int] = (120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(230, 230, 230)).save(path)


def _write_annotation(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_clamp_bbox_applies_padding_and_bounds() -> None:
    result = clamp_bbox((5, 6, 15, 16), pad=10, image_width=20, image_height=25)
    assert result == (0, 0, 20, 25)


def test_parse_bbox_rejects_invalid_shape_and_order() -> None:
    try:
        parse_bbox([1, 2, 3])
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        parse_bbox([10, 10, 5, 12])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_normalize_transcription_modes() -> None:
    assert normalize_transcription("  A B  ", "none") == "  A B  "
    assert normalize_transcription("  A B  ", "strip") == "A B"
    # fullwidth A => ASCII A under NFKC
    assert normalize_transcription(" Ａ ", "unicode") == "A"


def test_split_reproducibility_with_fixed_seed() -> None:
    image_keys = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]
    mapping1 = assign_image_level_splits(image_keys, train_ratio=0.6, seed=99)
    mapping2 = assign_image_level_splits(image_keys, train_ratio=0.6, seed=99)
    assert mapping1 == mapping2

    ann_keys = [("a.jpg", 0), ("a.jpg", 1), ("b.jpg", 0), ("c.jpg", 0)]
    crop_map1 = assign_crop_level_splits(ann_keys, train_ratio=0.5, seed=11)
    crop_map2 = assign_crop_level_splits(ann_keys, train_ratio=0.5, seed=11)
    assert crop_map1 == crop_map2


def test_deterministic_file_naming() -> None:
    assert build_main_crop_filename("sub/folder/sample_name.png", 7) == "sample_name__ann7.jpg"


def test_happy_path_creates_ppocr_outputs_and_metadata(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")
    _write_image(images_dir / "img2.jpg")

    payload = {
        "img1.jpg": {
            "width": 120,
            "height": 80,
            "ann": [
                {"cls": "date", "bbox": [10, 10, 50, 30], "transcription": "EXP 01/01/26"},
                {"cls": "date", "bbox": [55, 10, 95, 30], "transcription": "LOT123"},
            ],
        },
        "img2.jpg": {
            "width": 120,
            "height": 80,
            "ann": [
                {"cls": "date", "bbox": [8, 35, 60, 60], "transcription": "BEST 03/02/26"},
            ],
        },
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.67",
            "--seed",
            "42",
        ]
    )
    assert code == 0

    train_label_path = output_dir / "train_label.txt"
    val_label_path = output_dir / "val_label.txt"
    train_lines = [line for line in train_label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_lines = [line for line in val_label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(train_lines) + len(val_lines) == 3

    for line in train_lines + val_lines:
        rel_path, _text = line.split("\t", 1)
        assert (output_dir / rel_path).exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["exported_main_crops"] == 3

    train_meta = [json.loads(x) for x in (output_dir / "train_metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    val_meta = [json.loads(x) for x in (output_dir / "val_metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    assert len(train_meta) + len(val_meta) == 3

    # default policy: one source image must not appear in both splits among exported samples
    train_sources = {m["source_image_file"] for m in train_meta if m["output_crop_path"]}
    val_sources = {m["source_image_file"] for m in val_meta if m["output_crop_path"]}
    assert train_sources.isdisjoint(val_sources)


def test_missing_image_is_skipped_and_reported(tmp_path: Path, capsys) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "exists.jpg")

    payload = {
        "exists.jpg": {
            "width": 120,
            "height": 80,
            "ann": [{"cls": "date", "bbox": [10, 10, 50, 30], "transcription": "OK"}],
        },
        "missing.jpg": {
            "width": 120,
            "height": 80,
            "ann": [{"cls": "date", "bbox": [10, 10, 50, 30], "transcription": "MISS"}],
        },
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--train-ratio",
            "0.5",
            "--seed",
            "7",
        ]
    )
    assert code == 0

    stderr = capsys.readouterr().err
    assert "missing_source_image" in stderr

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped_by_reason"]["missing_source_image"] == 1


def test_malformed_json_exits_with_error(tmp_path: Path, capsys) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")

    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(bad_json),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "malformed JSON" in capsys.readouterr().err


def test_component_crop_optional_outputs(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    payload = {
        "img1.jpg": {
            "width": 120,
            "height": 80,
            "ann": [
                {
                    "cls": "date",
                    "bbox": [8, 8, 110, 50],
                    "transcription": "2026/03/22",
                    "dmy_ann": [
                        {"cls": "year", "bbox": [10, 10, 40, 25], "transcription": "2026"},
                        {"cls": "month", "bbox": [44, 10, 62, 25], "transcription": "03"},
                        {"cls": "day", "bbox": [65, 10, 83, 25], "transcription": "22"},
                    ],
                }
            ],
        }
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--generate-component-crops",
            "true",
        ]
    )
    assert code == 0

    comp_label = output_dir / "components_label.txt"
    assert comp_label.exists()
    comp_lines = [x for x in comp_label.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(comp_lines) == 3

    for line in comp_lines:
        rel_path, _text = line.split("\t", 1)
        assert (output_dir / rel_path).exists()


def test_candidate_crop_optional_outputs(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    payload = {
        "img1.jpg": {
            "width": 120,
            "height": 80,
            "ann": [{"cls": "date", "bbox": [20, 20, 70, 50], "transcription": "X"}],
        }
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--generate-candidate-crops",
            "true",
            "--candidate-jitter-count",
            "2",
            "--candidate-max-pad-px",
            "4",
        ]
    )
    assert code == 0

    candidates_metadata = output_dir / "candidates_metadata.jsonl"
    assert candidates_metadata.exists()
    records = [json.loads(x) for x in candidates_metadata.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(records) == 2
    assert all(r["is_candidate_crop"] for r in records)
    for rec in records:
        assert (output_dir / rec["output_crop_path"]).exists()


def test_dry_run_writes_no_dataset_files(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    payload = {
        "img1.jpg": {
            "width": 120,
            "height": 80,
            "ann": [{"cls": "date", "bbox": [10, 10, 50, 30], "transcription": "OK"}],
        }
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not output_dir.exists()


def test_allow_cross_image_split_can_split_same_source_between_train_and_val(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")
    _write_image(images_dir / "img2.jpg")

    payload = {
        "img1.jpg": {
            "width": 120,
            "height": 80,
            "ann": [
                {"cls": "date", "bbox": [5, 5, 40, 25], "transcription": "A"},
                {"cls": "date", "bbox": [45, 5, 85, 25], "transcription": "B"},
            ],
        },
        "img2.jpg": {
            "width": 120,
            "height": 80,
            "ann": [
                {"cls": "date", "bbox": [5, 35, 40, 55], "transcription": "C"},
                {"cls": "date", "bbox": [45, 35, 85, 55], "transcription": "D"},
            ],
        },
    }
    annotation_json = tmp_path / "annotations.json"
    _write_annotation(annotation_json, payload)

    output_dir = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--annotation-json",
            str(annotation_json),
            "--output-dir",
            str(output_dir),
            "--allow-cross-image-split",
            "true",
            "--train-ratio",
            "0.5",
            "--seed",
            "3",
        ]
    )
    assert code == 0

    train_meta = [json.loads(x) for x in (output_dir / "train_metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    val_meta = [json.loads(x) for x in (output_dir / "val_metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]

    train_sources = {m["source_image_file"] for m in train_meta if m["output_crop_path"]}
    val_sources = {m["source_image_file"] for m in val_meta if m["output_crop_path"]}
    # with crop-level splitting enabled we expect overlap in this deterministic fixture
    assert train_sources.intersection(val_sources)
