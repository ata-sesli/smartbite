from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.scripts.build_ppocrv5_rec_date_dataset import (
    assign_image_level_splits,
    build_export_filename,
    compose_date_from_ann,
    normalize_transcription,
    run,
)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
    # Override global DB fixture: this module only tests filesystem/CLI behavior.
    yield


def _write_image(path: Path, size: tuple[int, int] = (128, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(220, 220, 220)).save(path)


def _read_label_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert_label_paths_exist(output_dir: Path, lines: list[str]) -> None:
    for line in lines:
        rel_path, text = line.split("\t", 1)
        assert text
        assert (output_dir / rel_path).exists()


def test_normalize_transcription_modes() -> None:
    assert normalize_transcription("  A B  ", "none") == "  A B  "
    assert normalize_transcription("  A B  ", "strip") == "A B"
    assert normalize_transcription(" Ａ ", "unicode") == "A"


def test_compose_date_from_ann_prefers_bbox_left_to_right() -> None:
    ann = [
        {"cls": "day", "bbox": [100, 1, 140, 22], "transcription": "23"},
        {"cls": "year", "bbox": [1, 1, 50, 22], "transcription": "2024"},
        {"cls": "month", "bbox": [60, 1, 90, 22], "transcription": "12"},
    ]
    assert compose_date_from_ann(ann, "/") == "2024/12/23"


def test_compose_date_from_ann_falls_back_to_year_month_day() -> None:
    ann = [
        {"cls": "day", "transcription": "09"},
        {"cls": "month", "transcription": "01"},
        {"cls": "year", "transcription": "2025"},
    ]
    assert compose_date_from_ann(ann, "-") == "2025-01-09"


def test_split_reproducibility_with_fixed_seed() -> None:
    keys = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    mapping1 = assign_image_level_splits(keys, train_ratio=0.5, seed=99)
    mapping2 = assign_image_level_splits(keys, train_ratio=0.5, seed=99)
    assert mapping1 == mapping2


def test_export_filename_is_deterministic_and_distinct() -> None:
    one = build_export_filename("nested/a.jpg")
    two = build_export_filename("nested/a.jpg")
    three = build_export_filename("a.jpg")
    assert one == two
    assert one != three
    assert one.endswith(".jpg")


def test_json_annotation_happy_path_exports_ppocr_layout(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")
    _write_image(images_dir / "img2.jpg")

    labels = {
        "img1.jpg": {
            "ann": [
                {"cls": "day", "bbox": [100, 1, 140, 22], "transcription": "23"},
                {"cls": "year", "bbox": [1, 1, 50, 22], "transcription": "2024"},
                {"cls": "month", "bbox": [60, 1, 90, 22], "transcription": "12"},
            ]
        },
        "img2.jpg": {"transcription": "2025/01/05"},
    }
    labels_json = tmp_path / "labels.json"
    labels_json.write_text(json.dumps(labels), encoding="utf-8")

    out = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-json",
            str(labels_json),
            "--output-dir",
            str(out),
            "--train-ratio",
            "0.5",
            "--seed",
            "7",
        ]
    )
    assert code == 0

    train_lines = _read_label_lines(out / "train_label.txt")
    val_lines = _read_label_lines(out / "val_label.txt")
    all_lines = train_lines + val_lines
    assert len(all_lines) == 2
    assert any(line.endswith("\t2024/12/23") for line in all_lines)
    _assert_label_paths_exist(out, all_lines)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["exported_total"] == 2


@pytest.mark.parametrize("mode", ["txt", "csv", "jsonl", "filename"])
def test_supported_label_sources_export_successfully(tmp_path: Path, mode: str) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img_a.jpg")
    _write_image(images_dir / "img_b.jpg")

    out = tmp_path / "out"
    args = ["--images-dir", str(images_dir), "--output-dir", str(out), "--train-ratio", "0.5", "--seed", "3"]

    if mode == "txt":
        labels = tmp_path / "labels.txt"
        labels.write_text("img_a.jpg\t2024/01/01\nimg_b.jpg\t2024/01/02\n", encoding="utf-8")
        args.extend(["--labels-txt", str(labels)])
    elif mode == "csv":
        labels = tmp_path / "labels.csv"
        labels.write_text("image_path,transcription\nimg_a.jpg,2024/01/01\nimg_b.jpg,2024/01/02\n", encoding="utf-8")
        args.extend(["--labels-csv", str(labels)])
    elif mode == "jsonl":
        labels = tmp_path / "labels.jsonl"
        labels.write_text(
            "\n".join(
                [
                    json.dumps({"image_path": "img_a.jpg", "transcription": "2024/01/01"}),
                    json.dumps({"image_path": "img_b.jpg", "transcription": "2024/01/02"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        args.extend(["--labels-jsonl", str(labels)])
    else:
        # filename parser mode
        (images_dir / "img_a.jpg").unlink()
        (images_dir / "img_b.jpg").unlink()
        _write_image(images_dir / "2024-01-01.jpg")
        _write_image(images_dir / "2024-01-02.jpg")
        args.extend(["--labels-from-filename", "true", "--filename-label-regex", r"(?P<label>\d{4}-\d{2}-\d{2})"])

    code = run(args)
    assert code == 0

    train_lines = _read_label_lines(out / "train_label.txt")
    val_lines = _read_label_lines(out / "val_label.txt")
    all_lines = train_lines + val_lines
    assert len(all_lines) == 2
    _assert_label_paths_exist(out, all_lines)


def test_split_file_override_skips_unassigned_samples(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")
    _write_image(images_dir / "img2.jpg")
    _write_image(images_dir / "img3.jpg")

    labels = tmp_path / "labels.txt"
    labels.write_text(
        "img1.jpg\t2024/01/01\nimg2.jpg\t2024/01/02\nimg3.jpg\t2024/01/03\n",
        encoding="utf-8",
    )

    split = tmp_path / "split.txt"
    split.write_text("img1.jpg\ttrain\nimg2.jpg\tval\n", encoding="utf-8")

    out = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-txt",
            str(labels),
            "--split-file",
            str(split),
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["exported_total"] == 2
    assert summary["skipped_by_reason"]["split_unassigned"] == 1


def test_malformed_json_exits_with_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    labels_json = tmp_path / "bad.json"
    labels_json.write_text("{bad json", encoding="utf-8")

    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-json",
            str(labels_json),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "malformed JSON" in capsys.readouterr().err


def test_dry_run_does_not_write_dataset_artifacts(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    labels = tmp_path / "labels.txt"
    labels.write_text("img1.jpg\t2024/01/01\n", encoding="utf-8")

    out = tmp_path / "out"
    code = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-txt",
            str(labels),
            "--output-dir",
            str(out),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (out / "train_label.txt").exists()
    assert not (out / "val_label.txt").exists()
    assert not (out / "summary.json").exists()


def test_repeated_runs_are_deterministic(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")
    _write_image(images_dir / "img2.jpg")
    _write_image(images_dir / "img3.jpg")

    labels = tmp_path / "labels.txt"
    labels.write_text(
        "img1.jpg\t2024/01/01\nimg2.jpg\t2024/01/02\nimg3.jpg\t2024/01/03\n",
        encoding="utf-8",
    )

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    code1 = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-txt",
            str(labels),
            "--output-dir",
            str(out1),
            "--train-ratio",
            "0.67",
            "--seed",
            "11",
        ]
    )
    code2 = run(
        [
            "--images-dir",
            str(images_dir),
            "--labels-txt",
            str(labels),
            "--output-dir",
            str(out2),
            "--train-ratio",
            "0.67",
            "--seed",
            "11",
        ]
    )

    assert code1 == 0
    assert code2 == 0
    assert (out1 / "train_label.txt").read_text(encoding="utf-8") == (out2 / "train_label.txt").read_text(encoding="utf-8")
    assert (out1 / "val_label.txt").read_text(encoding="utf-8") == (out2 / "val_label.txt").read_text(encoding="utf-8")
