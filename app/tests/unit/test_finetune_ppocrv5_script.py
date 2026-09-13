from __future__ import annotations

from pathlib import Path

from app.scripts.finetune_ppocrv5_rec import run


def _build_valid_dataset(root: Path) -> tuple[Path, Path, Path]:
    train_images = root / "train_images"
    val_images = root / "val_images"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)

    train_image = train_images / "sample1.jpg"
    val_image = val_images / "sample2.jpg"
    train_image.write_bytes(b"fake-image")
    val_image.write_bytes(b"fake-image")

    (root / "train_label.txt").write_text("train_images/sample1.jpg\tEXP 29/03/26\n")
    (root / "val_label.txt").write_text("val_images/sample2.jpg\tBEST BEFORE 10/04/26\n")

    pretrained_model = root / "pretrained" / "best_accuracy"
    pretrained_model.parent.mkdir(parents=True)
    pretrained_model.write_text("checkpoint-placeholder")

    base_config = root / "rec_train.yml"
    base_config.write_text("Global:\n  use_gpu: false\n")

    return root, pretrained_model, base_config


def test_finetune_dry_run_passes_with_valid_dataset(tmp_path: Path) -> None:
    data_root, pretrained_model, base_config = _build_valid_dataset(tmp_path / "data")
    output_dir = tmp_path / "outputs"

    code = run(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--pretrained-model",
            str(pretrained_model),
            "--base-config",
            str(base_config),
            "--dry-run",
        ]
    )

    assert code == 0


def test_finetune_dry_run_fails_on_bad_label_line(tmp_path: Path, capsys) -> None:
    data_root, pretrained_model, base_config = _build_valid_dataset(tmp_path / "data")
    (data_root / "train_label.txt").write_text("train_images/sample1.jpg EXP 29/03/26\n")

    code = run(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--pretrained-model",
            str(pretrained_model),
            "--base-config",
            str(base_config),
            "--dry-run",
        ]
    )

    stderr = capsys.readouterr().err
    assert code == 2
    assert "must contain tab separator" in stderr


def test_finetune_dry_run_fails_when_image_missing(tmp_path: Path, capsys) -> None:
    data_root, pretrained_model, base_config = _build_valid_dataset(tmp_path / "data")
    (data_root / "train_images" / "sample1.jpg").unlink()

    code = run(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--pretrained-model",
            str(pretrained_model),
            "--base-config",
            str(base_config),
            "--dry-run",
        ]
    )

    stderr = capsys.readouterr().err
    assert code == 2
    assert "references missing image" in stderr
