from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.scripts import generate_ppocrv5_rec_candidates as script


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment() -> None:
    # Override global DB fixture: this module only tests local CLI/filesystem behavior.
    yield


class FakeExtractor:
    def __init__(self, responses: list[list[script.OCRLine] | tuple[list[script.OCRLine], str | None]]) -> None:
        self.responses = responses
        self.index = 0

    def extract_lines(self, image_np):
        _ = image_np
        if self.index >= len(self.responses):
            return [], None

        response = self.responses[self.index]
        self.index += 1

        if isinstance(response, tuple):
            return response
        return response, None


def _patch_extractor(monkeypatch: pytest.MonkeyPatch, responses: list[list[script.OCRLine] | tuple[list[script.OCRLine], str | None]]) -> None:
    monkeypatch.setattr(script, "create_ocr_extractor", lambda _args: FakeExtractor(responses))


def _write_image(path: Path, size: tuple[int, int] = (140, 90), color: tuple[int, int, int] = (240, 240, 240)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_yolo_jsonl_validation_rejects_missing_fields(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True)

    bad = tmp_path / "det.jsonl"
    bad.write_text(json.dumps({"image_path": "a.jpg", "product_id": "p1", "bbox_xyxy": [1, 2, 3, 4], "score": 0.9}) + "\n")

    with pytest.raises(ValueError, match="missing fields"):
        script.load_yolo_detections_jsonl(bad, images_dir)


def test_candidate_id_is_deterministic() -> None:
    first = script.build_candidate_id(
        seed=42,
        source_image="img1.jpg",
        source_product_id="prod_1",
        source_product_bbox=(0, 0, 100, 50),
        text_region_bbox_in_source=(10, 12, 40, 28),
        text="EXP 12/04/26",
        line_index=0,
    )
    second = script.build_candidate_id(
        seed=42,
        source_image="img1.jpg",
        source_product_id="prod_1",
        source_product_bbox=(0, 0, 100, 50),
        text_region_bbox_in_source=(10, 12, 40, 28),
        text="EXP 12/04/26",
        line_index=0,
    )
    assert first == second


def test_heuristic_feature_extraction_detects_expiry_and_hardness() -> None:
    crop = Image.new("RGB", (12, 10), color=(128, 128, 128))
    expiry, confuser, hardness, priority, tags, suggested = script.evaluate_candidate(
        text="BEST BEFORE 12/05/26",
        confidence=0.55,
        crop_width=12,
        crop_height=10,
        crop_image=crop,
    )
    crop.close()

    assert suggested == "expiry"
    assert expiry > confuser
    assert hardness > 0
    assert priority > 0
    assert "date_like" in tags
    assert "expiry_keyword" in tags


def test_score_and_class_are_deterministic() -> None:
    crop1 = Image.new("RGB", (20, 14), color=(120, 120, 120))
    crop2 = Image.new("RGB", (20, 14), color=(120, 120, 120))

    first = script.evaluate_candidate("LOT 12345", 0.7, 20, 14, crop1)
    second = script.evaluate_candidate("LOT 12345", 0.7, 20, 14, crop2)

    crop1.close()
    crop2.close()
    assert first == second


def test_limit_candidates_respects_bucket_caps(tmp_path: Path) -> None:
    args = script.parse_args(["--images-dir", str(tmp_path), "--output-dir", str(tmp_path / "out")])
    args.max_candidates_per_source = 3
    args.max_expiry_per_source = 1
    args.max_confuser_per_source = 1
    args.max_other_per_source = 1

    def candidate(cid: str, suggested: str, priority: float) -> script.Candidate:
        return script.Candidate(
            candidate_id=cid,
            source_image="img1.jpg",
            source_image_path=tmp_path / "img1.jpg",
            source_product_id="p1",
            source_product_bbox=(0, 0, 60, 30),
            source_product_score=1.0,
            source_product_class="whole",
            text_region_bbox_in_product=(0, 0, 10, 10),
            text_region_bbox_in_source=(0, 0, 10, 10),
            ocr_prefill=cid,
            ocr_confidence=0.9,
            crop_width=10,
            crop_height=10,
            expiry_score=priority,
            confuser_score=0.0,
            hardness_score=0.0,
            priority_score=priority,
            heuristic_tags=[],
            suggested_class=suggested,
            tie_break=0,
            crop_image=Image.new("RGB", (10, 10), color=(255, 255, 255)),
        )

    items = [
        candidate("e1", "expiry", 9.0),
        candidate("e2", "expiry", 8.0),
        candidate("c1", "lot", 7.0),
        candidate("o1", "other", 6.0),
        candidate("o2", "brand", 5.0),
    ]

    selected = script.limit_candidates_for_source(items, args)
    assert len(selected) == 3
    assert sum(1 for x in selected if script.bucket_for_class(x.suggested_class) == "expiry") == 1
    assert sum(1 for x in selected if script.bucket_for_class(x.suggested_class) == "confuser") == 1
    assert sum(1 for x in selected if script.bucket_for_class(x.suggested_class) == "other") == 1

    for item in items:
        item.crop_image.close()


def test_full_images_without_yolo_uses_whole_image_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    _patch_extractor(
        monkeypatch,
        [
            [
                script.OCRLine((10, 10, 60, 30), "EXP 12/05/26", 0.91),
                script.OCRLine((62, 12, 110, 28), "LOT 4321", 0.88),
            ]
        ],
    )

    out = tmp_path / "out"
    code = script.run(["--images-dir", str(images_dir), "--output-dir", str(out), "--seed", "9"])
    assert code == 0

    manifest = _read_jsonl(out / "candidates" / "manifest.jsonl")
    assert len(manifest) == 2
    assert {row["source_product_class"] for row in manifest} == {"whole_image_fallback"}

    for row in manifest:
        assert (out / row["crop_path"]).exists()


def test_full_images_with_yolo_jsonl_uses_detected_regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg", size=(200, 120))

    yolo = tmp_path / "det.jsonl"
    yolo.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_path": "img1.jpg",
                        "product_id": "p1",
                        "bbox_xyxy": [10, 10, 90, 70],
                        "score": 0.95,
                        "class_name": "product",
                    }
                ),
                json.dumps(
                    {
                        "image_path": "img1.jpg",
                        "product_id": "p2",
                        "bbox_xyxy": [100, 20, 180, 90],
                        "score": 0.9,
                        "class_name": "product",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _patch_extractor(
        monkeypatch,
        [
            [script.OCRLine((5, 5, 40, 20), "BEST BEFORE 01/01/27", 0.9)],
            [script.OCRLine((8, 8, 35, 22), "LOT 55", 0.8)],
        ],
    )

    out = tmp_path / "out"
    code = script.run(
        [
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(out),
            "--yolo-detections-jsonl",
            str(yolo),
        ]
    )
    assert code == 0

    manifest = _read_jsonl(out / "candidates" / "manifest.jsonl")
    assert len(manifest) == 2
    assert {row["source_product_id"] for row in manifest} == {"p1", "p2"}


def test_product_crops_mode_treats_each_image_as_one_region(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "crop1.jpg")
    _write_image(images_dir / "crop2.jpg")

    _patch_extractor(
        monkeypatch,
        [
            [script.OCRLine((6, 6, 40, 20), "ABC", 0.9)],
            [script.OCRLine((5, 8, 50, 25), "EXP 02/02/27", 0.92)],
        ],
    )

    out = tmp_path / "out"
    code = script.run(
        [
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(out),
            "--input-mode",
            "product_crops",
        ]
    )
    assert code == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["images_processed"] == 2
    assert summary["product_regions_processed"] == 2


def test_dry_run_produces_summary_only_on_stdout_and_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    _patch_extractor(monkeypatch, [[script.OCRLine((10, 10, 40, 25), "EXP 10/10/27", 0.9)]])

    out = tmp_path / "out"
    code = script.run(["--images-dir", str(images_dir), "--output-dir", str(out), "--dry-run"])
    assert code == 0
    assert not out.exists()


def test_csv_and_overlay_exports_are_generated_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    _patch_extractor(monkeypatch, [[script.OCRLine((10, 10, 50, 28), "LOT 123", 0.85)]])

    out = tmp_path / "out"
    code = script.run(
        [
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(out),
            "--export-csv",
            "true",
            "--export-overlays",
            "true",
        ]
    )
    assert code == 0

    assert (out / "candidates" / "review.csv").exists()
    overlays = list((out / "candidates" / "overlays").glob("*.jpg"))
    assert overlays


def test_manifest_paths_exist_and_candidate_ids_are_unique_and_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images_dir = tmp_path / "images"
    _write_image(images_dir / "img1.jpg")

    responses = [
        [
            script.OCRLine((10, 10, 60, 30), "EXP 03/03/27", 0.9),
            script.OCRLine((65, 10, 115, 30), "BRANDNAME", 0.95),
        ]
    ]

    out1 = tmp_path / "out1"
    _patch_extractor(monkeypatch, responses)
    code1 = script.run(["--images-dir", str(images_dir), "--output-dir", str(out1), "--seed", "42"])
    assert code1 == 0

    manifest1 = _read_jsonl(out1 / "candidates" / "manifest.jsonl")
    ids1 = [row["candidate_id"] for row in manifest1]
    assert len(ids1) == len(set(ids1))
    for row in manifest1:
        assert (out1 / row["crop_path"]).exists()

    out2 = tmp_path / "out2"
    _patch_extractor(monkeypatch, responses)
    code2 = script.run(["--images-dir", str(images_dir), "--output-dir", str(out2), "--seed", "42"])
    assert code2 == 0

    manifest2 = _read_jsonl(out2 / "candidates" / "manifest.jsonl")
    ids2 = [row["candidate_id"] for row in manifest2]
    assert ids1 == ids2
