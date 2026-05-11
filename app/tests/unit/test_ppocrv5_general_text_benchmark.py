from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.ai.ppocrv5_general_text import GeneralTextExtraction, GeneralTextLine
from app.scripts import ppocrv5_general_text_benchmark as script


def _write_image(path: Path) -> None:
    Image.new("RGB", (12, 8), color=(255, 255, 255)).save(path, format="JPEG")


def test_run_benchmark_writes_reports_and_prints_visible_text(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    for index in range(12):
        _write_image(input_dir / f"{index:02d}.jpg")

    class FakeRunner:
        engine_name = "ppocrv5_general"
        runtime_device = "cpu"

        def extract(self, image_bytes: bytes) -> GeneralTextExtraction:
            assert image_bytes
            return GeneralTextExtraction(
                lines=[
                    GeneralTextLine(text="HELLO", confidence=0.9, bbox_xyxy=[0, 0, 10, 10]),
                    GeneralTextLine(text="WORLD", confidence=0.8, bbox_xyxy=[0, 12, 10, 22]),
                ],
                full_text="HELLO\nWORLD",
                confidence=0.85,
                engine_name=self.engine_name,
                runtime_device=self.runtime_device,
                reason=None,
                inference_ms=7,
            )

    report = script.run_benchmark(
        input_dir=input_dir,
        output_dir=tmp_path / "reports",
        limit=10,
        runner=FakeRunner(),
    )

    assert report.summary["total_images"] == 10
    assert report.summary["successful_images"] == 10
    assert (report.report_dir / "ppocrv5_general_text_results.jsonl").exists()
    assert (report.report_dir / "ppocrv5_general_text_report.json").exists()
    assert (report.report_dir / "ppocrv5_general_text_summary.md").exists()

    payload = json.loads((report.report_dir / "ppocrv5_general_text_report.json").read_text(encoding="utf-8"))
    assert payload["results"][0]["full_text"] == "HELLO\nWORLD"
    assert payload["results"][0]["lines"][0]["text"] == "HELLO"

    output = capsys.readouterr().out
    assert "00.jpg" in output
    assert "HELLO" in output
    assert "WORLD" in output
