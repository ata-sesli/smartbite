from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Protocol

from app.ai.ppocrv5_general_text import GeneralTextExtraction, PPOCRV5GeneralTextRunner


@dataclass(slots=True)
class BenchmarkItem:
    filename: str
    image_path: str
    full_text: str
    confidence: float | None
    reason: str | None
    inference_ms: int
    lines: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkReport:
    report_dir: Path
    summary: dict[str, Any]
    results: list[BenchmarkItem]


class GeneralTextRunner(Protocol):
    engine_name: str
    runtime_device: str

    def extract(self, image_bytes: bytes) -> GeneralTextExtraction:
        ...


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_images(input_dir: Path, *, limit: int | None) -> list[Path]:
    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed)
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def _summary(results: list[BenchmarkItem], runner: GeneralTextRunner) -> dict[str, Any]:
    total_ms = sum(item.inference_ms for item in results)
    return {
        "engine_name": runner.engine_name,
        "runtime_device": runner.runtime_device,
        "total_images": len(results),
        "successful_images": sum(1 for item in results if item.reason is None),
        "failed_images": sum(1 for item in results if item.reason is not None),
        "total_lines": sum(len(item.lines) for item in results),
        "average_inference_ms": round(total_ms / len(results), 2) if results else None,
    }


def _write_report(report_dir: Path, summary: dict[str, Any], results: list[BenchmarkItem]) -> None:
    _ensure_dir(report_dir)
    jsonl_path = report_dir / "ppocrv5_general_text_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    payload = {"summary": summary, "results": [item.to_dict() for item in results]}
    (report_dir / "ppocrv5_general_text_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# PP-OCRv5 General Text Benchmark",
        "",
        f"- engine_name: `{summary['engine_name']}`",
        f"- runtime_device: `{summary['runtime_device']}`",
        f"- total_images: `{summary['total_images']}`",
        f"- successful_images: `{summary['successful_images']}`",
        f"- failed_images: `{summary['failed_images']}`",
        f"- total_lines: `{summary['total_lines']}`",
        f"- average_inference_ms: `{summary['average_inference_ms']}`",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item.filename}",
                "",
                "```text",
                item.full_text,
                "```",
                "",
            ]
        )
    (report_dir / "ppocrv5_general_text_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(
    *,
    input_dir: Path,
    output_dir: Path,
    limit: int,
    runner: GeneralTextRunner,
) -> BenchmarkReport:
    image_paths = _iter_images(input_dir, limit=limit)
    report_dir = output_dir / f"ppocrv5_general_text_{_utc_stamp()}"
    results: list[BenchmarkItem] = []

    for image_path in image_paths:
        extraction = runner.extract(image_path.read_bytes())
        item = BenchmarkItem(
            filename=image_path.name,
            image_path=str(image_path),
            full_text=extraction.full_text,
            confidence=extraction.confidence,
            reason=extraction.reason,
            inference_ms=extraction.inference_ms,
            lines=[line.to_dict() for line in extraction.lines],
        )
        results.append(item)
        print(f"{item.filename}\n{item.full_text}\n")

    summary = _summary(results, runner)
    _write_report(report_dir, summary, results)
    return BenchmarkReport(report_dir=report_dir, summary=summary, results=results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PP-OCRv5 general visible-text benchmark")
    parser.add_argument("--input-dir", type=Path, default=Path("test-images"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device-mode", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-image-side", type=int, default=1600)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forensics"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise SystemExit(f"input directory not found: {args.input_dir}")
    runner = PPOCRV5GeneralTextRunner(device_mode=args.device_mode, max_image_side=args.max_image_side)
    run_benchmark(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        runner=runner,
    )


if __name__ == "__main__":
    main()
