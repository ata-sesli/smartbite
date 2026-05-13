from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _winner_by_filename(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in report.get("items", []):
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        winner = item.get("winner")
        if isinstance(filename, str) and isinstance(winner, dict):
            out[filename] = winner
    return out


def _confidence_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return None


def summarize_parity(paddle_report: dict[str, Any], onnx_report: dict[str, Any]) -> dict[str, Any]:
    paddle = _winner_by_filename(paddle_report)
    onnx = _winner_by_filename(onnx_report)
    filenames = sorted(set(paddle) | set(onnx))
    mismatches: list[dict[str, Any]] = []
    confidence_deltas: list[float] = []
    text_matches = 0
    parsed_date_matches = 0
    exact_match_agreements = 0

    for filename in filenames:
        paddle_winner = paddle.get(filename, {})
        onnx_winner = onnx.get(filename, {})
        paddle_text = paddle_winner.get("parseq_normalized_output")
        onnx_text = onnx_winner.get("parseq_normalized_output")
        paddle_date = paddle_winner.get("parser_parsed_date")
        onnx_date = onnx_winner.get("parser_parsed_date")
        paddle_exact = paddle_winner.get("exact_match")
        onnx_exact = onnx_winner.get("exact_match")
        delta = _confidence_delta(paddle_winner.get("parseq_confidence"), onnx_winner.get("parseq_confidence"))
        if delta is not None:
            confidence_deltas.append(delta)

        text_match = paddle_text == onnx_text
        parsed_date_match = paddle_date == onnx_date
        exact_match_agreement = paddle_exact == onnx_exact
        text_matches += int(text_match)
        parsed_date_matches += int(parsed_date_match)
        exact_match_agreements += int(exact_match_agreement)
        if not (text_match and parsed_date_match and exact_match_agreement):
            mismatches.append(
                {
                    "filename": filename,
                    "paddle_text": paddle_text,
                    "onnx_text": onnx_text,
                    "paddle_parsed_date": paddle_date,
                    "onnx_parsed_date": onnx_date,
                    "paddle_exact_match": paddle_exact,
                    "onnx_exact_match": onnx_exact,
                    "confidence_delta": round(delta, 6) if delta is not None else None,
                }
            )

    return {
        "total": len(filenames),
        "text_matches": text_matches,
        "parsed_date_matches": parsed_date_matches,
        "exact_match_agreements": exact_match_agreements,
        "max_confidence_delta": round(max(confidence_deltas), 6) if confidence_deltas else None,
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Paddle SVTR and ONNX SVTR manual-crop benchmark reports")
    parser.add_argument("--paddle-report", type=Path, required=True)
    parser.add_argument("--onnx-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paddle_report = json.loads(args.paddle_report.read_text(encoding="utf-8"))
    onnx_report = json.loads(args.onnx_report.read_text(encoding="utf-8"))
    summary = summarize_parity(paddle_report, onnx_report)
    payload = json.dumps(summary, ensure_ascii=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
