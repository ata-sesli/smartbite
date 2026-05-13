from __future__ import annotations

from app.scripts.svtr_onnx_parity import summarize_parity


def test_summarize_parity_tracks_text_and_confidence_deltas() -> None:
    paddle_report = {
        "items": [
            {
                "filename": "a.jpg",
                "winner": {
                    "variant_name": "original/original_color_tight",
                    "parseq_normalized_output": "EXP 01/02/2027",
                    "parseq_confidence": 0.90,
                    "parser_parsed_date": "2027-02-01",
                    "exact_match": True,
                },
            },
            {
                "filename": "b.jpg",
                "winner": {
                    "variant_name": "original/original_color_tight",
                    "parseq_normalized_output": "LOT 123",
                    "parseq_confidence": 0.50,
                    "parser_parsed_date": None,
                    "exact_match": False,
                },
            },
        ]
    }
    onnx_report = {
        "items": [
            {
                "filename": "a.jpg",
                "winner": {
                    "variant_name": "original/original_color_tight",
                    "parseq_normalized_output": "EXP 01/02/2027",
                    "parseq_confidence": 0.86,
                    "parser_parsed_date": "2027-02-01",
                    "exact_match": True,
                },
            },
            {
                "filename": "b.jpg",
                "winner": {
                    "variant_name": "original/original_color_tight",
                    "parseq_normalized_output": "LOT 12",
                    "parseq_confidence": 0.51,
                    "parser_parsed_date": None,
                    "exact_match": False,
                },
            },
        ]
    }

    summary = summarize_parity(paddle_report, onnx_report)

    assert summary["total"] == 2
    assert summary["text_matches"] == 1
    assert summary["parsed_date_matches"] == 2
    assert summary["exact_match_agreements"] == 2
    assert summary["max_confidence_delta"] == 0.04
    assert [item["filename"] for item in summary["mismatches"]] == ["b.jpg"]
