from __future__ import annotations

from app.ai.ppocrv5_general_text import PPOCRV5GeneralTextRunner


def test_parse_result_orders_lines_and_builds_full_text() -> None:
    raw_result = [
        {
            "rec_texts": ["BOTTOM", "TOP"],
            "rec_scores": [0.8, 0.9],
            "dt_polys": [
                [[0, 40], [80, 40], [80, 60], [0, 60]],
                [[0, 5], [60, 5], [60, 20], [0, 20]],
            ],
        }
    ]

    lines = PPOCRV5GeneralTextRunner.parse_result(raw_result)

    assert [line.text for line in lines] == ["TOP", "BOTTOM"]
    assert [line.bbox_xyxy for line in lines] == [[0, 5, 60, 20], [0, 40, 80, 60]]
    assert lines[0].confidence == 0.9
    assert PPOCRV5GeneralTextRunner.join_lines(lines) == "TOP\nBOTTOM"


def test_parse_result_accepts_paddlex_result_object_with_res() -> None:
    class ResultObject:
        res = {
            "rec_texts": ["MILK"],
            "rec_scores": [0.95],
            "dt_polys": [[[1, 2], [30, 2], [30, 14], [1, 14]]],
        }

    lines = PPOCRV5GeneralTextRunner.parse_result([ResultObject()])

    assert len(lines) == 1
    assert lines[0].text == "MILK"
    assert lines[0].bbox_xyxy == [1, 2, 30, 14]
    assert PPOCRV5GeneralTextRunner.average_confidence(lines) == 0.95


def test_parse_result_returns_empty_for_invalid_output() -> None:
    assert PPOCRV5GeneralTextRunner.parse_result(None) == []
    assert PPOCRV5GeneralTextRunner.parse_result([{"unexpected": "shape"}]) == []
