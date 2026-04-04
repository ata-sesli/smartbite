from __future__ import annotations

from datetime import date

from app.ai.parser import ExpiryDateParser


def test_parser_supports_prefixed_dd_mm_yy() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("EXP 29/03/26")

    assert result.parsed_date == date(2026, 3, 29)
    assert result.date_format_detected == "DD/MM/YY"
    assert result.confidence > 0.8


def test_parser_supports_dd_mon_yyyy() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("BEST BEFORE 01 JAN 2027")

    assert result.parsed_date == date(2027, 1, 1)
    assert result.date_format_detected == "DD MON YYYY"


def test_parser_rejects_unparseable_text() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("LOT 554433 MFG 2026")

    assert result.parsed_date is None
    assert result.reason == "no valid date parsed"


def test_parser_rejects_implausible_old_year() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("EXP 01/01/2010", reference_date=date(2026, 4, 4))

    assert result.parsed_date is None
    assert result.reason == "no valid date parsed"


def test_parser_accepts_recent_expired_date_within_window() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("SNTT 25/11/21", reference_date=date(2026, 4, 4))

    assert result.parsed_date == date(2021, 11, 25)
    assert result.date_format_detected == "DD/MM/YY"
