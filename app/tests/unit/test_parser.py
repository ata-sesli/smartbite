from __future__ import annotations

from datetime import date

from app.ai.parser import ExpiryDateParser, normalize_ocr_date_tokens


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
    assert result.date_precision == "day"
    assert (result.parsed_day, result.parsed_month, result.parsed_year) == (1, 1, 2027)


def test_parser_supports_month_name_year_as_month_precision() -> None:
    parser = ExpiryDateParser()

    nov = parser.parse("NOV 2025", reference_date=date(2026, 1, 1))
    jan = parser.parse("JAN 2027", reference_date=date(2026, 1, 1))
    feb = parser.parse("FEB 2028", reference_date=date(2026, 1, 1))

    assert nov.date_format_detected == "MON YYYY"
    assert nov.date_precision == "month"
    assert (nov.parsed_day, nov.parsed_month, nov.parsed_year) == (None, 11, 2025)
    assert jan.date_precision == "month"
    assert (jan.parsed_day, jan.parsed_month, jan.parsed_year) == (None, 1, 2027)
    assert feb.date_precision == "month"
    assert (feb.parsed_day, feb.parsed_month, feb.parsed_year) == (None, 2, 2028)


def test_parser_recovers_trailing_month_year_after_noisy_prefix() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("1L/1-03/2029", reference_date=date(2026, 1, 1))

    assert result.date_format_detected == "TRAILING_MM/YYYY"
    assert result.date_precision == "month"
    assert (result.parsed_day, result.parsed_month, result.parsed_year) == (None, 3, 2029)


def test_parser_supports_slash_only_mm_yy_as_month_precision() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("05/26", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 5, 31)
    assert result.date_format_detected == "MM/YY"
    assert result.date_precision == "month"
    assert (result.parsed_day, result.parsed_month, result.parsed_year) == (None, 5, 2026)


def test_parser_supports_unambiguous_us_month_day_year() -> None:
    parser = ExpiryDateParser()

    short_year = parser.parse("05/25/27", reference_date=date(2026, 1, 1))
    full_year = parser.parse("EXP 05/25/2027", reference_date=date(2026, 1, 1))
    ambiguous = parser.parse("05/07/27", reference_date=date(2026, 1, 1))

    assert short_year.parsed_date == date(2027, 5, 25)
    assert short_year.date_format_detected == "MM/DD/YY_UNAMBIGUOUS"
    assert full_year.parsed_date == date(2027, 5, 25)
    assert full_year.date_format_detected == "MM/DD/YYYY_UNAMBIGUOUS"
    assert ambiguous.parsed_date == date(2027, 7, 5)
    assert ambiguous.date_format_detected == "DD/MM/YY"


def test_parser_supports_expiry_date_followed_by_glued_time() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("TETT:22.02.2611:49", reference_date=date(2026, 1, 1))
    noisy_year = parser.parse("1EY.22.02.6.1.19", reference_date=date(2026, 1, 1))
    missing_lead = parser.parse("ET.22.02.6.1.19", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 2, 22)
    assert result.date_format_detected == "DD/MM/YY_WITH_TIME"
    assert noisy_year.parsed_date == date(2026, 2, 22)
    assert noisy_year.date_format_detected == "DD/MM/Y_WITH_TIME"
    assert missing_lead.parsed_date == date(2026, 2, 22)
    assert missing_lead.date_format_detected == "DD/MM/Y_WITH_TIME"


def test_parser_does_not_enable_dot_or_dash_mm_yy() -> None:
    parser = ExpiryDateParser()

    assert parser.parse("01.28", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("02-02", reference_date=date(2026, 1, 1)).parsed_date is None


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


def test_parser_supports_common_ocr_month_confusion_for_dot_matrix() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("EXP 23/17/26", reference_date=date(2026, 4, 4))

    assert result.parsed_date == date(2026, 7, 23)
    assert result.date_format_detected in {"DD/MM/YY_FUZZY_MONTH", "DD/MM/YY", "D/M/YY_FUZZY_MONTH", "D/M/YY"}


def test_parser_supports_one_digit_day_month_and_dot_separator() -> None:
    parser = ExpiryDateParser(min_candidate_confidence=0.4)
    result = parser.parse("EXP 7.8.2026", reference_date=date(2026, 4, 4))

    assert result.parsed_date == date(2026, 8, 7)
    assert result.date_format_detected == "D/M/YYYY"


def test_normalize_ocr_date_tokens_replaces_common_confusions() -> None:
    normalized = normalize_ocr_date_tokens("EXP O7/I1/2S")
    assert normalized == "EXP 07/11/25"


def test_parser_prefers_expiry_keyword_over_production_date() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("UR.T. 12/12/2025 SKT 12/12/2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 12, 12)
    assert "role=expiry_keyword" in result.reason


def test_parser_prefers_expiry_keyword_even_when_production_date_has_prefix() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("MFG 12/12/2025 EXP 12/12/2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 12, 12)
    assert "role=expiry_keyword" in result.reason


def test_parser_keeps_single_production_like_date_as_expiry_candidate() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("UR.T. 12/12/2025", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2025, 12, 12)
    assert "role=single_candidate" in result.reason


def test_parser_prefers_later_date_when_multiple_dates_have_no_role_keyword() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("12/12/2025 12/12/2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 12, 12)
    assert "role=latest_non_production" in result.reason


def test_parser_uses_expiry_keyword_with_lot_noise_present() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("LOT 12345 EXP 18/02/2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 2, 18)
    assert "role=expiry_keyword" in result.reason


def test_parser_supports_compact_date_with_weak_expiry_context() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("YEYT 031026", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 10, 3)
    assert result.date_format_detected == "DDMMYY"
    assert "role=weak_expiry_keyword" in result.reason


def test_parser_supports_yett_compact_date_as_weak_expiry_context() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("YETT 031026", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 10, 3)
    assert result.date_format_detected == "DDMMYY"
    assert "role=weak_expiry_keyword" in result.reason


def test_parser_supports_feyy_as_weak_expiry_context() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("FEYY 11 04 2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 4, 11)
    assert result.date_format_detected == "DD MM YYYY"
    assert "role=weak_expiry_keyword" in result.reason


def test_parser_supports_compact_date_with_strong_expiry_context() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("STT:080426", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 4, 8)
    assert result.date_format_detected == "DDMMYY"


def test_parser_supports_spaced_numeric_date_with_weak_expiry_context() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("FELT 11 10 2022", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2022, 10, 11)
    assert result.date_format_detected == "DD MM YYYY"
    assert "role=weak_expiry_keyword" in result.reason


def test_parser_recovers_repeated_separators_before_month_year_fallback() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("2ETT.29..01.2028", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2028, 1, 29)
    assert result.date_format_detected == "DD/MM/YYYY_RECOVERED_SEPARATOR"
    assert "2028-01" in result.candidates


def test_parser_rejects_long_serial_like_digit_runs() -> None:
    parser = ExpiryDateParser()

    assert parser.parse("2701261251", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("90412065", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("LOT 080426", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("12 01 19", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("S11-30/04..29", reference_date=date(2026, 1, 1)).parsed_date is None


def test_parser_does_not_promote_production_date_above_expiry_after_recovery() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("URT 121225 TETT 121227", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 12, 12)
    assert result.date_format_detected == "DDMMYY"
    assert "role=expiry_keyword" in result.reason


def test_parser_recovers_complete_trailing_date_after_noisy_prefix() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("1.EB.19.05.27", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 5, 19)
    assert result.date_format_detected == "TRAILING_DD/MM/YY"


def test_parser_recovers_trailing_date_after_expiry_keyword() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("TETT.19.05.27", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 5, 19)
    assert result.date_format_detected == "TRAILING_DD/MM/YY"


def test_parser_recovers_complete_trailing_date_after_confusable_prefix() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("S.11.04.26", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 4, 11)
    assert result.date_format_detected == "TRAILING_DD/MM/YY"
    assert result.date_precision == "day"


def test_parser_recovers_clean_rightmost_date_after_numeric_noisy_prefix() -> None:
    parser = ExpiryDateParser()

    result = parser.parse("S11/30/04/26", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2026, 4, 30)
    assert result.date_format_detected == "TRAILING_DD/MM/YY"
    assert result.date_precision == "day"


def test_parser_rejects_mixed_separator_rightmost_suffix() -> None:
    parser = ExpiryDateParser()

    assert parser.parse("11/30-04/28", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("30/04-26", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("S11/30-04/26", reference_date=date(2026, 1, 1)).parsed_date is None


def test_parser_accepts_clean_mixed_separator_dates() -> None:
    parser = ExpiryDateParser()

    full_year = parser.parse("24.05/2026", reference_date=date(2026, 1, 1))
    short_year = parser.parse("21.07/27", reference_date=date(2026, 1, 1))
    slash_dot = parser.parse("24/05.2026", reference_date=date(2026, 1, 1))

    assert full_year.parsed_date == date(2026, 5, 24)
    assert full_year.date_format_detected == "DD/MM/YYYY_MIXED_SEPARATOR"
    assert short_year.parsed_date == date(2027, 7, 21)
    assert short_year.date_format_detected == "DD/MM/YY_MIXED_SEPARATOR"
    assert slash_dot.parsed_date == date(2026, 5, 24)
    assert slash_dot.date_format_detected == "DD/MM/YYYY_MIXED_SEPARATOR"


def test_parser_recovers_single_noise_digit_between_month_and_short_year() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("04.03.2.28", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2028, 3, 4)
    assert result.date_format_detected == "DD/MM/YY_NOISE_DIGIT"


def test_parser_recovers_e_as_two_in_noisy_day_prefix() -> None:
    parser = ExpiryDateParser()
    result = parser.parse("72E3/03/2027", reference_date=date(2026, 1, 1))

    assert result.parsed_date == date(2027, 3, 23)
    assert result.date_format_detected == "NOISY_DD/MM/YYYY_E_AS_TWO"


def test_parser_still_rejects_noisy_mixed_separator_dates() -> None:
    parser = ExpiryDateParser()

    assert parser.parse("S11/30-04/26", reference_date=date(2026, 1, 1)).parsed_date is None


def test_parser_rejects_incomplete_noisy_trailing_date() -> None:
    parser = ExpiryDateParser()

    assert parser.parse("1E.19.2027", reference_date=date(2026, 1, 1)).parsed_date is None
    assert parser.parse("1.EB.19.05", reference_date=date(2026, 1, 1)).parsed_date is None
