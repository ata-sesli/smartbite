from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

from app.ai.date_role import (
    has_expiry_keyword,
    has_noise_keyword,
    has_production_keyword,
    has_weak_expiry_keyword,
    role_selection_key,
    score_date_role,
)
from app.ai.types import ParsedDateData

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

PREFIX_HINTS = ("EXP", "USE BY", "BEST BEFORE")
LOW_PRIORITY_TOKENS = ("LOT", "BATCH", "MFG", "PROD")
SEPARATOR_PATTERN = r"[\\/\\.-]"
OCR_CONFUSION_MAP = str.maketrans(
    {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "B": "8",
    }
)


@dataclass(slots=True)
class Candidate:
    dt: date
    fmt: str
    confidence: float
    source: str
    span: tuple[int, int]
    precision: str = "day"

    @property
    def parsed_day(self) -> int | None:
        return None if self.precision == "month" else self.dt.day

    @property
    def parsed_month(self) -> int:
        return self.dt.month

    @property
    def parsed_year(self) -> int:
        return self.dt.year


def _normalize_year(value: int) -> int:
    if value >= 100:
        return value
    return 2000 + value if value < 70 else 1900 + value


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def normalize_ocr_date_tokens(text: str) -> str:
    def replace_token(match: "re.Match[str]") -> str:
        token = match.group(0)
        has_digit_or_separator = any(ch.isdigit() or ch in "/.-:" for ch in token)
        has_confusable_chars = any(ch in "OILSB" for ch in token)
        if has_digit_or_separator and has_confusable_chars:
            return token.translate(OCR_CONFUSION_MAP)
        return token

    return re.sub(r"[A-Z0-9/\\.-:]+", replace_token, text.upper())


class ExpiryDateParser:
    def __init__(
        self,
        *,
        min_year: int = 2000,
        max_year: int = 2099,
        max_past_years: int = 7,
        max_future_years: int = 15,
        min_candidate_confidence: float = 0.40,
    ) -> None:
        self.min_year = min_year
        self.max_year = max_year
        self.max_past_years = max_past_years
        self.max_future_years = max_future_years
        self.min_candidate_confidence = min_candidate_confidence

    def parse(self, normalized_text: str, *, reference_date: date | None = None) -> ParsedDateData:
        text = normalize_ocr_date_tokens(normalized_text)
        today = reference_date or date.today()
        candidates: list[Candidate] = []

        candidates.extend(self._extract_trailing_noisy_prefix_d_m_y(text))
        candidates.extend(self._extract_trailing_noisy_prefix_mm_yyyy(text))
        candidates.extend(self._extract_d_m_yyyy(text))
        candidates.extend(self._extract_d_m_yy(text))
        candidates.extend(self._extract_d_m_yy_with_common_ocr_month_confusion(text))
        candidates.extend(self._extract_malformed_separator_d_m_y(text))
        candidates.extend(self._extract_compact_d_m_y(text))
        candidates.extend(self._extract_spaced_d_m_y(text))
        candidates.extend(self._extract_yyyy_mm_dd(text))
        candidates.extend(self._extract_mm_yyyy(text))
        candidates.extend(self._extract_dd_mon_yyyy(text))
        candidates.extend(self._extract_mon_yyyy(text))
        candidates = self._prefer_clean_trailing_month_year(candidates)

        candidates = [
            c for c in candidates if self._is_plausible_date(c.dt, today) and c.confidence >= self.min_candidate_confidence
        ]
        if not candidates:
            return ParsedDateData(None, None, 0.0, [], "no valid date parsed")

        primary_count = sum(1 for candidate in candidates if self._format_specificity(candidate.fmt) > 0)
        single_primary_candidate = primary_count == 1
        ranked = sorted(
            candidates,
            key=lambda c: self._candidate_selection_key(c, text, single_primary_candidate=single_primary_candidate),
            reverse=True,
        )
        best = ranked[0]
        best_role = score_date_role(text, span=best.span)
        role_reason = (
            "role=single_candidate"
            if single_primary_candidate and not best_role.expiry_keyword and best_role.role != "weak_expiry_keyword"
            else best_role.reason
        )

        reason_parts = []
        if has_expiry_keyword(text):
            reason_parts.append("prefix_detected")
        if has_production_keyword(text) or has_noise_keyword(text):
            reason_parts.append("contains_low_priority_token")
        reason_parts.append(f"selected_{best.fmt}")
        reason_parts.append(role_reason)

        return ParsedDateData(
            parsed_date=best.dt,
            date_format_detected=best.fmt,
            confidence=round(best.confidence, 4),
            candidates=[self._candidate_display_value(c) for c in ranked],
            reason=";".join(reason_parts),
            date_precision=best.precision,
            parsed_day=best.parsed_day,
            parsed_month=best.parsed_month,
            parsed_year=best.parsed_year,
        )

    @staticmethod
    def _candidate_display_value(candidate: Candidate) -> str:
        if candidate.precision == "month":
            return f"{candidate.dt.year:04d}-{candidate.dt.month:02d}"
        return candidate.dt.isoformat()

    def _score(self, text: str, source: str, base: float = 0.8) -> float:
        score = base
        if has_expiry_keyword(text):
            score += 0.15
        if has_production_keyword(source) or has_noise_keyword(source):
            score -= 0.3
        date_char_ratio = self._date_char_ratio(text)
        if date_char_ratio < 0.35:
            score -= 0.1
        return max(0.0, min(score, 0.99))

    def _candidate_selection_key(
        self,
        candidate: Candidate,
        text: str,
        *,
        single_primary_candidate: bool,
    ) -> tuple[int, int, int, int, float]:
        evidence = score_date_role(text, span=candidate.span)
        if single_primary_candidate:
            evidence_priority = 2
            non_production = 1
        else:
            evidence_priority = evidence.priority
            non_production = evidence.non_production
        return role_selection_key(
            evidence=evidence,
            parsed_date=candidate.dt,
            specificity=self._format_specificity(candidate.fmt),
            confidence=candidate.confidence,
        ) if not single_primary_candidate else (
            evidence_priority,
            non_production,
            self._format_specificity(candidate.fmt),
            candidate.dt.toordinal(),
            candidate.confidence,
        )

    @staticmethod
    def _format_specificity(fmt: str) -> int:
        if fmt in {
            "DD/MM/YYYY",
            "D/M/YYYY",
            "DD/MM/YY",
            "D/M/YY",
            "YYYY-MM-DD",
            "DD MON YYYY",
            "DDMMYYYY",
            "DD MM YYYY",
            "DD/MM/YYYY_RECOVERED_SEPARATOR",
            "TRAILING_DD/MM/YYYY",
            "TRAILING_DD/MM/YY",
        }:
            return 2
        if fmt in {
            "D/M/YY_FUZZY_MONTH",
            "DDMMYY",
            "DD MM YY",
            "DD/MM/YY_RECOVERED_SEPARATOR",
        }:
            return 1
        return 0

    @staticmethod
    def _date_char_ratio(text: str) -> float:
        compact = [ch for ch in text if not ch.isspace()]
        if not compact:
            return 0.0
        date_like = sum(ch.isdigit() or ch in {"/", "-", ".", ":"} for ch in compact)
        return date_like / len(compact)

    def _is_plausible_date(self, parsed: date, today: date) -> bool:
        if parsed.year < self.min_year or parsed.year > self.max_year:
            return False
        min_allowed = date(today.year - self.max_past_years, 1, 1)
        max_allowed = date(today.year + self.max_future_years, 12, 31)
        return min_allowed <= parsed <= max_allowed

    def _extract_d_m_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(\d{{1,2}})({SEPARATOR_PATTERN})(\d{{1,2}})\2(\d{{4}})", text):
            d, m, y = (int(match.group(1)), int(match.group(3)), int(match.group(4)))
            parsed = _safe_date(y, m, d)
            if parsed:
                day_len = len(match.group(1))
                month_len = len(match.group(3))
                base = 0.95 if day_len == 2 and month_len == 2 else 0.82
                fmt = "DD/MM/YYYY" if day_len == 2 and month_len == 2 else "D/M/YYYY"
                out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _extract_trailing_noisy_prefix_d_m_y(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        pattern = fr"(\d{{1,2}})({SEPARATOR_PATTERN})(\d{{1,2}})\2(\d{{2}}|\d{{4}})\s*$"
        for match in re.finditer(pattern, text):
            prefix = text[: match.start()]
            if not self._has_noisy_trailing_date_context(prefix):
                continue
            d, m = int(match.group(1)), int(match.group(3))
            y_raw = match.group(4)
            y = _normalize_year(int(y_raw)) if len(y_raw) == 2 else int(y_raw)
            parsed = _safe_date(y, m, d)
            if not parsed:
                continue
            fmt = "TRAILING_DD/MM/YY" if len(y_raw) == 2 else "TRAILING_DD/MM/YYYY"
            base = 0.93 if len(y_raw) == 2 else 0.96
            out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _extract_trailing_noisy_prefix_mm_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(?<!\d)(\d{{1,2}}){SEPARATOR_PATTERN}(\d{{4}})\s*$", text):
            prefix = text[: match.start()]
            if not self._has_noisy_trailing_month_year_context(prefix, separator=match.group(0)[len(match.group(1))]):
                continue
            m, y = int(match.group(1)), int(match.group(2))
            if m < 1 or m > 12:
                continue
            parsed = _safe_date(y, m, calendar.monthrange(y, m)[1])
            if not parsed:
                continue
            out.append(
                Candidate(
                    parsed,
                    "TRAILING_MM/YYYY",
                    self._score(text, match.group(0), 0.88),
                    match.group(0),
                    match.span(),
                    "month",
                )
            )
        return out

    @staticmethod
    def _has_noisy_trailing_date_context(prefix: str) -> bool:
        if not prefix:
            return False
        if has_production_keyword(prefix) or has_noise_keyword(prefix):
            return False
        stripped = prefix.rstrip()
        compact = re.sub(r"[^A-Z0-9]", "", prefix.upper())
        if has_weak_expiry_keyword(prefix):
            return True
        if has_expiry_keyword(prefix) and stripped.endswith((".", "/", "-")) and any(
            token in compact for token in ("TETT", "SKT", "STT")
        ):
            return True
        if any(ch.isalpha() for ch in compact) and any(ch in prefix for ch in ".-/") and len(compact) <= 8:
            return True
        if compact.isdigit() and len(compact) <= 2 and any(ch in prefix for ch in ".-/") and len(prefix.strip()) <= 4:
            return True
        if compact.isdigit() and len(compact) <= 3 and stripped.endswith((".", "/", "-")) and len(prefix.strip()) <= 5:
            return True
        return False

    @staticmethod
    def _has_noisy_trailing_month_year_context(prefix: str, *, separator: str) -> bool:
        if not prefix:
            return False
        if has_production_keyword(prefix) or has_noise_keyword(prefix):
            return False
        if prefix.endswith(separator * 2):
            return False
        day_prefix = re.search(rf"(\d{{1,2}}){re.escape(separator)}$", prefix)
        if day_prefix:
            before_day = prefix[: day_prefix.start(1)].rstrip()
            before_day_compact = re.sub(r"[^A-Z0-9]", "", before_day.upper())
            if any(ch.isalpha() for ch in before_day_compact):
                return False
            if not before_day or len(before_day_compact) < 2:
                return False
            if before_day and before_day[-1] not in "/.-":
                return False
        compact = re.sub(r"[^A-Z0-9]", "", prefix.upper())
        if has_expiry_keyword(prefix) or has_weak_expiry_keyword(prefix):
            return True
        if any(ch in prefix for ch in ".-/") and len(compact) <= 8:
            return True
        return False

    @staticmethod
    def _prefer_clean_trailing_month_year(candidates: list[Candidate]) -> list[Candidate]:
        trailing_months = [candidate for candidate in candidates if candidate.fmt == "TRAILING_MM/YYYY"]
        if not trailing_months:
            return candidates
        cleaned = list(candidates)
        for month_candidate in trailing_months:
            month_start = month_candidate.span[0]
            month_end = month_candidate.span[1]
            cleaned = [
                candidate
                for candidate in cleaned
                if not (
                    candidate.precision == "day"
                    and candidate.span[1] == month_end
                    and candidate.span[0] < month_start
                    and candidate.dt.month == month_candidate.dt.month
                    and candidate.dt.year == month_candidate.dt.year
                )
            ]
        return cleaned

    def _extract_d_m_yy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(\d{{1,2}})({SEPARATOR_PATTERN})(\d{{1,2}})\2(\d{{2}})(?!\d)", text):
            d, m, y = (int(match.group(1)), int(match.group(3)), _normalize_year(int(match.group(4))))
            parsed = _safe_date(y, m, d)
            if parsed:
                day_len = len(match.group(1))
                month_len = len(match.group(3))
                base = 0.88 if day_len == 2 and month_len == 2 else 0.76
                fmt = "DD/MM/YY" if day_len == 2 and month_len == 2 else "D/M/YY"
                out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _extract_d_m_yy_with_common_ocr_month_confusion(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(\d{{1,2}})({SEPARATOR_PATTERN})(\d{{1,2}})\2(\d{{2}})(?!\d)", text):
            d_raw = int(match.group(1))
            m_raw = int(match.group(3))
            y = _normalize_year(int(match.group(4)))

            # Common OCR confusion on dot-matrix lines: month "07" can become "17".
            month_candidates: list[int] = []
            if 13 <= m_raw <= 19:
                month_candidates.append(m_raw - 10)
            if not month_candidates:
                continue

            for month in month_candidates:
                parsed = _safe_date(y, month, d_raw)
                if parsed:
                    out.append(
                        Candidate(
                            parsed,
                            "D/M/YY_FUZZY_MONTH",
                            self._score(text, match.group(0), 0.68),
                            match.group(0),
                            match.span(),
                        )
                    )
        return out

    def _extract_yyyy_mm_dd(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(\d{{4}})({SEPARATOR_PATTERN})(\d{{1,2}})\2(\d{{1,2}})", text):
            y, m, d = (int(match.group(1)), int(match.group(3)), int(match.group(4)))
            parsed = _safe_date(y, m, d)
            if parsed:
                out.append(Candidate(parsed, "YYYY-MM-DD", self._score(text, match.group(0), 0.9), match.group(0), match.span()))
        return out

    def _extract_compact_d_m_y(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(?<!\d)(\d{6}|\d{8})(?!\d)", text):
            token = match.group(1)
            parsed: date | None = None
            fmt = ""
            if len(token) == 6:
                parsed = _safe_date(_normalize_year(int(token[4:6])), int(token[2:4]), int(token[0:2]))
                fmt = "DDMMYY"
                base = 0.72
            else:
                parsed = _safe_date(int(token[4:8]), int(token[2:4]), int(token[0:2]))
                fmt = "DDMMYYYY"
                base = 0.82
            if parsed and self._has_safe_compact_context(text, match):
                out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _extract_spaced_d_m_y(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(?<!\d)(\d{1,2})\s+(\d{1,2})\s+(\d{2}|\d{4})(?!\d)", text):
            d, m = int(match.group(1)), int(match.group(2))
            y_raw = match.group(3)
            y = _normalize_year(int(y_raw)) if len(y_raw) == 2 else int(y_raw)
            parsed = _safe_date(y, m, d)
            context = text[max(0, match.start() - 12) : min(len(text), match.end() + 12)]
            if len(y_raw) == 2 and y < 2020 and not (
                has_expiry_keyword(context) or has_weak_expiry_keyword(context)
            ):
                continue
            if parsed and self._has_safe_compact_context(text, match):
                fmt = "DD MM YY" if len(y_raw) == 2 else "DD MM YYYY"
                base = 0.70 if len(y_raw) == 2 else 0.80
                out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _extract_malformed_separator_d_m_y(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        pattern = r"(?<!\d)(\d{1,2})[\/\.-]{2,}(\d{1,2})[\/\.-]+(\d{2}|\d{4})(?!\d)|(?<!\d)(\d{1,2})[\/\.-]+(\d{1,2})[\/\.-]{2,}(\d{2}|\d{4})(?!\d)"
        for match in re.finditer(pattern, text):
            if match.start() >= 2 and text[match.start() - 1] in "/.-" and text[match.start() - 2].isdigit():
                continue
            groups = [group for group in match.groups() if group is not None]
            if len(groups) != 3:
                continue
            d, m = int(groups[0]), int(groups[1])
            y_raw = groups[2]
            y = _normalize_year(int(y_raw)) if len(y_raw) == 2 else int(y_raw)
            parsed = _safe_date(y, m, d)
            context = text[max(0, match.start() - 12) : min(len(text), match.end() + 12)]
            has_letters = any(ch.isalpha() for ch in text)
            if has_letters and not (has_expiry_keyword(context) or has_weak_expiry_keyword(context)):
                continue
            if parsed:
                fmt = "DD/MM/YY_RECOVERED_SEPARATOR" if len(y_raw) == 2 else "DD/MM/YYYY_RECOVERED_SEPARATOR"
                base = 0.70 if len(y_raw) == 2 else 0.82
                out.append(Candidate(parsed, fmt, self._score(text, match.group(0), base), match.group(0), match.span()))
        return out

    def _has_safe_compact_context(self, text: str, match: "re.Match[str]") -> bool:
        if self._looks_like_serial_context(text, match):
            return False
        context = text[max(0, match.start() - 12) : min(len(text), match.end() + 12)]
        if has_expiry_keyword(context) or has_weak_expiry_keyword(context):
            return True
        if has_production_keyword(context) or has_noise_keyword(context):
            return False
        return self._date_char_ratio(text) >= 0.70

    def _looks_like_serial_context(self, text: str, match: "re.Match[str]") -> bool:
        token = match.group(0)
        if len(re.sub(r"\D", "", token)) not in {6, 8}:
            return True
        context = text[max(0, match.start() - 12) : min(len(text), match.end() + 12)]
        if has_noise_keyword(context) and not (has_expiry_keyword(context) or has_weak_expiry_keyword(context)):
            return True
        return False

    def _extract_mm_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(fr"(\d{{1,2}}){SEPARATOR_PATTERN}(\d{{4}})", text):
            m, y = int(match.group(1)), int(match.group(2))
            if m < 1 or m > 12:
                continue
            day = calendar.monthrange(y, m)[1]
            parsed = _safe_date(y, m, day)
            if parsed:
                base = 0.75 if len(match.group(1)) == 2 else 0.65
                out.append(
                    Candidate(
                        parsed,
                        "MM/YYYY",
                        self._score(text, match.group(0), base),
                        match.group(0),
                        match.span(),
                        "month",
                    )
                )
        return out

    def _extract_dd_mon_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        pattern = r"(\d{1,2})\s*([A-Z]{3})\s*(\d{4})"
        for match in re.finditer(pattern, text):
            d = int(match.group(1))
            mon = MONTHS.get(match.group(2))
            y = int(match.group(3))
            if not mon:
                continue
            parsed = _safe_date(y, mon, d)
            if parsed:
                out.append(Candidate(parsed, "DD MON YYYY", self._score(text, match.group(0), 0.9), match.group(0), match.span()))
        return out

    def _extract_mon_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        pattern = r"(?<![A-Z0-9])([A-Z]{3})\s*(\d{4})(?![A-Z0-9])"
        for match in re.finditer(pattern, text):
            mon = MONTHS.get(match.group(1))
            y = int(match.group(2))
            if not mon:
                continue
            parsed = _safe_date(y, mon, calendar.monthrange(y, mon)[1])
            if parsed:
                out.append(
                    Candidate(
                        parsed,
                        "MON YYYY",
                        self._score(text, match.group(0), 0.78),
                        match.group(0),
                        match.span(),
                        "month",
                    )
                )
        return out
