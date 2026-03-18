from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

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


@dataclass(slots=True)
class Candidate:
    dt: date
    fmt: str
    confidence: float
    source: str


def _normalize_year(value: int) -> int:
    if value >= 100:
        return value
    return 2000 + value if value < 70 else 1900 + value


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


class ExpiryDateParser:
    def parse(self, normalized_text: str) -> ParsedDateData:
        text = normalized_text.upper()
        candidates: list[Candidate] = []

        candidates.extend(self._extract_dd_mm_yyyy(text))
        candidates.extend(self._extract_dd_mm_yy(text))
        candidates.extend(self._extract_yyyy_mm_dd(text))
        candidates.extend(self._extract_mm_yyyy(text))
        candidates.extend(self._extract_dd_mon_yyyy(text))

        if not candidates:
            return ParsedDateData(None, None, 0.0, [], "no valid date parsed")

        ranked = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        best = ranked[0]

        reason_parts = []
        if any(prefix in text for prefix in PREFIX_HINTS):
            reason_parts.append("prefix_detected")
        if any(token in text for token in LOW_PRIORITY_TOKENS):
            reason_parts.append("contains_low_priority_token")
        reason_parts.append(f"selected_{best.fmt}")

        return ParsedDateData(
            parsed_date=best.dt,
            date_format_detected=best.fmt,
            confidence=round(best.confidence, 4),
            candidates=[c.dt.isoformat() for c in ranked],
            reason=";".join(reason_parts),
        )

    def _score(self, text: str, source: str, base: float = 0.8) -> float:
        score = base
        if any(prefix in text for prefix in PREFIX_HINTS):
            score += 0.15
        if any(token in source for token in LOW_PRIORITY_TOKENS):
            score -= 0.3
        return max(0.0, min(score, 0.99))

    def _extract_dd_mm_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(\d{2})[/-](\d{2})[/-](\d{4})", text):
            d, m, y = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            parsed = _safe_date(y, m, d)
            if parsed:
                out.append(Candidate(parsed, "DD/MM/YYYY", self._score(text, match.group(0), 0.95), match.group(0)))
        return out

    def _extract_dd_mm_yy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(\d{2})[/-](\d{2})[/-](\d{2})(?!\d)", text):
            d, m, y = (int(match.group(1)), int(match.group(2)), _normalize_year(int(match.group(3))))
            parsed = _safe_date(y, m, d)
            if parsed:
                out.append(Candidate(parsed, "DD/MM/YY", self._score(text, match.group(0), 0.88), match.group(0)))
        return out

    def _extract_yyyy_mm_dd(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(\d{4})-(\d{2})-(\d{2})", text):
            y, m, d = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            parsed = _safe_date(y, m, d)
            if parsed:
                out.append(Candidate(parsed, "YYYY-MM-DD", self._score(text, match.group(0), 0.9), match.group(0)))
        return out

    def _extract_mm_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        for match in re.finditer(r"(\d{2})/(\d{4})", text):
            m, y = int(match.group(1)), int(match.group(2))
            if m < 1 or m > 12:
                continue
            day = calendar.monthrange(y, m)[1]
            parsed = _safe_date(y, m, day)
            if parsed:
                out.append(Candidate(parsed, "MM/YYYY", self._score(text, match.group(0), 0.75), match.group(0)))
        return out

    def _extract_dd_mon_yyyy(self, text: str) -> list[Candidate]:
        out: list[Candidate] = []
        pattern = r"(\d{2})\s*([A-Z]{3})\s*(\d{4})"
        for match in re.finditer(pattern, text):
            d = int(match.group(1))
            mon = MONTHS.get(match.group(2))
            y = int(match.group(3))
            if not mon:
                continue
            parsed = _safe_date(y, mon, d)
            if parsed:
                out.append(Candidate(parsed, "DD MON YYYY", self._score(text, match.group(0), 0.9), match.group(0)))
        return out
