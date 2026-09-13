from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
import unicodedata


EXPIRY_KEYWORDS = (
    "EXP",
    "EXPIRE",
    "EXPIRY",
    "BBE",
    "BEST BEFORE",
    "USE BY",
    "TETT",
    "TET",
    "SKT",
    "STT",
    "SON TUKETIM",
    "SON KULLANMA",
)
PRODUCTION_KEYWORDS = (
    "URT",
    "URETIM",
    "URET",
    "UFT",
    "MFG",
    "MFD",
    "PROD",
    "PRODUCTION",
    "MANUFACTURE",
    "MANUFACTURING",
    "PKD",
    "PACKED",
)
NOISE_KEYWORDS = ("LOT", "BATCH", "CODE")
WEAK_EXPIRY_KEYWORDS = (
    "YEYT",
    "YETT",
    "FEYT",
    "FEYY",
    "FELT",
    "1ET",
    "1EY",
    "2ETT",
)


@dataclass(frozen=True, slots=True)
class DateRoleEvidence:
    role: str
    priority: int
    non_production: int
    expiry_keyword: bool
    production_keyword: bool
    noise_keyword: bool
    context: str

    @property
    def reason(self) -> str:
        return f"role={self.role}"


def normalize_evidence_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", ascii_text.upper()).split())


def compact_evidence_text(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_evidence_text(text))


def has_expiry_keyword(text: str | None) -> bool:
    normalized = normalize_evidence_text(text)
    compact = compact_evidence_text(text)
    return any(keyword in normalized for keyword in EXPIRY_KEYWORDS) or any(
        token in compact
        for token in (
            "EXP",
            "EXPIRE",
            "EXPIRY",
            "BBE",
            "BESTBEFORE",
            "USEBY",
            "TETT",
            "TET",
            "SKT",
            "STT",
            "SONTUKETIM",
            "SONKULLANMA",
        )
    )


def has_weak_expiry_keyword(text: str | None) -> bool:
    compact = compact_evidence_text(text)
    return any(token in compact for token in WEAK_EXPIRY_KEYWORDS)


def has_production_keyword(text: str | None) -> bool:
    normalized = normalize_evidence_text(text)
    compact = compact_evidence_text(text)
    return (
        bool(re.search(r"\bU\s*T\b", normalized))
        or any(keyword in normalized for keyword in PRODUCTION_KEYWORDS)
        or any(
            token in compact
            for token in (
                "URT",
                "URETIM",
                "URET",
                "UFT",
                "MFG",
                "MFD",
                "PROD",
                "PRODUCTION",
                "MANUFACTURE",
                "MANUFACTURING",
                "PKD",
                "PACKED",
            )
        )
    )


def has_noise_keyword(text: str | None) -> bool:
    normalized = normalize_evidence_text(text)
    return any(keyword in normalized for keyword in NOISE_KEYWORDS)


def score_date_role(text: str, *, span: tuple[int, int] | None = None) -> DateRoleEvidence:
    context = _role_context(text, span=span)
    expiry_keyword = has_expiry_keyword(context)
    weak_expiry_keyword = has_weak_expiry_keyword(context)
    production_keyword = has_production_keyword(context)
    noise_keyword = has_noise_keyword(context)

    if expiry_keyword:
        role = "expiry_keyword"
        priority = 3
        non_production = 1
    elif weak_expiry_keyword:
        role = "weak_expiry_keyword"
        priority = 3
        non_production = 1
    elif production_keyword:
        role = "production_keyword"
        priority = 1
        non_production = 0
    elif noise_keyword:
        role = "noise_keyword"
        priority = 1
        non_production = 1
    else:
        role = "latest_non_production"
        priority = 2
        non_production = 1

    return DateRoleEvidence(
        role=role,
        priority=priority,
        non_production=non_production,
        expiry_keyword=expiry_keyword,
        production_keyword=production_keyword,
        noise_keyword=noise_keyword,
        context=normalize_evidence_text(context),
    )


def role_selection_key(
    *,
    evidence: DateRoleEvidence,
    parsed_date: date | None,
    specificity: int = 1,
    confidence: float = 0.0,
) -> tuple[int, int, int, int, float]:
    ordinal = parsed_date.toordinal() if parsed_date is not None else 0
    return (
        evidence.priority,
        evidence.non_production,
        specificity,
        ordinal,
        confidence,
    )


def _role_context(text: str, *, span: tuple[int, int] | None) -> str:
    if span is None:
        return text

    start, end = span
    before = text[max(0, start - 28) : start]
    source = text[start:end]
    return f"{before} {source}"
