from __future__ import annotations

from datetime import date, timedelta

from app.ai.decision import ExpiryDecisionEngine
from app.domain.enums import ExpiryClassification


def test_decision_expired() -> None:
    engine = ExpiryDecisionEngine(alert_threshold_days=3)
    today = date(2026, 3, 18)
    result = engine.decide(today - timedelta(days=1), today=today, parse_confidence=0.9)

    assert result.expiry_classification == ExpiryClassification.EXPIRED
    assert result.alert_required is True


def test_decision_expiring_soon() -> None:
    engine = ExpiryDecisionEngine(alert_threshold_days=3)
    today = date(2026, 3, 18)
    result = engine.decide(today + timedelta(days=2), today=today, parse_confidence=0.9)

    assert result.expiry_classification == ExpiryClassification.EXPIRING_SOON
    assert result.days_remaining == 2


def test_decision_safe() -> None:
    engine = ExpiryDecisionEngine(alert_threshold_days=3)
    today = date(2026, 3, 18)
    result = engine.decide(today + timedelta(days=10), today=today, parse_confidence=0.9)

    assert result.expiry_classification == ExpiryClassification.SAFE
    assert result.alert_required is False
