from __future__ import annotations

from datetime import date

from app.ai.types import DecisionResult
from app.domain.enums import ExpiryClassification, FinalResultStatus


class ExpiryDecisionEngine:
    def __init__(self, alert_threshold_days: int = 3) -> None:
        self.alert_threshold_days = alert_threshold_days

    def decide(self, parsed_date: date | None, *, today: date, parse_confidence: float) -> DecisionResult:
        if parsed_date is None:
            return DecisionResult(
                final_status=FinalResultStatus.MANUAL_REVIEW_REQUIRED,
                expiry_classification=ExpiryClassification.MANUAL_REVIEW_REQUIRED,
                days_remaining=None,
                alert_required=True,
                needs_review=True,
                reason="date_parse_failed",
            )

        days_remaining = (parsed_date - today).days

        if days_remaining < 0:
            return DecisionResult(
                final_status=FinalResultStatus.PARSED_SUCCESS,
                expiry_classification=ExpiryClassification.EXPIRED,
                days_remaining=days_remaining,
                alert_required=True,
                needs_review=False,
                reason="parsed_date_in_past",
            )

        if days_remaining <= self.alert_threshold_days:
            final_status = (
                FinalResultStatus.PARSED_WITH_LOW_CONFIDENCE if parse_confidence < 0.65 else FinalResultStatus.PARSED_SUCCESS
            )
            return DecisionResult(
                final_status=final_status,
                expiry_classification=ExpiryClassification.EXPIRING_SOON,
                days_remaining=days_remaining,
                alert_required=True,
                needs_review=parse_confidence < 0.5,
                reason="within_alert_threshold",
            )

        final_status = FinalResultStatus.PARSED_WITH_LOW_CONFIDENCE if parse_confidence < 0.65 else FinalResultStatus.PARSED_SUCCESS
        return DecisionResult(
            final_status=final_status,
            expiry_classification=ExpiryClassification.SAFE,
            days_remaining=days_remaining,
            alert_required=False,
            needs_review=parse_confidence < 0.5,
            reason="beyond_alert_threshold",
        )
