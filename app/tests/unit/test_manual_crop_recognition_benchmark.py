from __future__ import annotations

from datetime import date
import json

import cv2
import numpy as np

from app.ai.parser import ExpiryDateParser
from app.ai.preprocess import ImageVariant
from app.ai.ocr import RecognitionData
from app.ai.types import ParsedDateData
from app.scripts.manual_crop_recognition_benchmark import (
    EXPECTED_VARIANT_NAMES,
    BenchmarkRecognizer,
    TruthCropItem,
    VariantRunResult,
    _audit_one,
    analyze_compact_digits,
    classify_image_result,
    expected_label,
    generate_benchmark_variants,
    is_generic_text_output,
    load_annotated_truth_items,
    parser_matches_expected,
    resolve_recognizer_config,
    select_image_winner,
)
from app.infra.settings import get_settings


def _variant(
    name: str,
    *,
    raw_text: str,
    parsed: date | None,
    expected: date = date(2027, 12, 12),
    confidence: float = 0.5,
) -> VariantRunResult:
    return VariantRunResult(
        filename="sample.jpg",
        expected_date=expected.isoformat(),
        true_bbox_xyxy=[1, 2, 30, 40],
        variant_name=name,
        crop_path="/tmp/crop.png",
        parseq_raw_output=raw_text,
        parseq_normalized_output=raw_text.upper(),
        parseq_confidence=confidence,
        parseq_reason=None,
        parseq_runtime_ms=5,
        parser_parsed_date=parsed.isoformat() if parsed else None,
        parser_confidence=0.9 if parsed else 0.0,
        parser_reason="selected" if parsed else "no valid date parsed",
        parser_candidates=[parsed.isoformat()] if parsed else [],
        exact_match=parsed == expected,
        malformed_but_potentially_recoverable=False,
        generic_text_output=is_generic_text_output(raw_text),
        compact_digit_analysis=analyze_compact_digits(raw_text, expected),
        parser_date_precision="day" if parsed else None,
        parser_parsed_day=parsed.day if parsed else None,
        parser_parsed_month=parsed.month if parsed else None,
        parser_parsed_year=parsed.year if parsed else None,
    )


def test_load_annotated_truth_items_uses_all_manifest_items(tmp_path) -> None:
    manifest = {
        "version": 1,
        "items": {
            "a.jpg": {
                "filename": "a.jpg",
                "expected_day": 1,
                "expected_month": 2,
                "expected_year": 2027,
                "true_bbox_xyxy": [1, 2, 30, 40],
            },
            "b.jpg": {
                "filename": "b.jpg",
                "expected_day": 3,
                "expected_month": 4,
                "expected_year": 2028,
                "true_bbox_xyxy": [5, 6, 70, 80],
            },
            "missing.jpg": {
                "filename": "missing.jpg",
                "expected_day": 1,
                "expected_month": 1,
                "expected_year": 2029,
                "true_bbox_xyxy": None,
            },
        },
    }
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    items = load_annotated_truth_items(path)

    assert [item.filename for item in items] == ["a.jpg", "b.jpg"]
    assert [item.expected_date for item in items] == [date(2027, 2, 1), date(2028, 4, 3)]


def test_expected_label_and_matching_support_month_precision() -> None:
    assert expected_label(None, 11, 2025) == "11/2025"
    assert expected_label(4, 11, 2025) == "2025-11-04"

    month_precision = ParsedDateData(
        parsed_date=date(2025, 11, 30),
        date_format_detected="MON YYYY",
        confidence=0.9,
        candidates=["2025-11"],
        reason="selected_MON YYYY",
        date_precision="month",
        parsed_day=None,
        parsed_month=11,
        parsed_year=2025,
    )
    day_precision = ParsedDateData(
        parsed_date=date(2025, 11, 4),
        date_format_detected="DD/MM/YYYY",
        confidence=0.9,
        candidates=["2025-11-04"],
        reason="selected_DD/MM/YYYY",
    )

    assert parser_matches_expected(month_precision, None, 11, 2025)
    assert not parser_matches_expected(month_precision, 30, 11, 2025)
    assert parser_matches_expected(day_precision, 4, 11, 2025)
    assert parser_matches_expected(day_precision, None, 11, 2025)


def test_generate_benchmark_variants_includes_original_tight_and_each_recognition_variant_once() -> None:
    crop = np.zeros((20, 60, 3), dtype=np.uint8)

    variants = generate_benchmark_variants(crop)

    assert [variant.name for variant in variants] == list(EXPECTED_VARIANT_NAMES)
    assert len({variant.name for variant in variants}) == len(EXPECTED_VARIANT_NAMES)
    assert variants[0].image.shape == crop.shape


def test_winner_selection_prefers_exact_match_then_parseable_then_malformed_then_confidence() -> None:
    exact = _variant("adaptive_binary", raw_text="12/12/2027", parsed=date(2027, 12, 12), confidence=0.7)
    parseable_wrong = _variant("original_padded", raw_text="12/12/2025", parsed=date(2025, 12, 12), confidence=0.99)
    malformed = _variant("gray_upscaled", raw_text="12052027", parsed=None, confidence=0.8)
    malformed.malformed_but_potentially_recoverable = True
    generic = _variant("clahe_gray", raw_text="COMPANY", parsed=None, confidence=1.0)

    assert select_image_winner([generic, malformed, parseable_wrong, exact]) == exact
    assert select_image_winner([generic, malformed, parseable_wrong]) == parseable_wrong
    assert select_image_winner([generic, malformed]) == malformed
    assert select_image_winner([generic]) == generic


def test_generic_text_detection_catches_brand_like_outputs() -> None:
    assert is_generic_text_output("COMPANY")
    assert is_generic_text_output("COMPAGNER")
    assert is_generic_text_output("PROGRAPHY")
    assert not is_generic_text_output("EXP 12/12/2027")


def test_compact_digit_analysis_flags_plausible_dates_and_rejects_serials() -> None:
    expected = date(2026, 5, 6)

    compact = analyze_compact_digits("060528", expected)
    assert compact["has_compact_digit_candidate"] is True
    assert compact["candidate_lengths"] == [6]
    assert compact["matches_expected"] is False
    assert compact["rejected_as_serial"] is False

    exact = analyze_compact_digits("06052026", expected)
    assert exact["matches_expected"] is True
    assert exact["candidate_dates"] == ["2026-05-06"]

    serial = analyze_compact_digits("2701261251", expected)
    assert serial["has_compact_digit_candidate"] is False
    assert serial["rejected_as_serial"] is True


def test_outcome_classification_distinguishes_parser_preprocessing_and_wrong_date() -> None:
    expected = date(2027, 12, 12)
    original = _variant("original_color_tight", raw_text="COMPANY", parsed=None, expected=expected)
    variant_success = _variant("adaptive_binary", raw_text="12/12/2027", parsed=expected, expected=expected)
    assert classify_image_result([original, variant_success], expected) == "preprocessing_variant_needed"

    strict = _variant("original_color_tight", raw_text="12052027", parsed=None, expected=date(2027, 5, 12))
    strict.malformed_but_potentially_recoverable = True
    assert classify_image_result([strict], date(2027, 5, 12)) == "parser_too_strict"

    wrong = _variant("original_color_tight", raw_text="12/12/2025", parsed=date(2025, 12, 12), expected=expected)
    assert classify_image_result([wrong], expected) == "wrong_date_from_crop"

    generic = _variant("original_color_tight", raw_text="COMPAGNE", parsed=None, expected=expected)
    assert classify_image_result([generic], expected) == "recognizer_unreadable_crop"


def test_resolve_recognizer_config_selects_fine_tuned_ppocrv5_server_rec() -> None:
    settings = get_settings()
    config = resolve_recognizer_config(
        recognizer_name="ppocrv5_server_rec",
        settings=settings,
        ppocr_rec_model_dir=None,
        ppocr_rec_model_name=None,
    )

    assert config["recognizer"] == "ppocrv5_server_rec"
    assert config["model_name"] == "PP-OCRv5_server_rec"
    assert str(config["model_dir"]).endswith("models/fine-tuned-models/best_model_inference")


def test_resolve_recognizer_config_selects_svtrv2() -> None:
    settings = get_settings()
    config = resolve_recognizer_config(
        recognizer_name="svtrv2",
        settings=settings,
        ppocr_rec_model_dir=None,
        ppocr_rec_model_name=None,
    )

    assert config["recognizer"] == "svtrv2"
    assert config["model_name"] == "ch_SVTRv2_rec"
    assert str(config["model_dir"]).endswith("models/svtrv2/smartbite_svtrv2_expdate_rec")


def test_resolve_recognizer_config_selects_svtrv2_onnx() -> None:
    settings = get_settings()
    config = resolve_recognizer_config(
        recognizer_name="svtrv2_onnx",
        settings=settings,
        ppocr_rec_model_dir=None,
        ppocr_rec_model_name=None,
    )

    assert config["recognizer"] == "svtrv2_onnx"
    assert config["model_name"] == "svtrv2_onnx"
    assert str(config["model_dir"]).endswith("models/svtrv2/smartbite_svtrv2_expdate_rec")
    assert str(config["onnx_model_path"]).endswith("models/svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx")


class _OriginalOnlyPreprocessor:
    def recognition_variants(self, crop: np.ndarray, allowed_names=None) -> list[ImageVariant]:
        _ = crop, allowed_names
        return []


class _MarkerRecognizer(BenchmarkRecognizer):
    def __init__(self, *, original_text: str = "COMPANY") -> None:
        self.original_text = original_text
        self.calls = 0

    def recognize(self, image: np.ndarray) -> RecognitionData:
        self.calls += 1
        marker = int(image[0, 0, 0]) if image.size else 0
        if marker == 222:
            return RecognitionData(raw_text="EXP 12/12/2027", normalized_text="EXP 12/12/2027", confidence=0.95, reason=None)
        return RecognitionData(raw_text=self.original_text, normalized_text=self.original_text, confidence=0.95, reason=None)


def test_manual_benchmark_tries_rotate_180_only_after_original_fails_expected_date(tmp_path) -> None:
    test64_dir = tmp_path / "test64"
    test64_dir.mkdir()
    image = np.zeros((20, 80, 3), dtype=np.uint8)
    image[19, 79] = 222
    cv2.imwrite(str(test64_dir / "sample.jpg"), image)
    recognizer = _MarkerRecognizer()

    result = _audit_one(
        item=TruthCropItem(
            filename="sample.jpg",
            expected_date=date(2027, 12, 12),
            expected_day=12,
            expected_month=12,
            expected_year=2027,
            true_bbox_xyxy=[0, 0, 80, 20],
        ),
        test64_dir=test64_dir,
        report_root=tmp_path / "report",
        recognizer=recognizer,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        preprocessor=_OriginalOnlyPreprocessor(),  # type: ignore[arg-type]
        today=date(2026, 1, 1),
    )

    variant_names = [variant["variant_name"] for variant in result["variants"]]
    assert variant_names == ["original/original_color_tight", "rotate_180/original_color_tight"]
    assert result["winner"]["selected_orientation"] == "rotate_180"
    assert result["winner"]["exact_match"] is True
    assert recognizer.calls == 2


def test_manual_benchmark_does_not_try_rotate_180_when_original_matches_expected_date(tmp_path) -> None:
    test64_dir = tmp_path / "test64"
    test64_dir.mkdir()
    image = np.zeros((20, 80, 3), dtype=np.uint8)
    cv2.imwrite(str(test64_dir / "sample.jpg"), image)
    recognizer = _MarkerRecognizer(original_text="EXP 12/12/2027")

    result = _audit_one(
        item=TruthCropItem(
            filename="sample.jpg",
            expected_date=date(2027, 12, 12),
            expected_day=12,
            expected_month=12,
            expected_year=2027,
            true_bbox_xyxy=[0, 0, 80, 20],
        ),
        test64_dir=test64_dir,
        report_root=tmp_path / "report",
        recognizer=recognizer,
        parser=ExpiryDateParser(min_candidate_confidence=0.4),
        preprocessor=_OriginalOnlyPreprocessor(),  # type: ignore[arg-type]
        today=date(2026, 1, 1),
    )

    variant_names = [variant["variant_name"] for variant in result["variants"]]
    assert variant_names == ["original/original_color_tight"]
    assert result["winner"]["selected_orientation"] == "original"
    assert recognizer.calls == 1
