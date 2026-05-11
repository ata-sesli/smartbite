from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from app.infra.settings import Settings
from app.scripts.plain_test64_benchmark import _detector_stats, apply_detector_overrides, detector_config_payload


def test_fine_tuned_ppocrv5_only_override_disables_craft_and_uses_custom_detector() -> None:
    settings = Settings(
        text_detector_mode="ensemble",
        craft_enabled=True,
        ocr_ppocrv5_use_custom_text_det_model=False,
        ocr_ppocrv5_text_det_model_dir=Path("models/default-det"),
    )
    args = Namespace(
        fine_tuned_ppocrv5_only=True,
        text_detector_mode=None,
        ppocrv5_text_det_model_dir=None,
        ppocrv5_det_limit_side_len=None,
        ppocrv5_det_limit_type=None,
        ppocrv5_det_db_thresh=None,
        ppocrv5_det_db_box_thresh=None,
        ppocrv5_det_db_unclip_ratio=None,
        use_custom_ppocrv5_text_det_model=False,
        disable_craft=False,
    )

    overridden = apply_detector_overrides(settings, args)
    payload = detector_config_payload(overridden)

    assert overridden.text_detector_mode == "ppocrv5_server"
    assert overridden.craft_enabled is False
    assert overridden.ocr_ppocrv5_use_custom_text_det_model is True
    assert overridden.ocr_ppocrv5_text_det_model_dir == Path("models/pp-ocrv5-text-detection/best_model_inference")
    assert payload["text_detector_mode"] == "ppocrv5_server"
    assert payload["ppocrv5_effective_text_det_model_dir"] == "models/pp-ocrv5-text-detection/best_model_inference"
    assert payload["craft_enabled"] is False


def test_detector_config_payload_marks_builtin_detector_when_custom_detector_disabled() -> None:
    settings = Settings(
        text_detector_mode="ppocrv5_server",
        craft_enabled=False,
        ocr_ppocrv5_use_custom_text_det_model=False,
        ocr_ppocrv5_text_det_model_dir=Path("models/unused-custom-det"),
    )

    payload = detector_config_payload(settings)

    assert payload["ppocrv5_use_custom_text_det_model"] is False
    assert payload["ppocrv5_effective_text_det_model_dir"] is None
    assert payload["ppocrv5_configured_text_det_model_dir"] == "models/unused-custom-det"


def test_detector_config_payload_includes_ppocrv5_db_recovery_knobs() -> None:
    settings = Settings(
        ocr_ppocrv5_det_db_thresh=0.25,
        ocr_ppocrv5_det_db_box_thresh=0.35,
        ocr_ppocrv5_det_limit_side_len=1216,
        ocr_ppocrv5_det_limit_type="max",
        ocr_ppocrv5_det_db_unclip_ratio=1.8,
    )

    payload = detector_config_payload(settings)

    assert payload["ppocrv5_det_db_thresh"] == 0.25
    assert payload["ppocrv5_det_db_box_thresh"] == 0.35
    assert payload["ppocrv5_det_limit_side_len"] == 1216
    assert payload["ppocrv5_det_limit_type"] == "max"
    assert payload["ppocrv5_det_db_unclip_ratio"] == 1.8


def test_apply_detector_overrides_accepts_ppocrv5_db_cli_knobs() -> None:
    settings = Settings()
    args = Namespace(
        fine_tuned_ppocrv5_only=False,
        text_detector_mode=None,
        ppocrv5_text_det_model_dir=None,
        ppocrv5_det_limit_side_len=1536,
        ppocrv5_det_limit_type="max",
        ppocrv5_det_db_thresh=0.25,
        ppocrv5_det_db_box_thresh=0.35,
        ppocrv5_det_db_unclip_ratio=1.8,
        use_custom_ppocrv5_text_det_model=False,
        disable_craft=False,
    )

    overridden = apply_detector_overrides(settings, args)

    assert overridden.ocr_ppocrv5_det_limit_side_len == 1536
    assert overridden.ocr_ppocrv5_det_limit_type == "max"
    assert overridden.ocr_ppocrv5_det_db_thresh == 0.25
    assert overridden.ocr_ppocrv5_det_db_box_thresh == 0.35
    assert overridden.ocr_ppocrv5_det_db_unclip_ratio == 1.8


def test_detector_stats_extracts_detector_box_geometry() -> None:
    class Output:
        debug_candidates_json_bytes = json.dumps(
            {
                "variants": [
                    {
                        "detector_mode": "ppocrv5_server",
                        "detector_counts": {"ppocrv5_server": 1},
                        "detected_boxes": [
                            {
                                "bbox_xyxy": [10, 20, 110, 50],
                                "polygon_xy": [[10, 20], [110, 20], [110, 50], [10, 50]],
                                "confidence": 0.91,
                                "source": "ppocrv5_server",
                                "sources": ["ppocrv5_server"],
                                "variant_name": "original",
                            }
                        ],
                    }
                ]
            }
        ).encode("utf-8")

    stats = _detector_stats(Output())

    assert stats["detector_boxes"] == [
        {
            "bbox_xyxy": [10.0, 20.0, 110.0, 50.0],
            "polygon_xy": [[10.0, 20.0], [110.0, 20.0], [110.0, 50.0], [10.0, 50.0]],
            "confidence": 0.91,
            "source": "ppocrv5_server",
            "sources": ["ppocrv5_server"],
            "variant_name": "original",
        }
    ]
