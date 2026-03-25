from __future__ import annotations

import json
from pathlib import Path

from app.ai.ocr import OCRConfig, OCRRouter, detect_runtime_device


def test_detect_runtime_explicit_override() -> None:
    assert detect_runtime_device("cpu") == "cpu"
    assert detect_runtime_device("mps") == "mps"
    assert detect_runtime_device("cuda") == "cuda"


def test_router_selects_ppocrv5_main_on_all_devices(tmp_path: Path) -> None:
    model_dir = tmp_path / "main-model"
    model_dir.mkdir()
    char_dict = tmp_path / "chars.txt"
    char_dict.write_text("A\nB\nC\n")

    for mode in ("cpu", "mps", "cuda"):
        router = OCRRouter(
            OCRConfig(
                ppocrv5_main_model_dir=model_dir,
                ppocrv5_main_char_dict_path=char_dict,
                paddle_lang="en",
                device_mode=mode,
                ppocrv5_use_angle_cls=True,
                ppocrv5_det_db_thresh=0.3,
                substitute_config_path=None,
                enable_substitute_model=False,
            )
        )
        assert router.active_engine_name == "ppocrv5_main"


def test_substitute_config_is_discoverable_but_disabled_by_default(tmp_path: Path) -> None:
    model_dir = tmp_path / "main-model"
    model_dir.mkdir()

    substitute_cfg_path = tmp_path / "substitute.json"
    substitute_cfg_path.write_text(
        json.dumps(
            {
                "engine_name": "ppocrv5_substitute",
                "model_dir": str(tmp_path / "sub-model"),
                "char_dict_path": str(tmp_path / "sub_chars.txt"),
                "use_angle_cls": True,
                "det_db_thresh": 0.3,
            }
        )
    )

    router = OCRRouter(
        OCRConfig(
            ppocrv5_main_model_dir=model_dir,
            ppocrv5_main_char_dict_path=None,
            paddle_lang="en",
            device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            substitute_config_path=substitute_cfg_path,
            enable_substitute_model=False,
        )
    )

    assert router.substitute_available is True
    assert router.active_engine_name == "ppocrv5_main"


def test_substitute_runner_is_used_when_enabled(tmp_path: Path) -> None:
    main_model_dir = tmp_path / "main-model"
    main_model_dir.mkdir()
    sub_model_dir = tmp_path / "sub-model"
    sub_model_dir.mkdir()
    sub_char_dict = tmp_path / "sub_chars.txt"
    sub_char_dict.write_text("A\nB\nC\n")

    substitute_cfg_path = tmp_path / "substitute.json"
    substitute_cfg_path.write_text(
        json.dumps(
            {
                "engine_name": "ppocrv5_substitute",
                "model_dir": str(sub_model_dir),
                "char_dict_path": str(sub_char_dict),
                "use_angle_cls": True,
                "det_db_thresh": 0.3,
            }
        )
    )

    router = OCRRouter(
        OCRConfig(
            ppocrv5_main_model_dir=main_model_dir,
            ppocrv5_main_char_dict_path=None,
            paddle_lang="en",
            device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            substitute_config_path=substitute_cfg_path,
            enable_substitute_model=True,
        )
    )

    assert router.substitute_available is True
    assert router.active_engine_name == "ppocrv5_substitute"


def test_missing_substitute_config_returns_explicit_reason_when_enabled(tmp_path: Path) -> None:
    main_model_dir = tmp_path / "main-model"
    main_model_dir.mkdir()

    router = OCRRouter(
        OCRConfig(
            ppocrv5_main_model_dir=main_model_dir,
            ppocrv5_main_char_dict_path=None,
            paddle_lang="en",
            device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            substitute_config_path=tmp_path / "missing-substitute.json",
            enable_substitute_model=True,
        )
    )

    import numpy as np

    sample = np.zeros((48, 320), dtype=np.uint8)
    result = router.run(sample)
    assert result.engine_name == "ppocrv5_substitute"
    assert result.reason is not None
    assert "config file not found" in result.reason


def test_missing_ppocrv5_assets_return_explicit_reason(tmp_path: Path) -> None:
    router = OCRRouter(
        OCRConfig(
            ppocrv5_main_model_dir=tmp_path / "missing-model",
            ppocrv5_main_char_dict_path=tmp_path / "missing-chars.txt",
            paddle_lang="en",
            device_mode="cpu",
            ppocrv5_use_angle_cls=True,
            ppocrv5_det_db_thresh=0.3,
            substitute_config_path=None,
            enable_substitute_model=False,
        )
    )

    import numpy as np

    sample = np.zeros((48, 320), dtype=np.uint8)
    result = router.run(sample)
    assert result.engine_name == "ppocrv5_main"
    assert result.reason is not None
    assert "model directory not found" in result.reason
