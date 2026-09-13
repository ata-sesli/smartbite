from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_any_file(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).exists() for name in names)


def validate_parseq_assets(model_dir: Path) -> list[str]:
    errors: list[str] = []
    if not model_dir.exists() or not model_dir.is_dir():
        return [f"missing PARSeq model directory: {model_dir}"]

    if not _has_any_file(model_dir, ("pytorch_model.bin", "model.safetensors")):
        errors.append(
            "PARSeq model directory is incomplete: expected one of "
            "`pytorch_model.bin` or `model.safetensors`"
        )
    return errors


def validate_ppocr_inference_assets(model_dir: Path, *, label: str) -> list[str]:
    errors: list[str] = []
    if not model_dir.exists() or not model_dir.is_dir():
        return [f"missing {label} model directory: {model_dir}"]

    if not (model_dir / "inference.yml").exists():
        errors.append(f"{label} model directory is incomplete: missing `inference.yml`")
    if not _has_any_file(model_dir, ("inference.pdmodel", "inference.json")):
        errors.append(f"{label} model directory is incomplete: missing inference model file")
    if not _has_any_file(model_dir, ("inference.pdiparams", "inference.pdiparams.info")):
        errors.append(f"{label} model directory is incomplete: missing inference parameter file")
    return errors


def validate_yolo_detector_assets(
    detector_path: Path,
    onnx_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    pt_exists = detector_path.exists() and detector_path.is_file()
    onnx_exists = onnx_path is not None and onnx_path.exists() and onnx_path.is_file()
    if not pt_exists and not onnx_exists:
        errors.append(f"missing YOLO detector weights at {detector_path} (or {onnx_path})")
    return errors


def are_active_models_present(
    *,
    detector_path: Path,
    detector_onnx_path: Path | None = None,
    svtr_model_dir: Path | None = None,
    svtr_onnx_path: Path | None = None,
) -> bool:
    detector_ok = (detector_path.exists() and detector_path.is_file()) or (
        detector_onnx_path is not None and detector_onnx_path.exists() and detector_onnx_path.is_file()
    )
    svtr_ok = False
    if svtr_model_dir is not None and svtr_model_dir.exists() and svtr_model_dir.is_dir():
        if len(validate_ppocr_inference_assets(svtr_model_dir, label="SVTRv2")) == 0:
            svtr_ok = True
    if not svtr_ok and svtr_onnx_path is not None and svtr_onnx_path.exists() and svtr_onnx_path.is_file():
        svtr_ok = True
    return detector_ok and svtr_ok


def ensure_smartbite_models(
    *,
    repo_id: str = "atasesli/smartbite-models",
    revision: str = "main",
    models_root: Path | None = None,
    detector_path: Path | None = None,
    detector_onnx_path: Path | None = None,
    svtr_model_dir: Path | None = None,
    svtr_onnx_path: Path | None = None,
    force: bool = False,
) -> bool:
    target_root = models_root or Path("models")
    det_path = detector_path or (target_root / "yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt")
    det_onnx = detector_onnx_path or (target_root / "yolo26s_obb_expdate2k_ft_after_brazil/weights/best.onnx")
    svtr_dir = svtr_model_dir or (target_root / "svtrv2/smartbite_svtrv2_expdate_rec")
    svtr_onnx = svtr_onnx_path or (target_root / "svtrv2/smartbite_svtrv2_expdate_rec_onnx/model.onnx")

    if not force and are_active_models_present(
        detector_path=det_path,
        detector_onnx_path=det_onnx,
        svtr_model_dir=svtr_dir,
        svtr_onnx_path=svtr_onnx,
    ):
        return False

    logger.info("downloading SmartBite models from Hugging Face (%s)...", repo_id)
    try:
        from huggingface_hub import snapshot_download

        target_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target_root),
            ignore_patterns=(".git*", "README.md"),
        )
        logger.info("SmartBite model download complete.")
        return True
    except Exception as exc:
        logger.warning("failed to download models from Hugging Face: %s", exc)
        return False


def assert_required_model_assets(
    *,
    parseq_model_dir: Path,
    check_parseq: bool,
    detector_path: Path | None = None,
    detector_onnx_path: Path | None = None,
    check_detector: bool = False,
    svtrv2_rec_model_dir: Path | None = None,
    context_svtrv2_rec_model_dir: Path | None = None,
    check_svtrv2_rec: bool = False,
    check_context_svtrv2_rec: bool = False,
    probe_mobile_rec_model_dir: Path | None = None,
    probe_mobile_det_model_dir: Path | None = None,
    check_probe_mobile_rec: bool = False,
    check_probe_mobile_det: bool = False,
    strict: bool,
) -> list[str]:
    errors: list[str] = []
    if check_detector:
        if detector_path is None:
            errors.append("missing detector model path setting")
        else:
            errors.extend(validate_yolo_detector_assets(detector_path, onnx_path=detector_onnx_path))
    if check_parseq:
        errors.extend(validate_parseq_assets(parseq_model_dir))
    if check_svtrv2_rec:
        if svtrv2_rec_model_dir is None:
            errors.append("missing SVTRv2 recognition model directory setting")
        else:
            errors.extend(validate_ppocr_inference_assets(svtrv2_rec_model_dir, label="SVTRv2 recognition"))
    if check_context_svtrv2_rec:
        if context_svtrv2_rec_model_dir is None:
            errors.append("missing context SVTRv2 recognition model directory setting")
        else:
            errors.extend(validate_ppocr_inference_assets(context_svtrv2_rec_model_dir, label="context SVTRv2 recognition"))
    if check_probe_mobile_rec:
        if probe_mobile_rec_model_dir is None:
            errors.append("missing PP-OCRv5 mobile recognition model directory setting")
        else:
            errors.extend(
                validate_ppocr_inference_assets(
                    probe_mobile_rec_model_dir,
                    label="PP-OCRv5 mobile recognition",
                )
            )
    if check_probe_mobile_det:
        if probe_mobile_det_model_dir is None:
            errors.append("missing PP-OCRv5 mobile detection model directory setting")
        else:
            errors.extend(
                validate_ppocr_inference_assets(
                    probe_mobile_det_model_dir,
                    label="PP-OCRv5 mobile detection",
                )
            )

    if errors and strict:
        joined = "; ".join(errors)
        raise RuntimeError(
            f"model asset validation failed: {joined}. "
            "Run `smartbite-prefetch-models` to download pinned model revisions."
        )
    return errors
