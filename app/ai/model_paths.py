from __future__ import annotations

from pathlib import Path


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


def assert_required_model_assets(
    *,
    parseq_model_dir: Path,
    check_parseq: bool,
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
