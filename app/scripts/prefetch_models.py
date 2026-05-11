from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ai.model_paths import assert_required_model_assets
from app.infra.settings import get_settings


def _download_model(*, repo_id: str, revision: str, local_dir: Path) -> str:
    from huggingface_hub import model_info, snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        ignore_patterns=(".git*",),
    )
    info = model_info(repo_id=repo_id, revision=revision)
    return info.sha


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch SmartBite OCR models into local models/ directory")
    parser.add_argument("--skip-parseq", action="store_true", help="Skip PARSeq download")
    parser.add_argument("--skip-mobile-rec", action="store_true", help="Skip PP-OCRv5 mobile recognition download")
    parser.add_argument("--include-mobile-det", action="store_true", help="Download unused legacy PP-OCRv5 mobile detection asset")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-download model folder validation")
    args = parser.parse_args()

    settings = get_settings()
    result: dict[str, dict[str, str]] = {}

    if not args.skip_parseq:
        sha = _download_model(
            repo_id=settings.parseq_hf_repo_id,
            revision=settings.parseq_hf_revision,
            local_dir=settings.parseq_model_dir,
        )
        result["parseq"] = {
            "repo_id": settings.parseq_hf_repo_id,
            "requested_revision": settings.parseq_hf_revision,
            "resolved_sha": sha,
            "local_dir": str(settings.parseq_model_dir),
        }

    if not args.skip_mobile_rec:
        sha = _download_model(
            repo_id=settings.expiry_probe_mobile_rec_hf_repo_id,
            revision=settings.expiry_probe_mobile_rec_hf_revision,
            local_dir=settings.expiry_probe_mobile_rec_model_dir,
        )
        result["ppocrv5_mobile_rec"] = {
            "repo_id": settings.expiry_probe_mobile_rec_hf_repo_id,
            "requested_revision": settings.expiry_probe_mobile_rec_hf_revision,
            "resolved_sha": sha,
            "local_dir": str(settings.expiry_probe_mobile_rec_model_dir),
        }

    if args.include_mobile_det:
        sha = _download_model(
            repo_id=settings.expiry_probe_mobile_det_hf_repo_id,
            revision=settings.expiry_probe_mobile_det_hf_revision,
            local_dir=settings.expiry_probe_mobile_det_model_dir,
        )
        result["ppocrv5_mobile_det"] = {
            "repo_id": settings.expiry_probe_mobile_det_hf_repo_id,
            "requested_revision": settings.expiry_probe_mobile_det_hf_revision,
            "resolved_sha": sha,
            "local_dir": str(settings.expiry_probe_mobile_det_model_dir),
        }

    if not args.no_validate:
        assert_required_model_assets(
            parseq_model_dir=settings.parseq_model_dir,
            check_parseq=not args.skip_parseq,
            probe_mobile_rec_model_dir=settings.expiry_probe_mobile_rec_model_dir,
            probe_mobile_det_model_dir=settings.expiry_probe_mobile_det_model_dir,
            check_probe_mobile_rec=not args.skip_mobile_rec,
            check_probe_mobile_det=args.include_mobile_det,
            strict=True,
        )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
