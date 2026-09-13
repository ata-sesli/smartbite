from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from app.infra.settings import Settings, get_settings
from app.scripts.mobile_test64_upload_benchmark import (
    IMAGE_SUFFIXES,
    _detected_matches_expected,
    _expected_label,
    _json_default,
    _load_expected_labels,
)
from app.workers.scan_jobs import build_pipeline


PROFILE_TIMING_KEYS = (
    "decode_ms",
    "yolo_ms",
    "rapidocr_primary_ms",
    "candidate_build_ms",
    "probe_svtr_ms",
    "context_probe_ms",
    "final_svtr_primary_ms",
    "hardcase_svtr_ms",
    "dot_matrix_ms",
    "pp_mobile_full_image_ms",
    "date_tail_crop_ms",
    "partial_expansion_ms",
    "production_sibling_ms",
    "anchor_local_sibling_ms",
    "detector_variant_rescue_ms",
    "rapidocr_rescue_ms",
    "total_ms",
)

PROFILE_COUNT_KEYS = (
    "num_yolo_boxes",
    "num_pp_boxes",
    "num_probe_candidates",
    "num_final_candidates",
    "num_primary_svtr_calls",
    "num_hardcase_svtr_calls",
    "num_dot_matrix_crops",
)


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _run_prefix(images_dir: Path) -> str:
    return "test_images_direct" if images_dir.name == "test-images" else "test64_mobile_direct"


def _blind_expected_label() -> dict[str, Any]:
    return {
        "label": None,
        "precision": None,
        "day": None,
        "month": None,
        "year": None,
    }


def _filenames_from_non_exact_report(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"report does not contain rows: {path}")
    filenames: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        if filename and row.get("exact_match") is not True:
            filenames.add(filename)
    return filenames


def _baseline_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("filename")): row
        for row in rows
        if isinstance(row, dict) and row.get("filename")
    }


def _expected_labels_from_report(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        filename = str(row.get("filename") or "").strip()
        if not filename:
            continue
        labels[filename] = {
            "filename": filename,
            "expected_day": row.get("expected_day"),
            "expected_month": row.get("expected_month"),
            "expected_year": row.get("expected_year"),
        }
    return labels


def _latency_summary(latencies_ms: list[float]) -> dict[str, float | None]:
    if not latencies_ms:
        return {"avg_ms": None, "p50_ms": None, "p90_ms": None, "min_ms": None, "max_ms": None}
    ordered = sorted(latencies_ms)
    p90_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.90)))
    return {
        "avg_ms": round(mean(ordered), 3),
        "p50_ms": round(median(ordered), 3),
        "p90_ms": round(ordered[p90_index], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _image_paths(images_dir: Path, *, only_filenames: set[str] | None) -> list[Path]:
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if only_filenames is None:
        return paths
    return [path for path in paths if path.name in only_filenames]


async def _labels_for_benchmark(args: argparse.Namespace, report_path: Path | None) -> dict[str, dict[str, Any]]:
    if args.blind:
        print("Blind run enabled; expected-label lookup is skipped.", flush=True)
        return {}
    if args.expected_from_report:
        return _expected_labels_from_report(Path(args.expected_from_report))
    try:
        return await _load_expected_labels()
    except Exception as exc:
        if report_path is None:
            raise
        print(f"Expected-label DB unavailable, using labels from prior report: {exc}", flush=True)
        return _expected_labels_from_report(report_path)


def _profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiled_rows = [
        row
        for row in rows
        if isinstance(row.get("debug_profile"), dict)
    ]
    timing_sums = {
        key: round(
            sum(float(row["debug_profile"].get(key) or 0.0) for row in profiled_rows),
            3,
        )
        for key in PROFILE_TIMING_KEYS
    }
    total_ms = timing_sums.get("total_ms") or sum(float(row["latency_ms"]) for row in rows)
    timing_percentages = {
        key: round((value / total_ms) * 100.0, 2) if total_ms else 0.0
        for key, value in timing_sums.items()
        if key != "total_ms"
    }
    count_sums = {
        key: int(sum(int(row["debug_profile"].get(key) or 0) for row in profiled_rows))
        for key in PROFILE_COUNT_KEYS
    }
    return {
        "total_rows": len(rows),
        "profiled_rows": len(profiled_rows),
        "latency": _latency_summary([float(row["latency_ms"]) for row in rows]),
        "slow_over_60000_ms": len([row for row in rows if float(row["latency_ms"]) > 60000.0]),
        "timing_sums_ms": timing_sums,
        "timing_percentages_of_profile_total": timing_percentages,
        "count_sums": count_sums,
        "rows": [
            {
                "filename": row.get("filename"),
                "latency_ms": row.get("latency_ms"),
                "response_status": row.get("response_status"),
                "detected_expiry_date": row.get("detected_expiry_date"),
                "debug_profile": row.get("debug_profile"),
            }
            for row in rows
        ],
    }


def _settings_for_benchmark(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    updates: dict[str, Any] = {}
    if getattr(args, "enable_product_cropper_rescue", False):
        updates["mobile_product_cropper_rescue_enabled"] = True
        updates["mobile_strict_evidence_acceptance_enabled"] = True
    if getattr(args, "enable_legacy_wide_group_rescue", False):
        updates["mobile_legacy_wide_group_rescue_enabled"] = True
        updates["mobile_strict_evidence_acceptance_enabled"] = True
    if getattr(args, "enable_strict_evidence_acceptance", False):
        updates["mobile_strict_evidence_acceptance_enabled"] = True
    if getattr(args, "enable_rapidocr_role_constraint", False):
        updates["mobile_rapidocr_role_constraint_enabled"] = True
    if getattr(args, "enable_production_anchor_sibling", False):
        updates["mobile_production_anchor_sibling_enabled"] = True
    if getattr(args, "enable_paired_crop_evidence", False):
        updates["mobile_paired_crop_evidence_enabled"] = True
    if getattr(args, "enable_local_group_wide_crop", False):
        updates["mobile_local_group_wide_crop_enabled"] = True
    if getattr(args, "enable_rotation_wide_crop", False):
        updates["mobile_rotation_wide_crop_enabled"] = True
    if getattr(args, "enable_hardcase_recognizer_rescue", False):
        updates["mobile_hardcase_recognizer_rescue_enabled"] = True
    if getattr(args, "hardcase_rec_model_dir", None):
        updates["mobile_hardcase_svtrv2_rec_model_dir"] = Path(args.hardcase_rec_model_dir)
    if getattr(args, "rotation_rescue_max_candidates", None) is not None:
        updates["mobile_full_image_rotation_detector_rescue_max_candidates"] = args.rotation_rescue_max_candidates
    if getattr(args, "rapidocr_primary_mode", None):
        updates["mobile_rapidocr_primary_mode"] = args.rapidocr_primary_mode
    return settings.model_copy(update=updates) if updates else settings


async def main_async(args: argparse.Namespace) -> int:
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"test64 images directory not found: {images_dir}")

    only_filenames = _filenames_from_non_exact_report(Path(args.only_from_report)) if args.only_from_report else None
    image_paths = _image_paths(images_dir, only_filenames=only_filenames)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("no benchmark images selected")

    report_path = Path(args.expected_from_report or args.only_from_report) if (args.expected_from_report or args.only_from_report) else None
    labels = await _labels_for_benchmark(args, report_path)
    baseline = _baseline_rows(Path(args.only_from_report) if args.only_from_report else None)
    pipeline = build_pipeline(_settings_for_benchmark(args))
    report_root = Path(args.output_dir).resolve() / f"{_run_prefix(images_dir)}_{_utc_stamp()}"
    report_root.mkdir(parents=True, exist_ok=True)
    debug_root = report_root / "per_image_debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths, start=1):
        started = perf_counter()
        try:
            output = pipeline.run(path.read_bytes(), today=date.today())
            error = None
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            output = None
            error = str(exc)
        latency_ms = (perf_counter() - started) * 1000.0

        expected = _blind_expected_label() if args.blind else _expected_label(labels.get(path.name))
        detected = output.detected_expiry_date.isoformat() if output and output.detected_expiry_date else None
        exact_match = None if args.blind else _detected_matches_expected(detected, expected)
        baseline_row = baseline.get(path.name, {})
        debug_candidates_path = None
        debug_bytes = getattr(output, "debug_candidates_json_bytes", None) if output else None
        if debug_bytes:
            debug_candidates_path = debug_root / f"{path.stem}.debug.json"
            debug_candidates_path.write_bytes(debug_bytes)
        row = {
            "filename": path.name,
            "latency_ms": round(latency_ms, 3),
            "expected_date": expected["label"],
            "expected_precision": expected["precision"],
            "expected_day": expected["day"],
            "expected_month": expected["month"],
            "expected_year": expected["year"],
            "detected_expiry_date": detected,
            "exact_match": exact_match,
            "response_status": output.status if output else "failed",
            "raw_text": output.raw_text if output else None,
            "normalized_text": output.normalized_text if output else None,
            "recognition_confidence": output.recognition_confidence if output else None,
            "detector_confidence": output.detector_confidence if output else None,
            "final_recognition_bbox_xyxy": output.final_recognition_bbox_xyxy if output else None,
            "final_recognition_polygon_json": output.final_recognition_polygon_json if output else None,
            "final_crop_policy": output.final_crop_policy if output else None,
            "final_crop_padding_px": output.final_crop_padding_px if output else None,
            "reason": output.reason if output else error,
            "date_evidence": output.date_evidence_json if output else None,
            "debug_profile": output.debug_profile if output else None,
            "debug_candidates_json_path": str(debug_candidates_path) if debug_candidates_path else None,
            "baseline_detected_expiry_date": baseline_row.get("detected_expiry_date"),
            "baseline_exact_match": baseline_row.get("exact_match"),
            "baseline_response_status": baseline_row.get("response_status"),
        }
        rows.append(row)
        print(
            f"[{index}/{len(image_paths)}] {path.name} status={row['response_status']} "
            f"detected={detected} expected={expected['label']} exact={exact_match} latency_ms={latency_ms:.1f}",
            flush=True,
        )

    parsed = [row for row in rows if row["response_status"] == "parsed_success"]
    manual = [row for row in rows if row["response_status"] == "manual_review_required"]
    exact = [row for row in rows if row["exact_match"]]
    wrong_dates = [
        row
        for row in rows
        if row["expected_date"]
        and row["detected_expiry_date"]
        and not row["exact_match"]
    ]
    fixed_from_baseline = [
        row
        for row in rows
        if row.get("baseline_exact_match") is False and row["exact_match"] is True
    ]
    summary = {
        "total": len(rows),
        "parsed_success": len(parsed),
        "manual_review_required": len(manual),
        "exact_matches": None if args.blind else len(exact),
        "accuracy": None if args.blind else (round(len(exact) / len(rows), 4) if rows else 0.0),
        "wrong_parsed_dates": len(wrong_dates),
        "fixed_from_baseline": len(fixed_from_baseline),
        "latency": _latency_summary([float(row["latency_ms"]) for row in rows]),
    }
    payload = {
        "mode": "direct_mobile_pipeline",
        "blind": bool(args.blind),
        "images_dir": str(images_dir),
        "only_from_report": str(args.only_from_report) if args.only_from_report else None,
        "created_at": datetime.utcnow(),
        "summary": summary,
        "rows": rows,
        "wrong_dates": wrong_dates,
        "manual_review": manual,
        "fixed_from_baseline": fixed_from_baseline,
    }

    report_json = report_root / "mobile_test64_direct_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    profile_json = report_root / "mobile_test64_profile_results.json"
    profile_json.write_text(json.dumps(_profile_summary(rows), indent=2, default=_json_default), encoding="utf-8")
    summary_md = report_root / "mobile_test64_direct_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Mobile Test64 Direct Benchmark",
                "",
                f"- mode: `direct_mobile_pipeline`",
                f"- total: `{summary['total']}`",
                f"- parsed_success: `{summary['parsed_success']}`",
                f"- manual_review_required: `{summary['manual_review_required']}`",
                f"- exact_matches: `{summary['exact_matches']}`",
                f"- accuracy: `{summary['accuracy']}`",
                f"- wrong_parsed_dates: `{summary['wrong_parsed_dates']}`",
                f"- fixed_from_baseline: `{summary['fixed_from_baseline']}`",
                f"- latency: `{summary['latency']}`",
            ]
        ),
        encoding="utf-8",
    )
    print("Report root:", report_root)
    print(report_json)
    print(profile_json)
    print(summary_md)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local test64 images directly through MobileExpiryPipeline")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--output-dir", default="artifacts/onnx_parity")
    parser.add_argument("--only-from-report", default=None)
    parser.add_argument(
        "--expected-from-report",
        default=None,
        help="Use expected labels from a prior report without filtering the image set.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--blind", action="store_true", help="Run without expected labels; exact-match metrics are omitted.")
    parser.add_argument("--enable-product-cropper-rescue", action="store_true")
    parser.add_argument("--enable-legacy-wide-group-rescue", action="store_true")
    parser.add_argument("--enable-strict-evidence-acceptance", action="store_true")
    parser.add_argument("--enable-rapidocr-role-constraint", action="store_true")
    parser.add_argument("--enable-production-anchor-sibling", action="store_true")
    parser.add_argument("--enable-paired-crop-evidence", action="store_true")
    parser.add_argument("--enable-local-group-wide-crop", action="store_true")
    parser.add_argument("--enable-rotation-wide-crop", action="store_true")
    parser.add_argument("--enable-hardcase-recognizer-rescue", action="store_true")
    parser.add_argument("--hardcase-rec-model-dir", default=None)
    parser.add_argument("--rotation-rescue-max-candidates", type=int, default=None)
    parser.add_argument("--rapidocr-primary-mode", choices=("gated", "always", "off"), default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
