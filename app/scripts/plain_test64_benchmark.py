from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

from app.infra.settings import get_settings
from app.infra.settings import Settings
from app.infra.storage import LocalStorage
from app.scripts.forensic_test64_topk import (
    _evaluate_run_outputs,
    _json_default,
    _load_expected_labels,
    _load_latest_batch,
)
from app.workers.scan_jobs import build_pipeline


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_detector_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    updates: dict[str, Any] = {}
    if getattr(args, "fine_tuned_ppocrv5_only", False):
        updates.update(
            {
                "text_detector_mode": "ppocrv5_server",
                "ocr_ppocrv5_use_custom_text_det_model": True,
                "ocr_ppocrv5_text_det_model_dir": Path("models/pp-ocrv5-text-detection/best_model_inference"),
                "craft_enabled": False,
            }
        )
    if getattr(args, "text_detector_mode", None):
        updates["text_detector_mode"] = args.text_detector_mode
    if getattr(args, "use_custom_ppocrv5_text_det_model", False):
        updates["ocr_ppocrv5_use_custom_text_det_model"] = True
    if getattr(args, "ppocrv5_text_det_model_dir", None):
        updates["ocr_ppocrv5_text_det_model_dir"] = Path(args.ppocrv5_text_det_model_dir)
        updates["ocr_ppocrv5_use_custom_text_det_model"] = True
    if getattr(args, "disable_craft", False):
        updates["craft_enabled"] = False
    if getattr(args, "ppocrv5_det_db_thresh", None) is not None:
        updates["ocr_ppocrv5_det_db_thresh"] = args.ppocrv5_det_db_thresh
    if getattr(args, "ppocrv5_det_db_box_thresh", None) is not None:
        updates["ocr_ppocrv5_det_db_box_thresh"] = args.ppocrv5_det_db_box_thresh
    if getattr(args, "ppocrv5_det_limit_side_len", None) is not None:
        updates["ocr_ppocrv5_det_limit_side_len"] = args.ppocrv5_det_limit_side_len
    if getattr(args, "ppocrv5_det_limit_type", None) is not None:
        updates["ocr_ppocrv5_det_limit_type"] = args.ppocrv5_det_limit_type
    if getattr(args, "ppocrv5_det_db_unclip_ratio", None) is not None:
        updates["ocr_ppocrv5_det_db_unclip_ratio"] = args.ppocrv5_det_db_unclip_ratio
    return settings.model_copy(update=updates) if updates else settings


def detector_config_payload(settings: Settings) -> dict[str, Any]:
    configured_det_dir = settings.ocr_ppocrv5_text_det_model_dir
    effective_det_dir = configured_det_dir if settings.ocr_ppocrv5_use_custom_text_det_model else None
    return {
        "text_detector_mode": settings.text_detector_mode,
        "ppocrv5_use_custom_text_det_model": settings.ocr_ppocrv5_use_custom_text_det_model,
        "ppocrv5_text_det_model_name": settings.ocr_ppocrv5_text_det_model_name,
        "ppocrv5_configured_text_det_model_dir": str(configured_det_dir) if configured_det_dir is not None else None,
        "ppocrv5_effective_text_det_model_dir": str(effective_det_dir) if effective_det_dir is not None else None,
        "ppocrv5_det_db_thresh": settings.ocr_ppocrv5_det_db_thresh,
        "ppocrv5_det_db_box_thresh": settings.ocr_ppocrv5_det_db_box_thresh,
        "ppocrv5_det_limit_side_len": settings.ocr_ppocrv5_det_limit_side_len,
        "ppocrv5_det_limit_type": settings.ocr_ppocrv5_det_limit_type,
        "ppocrv5_det_db_unclip_ratio": settings.ocr_ppocrv5_det_db_unclip_ratio,
        "craft_enabled": settings.craft_enabled,
        "craft_model_path": str(settings.craft_model_path),
        "craft_rescue_enabled": settings.craft_rescue_enabled,
    }


def _detector_stats(output: Any) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "variants": 0,
        "ppocrv5_server_boxes": 0,
        "craft_boxes": 0,
        "ensemble_boxes": 0,
        "detector_boxes": [],
        "craft_unavailable_reasons": [],
        "detector_modes": {},
        "performance": {},
    }
    payload_bytes = getattr(output, "debug_candidates_json_bytes", None)
    if not payload_bytes:
        return stats
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return stats

    summary = payload.get("summary") or {}
    if isinstance(summary, dict) and isinstance(summary.get("performance"), dict):
        stats["performance"] = summary["performance"]

    reasons: set[str] = set()
    for variant in payload.get("variants", []):
        if not isinstance(variant, dict):
            continue
        stats["variants"] += 1
        mode = str(variant.get("detector_mode") or "unknown")
        stats["detector_modes"][mode] = int(stats["detector_modes"].get(mode, 0)) + 1
        counts = variant.get("detector_counts") or {}
        if isinstance(counts, dict):
            stats["ppocrv5_server_boxes"] += int(counts.get("ppocrv5_server") or 0)
            stats["craft_boxes"] += int(counts.get("craft") or 0)
            stats["ensemble_boxes"] += int(counts.get("ensemble") or 0)
        for box in variant.get("detected_boxes") or []:
            normalized = _normalize_detector_box(box)
            if normalized is not None:
                stats["detector_boxes"].append(normalized)
        for reason in variant.get("detector_unavailable_reasons") or []:
            reasons.add(str(reason))

    stats["craft_unavailable_reasons"] = sorted(reasons)
    return stats


def _normalize_xyxy(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(coord == coord for coord in coords):
        return None
    return coords


def _normalize_polygon(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, list | tuple) or len(point) != 2:
            return None
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return None
    return points or None


def _normalize_detector_box(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    bbox_xyxy = _normalize_xyxy(value.get("bbox_xyxy"))
    if bbox_xyxy is None:
        return None
    confidence = value.get("confidence")
    try:
        normalized_confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        normalized_confidence = None
    sources = value.get("sources")
    return {
        "bbox_xyxy": bbox_xyxy,
        "polygon_xy": _normalize_polygon(value.get("polygon_xy")),
        "confidence": normalized_confidence,
        "source": str(value.get("source") or "unknown"),
        "sources": [str(source) for source in sources] if isinstance(sources, list) else [],
        "variant_name": str(value.get("variant_name")) if value.get("variant_name") is not None else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    settings = apply_detector_overrides(get_settings(), args)
    storage = LocalStorage(settings.storage_root)
    pipeline = build_pipeline(settings)

    rows = await _load_latest_batch(args.source, args.max_rows, args.batch_gap_seconds)
    if not rows:
        print("No rows found for source:", args.source)
        return 1
    labels = await _load_expected_labels()

    report_root = Path(args.output_dir).resolve() / f"test64_plain_{_utc_stamp()}"
    _ensure_dir(report_root)

    run_outputs: list[dict[str, Any]] = []
    detector_totals = {
        "ppocrv5_server_boxes": 0,
        "craft_boxes": 0,
        "ensemble_boxes": 0,
        "variants": 0,
        "craft_unavailable_reasons": set(),
        "detector_modes": {},
        "performance": {},
    }

    for index, row in enumerate([row for row in rows if row.get("filename")], start=1):
        image_bytes = storage.read_bytes(str(row["image_path"]))
        output = await asyncio.to_thread(pipeline.run, image_bytes, today=date.today())
        stats = _detector_stats(output)
        detector_totals["variants"] += int(stats["variants"])
        detector_totals["ppocrv5_server_boxes"] += int(stats["ppocrv5_server_boxes"])
        detector_totals["craft_boxes"] += int(stats["craft_boxes"])
        detector_totals["ensemble_boxes"] += int(stats["ensemble_boxes"])
        detector_totals["craft_unavailable_reasons"].update(stats["craft_unavailable_reasons"])
        for mode, count in stats["detector_modes"].items():
            detector_totals["detector_modes"][mode] = int(detector_totals["detector_modes"].get(mode, 0)) + int(count)
        if isinstance(stats.get("performance"), dict):
            detector_totals["performance"] = stats["performance"]
        run_outputs.append({"filename": row["filename"], "output": output, "detector_stats": stats})
        print(f"[{index}/{len(rows)}] {row['filename']} -> {output.final_status} {output.parsed_date}")

    evaluation = _evaluate_run_outputs(run_outputs, labels)
    detector_totals["craft_unavailable_reasons"] = sorted(detector_totals["craft_unavailable_reasons"])
    payload = {
        "source": args.source,
        "latest_batch_size": len(rows),
        "text_detector_mode": getattr(pipeline.ocr_router, "text_detector_mode", None),
        "detector_config": detector_config_payload(settings),
        "craft_model_path": str(settings.craft_model_path),
        "detector_totals": detector_totals,
        "evaluation": evaluation,
        "per_scan_detector_stats": [
            {"filename": item["filename"], **item["detector_stats"]} for item in run_outputs
        ],
    }

    report_json = report_root / "plain_test64_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    summary_md = report_root / "plain_test64_summary.md"
    lines = [
        "# Plain Test64 Benchmark",
        "",
        f"- source: `{args.source}`",
        f"- latest_batch_size: `{len(rows)}`",
        f"- text_detector_mode: `{payload['text_detector_mode']}`",
        f"- detector_config: `{payload['detector_config']}`",
        f"- craft_model_path: `{payload['craft_model_path']}`",
        f"- summary: `{evaluation['summary']}`",
        f"- parseable_candidates_total: `{evaluation['parseable_candidates_total']}`",
        f"- average_parseq_candidates_per_scan: `{evaluation['average_parseq_candidates_per_scan']}`",
        f"- average_runtime_ms_per_scan: `{evaluation['average_runtime_ms_per_scan']}`",
        f"- detector_totals: `{detector_totals}`",
        f"- performance: `{detector_totals.get('performance')}`",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("Report root:", report_root)
    print(report_json)
    print(summary_md)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain current-pipeline benchmark for SmartBite test64")
    parser.add_argument("--source", default="test64-bulk-queue")
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--batch-gap-seconds", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--text-detector-mode", choices=("ppocrv5_server", "craft", "ensemble"), default=None)
    parser.add_argument("--ppocrv5-text-det-model-dir", default=None)
    parser.add_argument("--ppocrv5-det-db-thresh", type=float, default=None)
    parser.add_argument("--ppocrv5-det-db-box-thresh", type=float, default=None)
    parser.add_argument("--ppocrv5-det-limit-side-len", type=int, default=None)
    parser.add_argument("--ppocrv5-det-limit-type", choices=("max", "min"), default=None)
    parser.add_argument("--ppocrv5-det-db-unclip-ratio", type=float, default=None)
    parser.add_argument("--use-custom-ppocrv5-text-det-model", action="store_true")
    parser.add_argument("--disable-craft", action="store_true")
    parser.add_argument(
        "--fine-tuned-ppocrv5-only",
        action="store_true",
        help="Force fine-tuned PP-OCRv5 server text detection and disable CRAFT for this benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
