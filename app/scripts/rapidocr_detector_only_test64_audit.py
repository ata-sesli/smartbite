from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ai.rapidocr_text_detector import RapidOCRTextProposalDetector
from app.domain.services import CROP_TRUTH_ANNOTATIONS_PATH
from app.scripts.detector_truth_ablation import _best_overlap, _clean_bbox, _ensure_dir, _load_truth_items


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _decode_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _selected_filenames(raw_names: str | None) -> set[str] | None:
    if not raw_names:
        return None
    return {Path(name.strip()).name for name in raw_names.split(",") if name.strip()}


def _bbox_center_inside(box: tuple[int, int, int, int], truth_bbox: tuple[int, int, int, int]) -> bool:
    cx = (truth_bbox[0] + truth_bbox[2]) / 2.0
    cy = (truth_bbox[1] + truth_bbox[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _proposal_to_dict(proposal: Any) -> dict[str, Any]:
    polygon = getattr(proposal, "polygon_xy", None)
    return {
        "bbox_xyxy": [int(value) for value in proposal.bbox_xyxy],
        "polygon_xy": [[float(x), float(y)] for x, y in polygon] if polygon else None,
        "confidence": float(proposal.confidence or 0.0),
        "source": str(proposal.source),
        "sources": list(proposal.sources),
        "variant_name": str(proposal.variant_name),
    }


def _audit_image(
    detector: RapidOCRTextProposalDetector,
    image: np.ndarray,
    truth_bbox: tuple[int, int, int, int],
    *,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    started = perf_counter()
    proposals, reason = detector.detect(image)
    runtime_ms = (perf_counter() - started) * 1000.0
    boxes = [tuple(int(round(float(value))) for value in proposal.bbox_xyxy) for proposal in proposals]
    overlap_all = _best_overlap(boxes, truth_bbox)
    center_hits = [idx for idx, box in enumerate(boxes) if _bbox_center_inside(box, truth_bbox)]
    by_top_k: dict[str, dict[str, Any]] = {}
    for top_k in top_k_values:
        by_top_k[str(top_k)] = _best_overlap(boxes[:top_k], truth_bbox)
    best_box_payload = None
    if overlap_all["best_box"] is not None:
        best_tuple = tuple(int(value) for value in overlap_all["best_box"])
        for proposal in proposals:
            if tuple(int(round(float(value))) for value in proposal.bbox_xyxy) == best_tuple:
                best_box_payload = _proposal_to_dict(proposal)
                break
    return {
        **overlap_all,
        "runtime_ms": runtime_ms,
        "reason": reason,
        "best_box": best_box_payload,
        "center_hit": bool(center_hits),
        "center_hit_indices": center_hits,
        "by_top_k": by_top_k,
        "detector_boxes": [_proposal_to_dict(proposal) for proposal in proposals],
    }


def _summary(results: list[dict[str, Any]], *, top_k_values: tuple[int, ...]) -> dict[str, Any]:
    images = len(results)
    summary: dict[str, Any] = {
        "images": images,
        "covered": 0,
        "tight": 0,
        "partial": 0,
        "missed_truth": 0,
        "no_box": 0,
        "center_hit": 0,
        "average_box_count": 0.0,
        "average_runtime_ms": 0.0,
        "p50_runtime_ms": 0.0,
        "p90_runtime_ms": 0.0,
        "mean_truth_coverage_ratio": 0.0,
        "mean_best_iou": 0.0,
        "top_k": {},
    }
    if not results:
        return summary
    runtimes = sorted(float(result.get("runtime_ms") or 0.0) for result in results)
    for result in results:
        verdict = str(result.get("verdict") or "missed_truth")
        if verdict in summary:
            summary[verdict] += 1
        if bool(result.get("center_hit")):
            summary["center_hit"] += 1
    summary["average_box_count"] = sum(float(result.get("box_count") or 0) for result in results) / images
    summary["average_runtime_ms"] = sum(runtimes) / images
    summary["p50_runtime_ms"] = runtimes[min(len(runtimes) - 1, int(round((len(runtimes) - 1) * 0.50)))]
    summary["p90_runtime_ms"] = runtimes[min(len(runtimes) - 1, int(round((len(runtimes) - 1) * 0.90)))]
    summary["mean_truth_coverage_ratio"] = sum(float(result.get("truth_coverage_ratio") or 0.0) for result in results) / images
    summary["mean_best_iou"] = sum(float(result.get("best_iou") or 0.0) for result in results) / images
    for top_k in top_k_values:
        key = str(top_k)
        top_results = [result.get("by_top_k", {}).get(key, {}) for result in results]
        summary["top_k"][key] = {
            "covered": sum(1 for result in top_results if result.get("verdict") == "covered"),
            "tight_or_better": sum(1 for result in top_results if result.get("verdict") in {"covered", "tight"}),
            "partial_or_better": sum(1 for result in top_results if result.get("verdict") in {"covered", "tight", "partial"}),
            "mean_truth_coverage_ratio": (
                sum(float(result.get("truth_coverage_ratio") or 0.0) for result in top_results) / images
            ),
        }
    return summary


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# RapidOCR Detector-Only Test64 Audit",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Truth manifest: `{payload['truth_manifest']}`",
        "",
        "## Summary",
        "",
        "| Config | Images | Covered | Tight | Partial | Missed | No box | Center hit | Avg boxes | Avg ms | P90 ms | Mean truth coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in payload["configs"]:
        name = config["name"]
        summary = payload["summary"]["configs"][name]
        lines.append(
            f"| {name} | {summary['images']} | {summary['covered']} | {summary['tight']} | "
            f"{summary['partial']} | {summary['missed_truth']} | {summary['no_box']} | "
            f"{summary['center_hit']} | {summary['average_box_count']:.2f} | "
            f"{summary['average_runtime_ms']:.1f} | {summary['p90_runtime_ms']:.1f} | "
            f"{summary['mean_truth_coverage_ratio']:.3f} |"
        )
    lines.extend(["", "## Top-K Recall", ""])
    for config in payload["configs"]:
        name = config["name"]
        summary = payload["summary"]["configs"][name]
        lines.extend(
            [
                f"### {name}",
                "",
                "| Top K | Covered | Tight+ | Partial+ | Mean truth coverage |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for top_k, item in summary["top_k"].items():
            lines.append(
                f"| {top_k} | {item['covered']} | {item['tight_or_better']} | "
                f"{item['partial_or_better']} | {item['mean_truth_coverage_ratio']:.3f} |"
            )
        lines.append("")
    lines.extend(["## Misses", ""])
    lines.append("| Filename | Config | Verdict | Boxes | Best IoU | Truth coverage | Runtime ms |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for item in payload["items"]:
        for result in item["configs"]:
            if result["verdict"] in {"covered", "tight"}:
                continue
            lines.append(
                f"| {item['filename']} | {result['config_name']} | {result['verdict']} | "
                f"{result['box_count']} | {result['best_iou']:.3f} | "
                f"{result['truth_coverage_ratio']:.3f} | {result['runtime_ms']:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _config_name(*, model_type: str, limit_side_len: int, max_candidates: int, max_boxes: int, min_confidence: float) -> str:
    return f"rapidocr_v5_{model_type}_limit{limit_side_len}_cand{max_candidates}_boxes{max_boxes}_conf{min_confidence:g}"


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _load_rows(args: argparse.Namespace, truth_items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected = _selected_filenames(args.filenames)
    images_dir = Path(args.images_dir)
    rows: list[dict[str, Any]] = []
    for filename, item in sorted(truth_items.items()):
        basename = Path(filename).name
        if selected is not None and basename not in selected:
            continue
        image_path = images_dir / basename
        if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        truth_bbox = _clean_bbox(item.get("true_bbox_xyxy"))
        if truth_bbox is None:
            continue
        rows.append({"filename": basename, "image_path": image_path, "truth_bbox": truth_bbox, "truth_item": item})
    return rows


def run(args: argparse.Namespace) -> int:
    truth_items = _load_truth_items(Path(args.truth_manifest))
    rows = _load_rows(args, truth_items)
    if not rows:
        print("No images with truth bboxes found.", flush=True)
        return 1

    top_k_values = _parse_int_list(args.top_k_values)
    configs = []
    for limit_side_len in _parse_int_list(args.limit_side_lens):
        for max_candidates in _parse_int_list(args.max_candidates_values):
            for max_boxes in _parse_int_list(args.max_boxes_values):
                configs.append(
                    {
                        "name": _config_name(
                            model_type=args.model_type,
                            limit_side_len=limit_side_len,
                            max_candidates=max_candidates,
                            max_boxes=max_boxes,
                            min_confidence=args.min_confidence,
                        ),
                        "settings": {
                            "ocr_version": "PP-OCRv5",
                            "model_type": args.model_type,
                            "lang_type": args.lang_type,
                            "limit_side_len": limit_side_len,
                            "limit_type": args.limit_type,
                            "max_candidates": max_candidates,
                            "max_boxes": max_boxes,
                            "min_confidence": args.min_confidence,
                            "success_overlap_rule": "covered if truth_coverage >= 0.75 or IoU >= 0.50",
                        },
                    }
                )

    print(f"Starting RapidOCR detector-only audit: images={len(rows)} configs={len(configs)}", flush=True)
    items_by_name: dict[str, dict[str, Any]] = {}
    config_results: dict[str, list[dict[str, Any]]] = {config["name"]: [] for config in configs}

    for config in configs:
        name = config["name"]
        settings = config["settings"]
        detector = RapidOCRTextProposalDetector(
            ocr_version=str(settings["ocr_version"]),
            model_type=str(settings["model_type"]),
            lang_type=str(settings["lang_type"]),
            limit_side_len=int(settings["limit_side_len"]),
            limit_type=str(settings["limit_type"]),
            max_candidates=int(settings["max_candidates"]),
            max_boxes=int(settings["max_boxes"]),
            min_confidence=float(settings["min_confidence"]),
            min_box_area=int(args.min_box_area),
        )
        print(f"[{name}] config start", flush=True)
        for index, row in enumerate(rows, start=1):
            filename = str(row["filename"])
            image = _decode_image(Path(row["image_path"]))
            if image is None:
                print(f"[{name} {index}/{len(rows)}] skipped {filename}: unreadable image", flush=True)
                continue
            result = _audit_image(detector, image, row["truth_bbox"], top_k_values=top_k_values)
            result = {"config_name": name, **result}
            config_results[name].append(result)
            item = items_by_name.setdefault(
                filename,
                {
                    "filename": filename,
                    "truth_bbox_xyxy": list(row["truth_bbox"]),
                    "expected": {
                        "day": row["truth_item"].get("expected_day"),
                        "month": row["truth_item"].get("expected_month"),
                        "year": row["truth_item"].get("expected_year"),
                    },
                    "configs": [],
                },
            )
            item["configs"].append(result)
            print(
                f"[{name} {index}/{len(rows)}] {filename} -> {result['verdict']} "
                f"boxes={result['box_count']} truth={result['truth_coverage_ratio']:.3f} "
                f"iou={result['best_iou']:.3f} center={result['center_hit']} "
                f"runtime_ms={result['runtime_ms']:.1f}",
                flush=True,
            )

    report_root = Path(args.output_dir).resolve() / f"test64_rapidocr_detector_only_{_utc_stamp()}"
    _ensure_dir(report_root)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "test64-files",
        "truth_manifest": str(Path(args.truth_manifest).resolve()),
        "backend": "rapidocr_ppocrv5_text_detector",
        "configs": configs,
        "summary": {
            "configs": {
                name: _summary(results, top_k_values=top_k_values)
                for name, results in config_results.items()
            }
        },
        "items": [items_by_name[name] for name in sorted(items_by_name)],
    }
    report_json = report_root / "rapidocr_detector_only_report.json"
    report_md = report_root / "rapidocr_detector_only_summary.md"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(report_md, payload)
    print("Summary:", json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Report root: {report_root}", flush=True)
    print(report_json, flush=True)
    print(report_md, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector-only RapidOCR audit against test64 manual truth bboxes.")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--truth-manifest", default=str(CROP_TRUTH_ANNOTATIONS_PATH))
    parser.add_argument("--output-dir", default="artifacts/forensics")
    parser.add_argument("--filenames", default=None, help="Comma-separated filenames for a targeted run.")
    parser.add_argument("--model-type", choices=("mobile", "server"), default="mobile")
    parser.add_argument("--lang-type", default="ch")
    parser.add_argument("--limit-type", default="max")
    parser.add_argument("--limit-side-lens", default="512")
    parser.add_argument("--max-candidates-values", default="64")
    parser.add_argument("--max-boxes-values", default="64")
    parser.add_argument("--min-confidence", type=float, default=0.08)
    parser.add_argument("--min-box-area", type=int, default=8)
    parser.add_argument("--top-k-values", default="1,3,5,10,64")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
