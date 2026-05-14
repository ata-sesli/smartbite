from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import httpx
from sqlalchemy import text

from app.infra.db import get_session_factory


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Type not JSON serializable: {type(value)}")


def _expected_label(label: dict[str, Any] | None) -> dict[str, Any]:
    if not label:
        return {"label": None, "precision": None, "day": None, "month": None, "year": None}
    day = label.get("expected_day")
    month = label.get("expected_month")
    year = label.get("expected_year")
    if month is None or year is None:
        return {"label": None, "precision": None, "day": None, "month": None, "year": None}
    if day is None:
        return {
            "label": f"{int(year):04d}-{int(month):02d}",
            "precision": "month",
            "day": None,
            "month": int(month),
            "year": int(year),
        }
    return {
        "label": f"{int(year):04d}-{int(month):02d}-{int(day):02d}",
        "precision": "day",
        "day": int(day),
        "month": int(month),
        "year": int(year),
    }


def _detected_matches_expected(detected: str | None, expected: dict[str, Any]) -> bool:
    if not detected or expected["month"] is None or expected["year"] is None:
        return False
    try:
        parsed = date.fromisoformat(detected)
    except ValueError:
        return False
    if expected["precision"] == "month":
        return parsed.year == expected["year"] and parsed.month == expected["month"]
    return (
        parsed.year == expected["year"]
        and parsed.month == expected["month"]
        and parsed.day == expected["day"]
    )


async def _load_expected_labels() -> dict[str, dict[str, Any]]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT filename, expected_day, expected_month, expected_year, updated_at
                FROM test64_expected_dates
                """
            )
        )
        rows = [dict(row) for row in result.mappings().all()]
    return {row["filename"]: row for row in rows}


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


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _post_image(client: httpx.Client, endpoint: str, path: Path) -> tuple[int | None, dict[str, Any] | None, str | None, float]:
    started = perf_counter()
    try:
        with path.open("rb") as handle:
            response = client.post(
                endpoint,
                files={"image": (path.name, handle, _content_type(path))},
                data={"metadata": json.dumps({"source": "test64-mobile-upload-benchmark", "original_filename": path.name})},
            )
        latency_ms = (perf_counter() - started) * 1000.0
    except Exception as exc:
        return None, None, str(exc), (perf_counter() - started) * 1000.0
    try:
        payload = response.json()
    except Exception:
        payload = None
    error = None if response.status_code < 400 else response.text
    return response.status_code, payload, error, latency_ms


async def main_async(args: argparse.Namespace) -> int:
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"test64 images directory not found: {images_dir}")
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    labels = await _load_expected_labels()

    report_root = Path(args.output_dir).resolve() / f"test64_mobile_upload_{_utc_stamp()}"
    report_root.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/mobile/expiry-scans"

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout_seconds) as client:
        for index, path in enumerate(image_paths, start=1):
            status_code, payload, error, latency_ms = _post_image(client, endpoint, path)
            expected = _expected_label(labels.get(path.name))
            detected = payload.get("detected_expiry_date") if isinstance(payload, dict) else None
            exact_match = _detected_matches_expected(detected, expected)
            row = {
                "filename": path.name,
                "http_status": status_code,
                "latency_ms": round(latency_ms, 3),
                "expected_date": expected["label"],
                "expected_precision": expected["precision"],
                "expected_day": expected["day"],
                "expected_month": expected["month"],
                "expected_year": expected["year"],
                "detected_expiry_date": detected,
                "exact_match": exact_match,
                "response_status": payload.get("status") if isinstance(payload, dict) else None,
                "raw_text": payload.get("raw_text") if isinstance(payload, dict) else None,
                "normalized_text": payload.get("normalized_text") if isinstance(payload, dict) else None,
                "recognition_confidence": payload.get("recognition_confidence") if isinstance(payload, dict) else None,
                "detector_confidence": payload.get("detector_confidence") if isinstance(payload, dict) else None,
                "final_recognition_bbox_xyxy": payload.get("final_recognition_bbox_xyxy") if isinstance(payload, dict) else None,
                "final_recognition_polygon_json": (
                    payload.get("final_recognition_polygon_json") if isinstance(payload, dict) else None
                ),
                "final_crop_policy": payload.get("final_crop_policy") if isinstance(payload, dict) else None,
                "final_crop_padding_px": payload.get("final_crop_padding_px") if isinstance(payload, dict) else None,
                "reason": payload.get("reason") if isinstance(payload, dict) else error,
            }
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] {path.name} http={status_code} status={row['response_status']} detected={detected} expected={expected['label']} latency_ms={latency_ms:.1f}", flush=True)

    successful_http = [row for row in rows if row["http_status"] == 201]
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
    summary = {
        "total": len(rows),
        "http_201": len(successful_http),
        "parsed_success": len(parsed),
        "manual_review_required": len(manual),
        "exact_matches": len(exact),
        "accuracy": round(len(exact) / len(rows), 4) if rows else 0.0,
        "parsed_success_rate": round(len(parsed) / len(rows), 4) if rows else 0.0,
        "wrong_parsed_dates": len(wrong_dates),
        "previous_mobile_onnx_baseline_exact_matches": "32/64",
        "latency": _latency_summary([float(row["latency_ms"]) for row in rows]),
    }
    payload = {
        "base_url": args.base_url,
        "endpoint": endpoint,
        "images_dir": str(images_dir),
        "created_at": datetime.utcnow(),
        "summary": summary,
        "rows": rows,
        "wrong_dates": wrong_dates,
        "manual_review": manual,
    }

    report_json = report_root / "mobile_test64_upload_report.json"
    report_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    summary_md = report_root / "mobile_test64_upload_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Mobile Test64 Upload Benchmark",
                "",
                f"- endpoint: `{endpoint}`",
                f"- total: `{summary['total']}`",
                f"- http_201: `{summary['http_201']}`",
                f"- parsed_success: `{summary['parsed_success']}`",
                f"- manual_review_required: `{summary['manual_review_required']}`",
                f"- exact_matches: `{summary['exact_matches']}`",
                f"- accuracy: `{summary['accuracy']}`",
                f"- wrong_parsed_dates: `{summary['wrong_parsed_dates']}`",
                f"- previous_mobile_onnx_baseline_exact_matches: `{summary['previous_mobile_onnx_baseline_exact_matches']}`",
                f"- latency: `{summary['latency']}`",
            ]
        ),
        encoding="utf-8",
    )
    print("Report root:", report_root)
    print(report_json)
    print(summary_md)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="POST local test64 images through the mobile expiry upload endpoint")
    parser.add_argument("--base-url", default="http://localhost:8005")
    parser.add_argument("--images-dir", default="test64")
    parser.add_argument("--output-dir", default="artifacts/onnx_parity")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import asyncio

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
