from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _bucket_for_row(row: dict[str, Any]) -> str:
    reason = str(row.get("reason") or "")
    raw_text = str(row.get("raw_text") or row.get("normalized_text") or "")
    status = str(row.get("response_status") or "")
    if status == "parsed_success" and not bool(row.get("exact_match")):
        return "candidate_selection_wrong"
    if status == "manual_review_required":
        digit_count = sum(ch.isdigit() for ch in raw_text)
        if digit_count and digit_count < 5:
            return "proposal_truncates_truth"
        if "no expiry region" in reason.lower():
            return "proposal_missing_truth"
        return "recognizer_misreads_truth_crop"
    return "matched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("test64"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/onnx_parity"))
    parser.add_argument("--mobile-report", type=Path, required=True)
    args = parser.parse_args()

    mobile = json.loads(args.mobile_report.read_text(encoding="utf-8"))
    rows = mobile["rows"]
    audited_rows = []
    buckets: Counter[str] = Counter()
    for row in rows:
        bucket = _bucket_for_row(row)
        buckets[bucket] += 1
        audited_rows.append({**row, "failure_bucket": bucket})
    summary = {
        "total": len(rows),
        "manual_review": sum(1 for row in rows if row["response_status"] == "manual_review_required"),
        "wrong_dates": sum(
            1 for row in rows if row["response_status"] == "parsed_success" and not row["exact_match"]
        ),
        "bucket_counts": dict(sorted(buckets.items())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "advanced_mobile_parity_audit.json"
    out.write_text(json.dumps({"summary": summary, "rows": audited_rows}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
