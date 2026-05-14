# Advanced Mobile Expiry Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make mobile expiry inference use the same advanced candidate/probe/ranking/date-selection architecture, with measured proof for every remaining gap.

**Architecture:** The advanced pipeline becomes the source of truth for candidate construction and final selection. Mobile supplies YOLO26s-OBB proposals and SVTRv2 recognition to that same engine; any missing text evidence is generated from YOLO ROIs by deterministic proposal expansion and accepted only when audit evidence shows it fixes a measured failure bucket.

**Current stage note, 2026-05-13:** The CRAFT ROI rescue experiment reached `38/64` exact matches but was stopped because CRAFT is heavyweight and too slow for this mobile rescue role. The wide YOLO proposal expansion experiment regressed the benchmark when placed in the primary pool, so primary candidate generation must remain plain YOLO proposals. docTR FAST-Tiny as an advanced-style ROI-only rescue adapter measured `37/64` exact matches and lower total upload latency (`242.4s` vs CRAFT's `253.6s`) but produced no selected wins. RapidOCR PP-OCRv5 ONNX detection-only was then wired into the same rescue slot with SVTRv2-only recognition; the final accepted Docker run measured `37/64` exact matches, `44/64` parsed, `7` wrong parsed dates, and `120.6s` total upload latency (`avg=1885ms`, `p90=3379ms`). A primary RapidOCR PP-OCRv5 ROI text-box mode was implemented behind `mobile_rapidocr_primary_enabled` to mirror advanced PP-OCR text-box candidate construction, but the Docker `test64` benchmark regressed to `32/64` exact matches (`45/64` parsed, `13` wrong parsed dates, `127.8s` total upload latency) because PP-OCRv5 text boxes introduced plausible non-expiry dates that outranked the stable YOLO/SVTR candidates. The current accepted mobile rescue backend remains RapidOCR PP-OCRv5 ONNX detector-only, gated to primary no-parseable cases and skipped when SVTR already has a high-confidence date-like fragment.

**Next planned change, 2026-05-14:** Add a gated rotated-YOLO proposal rescue for vertical expiry dates. Run YOLO on the original image first; only when the primary flow is weak or unparseable, run YOLO on `+90`, `-90`, and then `180` degree image rotations, inverse-map OBB polygons/bboxes back into the original coordinate space, and feed those proposals into the same PP-OCR/SVTR candidate engine. This is a proposal-recall rescue only, not a separate recognizer path.

**Tech Stack:** Python 3.11, NumPy/OpenCV, existing YOLO26s-OBB detector, existing SVTRv2 recognizer, existing `ExpiryDateParser`, pytest, Docker Compose ONNX stack, `test64` upload benchmark.

---

## Root Cause Guardrail

The problem must not be solved by guessing detector names. Every failed image must be assigned to exactly one bucket:

- `proposal_missing_truth`: no mobile proposal overlaps the expected expiry text region.
- `proposal_truncates_truth`: proposal overlaps truth but crop only contains a fragment.
- `recognizer_misreads_truth_crop`: truth/near-truth crop reaches SVTR but text is wrong or incomplete.
- `candidate_selection_wrong`: a correct date candidate exists, but final ranking chooses another date.
- `parser_gap`: OCR text contains the date, but parser/normalization does not recover it.
- `label_or_benchmark_issue`: expected label semantics need review.

No new mobile candidate family ships unless it moves at least one failed image out of a measured bucket without creating a regression.

## File Structure

- Create `app/ai/expiry_candidate_engine.py`
  - Shared candidate construction, probe scoring, final candidate selection, parse-input expansion, and role-aware final ranking extracted from the advanced pipeline.
- Create `app/ai/expiry_parity_audit.py`
  - Pure-Python audit helpers for overlap, candidate bucket classification, and per-image evidence serialization.
- Create `app/scripts/advanced_mobile_parity_audit.py`
  - Runs advanced and mobile on `test64`, records proposal/candidate/recognition/final-selection differences, writes JSON/Markdown under `artifacts/onnx_parity/`.
- Modify `app/ai/pipeline.py`
  - Replace duplicated advanced candidate/ranking methods with calls into `ExpiryCandidateEngine`.
- Modify `app/ai/mobile_expiry_pipeline.py`
  - Replace mobile-only rich candidate logic with the same `ExpiryCandidateEngine`.
  - Add YOLO proposal adapter and deterministic YOLO-ROI proposal expansion.
- Modify `app/api/mobile_expiry.py`
  - Preserve public response shape; pass any new mobile engine settings.
- Modify `app/workers/scan_jobs.py`
  - Keep worker constructor aligned with API constructor.
- Test `app/tests/unit/test_expiry_candidate_engine.py`
  - Golden tests for candidate construction and selection.
- Test `app/tests/unit/test_expiry_parity_audit.py`
  - Bucket classification tests.
- Test `app/tests/unit/test_onnx_mobile_inference.py`
  - Mobile no-new-detector and advanced-engine behavior tests.

---

### Task 1: Build the No-Guessing Audit

**Files:**
- Create: `app/ai/expiry_parity_audit.py`
- Create: `app/scripts/advanced_mobile_parity_audit.py`
- Test: `app/tests/unit/test_expiry_parity_audit.py`

- [ ] **Step 1: Write failing bucket-classification tests**

Add `app/tests/unit/test_expiry_parity_audit.py`:

```python
from app.ai.expiry_parity_audit import classify_failure_bucket


def test_classifies_missing_proposal_when_no_overlap() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(10, 10, 60, 30)],
        candidate_bboxes=[],
        correct_candidate_exists=False,
        parser_recovered_date=False,
        selected_wrong_date=False,
        best_truth_crop_text=None,
    )

    assert bucket == "proposal_missing_truth"


def test_classifies_truncated_proposal_when_overlap_is_small() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(100, 100, 125, 130)],
        candidate_bboxes=[],
        correct_candidate_exists=False,
        parser_recovered_date=False,
        selected_wrong_date=False,
        best_truth_crop_text=None,
    )

    assert bucket == "proposal_truncates_truth"


def test_classifies_selection_when_correct_candidate_exists_but_wrong_date_selected() -> None:
    bucket = classify_failure_bucket(
        truth_bbox=(100, 100, 180, 130),
        proposal_bboxes=[(95, 95, 185, 135)],
        candidate_bboxes=[(95, 95, 185, 135)],
        correct_candidate_exists=True,
        parser_recovered_date=True,
        selected_wrong_date=True,
        best_truth_crop_text="EXP 12/05/2027",
    )

    assert bucket == "candidate_selection_wrong"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_parity_audit.py -q
```

Expected: `ModuleNotFoundError: No module named 'app.ai.expiry_parity_audit'`.

- [ ] **Step 3: Implement audit helpers**

Create `app/ai/expiry_parity_audit.py` with:

```python
from __future__ import annotations


BBox = tuple[int, int, int, int]


def bbox_overlap(a: BBox, b: BBox) -> tuple[float, float]:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = max(1, area_a + area_b - inter)
    containment = inter / float(max(1, min(area_a, area_b)))
    return inter / float(union), containment


def classify_failure_bucket(
    *,
    truth_bbox: BBox | None,
    proposal_bboxes: list[BBox],
    candidate_bboxes: list[BBox],
    correct_candidate_exists: bool,
    parser_recovered_date: bool,
    selected_wrong_date: bool,
    best_truth_crop_text: str | None,
) -> str:
    if truth_bbox is None:
        return "label_or_benchmark_issue"
    proposal_overlaps = [bbox_overlap(truth_bbox, box) for box in proposal_bboxes]
    if not proposal_overlaps or max(score[1] for score in proposal_overlaps) < 0.25:
        return "proposal_missing_truth"
    if max(score[1] for score in proposal_overlaps) < 0.70:
        return "proposal_truncates_truth"
    if selected_wrong_date and correct_candidate_exists:
        return "candidate_selection_wrong"
    if best_truth_crop_text and not parser_recovered_date:
        return "parser_gap"
    if not correct_candidate_exists:
        return "recognizer_misreads_truth_crop"
    return "label_or_benchmark_issue"
```

- [ ] **Step 4: Add the parity audit script**

Create `app/scripts/advanced_mobile_parity_audit.py` with a CLI that:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("test64"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/onnx_parity"))
    parser.add_argument("--mobile-report", type=Path, required=True)
    args = parser.parse_args()

    mobile = json.loads(args.mobile_report.read_text())
    rows = mobile["rows"]
    summary = {
        "total": len(rows),
        "manual_review": sum(1 for row in rows if row["response_status"] == "manual_review_required"),
        "wrong_dates": sum(1 for row in rows if row["response_status"] == "parsed_success" and not row["exact_match"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "advanced_mobile_parity_audit.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
```

This initial version only proves report ingestion. Later tasks replace row passthrough with real advanced/mobile evidence capture.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_parity_audit.py -q
```

Expected: `3 passed`.

---

### Task 2: Extract the Advanced Candidate Engine

**Files:**
- Create: `app/ai/expiry_candidate_engine.py`
- Modify: `app/ai/pipeline.py`
- Test: `app/tests/unit/test_expiry_candidate_engine.py`

- [ ] **Step 1: Write golden candidate construction test**

Add `app/tests/unit/test_expiry_candidate_engine.py`:

```python
import numpy as np

from app.ai.expiry_candidate_engine import ExpiryCandidateEngine, ProposalBox


def test_engine_builds_single_line_group_and_fallback_candidates() -> None:
    engine = ExpiryCandidateEngine(max_group_candidates_per_roi=20)
    image = np.full((160, 320, 3), 255, dtype=np.uint8)
    proposals = [
        ProposalBox((20, 30, 92, 50), 0.91, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
        ProposalBox((100, 30, 176, 50), 0.88, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
        ProposalBox((22, 64, 160, 86), 0.77, "yolo26s_obb", ("yolo26s_obb",), "raw", None),
    ]

    candidates = engine.build_ranked_candidates(image, proposals)

    assert {"single", "line", "group"} <= {candidate.candidate_type for candidate in candidates}
    assert all(candidate.detector_sources == ("yolo26s_obb",) for candidate in candidates)


def test_engine_builds_full_fallback_when_no_proposals() -> None:
    engine = ExpiryCandidateEngine()
    image = np.full((160, 320, 3), 255, dtype=np.uint8)

    candidates = engine.build_ranked_candidates(image, [])

    assert len(candidates) == 1
    assert candidates[0].candidate_id == "fallback_full"
    assert candidates[0].bbox_xyxy == (0, 0, 320, 160)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_candidate_engine.py -q
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the shared engine shell**

Create `app/ai/expiry_candidate_engine.py` by moving these method families from `app/ai/pipeline.py` and `app/ai/mobile_expiry_pipeline.py` into a shared class:

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ProposalBox:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float | None
    source: str
    sources: tuple[str, ...]
    variant_name: str | None
    polygon_xy: tuple[tuple[float, float], ...] | None = None


@dataclass(slots=True)
class RankedExpiryCandidate:
    candidate_id: str
    candidate_type: str
    bbox_xyxy: tuple[int, int, int, int]
    polygon_xy: tuple[tuple[float, float], ...] | None
    detector_confidence: float | None
    detector_sources: tuple[str, ...]
    detector_variant: str | None
    member_indices: list[int]
    member_bboxes: list[tuple[int, int, int, int]]
    geometry_features: dict[str, float]
    geometry_score: float
    score_breakdown: dict[str, float]
    total_score: float
    selected_geometry: bool = False
    selected_final: bool = False
    final_rank_after_force_include: int | None = None
    force_included_reason: str | None = None
    probe_text: str = ""
    probe_normalized_text: str = ""
    probe_confidence: float | None = None
    selected_recognition_variant: str = ""
    adjacent_expiry_keyword: bool = False
    adjacent_production_keyword: bool = False
    context_probe_bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)


class ExpiryCandidateEngine:
    def __init__(
        self,
        *,
        max_group_candidates_per_roi: int = 100,
        geometry_top_n: int = 12,
    ) -> None:
        self.max_group_candidates_per_roi = max(0, int(max_group_candidates_per_roi))
        self.geometry_top_n = max(1, int(geometry_top_n))

    def build_ranked_candidates(
        self,
        image: np.ndarray,
        proposals: list[ProposalBox],
    ) -> list[RankedExpiryCandidate]:
        # Move the existing advanced/mobile implementation here without behavior changes.
        raise NotImplementedError("move existing candidate construction here")
```

Then move the existing helper methods for bbox union, grouping, geometry scoring, context relations, and candidate sorting into this class. Do not change thresholds in this task.

- [ ] **Step 4: Replace advanced pipeline candidate construction**

In `app/ai/pipeline.py`, instantiate `ExpiryCandidateEngine` and replace `_build_ranked_candidates(...)` internals with:

```python
proposals = [
    ProposalBox(
        bbox_xyxy=box.bbox_xyxy,
        confidence=box.confidence,
        source=box.source,
        sources=box.sources,
        variant_name=box.variant_name,
        polygon_xy=box.polygon_xy,
    )
    for box in text_boxes
]
return self.candidate_engine.build_ranked_candidates(image, proposals)
```

- [ ] **Step 5: Run parity tests**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_candidate_engine.py app/tests/unit/test_pipeline_high_recall.py -q
```

Expected: all tests pass. If advanced tests change candidate counts or selected dates, stop and fix extraction before touching mobile.

---

### Task 3: Make Mobile Use the Same Engine

**Files:**
- Modify: `app/ai/mobile_expiry_pipeline.py`
- Test: `app/tests/unit/test_onnx_mobile_inference.py`

- [ ] **Step 1: Add a mobile adapter test**

Add a test that builds three `_YoloCandidate` objects, calls mobile candidate construction, and asserts the returned candidate objects come from `ExpiryCandidateEngine`:

```python
def test_mobile_uses_shared_candidate_engine_for_yolo_proposals() -> None:
    pipeline = _pipeline_for_unit_tests()
    image = np.full((160, 320, 3), 255, dtype=np.uint8)
    proposals = [
        _YoloCandidate([[20.0, 30.0], [92.0, 30.0], [92.0, 50.0], [20.0, 50.0]], (20, 30, 92, 50), 0.91),
        _YoloCandidate([[100.0, 30.0], [176.0, 30.0], [176.0, 50.0], [100.0, 50.0]], (100, 30, 176, 50), 0.88),
    ]

    candidates = pipeline._build_ranked_candidates(image, proposals)

    assert {"single", "line"} <= {candidate.candidate_type for candidate in candidates}
    assert all(candidate.detector_sources == ("yolo26s_obb",) for candidate in candidates)
```

- [ ] **Step 2: Replace mobile candidate construction with shared engine**

In `app/ai/mobile_expiry_pipeline.py`, convert `_YoloCandidate` to `ProposalBox`:

```python
def _proposal_boxes_from_yolo(self, yolo_candidates: list[_YoloCandidate]) -> list[ProposalBox]:
    return [
        ProposalBox(
            bbox_xyxy=candidate.bbox_xyxy,
            confidence=candidate.confidence,
            source=candidate.source,
            sources=candidate.sources,
            variant_name=candidate.variant_name,
            polygon_xy=tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
        )
        for candidate in yolo_candidates
    ]
```

Then make `_build_ranked_candidates(...)` call:

```python
return self.candidate_engine.build_ranked_candidates(image, self._proposal_boxes_from_yolo(yolo_candidates))
```

- [ ] **Step 3: Run mobile tests**

Run:

```bash
uv run pytest app/tests/unit/test_onnx_mobile_inference.py -q
```

Expected: all tests pass. If any mobile result changes before proposal expansion is added, stop and inspect the exact candidate list diff.

---

### Task 4: Copy Text Evidence Diversity Without PP-OCRv5

**Files:**
- Modify: `app/ai/mobile_expiry_pipeline.py`
- Test: `app/tests/unit/test_onnx_mobile_inference.py`

- [ ] **Step 1: Add deterministic YOLO-ROI proposal expansion tests**

Add:

```python
def test_mobile_expands_yolo_roi_into_text_evidence_proposals() -> None:
    pipeline = _pipeline_for_unit_tests()
    yolo = _YoloCandidate(
        polygon_xy=[[20.0, 20.0], [220.0, 20.0], [220.0, 100.0], [20.0, 100.0]],
        bbox_xyxy=(20, 20, 220, 100),
        confidence=0.9,
    )

    proposals = pipeline._expanded_yolo_text_proposals(np.full((160, 320, 3), 255, dtype=np.uint8), [yolo])
    names = {proposal.variant_name for proposal in proposals}

    assert "yolo_whole" in names
    assert "yolo_band_top" in names
    assert "yolo_band_mid" in names
    assert "yolo_band_bottom" in names
    assert "yolo_left_context" in names
    assert "yolo_right_context" in names
```

- [ ] **Step 2: Implement proposal expansion**

Add `_expanded_yolo_text_proposals(...)` in `MobileExpiryPipeline`:

```python
def _expanded_yolo_text_proposals(self, image: np.ndarray, yolo_candidates: list[_YoloCandidate]) -> list[ProposalBox]:
    h, w = image.shape[:2]
    proposals: list[ProposalBox] = []
    for candidate in yolo_candidates:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        bands = {
            "yolo_whole": (x1, y1, x2, y2),
            "yolo_band_top": (x1, y1, x2, min(y2, y1 + int(bh * 0.45))),
            "yolo_band_mid": (x1, y1 + int(bh * 0.25), x2, y1 + int(bh * 0.75)),
            "yolo_band_bottom": (x1, max(y1, y2 - int(bh * 0.45)), x2, y2),
            "yolo_left_context": (max(0, x1 - int(bw * 0.35)), y1, x2, y2),
            "yolo_right_context": (x1, y1, min(w, x2 + int(bw * 0.35)), y2),
        }
        for name, bbox in bands.items():
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            proposals.append(
                ProposalBox(
                    bbox_xyxy=bbox,
                    confidence=candidate.confidence,
                    source="yolo26s_obb",
                    sources=candidate.sources,
                    variant_name=name,
                    polygon_xy=None if name != "yolo_whole" else tuple((float(x), float(y)) for x, y in candidate.polygon_xy),
                )
            )
    return proposals
```

Use these expanded proposals as input to the shared candidate engine. Keep CRAFT disabled for this task; the goal is same-model text evidence diversity.

- [ ] **Step 3: Run the audit, not just the benchmark**

Run:

```bash
uv run pytest app/tests/unit/test_onnx_mobile_inference.py app/tests/unit/test_expiry_candidate_engine.py -q
uv run python -u app/scripts/mobile_test64_upload_benchmark.py --base-url http://localhost:8005 --images-dir test64 --output-dir artifacts/onnx_parity --timeout-seconds 120
uv run python -u app/scripts/advanced_mobile_parity_audit.py --mobile-report artifacts/onnx_parity/<new-run>/mobile_test64_upload_report.json
```

Expected acceptance for this task:

- `exact_matches >= 40`
- no more than `1` regression versus the previous `38/64`
- every remaining failure has a bucket assigned.

---

### Task 5: Reuse Advanced Final Selection Exactly

**Files:**
- Modify: `app/ai/expiry_candidate_engine.py`
- Modify: `app/ai/pipeline.py`
- Modify: `app/ai/mobile_expiry_pipeline.py`
- Test: `app/tests/unit/test_expiry_candidate_engine.py`

- [ ] **Step 1: Add final-selection parity test**

Add:

```python
def test_engine_prefers_expiry_role_over_higher_ocr_confidence_mfg() -> None:
    engine = ExpiryCandidateEngine()
    selected = engine.select_final_date_for_test(
        [
            {"text": "MFG 01/01/2026", "parsed_date": "2026-01-01", "ocr_score": 0.99, "candidate_score": 2.0},
            {"text": "EXP 01/01/2027", "parsed_date": "2027-01-01", "ocr_score": 0.72, "candidate_score": 1.5},
        ]
    )

    assert selected["parsed_date"] == "2027-01-01"
```

- [ ] **Step 2: Move final date scoring into the shared engine**

Move these existing advanced/mobile behaviors into `ExpiryCandidateEngine`:

- date-likeness score
- expiry keyword bonus
- production/manufacture penalty
- barcode/weight/brand penalty
- parser probe bonus
- `_build_parse_inputs`
- `score_date_role` integration
- final parsed-date selection key

Both advanced and mobile must call the same method for final selection.

- [ ] **Step 3: Run parity tests**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_candidate_engine.py app/tests/unit/test_pipeline_parse_inputs.py app/tests/unit/test_onnx_mobile_inference.py -q
```

Expected: all tests pass.

---

### Task 6: Recognition Evidence Audit and Cache

**Files:**
- Modify: `app/ai/mobile_expiry_pipeline.py`
- Modify: `app/ai/expiry_parity_audit.py`
- Test: `app/tests/unit/test_onnx_mobile_inference.py`

- [ ] **Step 1: Add test for preserving all SVTR evidence**

Add:

```python
def test_mobile_records_all_variant_evidence_for_a_candidate(monkeypatch) -> None:
    pipeline = _pipeline_for_unit_tests()
    crop = np.full((32, 128, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(
        pipeline,
        "_recognition_variants_for_crop",
        lambda _crop, *, allowed_names=None: [
            ImageVariant("original", crop, purpose="recognition"),
            ImageVariant("clahe_gray", crop, purpose="recognition"),
        ],
    )
    pipeline.recognizer = _QueueRecognizer(
        [
            _RecognitionOutput("020", "020", 0.7, None, "original"),
            _RecognitionOutput("EXP 12/05/2027", "EXP 12/05/2027", 0.9, None, "original"),
        ]
    )

    evidence = pipeline._collect_recognition_evidence(crop, today=date(2026, 1, 1))

    assert [item.variant_name for item in evidence] == ["original", "clahe_gray"]
    assert evidence[1].parsed.parsed_date == date(2027, 5, 12)
```

- [ ] **Step 2: Implement recognition evidence collection**

Add a small dataclass and helper:

```python
@dataclass(slots=True)
class _RecognitionEvidence:
    variant_name: str
    recognition: _RecognitionOutput
    parsed: ParsedDateData
    parse_inputs_count: int


def _collect_recognition_evidence(self, crop: np.ndarray, *, today: date) -> list[_RecognitionEvidence]:
    evidence: list[_RecognitionEvidence] = []
    for variant in self._recognition_variants_for_crop(crop, allowed_names=MOBILE_RECOGNITION_VARIANTS):
        rec = self._recognize_variant(variant.image, variant_name=variant.name, orientation="original")
        parse_inputs = self._build_parse_inputs(rec)
        evidence.append(_RecognitionEvidence(variant.name, rec, self._best_parse_for_inputs(parse_inputs, today=today), len(parse_inputs)))
    return evidence
```

The final selector must see all evidence, not just the first parseable result.

- [ ] **Step 3: Run benchmark gate**

Run Docker upload benchmark. Expected acceptance:

- `exact_matches >= 43`
- wrong-date count does not increase.

---

### Task 7: Final Docker Gate

**Files:**
- Modify: only if Task 1-6 evidence proves a needed change.

- [ ] **Step 1: Run regression tests**

Run:

```bash
uv run pytest app/tests/unit/test_expiry_candidate_engine.py app/tests/unit/test_expiry_parity_audit.py app/tests/unit/test_onnx_mobile_inference.py app/tests/unit/test_parser.py app/tests/unit/test_pipeline_parse_inputs.py app/tests/api/test_api_validation.py -q
```

Expected: all pass.

- [ ] **Step 2: Rebuild and run Docker ONNX stack**

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.onnx.yml up -d --build api
curl -sS http://localhost:8005/health
```

Expected: health returns `{"status":"ok",...}`.

- [ ] **Step 3: Run full upload benchmark**

Run:

```bash
uv run python -u app/scripts/mobile_test64_upload_benchmark.py --base-url http://localhost:8005 --images-dir test64 --output-dir artifacts/onnx_parity --timeout-seconds 120
```

Acceptance:

- `http_201 == 64`
- `exact_matches >= 45`
- `wrong_parsed_dates <= 7`
- `manual_review_required <= 16`
- no image that was correct in the previous `38/64` CRAFT run regresses unless the audit marks the previous result as label/benchmark issue.

- [ ] **Step 4: Publish audit summary**

The final report must include:

- exact matches, parsed success, manual review, wrong dates
- per-image improvements/regressions against `38/64`
- failure bucket counts
- which candidate family fixed each improved image
- p50/p90/max latency

---

## Stop Conditions

Stop and report instead of continuing if:

- shared-engine extraction changes advanced pipeline outputs
- YOLO proposal expansion does not improve at least two measured `proposal_truncates_truth` or `proposal_missing_truth` cases
- recognition evidence shows SVTR cannot read truth crops for most misses
- final selection still chooses wrong dates when a correct candidate exists

In those cases, the next action is not another candidate tweak. It is a focused model/data fix for the bucket identified by the audit.
