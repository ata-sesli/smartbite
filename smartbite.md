# SMARTBITE V1 — Backend + AI Implementation Plan

## Purpose
Build the V1 backend and AI service for SmartBite.

V1 scope is intentionally limited to:
- accepting product scans from a mobile client,
- accepting a QR code or product identifier together with the image,
- detecting and reading the expiry date from packaging,
- parsing and normalizing the date,
- classifying expiry status,
- storing results,
- exposing backend APIs for retrieval and alert processing.

Do **not** implement the mobile application in this version.

---

## Core Product Behavior
Given a scan request containing an image and QR-linked product metadata, the system must:
1. persist the original image,
2. create a scan job,
3. run expiry-date region detection on the image,
4. crop the detected region of interest,
5. preprocess the crop for OCR,
6. extract text from the crop,
7. parse the extracted text into a normalized date,
8. classify the item as `safe`, `expiring_soon`, `expired`, or `manual_review_required`,
9. store all intermediate and final results,
10. expose the result through the API,
11. generate alert events for expiring or expired items.

---

## Technical Constraints
- Web framework: **Litestar**
- Backend language: **Python**
- AI inference is backend-owned and server-side for V1
- The AI pipeline must be modular and testable outside the web framework
- OCR must run only on a detected expiry-date region, not on the full image
- Date parsing must be rule-based and deterministic after OCR
- The backend must support asynchronous scan processing
- Every scan must produce a traceable result, including failure reasons

---

## Out of Scope for V1
- mobile UI or mobile scanning UX
- push notification client integration
- fridge hardware integration
- live video inference
- product freshness estimation without printed expiry dates
- end-to-end giant multimodal model

---

## Recommended High-Level Architecture

```text
Mobile Client
  -> Litestar API
  -> Image Storage + Scan Record
  -> Async Scan Worker
  -> Expiry Region Detector
  -> ROI Preprocessing
  -> OCR Engine
  -> Date Parser
  -> Expiry Decision Engine
  -> Database Update
  -> Alert Event Creation
  -> Result Retrieval API
```

---

## Required Components

### 1. API Layer
Implement HTTP APIs for:
- creating scans,
- checking scan status,
- retrieving scan results,
- listing expiring items,
- processing alert creation,
- manually correcting failed or low-confidence results,
- health/status checks.

### 2. Storage Layer
Persist:
- raw uploaded images,
- cropped ROI images,
- OCR text,
- parsed dates,
- confidence values,
- processing logs,
- failure reasons,
- alert events.

### 3. AI Pipeline Layer
Implement a modular pipeline with isolated components for:
- detection,
- preprocessing,
- OCR,
- parsing,
- decision.

### 4. Background Job Layer
Use background jobs or async workers for scan execution so upload endpoints remain fast.

### 5. Alert Layer
Implement backend-side alert eligibility and alert event creation. V1 can stop at event generation or webhook/email stub delivery.

---

## Repository Structure
Use a structure close to this:

```text
app/
  api/
    scans.py
    items.py
    expiry.py
    alerts.py
    admin.py
    health.py
  domain/
    models.py
    schemas.py
    enums.py
    repositories.py
    services.py
  ai/
    pipeline.py
    detector.py
    preprocess.py
    ocr.py
    parser.py
    decision.py
    types.py
  workers/
    scan_jobs.py
    alert_jobs.py
  infra/
    db.py
    storage.py
    logging.py
    settings.py
    clock.py
  tests/
    api/
    ai/
    integration/
```

The AI package must be runnable independently from Litestar for local testing and benchmarking.

---

## Core Domain Models

### Product
Represents product metadata, ideally derived or linked from QR input.

Fields:
- `id`
- `qr_code`
- `name`
- `brand`
- `category`
- `created_at`
- `updated_at`

### Scan
Represents one uploaded image and its processing lifecycle.

Fields:
- `id`
- `product_id`
- `user_id`
- `image_path`
- `roi_path` nullable
- `status` enum
- `created_at`
- `updated_at`
- `processed_at` nullable

### OCRResult
Stores OCR stage output.

Fields:
- `id`
- `scan_id`
- `raw_text`
- `normalized_text`
- `ocr_confidence` nullable
- `engine_name`
- `created_at`

### ParsedDateResult
Stores parsed date decision.

Fields:
- `id`
- `scan_id`
- `parsed_date` nullable
- `date_format_detected` nullable
- `parse_confidence` nullable
- `parser_reason`
- `candidate_dates_json`
- `created_at`

### ExpiryStatus
Stores expiry classification.

Fields:
- `id`
- `scan_id`
- `status` enum
- `days_remaining` nullable
- `alert_required`
- `needs_review`
- `created_at`

### AlertEvent
Represents a backend-generated alert record.

Fields:
- `id`
- `scan_id`
- `user_id`
- `alert_type`
- `delivery_status`
- `payload_json`
- `created_at`
- `sent_at` nullable

---

## Required Enums

### ScanStatus
- `queued`
- `processing`
- `done`
- `failed`

### FinalResultStatus
- `parsed_success`
- `parsed_with_low_confidence`
- `multiple_candidates`
- `detector_failed`
- `ocr_failed`
- `parser_failed`
- `manual_review_required`

### ExpiryClassification
- `safe`
- `expiring_soon`
- `expired`
- `manual_review_required`

### AlertType
- `expiring_soon`
- `expired`
- `manual_review`

---

## API Contract

### `POST /scans`
Create a new scan.

Input:
- multipart image file
- `qr_code`
- `user_id`
- optional metadata

Behavior:
- validate payload,
- persist image,
- create or resolve product from QR,
- create scan row with `queued`,
- enqueue async scan job,
- return `scan_id` and status.

Response shape:

```json
{
  "scan_id": "uuid",
  "status": "queued"
}
```

### `GET /scans/{scan_id}`
Return current processing state and final result if available.

Response shape:

```json
{
  "scan_id": "uuid",
  "status": "done",
  "result": {
    "final_status": "parsed_success",
    "detected": true,
    "detector_confidence": 0.91,
    "raw_text": "EXP 29/03/26",
    "parsed_date": "2026-03-29",
    "date_format_detected": "DD/MM/YY",
    "parse_confidence": 0.95,
    "expiry_classification": "expiring_soon",
    "days_remaining": 2,
    "needs_review": false
  }
}
```

### `GET /expiry`
List processed items with filters.

Supported filters:
- `status`
- `user_id`
- `product_id`
- `date_from`
- `date_to`

### `POST /alerts/process`
Create alert events for eligible scans/items.

Behavior:
- find items requiring alerts,
- create idempotent alert events,
- return summary.

### `PATCH /scans/{scan_id}`
Manual correction endpoint.

Used for:
- correcting parsed date,
- resolving ambiguous candidates,
- overriding unreadable results,
- forcing review resolution.

### `GET /health`
Simple health endpoint.

### `GET /admin/metrics`
Expose operational metrics for debugging and evaluation.

---

## Async Processing Contract
Every created scan must pass through this backend job pipeline:

1. load original image,
2. set scan status to `processing`,
3. run expiry-date region detection,
4. if no ROI found, mark `detector_failed`, persist reason, mark review,
5. crop ROI and persist crop,
6. preprocess crop,
7. run OCR,
8. if OCR fails or empty text, mark `ocr_failed`, persist reason, mark review,
9. normalize OCR text,
10. parse candidate dates,
11. if no valid date, mark `parser_failed`, persist reason, mark review,
12. compute expiry classification,
13. persist OCRResult, ParsedDateResult, ExpiryStatus,
14. mark scan `done`,
15. create alert event if required.

This pipeline must be idempotent for retries.

---

## AI Pipeline Requirements

### Detector
Goal:
- locate the expiry-date region on packaging.

Requirements:
- detector returns bounding box coordinates,
- detector returns confidence score,
- detector supports fallback to multiple candidate boxes if confidence is close,
- detector must be replaceable without changing the API layer.

Recommended initial implementation:
- YOLO-based expiry region detector.

Detector output type:

```python
DetectionResult(
    detected: bool,
    boxes: list[BoundingBox],
    best_box: BoundingBox | None,
    confidence: float | None,
    reason: str | None,
)
```

### Preprocessing
Goal:
- improve OCR quality on cropped region.

Required operations:
- grayscale conversion,
- resize,
- contrast enhancement,
- thresholding or binarization,
- denoising,
- optional deskew.

Preprocessing must be configurable and benchmarkable.

### OCR
Goal:
- read the date text from the cropped ROI.

Requirements:
- must accept only ROI image input,
- return raw text and optional confidence,
- support swappable engines,
- preserve original OCR output before normalization.

Recommended V1 engines:
- PaddleOCR or Tesseract.

OCR output type:

```python
OCRResultData(
    raw_text: str,
    normalized_text: str,
    confidence: float | None,
    engine_name: str,
    reason: str | None,
)
```

### Parser
Goal:
- convert OCR text into a valid normalized date.

Requirements:
- support multiple common expiry formats,
- detect and rank candidate dates,
- reject likely lot numbers and manufacturing dates when possible,
- normalize to ISO `YYYY-MM-DD`,
- return parse confidence and explanation.

Parser must support at least these patterns:
- `DD/MM/YYYY`
- `DD/MM/YY`
- `DD-MM-YYYY`
- `DD-MM-YY`
- `YYYY-MM-DD`
- `MM/YYYY`
- `DD MON YYYY`
- strings prefixed with `EXP`, `USE BY`, `BEST BEFORE`

Parser must filter or de-prioritize tokens like:
- `LOT`
- `BATCH`
- `MFG`
- `PROD`

Parser output type:

```python
ParsedDateData(
    parsed_date: date | None,
    date_format_detected: str | None,
    confidence: float,
    candidates: list[str],
    reason: str,
)
```

### Decision Engine
Goal:
- classify result for business usage.

Rules:
- if date parse failed -> `manual_review_required`
- if parsed date < today -> `expired`
- if parsed date within configurable threshold days -> `expiring_soon`
- else -> `safe`

Threshold must be configurable, defaulting to a small number such as `3` days.

Decision output type:

```python
DecisionResult(
    final_status: str,
    expiry_classification: str,
    days_remaining: int | None,
    alert_required: bool,
    needs_review: bool,
    reason: str,
)
```

---

## Configuration Requirements
All configuration must live outside code defaults where reasonable.

Config must include:
- database connection
- storage root
- OCR engine selection
- alert threshold days
- model paths
- scan retry settings
- log level
- image size limits
- accepted file types

Use environment-driven settings with a typed settings object.

---

## Storage Requirements
- Store original uploaded images
- Store cropped ROI images
- Use deterministic file naming by scan ID
- Keep storage abstraction independent from local disk so it can later move to object storage
- Persist image paths in database, not image blobs

---

## Logging and Observability
Every scan job must produce structured logs including:
- scan ID
- product ID
- user ID
- pipeline stage
- elapsed time per stage
- detector confidence
- OCR confidence
- parser outcome
- final classification
- failure reason if any

Expose aggregate metrics for:
- total scans
- detector failures
- OCR failures
- parser failures
- manual review rate
- average processing latency
- alert creation counts

---

## Failure Handling Rules
The system must never silently fail.

Failure cases must be persisted with explicit reasons:
- no expiry region detected
- multiple conflicting candidate regions
- OCR returned empty text
- OCR text unusable
- no valid date parsed
- multiple dates found with low certainty
- image unreadable

On failure:
- mark scan `done` if processing completed but result requires review,
- use `manual_review_required` for user-facing final classification,
- store full failure reason,
- allow manual override through API.

---

## Idempotency Rules
- `POST /scans` may create a new scan each time, but downstream alert generation must be idempotent
- scan job retries must not duplicate OCRResult, ParsedDateResult, or AlertEvent incorrectly
- `POST /alerts/process` must be safe to call repeatedly

---

## Security and Validation
- validate uploaded file type
- validate file size
- reject unsupported image formats
- sanitize metadata inputs
- never trust QR-derived product metadata blindly without schema validation
- authenticate API access if auth already exists in project scope

---

## Database Choice
Use a relational database for V1.
Recommended:
- PostgreSQL for main persistence
- SQLite only for very local prototyping if needed

Use migrations from the beginning.

---

## Minimal Coding Order
The coding agent should implement in this order:

1. Litestar app skeleton
2. typed settings and database connection
3. core domain models and migrations
4. image storage abstraction
5. `POST /scans` and `GET /scans/{scan_id}`
6. background scan worker skeleton
7. manual-crop OCR baseline service
8. parser service with tests
9. detector interface and YOLO-backed implementation
10. preprocessing module
11. full `pipeline.py` orchestration
12. persistence of OCRResult, ParsedDateResult, ExpiryStatus
13. `GET /expiry`
14. alert event generation and `POST /alerts/process`
15. manual correction endpoint
16. metrics and structured logging

---

## Test Requirements

### Unit tests
Must cover:
- parser formats
- parser rejection of lot/manufacture patterns
- decision engine classification
- API schema validation
- storage path generation

### Integration tests
Must cover:
- upload image -> create scan -> process -> retrieve result
- detector failure path
- OCR failure path
- parser failure path
- alert creation path
- manual correction flow

### AI evaluation scripts
Must produce:
- detection precision/recall/F1
- OCR character accuracy
- exact date match accuracy
- end-to-end expiry classification accuracy

---

## Definition of Done for V1
V1 is complete when the system can:
- accept an image + QR payload,
- asynchronously process the scan,
- detect an expiry date region,
- run OCR on the ROI,
- parse a normalized date,
- classify expiry state,
- persist all results,
- expose the result via API,
- create alert events for eligible items,
- return review-required states for low-confidence failures.

---

## Final Output Contract
Every completed scan must end in one normalized backend result object with the following conceptual fields:

```json
{
  "scan_id": "uuid",
  "final_status": "parsed_success",
  "detected": true,
  "detector_confidence": 0.91,
  "raw_text": "EXP 29/03/26",
  "parsed_date": "2026-03-29",
  "date_format_detected": "DD/MM/YY",
  "parse_confidence": 0.95,
  "expiry_classification": "expiring_soon",
  "days_remaining": 2,
  "alert_required": true,
  "needs_review": false,
  "reason": "parsed_from_exp_prefix"
}
```

This normalized shape is the contract the mobile client and future hardware integrations should depend on.
