# Test64 YOLO Expiry-Date Detection Research Brief

Date: 2026-05-11

## Purpose

We are trying to increase expiry-date localization accuracy on `test64`. The current detector is a YOLO26s OBB expiry-date detector trained from the ExpDate-derived dataset, and we also tested whether the existing YOLO26 product detector can improve results by first cropping product regions and then running the expiry-date detector inside those crops.

Success in this report means:

`truth_coverage_ratio > 0.20`

This is intentionally permissive. It counts a detection as useful if at least 20% of the manually labeled expiry-date truth box is covered by the best generated candidate.

## Current Conclusion

The main blocker is not candidate count alone. It is a domain gap concentrated in the `IMG_*` phone-photo subset.

Tiled expiry-date inference improves the non-phone/UUID-style images, but it still gets `0/28` on the `IMG_*` block. Product-first cropping is faster and rescues one phone-photo image, but it lowers overall accuracy and only improves the union by one image.

Best current candidate-generator result:

`tiled expiry YOLO + product ROI union = 31/64 = 48.4%`

That is only a small gain over tiled expiry YOLO alone:

`tiled expiry YOLO = 30/64 = 46.9%`

## Experiment Summary

| Experiment | Success | IMG_* phone block | UUID jpgs | Manual jpeg names | Avg boxes/image | Avg runtime/image | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Expiry YOLO OBB, conf 0.10 | 25/64 = 39.1% | 0/28 = 0.0% | 22/33 = 66.7% | 3/3 = 100.0% | 0.62 | 348 ms | Baseline-ish detector threshold. Many no-box cases. |
| Expiry YOLO OBB, conf 0.01 | 27/64 = 42.2% | 0/28 = 0.0% | 24/33 = 72.7% | 3/3 = 100.0% | 1.88 | 346 ms | Lowering confidence helps a little, but does not solve `IMG_*`. |
| Tiled expiry YOLO OBB | 30/64 = 46.9% | 0/28 = 0.0% | 27/33 = 81.8% | 3/3 = 100.0% | 11.05 | 8482 ms | Better recall on UUID images, expensive, still fails phone block. |
| Product YOLO -> expiry YOLO OBB | 23/64 = 35.9% | 1/28 = 3.6% | 19/33 = 57.6% | 3/3 = 100.0% | 2.92 | 878 ms | Faster, but product cropping drops too many otherwise-successful cases. |
| Tiled + product ROI union | 31/64 = 48.4% | 1/28 = 3.6% | 27/33 = 81.8% | 3/3 = 100.0% | n/a | n/a | Product ROI adds only one new success: `IMG_0896.JPG`. |

## Product Detector Test

The product detector used:

`models/yolo20n/yolo26s/yolo26s-best.pt`

The expiry detector used inside product crops:

`models/yolo26s_obb_expdate_det/weights/best.pt`

Run configuration:

- Product confidence: `0.15`
- Product top-k: `4`
- Product padding ratio: `0.12`
- Product image size: `1024`
- Expiry confidence: `0.01`
- Expiry image size: `1024`
- Max expiry candidates: `12`

Result:

`23/64 = 35.9%`

The only product-ROI-only rescue over tiled inference was:

`IMG_0896.JPG`, truth coverage `0.201`

This is barely over the threshold, so product ROI is not a strong solution by itself.

## What The Numbers Suggest

The detector already works reasonably on the UUID-style images:

- Low confidence full-image YOLO: `24/33 = 72.7%`
- Tiled YOLO: `27/33 = 81.8%`

The detector essentially does not work on the `IMG_*` phone-photo subset:

- Low confidence full-image YOLO: `0/28 = 0.0%`
- Tiled YOLO: `0/28 = 0.0%`
- Product ROI: `1/28 = 3.6%`

Because tiled inference generates many candidates on `IMG_*` but still produces zero overlap, this looks less like a scale-only issue and more like a distribution shift:

- Different phone camera characteristics
- Different lighting/glare
- Expiry text may be embossed, low contrast, curved, dot-matrix, or printed on reflective packaging
- Different product composition/backgrounds
- Date locations not represented well by the ExpDate-derived training set

## Recognizer Reranking Implication

Recognizer reranking cannot create boxes where the detector has no useful candidate.

It may improve final OCR/date selection when a candidate already overlaps the expiry date, but it will not solve the `IMG_*` problem unless the detector produces at least one near-date candidate. In the current tiled run, the problem is mostly that candidates do not overlap the truth box.

## Recommended Research Direction

The highest-value research question is:

How do we train or adapt an expiry-date region detector for handheld grocery/product phone photos with fewer than 200 real labeled samples?

Research should focus on:

1. Small-data object detection/domain adaptation for packaging expiry-date regions.
2. Synthetic data generation that matches phone-photo expiry-date styles.
3. Copy-paste/date-stamp augmentation onto real product photos.
4. Tiled/SAHI-style training, not just tiled inference.
5. High-resolution small-object detection improvements for YOLO-style OBB models.
6. Whether axis-aligned boxes or segmentation masks are more stable than OBB for this task.
7. Active learning: label only the most informative new phone-photo failures.

Avoid Roboflow datasets and avoid PP-OCRv5 as a future path.

## Practical Next Steps

1. Inspect `IMG_*` failures visually with truth box and top 12 tiled candidates overlaid.
2. Build a phone-photo-specific mini-dev split, but do not train on all of `test64` unless a separate untouched holdout is created.
3. Collect or label at least 50-100 more phone-style product images if possible.
4. Try synthetic stamp augmentation on existing product/package images:
   - dot-matrix dates
   - black ink on transparent/reflective packaging
   - embossed/low-contrast dates
   - curved bottle/can surfaces
   - motion blur, glare, compression, phone perspective
5. Train a detector with explicit phone-style augmentations and evaluate on the current `test64` holdout.
6. Keep product-ROI as an optional auxiliary branch only if speed matters, because it is not currently a recall solution.

## Evidence Files

Scripts:

- `app/scripts/tiled_yolo_test64_audit.py`
- `app/scripts/product_roi_yolo_test64_audit.py`

Reports:

- `artifacts/forensics/test64_detector_audit_20260510T223127Z/detector_truth_ablation_report.json`
- `artifacts/forensics/test64_detector_audit_20260510T223249Z/detector_truth_ablation_report.json`
- `artifacts/forensics/test64_tiled_yolo_audit_20260511T104615Z/tiled_yolo_test64_report.json`
- `artifacts/forensics/test64_product_roi_yolo_audit_20260511T113609Z/product_roi_yolo_test64_report.json`

## Deep Research Query Starters

- "small dataset object detection domain adaptation for product expiry date localization"
- "synthetic data generation for expiry date detection on product packaging"
- "YOLO small object detection high resolution tiling training SAHI"
- "copy paste augmentation text date stamp detection packaging"
- "few-shot object detection for retail product label text regions"
- "oriented bounding box detection for low contrast printed text on packaging"
- "active learning for object detection with small labeled dataset"

