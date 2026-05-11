# OCR Detection Handoff - 2026-05-11

## Current State

We are working on SmartBite expiry-date OCR. Recognition/parser is mostly good now; detection is the active bottleneck.

Manual crop recognition with the fine-tuned SVTRv2 became strong after removing extra `SKT`/`TETT` context from some crops:

- Manual-crop recognizer benchmark reached about `60/64`.
- Parser/recognizer logic is considered frozen for now.
- Fine-tuned recognizer model path:
  - `models/svtrv2/smartbite_svtrv2_expdate_rec`

The current problem is finding the expiry-date region automatically.

## PP-OCRv5 Detector

Fine-tuned PP-OCRv5 detector was trained in Colab and exported locally:

- `models/pp-ocrv5-text-detection/best_model_inference`

Training metrics looked good, but detector-only audit on `test64` was weak.

Best PP-OCRv5 runtime config chosen after ablation:

- `SMARTBITE_OCR_PPOCRV5_USE_CUSTOM_TEXT_DET_MODEL=true`
- `SMARTBITE_OCR_PPOCRV5_TEXT_DET_MODEL_DIR=models/pp-ocrv5-text-detection/best_model_inference`
- `SMARTBITE_OCR_PPOCRV5_DET_DB_THRESH=0.25`
- `SMARTBITE_OCR_PPOCRV5_DET_DB_BOX_THRESH=0.35`
- `SMARTBITE_OCR_PPOCRV5_DET_LIMIT_SIDE_LEN=1216`
- `SMARTBITE_OCR_PPOCRV5_DET_LIMIT_TYPE=max`
- `SMARTBITE_OCR_PPOCRV5_DET_DB_UNCLIP_RATIO=1.8`

PP-OCRv5 detector audit report:

- `artifacts/forensics/test64_detector_audit_20260510T124043Z/detector_truth_ablation_report.json`

Results:

- `scale_1216`: covered `5`, tight `3`, partial `25`, missed `13`, no_box `18`
- `recall_1216_unclip18`: covered `6`, tight `4`, partial `23`, missed `14`, no_box `17`

Conclusion: PP-OCRv5 is still too blind/tight/fragmentary for many test64 cases.

## YOLO26s-OBB Experiment

Created notebook:

- `notebooks/smartbite_yolo26s_obb_products_date_det_colab.ipynb`

It uses the same dataset zip as PP-OCRv5:

- `/content/drive/My Drive/sb-colab/products_date_detection_balanced.zip`

It converts PaddleOCR detector labels to Ultralytics OBB labels and trains:

- `yolo26s-obb.pt`

YOLO run copied locally:

- `models/yolo26s_obb_expdate_det`
- `models/yolo26s_obb_expdate_det/weights/best.pt`
- `models/yolo26s_obb_expdate_det/weights/last.pt`
- `models/yolo26s_obb_expdate_det/results.csv`

Important note:

- Training stopped at epoch `89`.
- `save_period=-1`, so only `best.pt` and `last.pt` were saved.
- Epoch `61` looked more recall-balanced, but was not checkpointed.
- Ultralytics selected epoch `64` as `best.pt`.

Best training/val signals during YOLO training:

- Epoch 61: precision `.969`, recall `.964`, mAP50 `.980`, mAP50-95 `.961`
- Epoch 64/best.pt: precision `.979`, recall `.940`, mAP50 `.989`, mAP50-95 `.962`
- Epoch 89/last.pt: precision `.945`, recall `.946`, mAP50 `.959`, mAP50-95 `.946`

## YOLO26s-OBB Test64 Audit

I extended `app/scripts/detector_truth_ablation.py` to support:

- `--backend yolo_obb`
- `--yolo-model-path`
- direct OBB polygon extraction from Ultralytics result output

Tests run:

```bash
uv run pytest app/tests/unit/test_detector_truth_ablation.py -q
python -m py_compile app/scripts/detector_truth_ablation.py
```

Both passed.

YOLO audit command:

```bash
uv run python -m app.scripts.detector_truth_ablation \
  --backend yolo_obb \
  --yolo-model-path models/yolo26s_obb_expdate_det/weights/best.pt \
  --yolo-conf 0.10 \
  --yolo-config-name yolo26s_obb_best
```

Result report:

- `artifacts/forensics/test64_detector_audit_20260510T223127Z/detector_truth_ablation_report.json`

Results:

- covered `8`
- tight `6`
- partial `11`
- missed_truth `6`
- no_box `33`
- avg runtime about `348 ms/image`
- mean truth coverage `0.197`

Second low-confidence audit:

```bash
uv run python -m app.scripts.detector_truth_ablation \
  --backend yolo_obb \
  --yolo-model-path models/yolo26s_obb_expdate_det/weights/best.pt \
  --yolo-conf 0.01 \
  --yolo-config-name yolo26s_obb_best_conf001
```

Result report:

- `artifacts/forensics/test64_detector_audit_20260510T223249Z/detector_truth_ablation_report.json`

Results:

- covered `8`
- tight `7`
- partial `14`
- missed_truth `18`
- no_box `17`
- avg runtime about `346 ms/image`
- mean truth coverage `0.215`

Conclusion:

- YOLO26s-OBB is much faster than PP-OCRv5.
- But this checkpoint does **not** generalize to test64 well enough.
- Training validation looked excellent, but test64 reveals a domain gap.
- The dataset used for YOLO/PP-OCRv5 training is too different from test64 phone/product cases.

## Website / API State

Detection Review section now reads latest detector-only audit:

- endpoint: `/test64/detection-review`
- frontend proxy: `/api/test64/detection-review`

It should now show the latest YOLO detector audit boxes.

Full Pipeline report was separated into its own section:

- endpoint: `/test64/full-pipeline-review`
- frontend proxy: `/api/test64/full-pipeline-review`

API/Postgres/Redis must run in Docker. Do not run API outside Docker.

Docker API was running, though local `curl localhost:8005` may sometimes fail from host networking weirdness; browser/API logs showed successful requests.

## Important Files Changed

- `app/scripts/detector_truth_ablation.py`
  - now supports PP-OCRv5 and YOLO OBB detector audits.
- `notebooks/smartbite_yolo26s_obb_products_date_det_colab.ipynb`
  - Colab notebook for YOLO26s-OBB fine-tuning.
- `web/src/components/DetectionReviewPanel.svelte`
  - detector audit UI.
- `web/src/components/FullPipelineReviewPanel.svelte`
  - full pipeline UI.
- `app/domain/services.py`
  - split detector-only report from full pipeline report.

## Recommended Next Move

Do not keep tuning thresholds first. The failure is mainly domain gap.

Best next action:

1. Build a new YOLO26s-OBB training dataset that includes `test64`-style images and manual boxes.
2. Export the current manual `test64` truth boxes into YOLO OBB format.
3. Add them to training/validation, or create a real-phone fine-tuning split.
4. Retrain YOLO26s-OBB with:
   - `save_period=5` or `save_period=1`
   - maybe lower patience or keep `patience=25`
5. Re-run detector-only audit on test64.

Potential next script:

- `app/scripts/export_test64_yolo_obb_dataset.py`

Possible dataset strategy:

- Existing Products date detection balanced dataset remains base.
- Add test64 manual truth boxes as a real-phone validation/test split or small fine-tune training split.
- If using test64 for training, keep a separate holdout set. Otherwise we will overfit and fool ourselves.

## Emotional/Practical Summary

Recognition is good. Parser is good enough for now. Detection is the wall.

PP-OCRv5 and YOLO both looked good in training but failed on test64-style phone images. YOLO is faster and promising, but needs domain-matched data. The next serious improvement is not more parser work or more recognizer work; it is detector data curation/fine-tuning with real phone expiry-region boxes.
