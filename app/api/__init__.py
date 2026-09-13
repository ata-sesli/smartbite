from app.api.admin import metrics
from app.api.alerts import process_alerts
from app.api.expiry import list_expiry
from app.api.health import health
from app.api.history import list_scan_history
from app.api.mobile_expiry import correct_mobile_expiry_scan, create_mobile_expiry_scan
from app.api.reviews import export_training_data, get_scan_review, put_scan_review
from app.api.scans import (
    create_one_shot_scan,
    create_scan,
    get_manual_crop_recognition_review,
    get_manual_crop_recognition_review_asset,
    get_mobile_oracle_forensic_audit_asset,
    get_scan,
    get_scan_image,
    get_scan_roi,
    get_test64_mobile_oracle_forensic_audit,
    get_test64_detection_review,
    get_test64_full_pipeline_results,
    get_test64_full_pipeline_review,
    get_test64_image,
    get_test64_product_cropper_review,
    list_test64_truth_bboxes,
    list_test64_labels,
    patch_scan,
    put_test64_label,
    put_test64_truth_bbox,
    queue_test_images_standalone,
)

ROUTES = [
    create_mobile_expiry_scan,
    correct_mobile_expiry_scan,
    list_scan_history,
    get_scan_review,
    put_scan_review,
    export_training_data,
    create_scan,
    queue_test_images_standalone,
    list_test64_labels,
    put_test64_label,
    get_test64_image,
    get_test64_detection_review,
    get_test64_product_cropper_review,
    get_test64_full_pipeline_results,
    get_test64_full_pipeline_review,
    get_test64_mobile_oracle_forensic_audit,
    get_mobile_oracle_forensic_audit_asset,
    get_manual_crop_recognition_review,
    get_manual_crop_recognition_review_asset,
    list_test64_truth_bboxes,
    put_test64_truth_bbox,
    create_one_shot_scan,
    get_scan,
    get_scan_image,
    get_scan_roi,
    patch_scan,
    list_expiry,
    process_alerts,
    health,
    metrics,
]
