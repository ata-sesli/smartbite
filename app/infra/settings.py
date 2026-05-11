from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="SMARTBITE_",
        extra="ignore",
        enable_decoding=False,
    )

    app_name: str = "smartbite"
    environment: str = "dev"
    log_level: str = "INFO"
    api_port: int = 8005
    one_shot_timeout_seconds: float = 45.0
    one_shot_poll_interval_seconds: float = 0.5

    database_url: str = "postgresql+asyncpg://smartbite:smartbite@localhost:15432/smartbite"
    redis_url: str = "redis://localhost:16379/0"

    storage_root: Path = Path("data/storage")
    max_upload_size_bytes: int = 10 * 1024 * 1024
    accepted_file_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    accepted_file_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    alert_threshold_days: int = 3
    scan_retry_count: int = 3
    worker_max_jobs: int = 1
    scan_job_expires_seconds: int = 24 * 60 * 60

    detector_model_path: Path = Path("models/yolo20n/yolo26s/yolo26s-best.pt")
    pipeline_high_recall_mode: bool = True
    detector_top_k: int = 4
    detector_box_padding_ratio: float = 0.12
    detector_confidence_threshold: float = 0.15
    parser_min_candidate_confidence: float = 0.40

    mobile_expiry_detector_model_path: Path = Path("models/yolo26s_obb_expdate2k_ft_after_brazil/weights/best.pt")
    mobile_expiry_detector_confidence_threshold: float = 0.05
    mobile_expiry_detector_imgsz: int = 1024
    mobile_expiry_max_candidates: int = 12
    mobile_expiry_crop_padding_px: int = 4

    # Expiry OCR lane: PP-OCRv5 text detection + configurable final recognition.
    text_detector_mode: str = "ensemble"
    expiry_recognizer: str = "svtrv2"
    svtrv2_rec_model_name: str = "ch_SVTRv2_rec"
    svtrv2_rec_model_dir: Path = Path("models/svtrv2/smartbite_svtrv2_expdate_rec")
    svtrv2_device_mode: str = "cpu"
    context_probe_enabled: bool = True
    context_recognizer: str = "svtrv2"
    context_svtrv2_rec_model_name: str = "ch_SVTRv2_rec"
    context_svtrv2_rec_model_dir: Path = Path("models/svtrv2/ch_SVTRv2_rec")
    context_svtrv2_device_mode: str = "cpu"
    context_max_boxes_per_candidate: int = 2
    context_max_candidates_per_scan: int = 20
    # Deprecated runtime backend, retained for benchmarks/comparison.
    parseq_model_dir: Path = Path("models/parseq-small")
    parseq_hf_repo_id: str = "baudm/parseq-small"
    parseq_hf_revision: str = "main"
    # Forced to CPU by default for stability while debugging/fine-tuning PARSeq.
    parseq_device_mode: str = "cpu"
    # Deprecated compatibility knobs from the former PP-OCR recognition lane.
    ocr_ppocrv5_main_model_dir: Path = Path("models/fine-tuned-models/best_model_inference")
    ocr_ppocrv5_main_char_dict_path: Path | None = Path("ppocr/utils/dict/ppocrv5_dict.txt")
    ocr_substitute_config_path: Path = Path("app/ai/ocr_substitute_config.json")
    ocr_enable_substitute_model: bool = False
    ocr_ppocrv5_use_custom_text_det_model: bool = False
    ocr_ppocrv5_text_det_model_name: str = "PP-OCRv5_server_det"
    ocr_ppocrv5_text_det_model_dir: Path | None = Path("models/fine-tuned-models/ppocr-detection-best-model-inference")
    ocr_ppocrv5_use_angle_cls: bool = True
    ocr_ppocrv5_det_db_thresh: float = 0.3
    ocr_ppocrv5_det_db_box_thresh: float = 0.5
    ocr_ppocrv5_det_limit_side_len: int | None = None
    ocr_ppocrv5_det_limit_type: str | None = None
    ocr_ppocrv5_det_db_unclip_ratio: float | None = None
    ocr_paddle_lang: str = "en"
    craft_enabled: bool = True
    craft_model_path: Path = Path("models/craft/craft_mlt_25k.pth")
    craft_text_threshold: float = 0.7
    craft_link_threshold: float = 0.4
    craft_low_text: float = 0.4
    craft_canvas_size: int = 1280
    craft_mag_ratio: float = 1.5
    craft_variant_names: tuple[str, ...] = ("raw", "raw_upscaled", "luma_clahe")
    craft_rescue_enabled: bool = True
    craft_rescue_variant_name: str = "raw"
    craft_rescue_canvas_size: int = 640
    craft_rescue_mag_ratio: float = 1.0
    craft_max_rois_per_scan: int = 1
    craft_max_boxes_accepted: int = 40
    craft_trigger_min_ppocr_boxes: int = 3
    craft_trigger_min_best_score: float = 1.6
    craft_trigger_large_box_area_ratio: float = 0.15
    craft_timeout_seconds: float = 12.0

    expiry_filter_geometry_top_n: int = 12
    expiry_filter_final_top_k: int = 4
    expiry_global_probe_top_k: int = 20
    expiry_global_final_top_k: int = 6
    expiry_global_debug_final_top_k: int = 12
    expiry_global_max_per_roi: int = 2
    expiry_global_max_per_variant: int = 3
    expiry_max_detector_variants: int = 1
    expiry_detector_variants: tuple[str, ...] = ("raw",)
    expiry_max_rois_per_scan: int = 1
    expiry_probe_variants: tuple[str, ...] = ("original_padded", "clahe_gray", "adaptive_binary")
    expiry_parseq_variant_policy: str = "best_probe"
    expiry_parseq_max_variants_per_candidate: int = 1
    expiry_debug_all_recognition_variants: bool = False
    expiry_final_crop_padding_ratio: float = 0.04
    expiry_final_crop_min_padding_px: int = 3
    expiry_final_crop_max_padding_px: int = 18
    expiry_debug_parseq_whole_groups: bool = False
    max_group_candidates_per_roi: int = 100
    expiry_probe_rec_model_name: str = "PP-OCRv5_mobile_rec"
    expiry_probe_mobile_rec_model_dir: Path = Path("models/ppocrv5/mobile_rec")
    expiry_probe_mobile_rec_hf_repo_id: str = "PaddlePaddle/PP-OCRv5_mobile_rec"
    expiry_probe_mobile_rec_hf_revision: str = "main"
    # Stored for future use but intentionally unused in the current pipeline.
    expiry_probe_mobile_det_model_dir: Path = Path("models/ppocrv5/mobile_det")
    expiry_probe_mobile_det_hf_repo_id: str = "PaddlePaddle/PP-OCRv5_mobile_det"
    expiry_probe_mobile_det_hf_revision: str = "main"
    expiry_debug_save_candidates: bool = True
    expiry_debug_save_overlays: bool = True

    # Legacy device mode alias used by previous OCR pipeline components.
    ocr_device_mode: str = "auto"

    # General text OCR lane: PP-OCRv5 visible text extraction (separate sync endpoint)
    general_text_ocr_timeout_seconds: float = 45.0

    model_startup_strict_validation: bool = True

    webhook_stub_url: str = "http://localhost:9999/webhook/alerts"

    @field_validator("ocr_device_mode")
    @classmethod
    def validate_ocr_device_mode(cls, value: str) -> str:
        valid = {"auto", "cpu", "mps", "cuda"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"ocr_device_mode must be one of {sorted(valid)}")
        return normalized

    @field_validator("text_detector_mode")
    @classmethod
    def validate_text_detector_mode(cls, value: str) -> str:
        valid = {"ppocrv5_server", "craft", "ensemble"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"text_detector_mode must be one of {sorted(valid)}")
        return normalized

    @field_validator("expiry_recognizer")
    @classmethod
    def validate_expiry_recognizer(cls, value: str) -> str:
        valid = {"svtrv2", "parseq"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"expiry_recognizer must be one of {sorted(valid)}")
        return normalized

    @field_validator("context_recognizer")
    @classmethod
    def validate_context_recognizer(cls, value: str) -> str:
        valid = {"svtrv2"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"context_recognizer must be one of {sorted(valid)}")
        return normalized

    @field_validator("parseq_device_mode", "svtrv2_device_mode", "context_svtrv2_device_mode")
    @classmethod
    def validate_runtime_device_mode(cls, value: str) -> str:
        valid = {"auto", "cpu", "mps", "cuda"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"device mode must be one of {sorted(valid)}")
        return normalized

    @field_validator(
        "accepted_file_types",
        "accepted_file_extensions",
        "craft_variant_names",
        "expiry_detector_variants",
        "expiry_probe_variants",
        mode="before",
    )
    @classmethod
    def split_csv(cls, value: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(str(part).strip() for part in value if str(part).strip())
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        raise TypeError("value must be tuple[str, ...] or comma-separated string")

    @field_validator("detector_top_k")
    @classmethod
    def validate_detector_top_k(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("expiry_filter_geometry_top_n")
    @classmethod
    def validate_expiry_filter_geometry_top_n(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("expiry_filter_final_top_k")
    @classmethod
    def validate_expiry_filter_final_top_k(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator(
        "expiry_global_final_top_k",
        "expiry_global_probe_top_k",
        "expiry_global_debug_final_top_k",
        "expiry_global_max_per_roi",
        "expiry_global_max_per_variant",
        "expiry_max_detector_variants",
        "expiry_max_rois_per_scan",
        "expiry_parseq_max_variants_per_candidate",
        "max_group_candidates_per_roi",
        "context_max_boxes_per_candidate",
        "context_max_candidates_per_scan",
    )
    @classmethod
    def validate_expiry_global_positive_ints(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("expiry_final_crop_min_padding_px", "expiry_final_crop_max_padding_px")
    @classmethod
    def validate_expiry_final_crop_padding_px(cls, value: int) -> int:
        return max(0, int(value))

    @field_validator("expiry_parseq_variant_policy")
    @classmethod
    def validate_expiry_parseq_variant_policy(cls, value: str) -> str:
        normalized = value.strip().lower()
        valid = {"best_probe", "all"}
        if normalized not in valid:
            raise ValueError(f"expiry_parseq_variant_policy must be one of {sorted(valid)}")
        return normalized

    @field_validator("expiry_final_crop_padding_ratio")
    @classmethod
    def validate_expiry_final_crop_padding_ratio(cls, value: float) -> float:
        return max(0.0, min(float(value), 0.5))

    @field_validator("craft_rescue_variant_name")
    @classmethod
    def validate_craft_rescue_variant_name(cls, value: str) -> str:
        return value.strip() or "raw"

    @field_validator("craft_rescue_canvas_size")
    @classmethod
    def validate_craft_rescue_canvas_size(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("craft_rescue_mag_ratio")
    @classmethod
    def validate_craft_rescue_mag_ratio(cls, value: float) -> float:
        return max(0.1, float(value))

    @field_validator("craft_max_rois_per_scan")
    @classmethod
    def validate_craft_max_rois_per_scan(cls, value: int) -> int:
        return max(0, int(value))

    @field_validator("craft_max_boxes_accepted")
    @classmethod
    def validate_craft_max_boxes_accepted(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("craft_trigger_min_ppocr_boxes")
    @classmethod
    def validate_craft_trigger_min_ppocr_boxes(cls, value: int) -> int:
        return max(0, int(value))

    @field_validator("craft_trigger_large_box_area_ratio")
    @classmethod
    def validate_craft_trigger_large_box_area_ratio(cls, value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @field_validator("craft_timeout_seconds")
    @classmethod
    def validate_craft_timeout_seconds(cls, value: float) -> float:
        return max(0.1, float(value))

    @field_validator("detector_box_padding_ratio")
    @classmethod
    def validate_detector_box_padding_ratio(cls, value: float) -> float:
        return max(0.0, min(float(value), 0.5))

    @field_validator("detector_confidence_threshold")
    @classmethod
    def validate_detector_confidence_threshold(cls, value: float) -> float:
        return max(0.01, min(float(value), 0.99))

    @field_validator("mobile_expiry_detector_confidence_threshold")
    @classmethod
    def validate_mobile_expiry_detector_confidence_threshold(cls, value: float) -> float:
        return max(0.01, min(float(value), 0.99))

    @field_validator("mobile_expiry_detector_imgsz", "mobile_expiry_max_candidates")
    @classmethod
    def validate_mobile_expiry_positive_ints(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator("mobile_expiry_crop_padding_px")
    @classmethod
    def validate_mobile_expiry_crop_padding_px(cls, value: int) -> int:
        return max(0, int(value))

    @field_validator("parser_min_candidate_confidence")
    @classmethod
    def validate_parser_min_candidate_confidence(cls, value: float) -> float:
        return max(0.01, min(float(value), 0.99))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
