from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_prefix="SMARTBITE_", extra="ignore")

    app_name: str = "smartbite"
    environment: str = "dev"
    log_level: str = "INFO"
    api_port: int = 8005
    one_shot_timeout_seconds: float = 30.0
    one_shot_poll_interval_seconds: float = 0.5

    database_url: str = "postgresql+asyncpg://smartbite:smartbite@localhost:15432/smartbite"
    redis_url: str = "redis://localhost:16379/0"

    storage_root: Path = Path("data/storage")
    max_upload_size_bytes: int = 10 * 1024 * 1024
    accepted_file_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    accepted_file_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

    alert_threshold_days: int = 3
    scan_retry_count: int = 3

    detector_model_path: Path = Path("yolo26n.pt")
    # Easiest model switch point for backend OCR:
    # SMARTBITE_OCR_PPOCRV5_MAIN_MODEL_DIR in .env / docker-compose.
    ocr_ppocrv5_main_model_dir: Path = Path("models/fine-tuned-models/best_model_inference")
    ocr_ppocrv5_main_char_dict_path: Path | None = Path("ppocr/utils/dict/ppocrv5_dict.txt")
    ocr_ppocrv5_use_angle_cls: bool = True
    ocr_ppocrv5_det_db_thresh: float = 0.3
    ocr_ppocrv5_det_db_box_thresh: float = 0.5
    # Note: in PaddleOCR 3.x TextRecognition with explicit local model_dir,
    # `lang` may be ignored by runtime.
    ocr_paddle_lang: str = "en"
    ocr_substitute_config_path: Path = Path("app/ai/ocr_substitute_config.json")
    ocr_enable_substitute_model: bool = False

    # auto | cpu | mps | cuda
    ocr_device_mode: str = "auto"

    webhook_stub_url: str = "http://localhost:9999/webhook/alerts"

    @field_validator("ocr_device_mode")
    @classmethod
    def validate_ocr_device_mode(cls, value: str) -> str:
        valid = {"auto", "cpu", "mps", "cuda"}
        normalized = value.lower().strip()
        if normalized not in valid:
            raise ValueError(f"ocr_device_mode must be one of {sorted(valid)}")
        return normalized

    @field_validator("accepted_file_types", "accepted_file_extensions", mode="before")
    @classmethod
    def split_csv(cls, value: tuple[str, ...] | str) -> tuple[str, ...]:
        if isinstance(value, tuple):
            return value
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        raise TypeError("value must be tuple[str, ...] or comma-separated string")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
