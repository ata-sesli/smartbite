from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from app.ai.runtime_device import resolve_runtime_device


PARSEQ_CHARSET_94 = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

logger = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - runtime dependency path
        raise RuntimeError(
            "OpenCV is required for PARSeq preprocessing. "
            "Install opencv runtime libs (e.g. libGL) or run PARSeq outside this container."
        ) from exc
    return cv2


@dataclass(slots=True)
class ParSeqOutput:
    text: str
    confidence: float | None
    reason: str | None


class ParSeqRecognizer:
    def __init__(self, *, model_dir: Path, device_mode: str = "auto", charset: str = PARSEQ_CHARSET_94) -> None:
        self.model_dir = model_dir
        self.runtime_device = resolve_runtime_device(device_mode)
        self.requested_device_mode = device_mode
        self.charset = charset
        self._model = None
        self._device = None
        self._load_error: str | None = None
        self._model_file: Path | None = None
        self._backend: str = "unknown"
        self._mps_fallback_triggered = False

    def _resolve_model_file(self) -> Path | None:
        candidates = (
            self.model_dir / "pytorch_model.bin",
            self.model_dir / "model.safetensors",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        if not self.model_dir.exists() or not self.model_dir.is_dir():
            self._load_error = f"PARSeq model directory not found: {self.model_dir}"
            return

        model_file = self._resolve_model_file()
        if model_file is None:
            self._load_error = (
                f"PARSeq model file missing under {self.model_dir} "
                "(expected pytorch_model.bin or model.safetensors)"
            )
            return
        self._model_file = model_file

        try:
            import torch

            if self.runtime_device == "cuda":
                device = torch.device("cuda")
            elif self.runtime_device == "mps":
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            # Build eager PARSeq model code and load state_dict weights.
            # This path is suitable for fine-tuning/debug workflows.
            try:
                model = torch.hub.load(
                    "baudm/parseq",
                    "parseq",
                    pretrained=False,
                    trust_repo=True,
                    skip_validation=True,
                )
            except TypeError:
                # Older torch versions may not support trust_repo/skip_validation args.
                model = torch.hub.load("baudm/parseq", "parseq", pretrained=False)

            if model_file.name.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file as load_safetensors
                except Exception as exc:  # pragma: no cover - optional dependency path
                    raise RuntimeError(
                        "safetensors weights detected but safetensors is not installed. "
                        "Install `safetensors` or provide pytorch_model.bin."
                    ) from exc
                state_dict = load_safetensors(str(model_file), device="cpu")
            else:
                state_dict = torch.load(str(model_file), map_location="cpu")

            if not isinstance(state_dict, dict):
                raise RuntimeError("PARSeq state_dict is invalid (expected dict)")

            target_model = model
            sample_key = next(iter(state_dict.keys()), "")
            if hasattr(model, "model") and isinstance(sample_key, str) and not sample_key.startswith("model."):
                target_model = model.model

            missing, unexpected = target_model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("PARSeq state_dict missing keys: %s", missing[:10])
            if unexpected:
                logger.warning("PARSeq state_dict unexpected keys: %s", unexpected[:10])

            model.to(device)
            model.eval()
            self._model = model
            self._device = device
            self._backend = "eager_pytorch"
        except Exception as exc:  # pragma: no cover - runtime dependency path
            self._load_error = f"failed to initialize PARSeq recognizer: {exc}"

    @staticmethod
    def _prepare_tensor(image: np.ndarray) -> np.ndarray:
        cv2 = _require_cv2()
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_h, target_w = 32, 128
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((3, target_h, target_w), dtype=np.float32)

        scale = min(target_w / float(w), target_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        y_offset = max(0, (target_h - new_h) // 2)
        x_offset = max(0, (target_w - new_w) // 2)
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        normalized = canvas.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        return np.transpose(normalized, (2, 0, 1))

    def _decode_with_model_tokenizer(self, probs) -> tuple[str | None, float | None]:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "decode"):
            return None, None

        try:
            labels, confidences = tokenizer.decode(probs)
            if isinstance(labels, list) and labels:
                label = str(labels[0])
            else:
                label = str(labels)

            score: float | None = None
            if isinstance(confidences, list) and confidences:
                score = float(confidences[0])
            elif confidences is not None:
                score = float(confidences)
            return label.strip(), score
        except Exception:
            return None, None

    def _decode_fallback(self, probs) -> tuple[str, float | None]:
        # PARSeq logits: [B, T, C] where class index 0 is EOS and 1..N map
        # to charset symbols.
        best = probs.argmax(dim=-1)[0].tolist()
        char_scores = probs.max(dim=-1).values[0].tolist()

        chars: list[str] = []
        scores: list[float] = []
        for idx, score in zip(best, char_scores, strict=False):
            if idx == 0:
                break
            char_idx = idx - 1
            if 0 <= char_idx < len(self.charset):
                chars.append(self.charset[char_idx])
                scores.append(float(score))

        text = "".join(chars).strip()
        confidence = float(sum(scores) / len(scores)) if scores else None
        return text, confidence

    def _infer(self, image: np.ndarray) -> ParSeqOutput:
        import torch

        input_arr = self._prepare_tensor(image)
        tensor = torch.from_numpy(input_arr).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probs = logits.softmax(-1)

        text, confidence = self._decode_with_model_tokenizer(probs)
        if text is None:
            text, confidence = self._decode_fallback(probs)

        if not text:
            return ParSeqOutput(text="", confidence=None, reason="parseq returned empty text")
        return ParSeqOutput(text=text, confidence=confidence, reason=None)

    def _switch_to_cpu(self) -> None:
        self.runtime_device = "cpu"
        self._model = None
        self._device = None
        self._load_error = None
        self._backend = "unknown"

    @staticmethod
    def _is_mps_runtime_error(exc: Exception) -> bool:
        message = str(exc)
        return "mps" in message.lower() or "Placeholder storage has not been allocated on MPS device" in message

    def runtime_info(self) -> dict[str, object]:
        return {
            "model_dir": str(self.model_dir),
            "model_file": str(self._model_file) if self._model_file else None,
            "requested_device_mode": self.requested_device_mode,
            "runtime_device": self.runtime_device,
            "backend": self._backend,
            "charset_length": len(self.charset),
            "model_loaded": self._model is not None and self._device is not None,
            "mps_fallback_triggered": self._mps_fallback_triggered,
        }

    def recognize(self, image: np.ndarray) -> ParSeqOutput:
        self._ensure_loaded()

        if self._load_error is not None:
            return ParSeqOutput(text="", confidence=None, reason=self._load_error)
        if self._model is None or self._device is None:
            return ParSeqOutput(text="", confidence=None, reason="PARSeq recognizer unavailable")

        try:
            return self._infer(image)
        except Exception as exc:  # pragma: no cover - runtime inference path
            if self.runtime_device == "mps" and self._is_mps_runtime_error(exc):
                self._mps_fallback_triggered = True
                logger.warning("PARSeq MPS inference failed; retrying on CPU. reason=%s", exc)
                self._switch_to_cpu()
                self._ensure_loaded()
                if self._load_error is not None or self._model is None or self._device is None:
                    return ParSeqOutput(
                        text="",
                        confidence=None,
                        reason=f"parseq mps failure and cpu fallback load failed: {self._load_error or exc}",
                    )
                try:
                    output = self._infer(image)
                    if output.reason is None:
                        logger.warning("PARSeq CPU fallback succeeded after MPS failure.")
                    return output
                except Exception as cpu_exc:  # pragma: no cover - runtime inference path
                    return ParSeqOutput(
                        text="",
                        confidence=None,
                        reason=f"parseq inference failed on mps ({exc}) and cpu fallback ({cpu_exc})",
                    )
            return ParSeqOutput(text="", confidence=None, reason=f"parseq inference failed: {exc}")
