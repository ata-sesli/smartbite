from __future__ import annotations


def resolve_runtime_device(mode: str) -> str:
    explicit = mode.lower()
    if explicit in {"cpu", "mps", "cuda"}:
        return explicit

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
