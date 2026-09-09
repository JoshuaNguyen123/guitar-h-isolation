from __future__ import annotations

import os

# StemSplitio hosts the htdemucs_6s ONNX used for guitar isolation.
HF_REPO_ID = "StemSplitio/htdemucs-6s-onnx"
HF_FILENAME_FP32 = "htdemucs_6s.onnx"
HF_FILENAME_FP16 = "htdemucs_6s_fp16weights.onnx"

_CONFIGURED = False


def configure_fast_hf() -> None:
    """Speed up Hugging Face Xet chunk downloads. Must run before hub import."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "32")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _CONFIGURED = True


def _cached_path(filename: str) -> str | None:
    configure_fast_hf()
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(HF_REPO_ID, filename)
    if path is None:
        return None
    return str(path)


def preferred_precision() -> str:
    """Reuse whatever variant is already on disk; otherwise download the smaller one."""
    if _cached_path(HF_FILENAME_FP32):
        return "fp32"
    if _cached_path(HF_FILENAME_FP16):
        return "fp16weights"
    return "fp16weights"


def model_is_cached(precision: str | None = None) -> bool:
    precision = precision or preferred_precision()
    filename = HF_FILENAME_FP16 if precision == "fp16weights" else HF_FILENAME_FP32
    return _cached_path(filename) is not None


def prefetch_separator_model() -> str:
    """Download the guitar model now so Generate is not waiting on Hub chunks.

    Returns ``cached``, ``downloaded``, or raises.
    """
    configure_fast_hf()
    from demucs_onnx._hub import download_single_model

    precision = preferred_precision()
    already = model_is_cached(precision)
    download_single_model("htdemucs_6s", precision=precision)  # type: ignore[arg-type]
    return "cached" if already else "downloaded"
