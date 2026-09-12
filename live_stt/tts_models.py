"""TTS (Text-to-Speech) model registry -- mirrors ``live_stt/models.py`` and
``live_stt/diarization_models.py``.

Provides metadata and validation for supported TTS models. Default model is
``kokoro-82m-v1`` (Apache-2.0, ~82M parameters, fast inference, high quality,
small VRAM footprint), with ``piper-en-us`` (Piper ONNX, CPU-only fallback).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TTSModelSpec:
    key: str
    hf_repo_id: str
    engine: str  # "kokoro" | "piper"
    sample_rate: int  # e.g. 24000 for Kokoro, 22050 for Piper
    default_voice: str
    supported_voices: tuple[str, ...]
    supports_streaming: bool
    measured_peak_vram_mb: int | None = None


TTS_MODELS: dict[str, TTSModelSpec] = {
    "kokoro-82m-v1": TTSModelSpec(
        key="kokoro-82m-v1",
        hf_repo_id="hexgrad/Kokoro-82M",
        engine="kokoro",
        sample_rate=24000,
        default_voice="af_heart",
        supported_voices=(
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_aoede",
            "af_kore",
            "af_sarah",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ),
        supports_streaming=True,
        measured_peak_vram_mb=1200,
    ),
    "piper-en-us": TTSModelSpec(
        key="piper-en-us",
        hf_repo_id="rhasspy/piper-voices",
        engine="piper",
        sample_rate=22050,
        default_voice="en_US-lessac-medium",
        supported_voices=(
            "en_US-lessac-medium",
            "en_US-amy-medium",
            "en_US-danny-low",
            "en_US-ryan-medium",
        ),
        supports_streaming=True,
        measured_peak_vram_mb=0,
    ),
}

DEFAULT_TTS_MODEL_KEY = "kokoro-82m-v1"


def resolve(model_key: str | None) -> TTSModelSpec:
    key = model_key or DEFAULT_TTS_MODEL_KEY
    try:
        return TTS_MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown TTS model {key!r}; available: {sorted(TTS_MODELS)}"
        ) from exc
