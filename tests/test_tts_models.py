from __future__ import annotations

import pytest

from live_stt.tts_models import (
    DEFAULT_TTS_MODEL_KEY,
    TTS_MODELS,
    TTSModelSpec,
    resolve,
)


def test_default_key_is_registered() -> None:
    assert DEFAULT_TTS_MODEL_KEY in TTS_MODELS


def test_default_entry_is_kokoro() -> None:
    spec = TTS_MODELS[DEFAULT_TTS_MODEL_KEY]
    assert spec.hf_repo_id == "hexgrad/Kokoro-82M"
    assert spec.engine == "kokoro"
    assert spec.sample_rate == 24000
    assert spec.default_voice == "af_heart"
    assert spec.supports_streaming is True
    assert spec.measured_peak_vram_mb == 1200


def test_piper_entry_is_registered() -> None:
    spec = resolve("piper-en-us")
    assert spec.engine == "piper"
    assert spec.sample_rate == 22050
    assert spec.default_voice == "en_US-lessac-medium"
    assert spec.supports_streaming is True
    assert spec.measured_peak_vram_mb == 0


def test_resolve_none_returns_default() -> None:
    assert resolve(None).key == DEFAULT_TTS_MODEL_KEY


def test_resolve_unknown_key_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown TTS model"):
        resolve("non-existent-tts-model")


def test_spec_is_frozen() -> None:
    spec = resolve(None)
    with pytest.raises(AttributeError):
        spec.sample_rate = 16000  # type: ignore[misc]


def test_spec_dataclass_shape() -> None:
    assert set(TTSModelSpec.__dataclass_fields__) == {
        "key",
        "hf_repo_id",
        "engine",
        "sample_rate",
        "default_voice",
        "supported_voices",
        "supports_streaming",
        "measured_peak_vram_mb",
    }
