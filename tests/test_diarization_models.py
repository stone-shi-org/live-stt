from __future__ import annotations

import pytest

from live_stt.diarization_models import (
    DEFAULT_DIARIZATION_MODEL_KEY,
    DIARIZATION_MODELS,
    DiarizationModelSpec,
    resolve,
)


def test_default_key_is_registered():
    assert DEFAULT_DIARIZATION_MODEL_KEY in DIARIZATION_MODELS


def test_default_entry_is_the_real_measured_community_1_model():
    spec = DIARIZATION_MODELS[DEFAULT_DIARIZATION_MODEL_KEY]
    assert spec.hf_repo_id == "pyannote/speaker-diarization-community-1"
    assert spec.gated is True
    assert spec.supports_num_speakers_hint is True
    # The real measurement from CLAUDE.md's duration-scaling entry.
    assert spec.measured_peak_vram_mb == 11424


def test_resolve_none_returns_the_default():
    assert resolve(None).key == DEFAULT_DIARIZATION_MODEL_KEY


def test_resolve_known_key_returns_its_spec():
    spec = resolve("pyannote/speaker-diarization-community-1")
    assert spec.hf_repo_id == "pyannote/speaker-diarization-community-1"


def test_resolve_unknown_key_raises_with_available_keys_listed():
    with pytest.raises(KeyError, match="unknown diarization model"):
        resolve("not-a-real-model")
    # KeyError's str() includes surrounding quotes from repr() -- match
    # the substring directly instead of over-fitting to that formatting.
    try:
        resolve("not-a-real-model")
    except KeyError as exc:
        assert "pyannote/speaker-diarization-community-1" in str(exc)


def test_spec_is_frozen():
    spec = resolve(None)
    with pytest.raises(AttributeError):
        spec.gated = False  # type: ignore[misc]


def test_spec_dataclass_shape_has_no_surprise_fields():
    # A cheap guard against silent field drift -- if this fails after
    # adding a field, update it deliberately rather than by accident.
    assert set(DiarizationModelSpec.__dataclass_fields__) == {
        "key",
        "hf_repo_id",
        "gated",
        "supports_num_speakers_hint",
        "measured_peak_vram_mb",
    }
