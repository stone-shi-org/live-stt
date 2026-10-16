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
        "vram_mb",
        "engine",
        "max_speakers_supported",
    }


def test_pyannote_entry_defaults_engine_to_pyannote_and_no_speaker_cap():
    # The pre-existing entry's construction call never set engine=/
    # max_speakers_supported= -- confirms both new fields' defaults
    # preserve it unchanged, per LS-4's "additive default" design.
    spec = DIARIZATION_MODELS[DEFAULT_DIARIZATION_MODEL_KEY]
    assert spec.engine == "pyannote"
    assert spec.max_speakers_supported is None


def test_sortformer_entry_is_registered():
    spec = resolve("nvidia/diar_streaming_sortformer_4spk-v2.1")
    assert spec.hf_repo_id == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert spec.engine == "nemo_sortformer"
    # Confirmed via the HF API during the LS-4 spike -- not gated, unlike
    # the pyannote entry.
    assert spec.gated is False
    # End-to-end model, no clustering step to hand a hint to -- confirmed
    # during the LS-4 spike, not merely "unwired".
    assert spec.supports_num_speakers_hint is False
    # Hard cap confirmed during the LS-4 spike (degrades/silently caps at
    # 5+ real speakers).
    assert spec.max_speakers_supported == 4
    # Rough floor from the LS-4 GPU spike, not a calibrated peak -- see
    # that field's own docstring.
    assert spec.measured_peak_vram_mb == 1402


def test_default_diarization_model_is_unaffected_by_the_new_entry():
    # Sortformer is opt-in via model=, not a change to what an unqualified
    # diarization request does.
    assert DEFAULT_DIARIZATION_MODEL_KEY == "pyannote/speaker-diarization-community-1"
    assert resolve(None).engine == "pyannote"
