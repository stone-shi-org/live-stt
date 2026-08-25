"""Diarization model registry -- mirrors ``live_stt/models.py``'s ASR
registry pattern.

Before this existed, ``Settings.diarization_model`` was a bare string
passed straight to ``pyannote.audio.Pipeline.from_pretrained`` with no
validation at all -- unlike ASR's ``live_stt.models.resolve()``, which
rejects an unknown key with a clear error before ever touching the engine.
This closes that gap the same way: a known-keys dict plus a ``resolve()``
that raises ``KeyError`` (wrapped as ``DiarizationError`` by
``live_stt/diarization.py::load_pipeline``) on anything else.

Where ``live_stt/models.py`` hardcodes parakeet.cpp's per-GGUF latency/
language metadata (undiscoverable via its C API), this hardcodes what's
known about each diarization pipeline: whether it's gated on HuggingFace,
whether it accepts a ``num_speakers`` hint, and the one real GPU VRAM
measurement this codebase has made so far (see CLAUDE.md's "why 12GB" /
duration-scaling entries) -- informative metadata, not yet wired into a
per-model admission threshold. ``Settings.diarization_vram_mb`` stays one
global, operator-tunable value used at admission time regardless of which
model is selected; a per-model lookup isn't worth the complexity with only
one entry in this registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiarizationModelSpec:
    key: str
    hf_repo_id: str  # passed verbatim to pyannote.audio.Pipeline.from_pretrained
    gated: bool  # requires an accepted-terms HF token -- see Settings.diarization_hf_token
    # Accepts num_speakers=/min_speakers=/max_speakers= as __call__ kwargs --
    # one flag for all three since they're the same pyannote pipeline kwarg
    # surface (see pyannote.audio.pipelines.speaker_diarization.apply()); a
    # model that doesn't take one almost certainly doesn't take any of them.
    # False for a model withholds ALL THREE, not just the exact hint.
    supports_num_speakers_hint: bool
    # Real measurement, one file, one box (RTX 3090) -- the highest of the
    # 6/10/20-minute peaks in CLAUDE.md's duration-scaling entry (the
    # 40-minute result was reproducibly LOWER, not a worse case; see that
    # entry for why this doesn't just take the max across all four). None
    # for a model nobody has measured yet -- not a promise of "small".
    measured_peak_vram_mb: int | None = None


DIARIZATION_MODELS: dict[str, DiarizationModelSpec] = {
    "pyannote/speaker-diarization-community-1": DiarizationModelSpec(
        key="pyannote/speaker-diarization-community-1",
        hf_repo_id="pyannote/speaker-diarization-community-1",
        gated=True,
        supports_num_speakers_hint=True,
        measured_peak_vram_mb=11424,
    ),
}

DEFAULT_DIARIZATION_MODEL_KEY = "pyannote/speaker-diarization-community-1"


def resolve(model_key: str | None) -> DiarizationModelSpec:
    key = model_key or DEFAULT_DIARIZATION_MODEL_KEY
    try:
        return DIARIZATION_MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown diarization model {key!r}; available: {sorted(DIARIZATION_MODELS)}"
        ) from exc
