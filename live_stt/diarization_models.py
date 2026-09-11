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
one pyannote entry needing it.

**Second engine, NVIDIA Sortformer (LS-4)**: unlike ASR's parakeet/whisper
split (a ggml C++ worker binary per engine), diarization engines are both
plain Python/torch running in this process (or the subprocess spawned by
``diarize_in_subprocess``) -- so a second engine here is a registry entry
plus a dispatch branch in ``live_stt/diarization.py``, not a new binary.
See that module's docstring and CLAUDE.md's LS-4 entry for the full spike
writeup (license, real install path, accuracy/VRAM measurements on both
CPU and GPU) this entry is built from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiarizationModelSpec:
    key: str
    hf_repo_id: str  # passed verbatim to Pipeline.from_pretrained / SortformerEncLabelModel.from_pretrained
    gated: bool  # requires an accepted-terms HF token -- see Settings.diarization_hf_token
    # Accepts num_speakers=/min_speakers=/max_speakers= as __call__ kwargs --
    # one flag for all three since they're the same pyannote pipeline kwarg
    # surface (see pyannote.audio.pipelines.speaker_diarization.apply()); a
    # model that doesn't take one almost certainly doesn't take any of them.
    # False for a model withholds ALL THREE, not just the exact hint. Also
    # False for Sortformer, but for a different reason: it's a genuinely
    # end-to-end model with no clustering step to hand a hint to at all, not
    # merely an unwired kwarg (confirmed during the LS-4 spike).
    supports_num_speakers_hint: bool
    # Real measurement, one file, one box (RTX 3090) -- the highest of the
    # 6/10/20-minute peaks in CLAUDE.md's duration-scaling entry (the
    # 40-minute result was reproducibly LOWER, not a worse case; see that
    # entry for why this doesn't just take the max across all four). None
    # for a model nobody has measured yet -- not a promise of "small".
    # Sortformer's 1402 MiB is from the LS-4 GPU spike and is explicitly
    # flagged there as a rough floor, not a calibrated peak -- both fixtures
    # finished in ~1-1.4s, too fast for the 0.5s-interval sampler used to
    # catch more than ~3 readings.
    measured_peak_vram_mb: int | None = None
    # Which loader/runner path live_stt/diarization.py dispatches to.
    # "pyannote" (default, preserves the one pre-existing entry unchanged)
    # or "nemo_sortformer". Mirrors ASR's ModelSpec.engine field/reasoning.
    engine: str = "pyannote"
    # Hard cap on detected speakers, when the engine has one (Sortformer:
    # 4, confirmed during the LS-4 spike -- performance degrades at 5+ and
    # the model silently returns at most this many clusters, no error, no
    # warning of its own). None for a model with no such cap (pyannote's
    # clustering has no fixed ceiling). live_stt/diarization.py uses this to
    # attach an advisory "warnings" entry to the house JSON when detected
    # count reaches the cap -- a heuristic, not proof extra speakers were
    # actually dropped (a real N-speaker call at exactly the cap looks
    # identical), but the best signal available without engine-level
    # ground truth.
    max_speakers_supported: int | None = None


DIARIZATION_MODELS: dict[str, DiarizationModelSpec] = {
    "pyannote/speaker-diarization-community-1": DiarizationModelSpec(
        key="pyannote/speaker-diarization-community-1",
        hf_repo_id="pyannote/speaker-diarization-community-1",
        gated=True,
        supports_num_speakers_hint=True,
        measured_peak_vram_mb=11424,
    ),
    # NVIDIA Sortformer v2.1 (LS-4) -- streaming-capable end-to-end neural
    # diarizer, run here in fully-batch mode (one diarize() call over the
    # whole recorded WAV) since true inline/live diarization during an
    # active call is a materially bigger change than this entry, deferred
    # to a future ticket per the LS-4 plan's own scoping. NOT gated on
    # HuggingFace (confirmed via the HF API during the spike) and licensed
    # under the NVIDIA Open Model License Agreement (commercially usable,
    # confirmed against the actual PDF text, not a summary -- see
    # CLAUDE.md). Requires requirements-diarization-sortformer.txt, kept
    # separate from requirements-diarization.txt -- nemo_toolkit's
    # dependency tree is meaningfully heavier than pyannote.audio's own.
    "nvidia/diar_streaming_sortformer_4spk-v2.1": DiarizationModelSpec(
        key="nvidia/diar_streaming_sortformer_4spk-v2.1",
        hf_repo_id="nvidia/diar_streaming_sortformer_4spk-v2.1",
        gated=False,
        supports_num_speakers_hint=False,
        measured_peak_vram_mb=1402,
        engine="nemo_sortformer",
        max_speakers_supported=4,
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
