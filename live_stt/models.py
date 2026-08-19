"""Model registry.

parakeet.cpp bakes the streaming latency operating point (att_context /
model_chunk_ms) and language/EOU support into the GGUF at conversion time --
none of it is discoverable through the C API -- so it is hardcoded here,
verified against the published GGUF headers and HF model cards during design.

Default is realtime_eou_120m-v1: English only and no punctuation, but it is
the only one of the two with a real <EOU> token, which is what makes the
worker-rotation seam in live_stt/session.py exact (cut at EOU, no fuzzy
word-timestamp dedup needed) and it is ~5x cheaper per stream. nemotron is the
opt-in multilingual path (StreamConfig.language -> stream_begin_lang) and has
no <EOU>/<EOB> vocab entries at all -- eou_id_ stays -1 in parakeet.cpp, so
*eou_out is always 0. Do not build turn detection on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# nemotron leaks a BCP-47-looking language tag into the transcribed text
# itself, e.g. "her eyes. <en-US> It is certainly ...". Confirmed in the
# parakeet.cpp README benchmark output and independently by
# my-meeting-notes/app/services/diarize.py's _LANGUAGE_TAG_RE, which exists
# for the same reason against the same model family.
_LANGUAGE_TAG_RE = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2,4})?>")


def strip_language_tag(text: str) -> str:
    return _LANGUAGE_TAG_RE.sub("", text)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    gguf_filename: str
    model_chunk_ms: int  # baked into the GGUF's att_context; not discoverable via the C API
    has_eou: bool
    has_punctuation: bool
    multilingual: bool
    # RSS growth rate assumed for capacity planning (mb/s of audio fed), from
    # the #63 leak report. Overridden by the measured value from
    # tools/leak_curve.py (Phase 1 Gate A) once it exists -- this default is
    # deliberately conservative (upper end of the 19-41 MB/s reported range)
    # so an un-run Gate A fails safe rather than under-provisioning.
    assumed_leak_mb_per_audio_sec: float
    strip_language_tag: bool = False


MODELS: dict[str, ModelSpec] = {
    "realtime_eou_120m-v1": ModelSpec(
        key="realtime_eou_120m-v1",
        gguf_filename="realtime_eou_120m-v1-q8_0.gguf",
        model_chunk_ms=160,
        has_eou=True,
        has_punctuation=False,
        multilingual=False,
        assumed_leak_mb_per_audio_sec=35.0,
    ),
    "nemotron-3.5-asr-streaming-0.6b": ModelSpec(
        key="nemotron-3.5-asr-streaming-0.6b",
        gguf_filename="nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
        model_chunk_ms=320,
        has_eou=False,
        has_punctuation=True,
        multilingual=True,
        assumed_leak_mb_per_audio_sec=35.0,
        strip_language_tag=True,
    ),
}

DEFAULT_MODEL_KEY = "realtime_eou_120m-v1"


def resolve(model_key: str | None) -> ModelSpec:
    key = model_key or DEFAULT_MODEL_KEY
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {key!r}; available: {sorted(MODELS)}"
        ) from exc
