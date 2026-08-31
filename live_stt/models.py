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

Second engine: whisper.cpp (`worker/third_party/whisper.cpp`, vendored
alongside parakeet.cpp, statically linked into a SEPARATE worker binary --
`live_stt_worker_whisper`, see worker/CMakeLists.txt), added for the
whisper-large-v3-turbo-q8_0-and-family use case. Architecturally
incompatible with parakeet.cpp's streaming C API (different file format --
whisper.cpp uses its own legacy `ggml-*.bin` format, NOT gguf, confirmed
against the real ggerganov/whisper.cpp HF repo listing -- and no incremental
encoder/decoder state: examples/stream/stream.cpp in that repo just re-runs
whisper_full() over a manually reconstructed sliding window, not a real
carried state). `ModelSpec.gguf_filename` is a misnomer for these entries --
it holds a `.bin` path, not a GGUF one -- kept as one field rather than
splitting it because it is only ever used as "the model file name to join
onto models_dir", which doesn't care about the actual format. `engine` and
`streaming_capable` are what actually distinguish the two families:
`streaming_capable=False` models are batch-only (see
live_stt/servicer.py's gate, which rejects them on the streaming gRPC
Transcribe RPC with INVALID_ARGUMENT -- they are reachable only via
POST /v1/audio/transcriptions) and never rotate mid-call (see
live_stt/session.py -- whisper's worker only emits a transcript at
finalize(), so a rotation that SIGKILLs it mid-buffer would silently drop
whatever hadn't been transcribed yet; rotation is unconditionally skipped
for these instead of risking that).
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
    # so an un-run Gate A fails safe rather than under-provisioning. Not
    # really meaningful for engine="whisper" (a one-shot batch process --
    # #63 is a parakeet.cpp-specific streaming-session leak), kept at 0.0
    # for those entries rather than inventing a number nothing measures.
    assumed_leak_mb_per_audio_sec: float
    strip_language_tag: bool = False
    # Which worker binary/IPC contract this model needs -- see
    # live_stt/worker.py and live_stt/session.py's _spawn_worker. Additive
    # field: both existing entries above default to "parakeet" unchanged.
    engine: str = "parakeet"  # "parakeet" | "whisper"
    # False => batch-only: no gRPC Transcribe (live_stt/servicer.py aborts
    # INVALID_ARGUMENT before spawning a worker), reachable only via
    # POST /v1/audio/transcriptions, and never rotates mid-call
    # (live_stt/session.py skips the rotation-trigger check entirely) --
    # see this module's docstring for why.
    streaming_capable: bool = True


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
    # --- whisper.cpp family (batch-only, see this module's docstring) -----
    # Files/checksums verified live against the real ggerganov/whisper.cpp
    # HF repo listing while building this (see scripts/fetch_model.sh) --
    # q8_0 does not exist for every size upstream (large-v3 only has q5_0,
    # confirmed: ggml-large-v3-q8_0.bin 404s there), so that entry uses the
    # closest available quantization instead of the ~3GB unquantized file.
    "whisper-base.en-q8_0": ModelSpec(
        key="whisper-base.en-q8_0",
        gguf_filename="ggml-base.en-q8_0.bin",
        model_chunk_ms=1000,  # not baked into the model like Parakeet -- just an IPC buffering size
        has_eou=False,
        has_punctuation=True,
        multilingual=False,
        assumed_leak_mb_per_audio_sec=0.0,
        engine="whisper",
        streaming_capable=False,
    ),
    "whisper-small-q8_0": ModelSpec(
        key="whisper-small-q8_0",
        gguf_filename="ggml-small-q8_0.bin",
        model_chunk_ms=1000,
        has_eou=False,
        has_punctuation=True,
        multilingual=True,
        assumed_leak_mb_per_audio_sec=0.0,
        engine="whisper",
        streaming_capable=False,
    ),
    "whisper-medium-q8_0": ModelSpec(
        key="whisper-medium-q8_0",
        gguf_filename="ggml-medium-q8_0.bin",
        model_chunk_ms=1000,
        has_eou=False,
        has_punctuation=True,
        multilingual=True,
        assumed_leak_mb_per_audio_sec=0.0,
        engine="whisper",
        streaming_capable=False,
    ),
    "whisper-large-v3-q5_0": ModelSpec(
        key="whisper-large-v3-q5_0",
        gguf_filename="ggml-large-v3-q5_0.bin",
        model_chunk_ms=1000,
        has_eou=False,
        has_punctuation=True,
        multilingual=True,
        assumed_leak_mb_per_audio_sec=0.0,
        engine="whisper",
        streaming_capable=False,
    ),
    "whisper-large-v3-turbo-q8_0": ModelSpec(
        key="whisper-large-v3-turbo-q8_0",
        gguf_filename="ggml-large-v3-turbo-q8_0.bin",
        model_chunk_ms=1000,
        has_eou=False,
        has_punctuation=True,
        multilingual=True,
        assumed_leak_mb_per_audio_sec=0.0,
        engine="whisper",
        streaming_capable=False,
    ),
}

DEFAULT_MODEL_KEY = "realtime_eou_120m-v1"
# Not wired into Settings.default_model (that stays realtime_eou_120m-v1,
# the streaming default) -- this exists purely so other code/docs/tests have
# one canonical whisper key to point at without repeating the string. This
# is also the exact model the user asked to add support for.
DEFAULT_WHISPER_MODEL_KEY = "whisper-large-v3-turbo-q8_0"


def resolve(model_key: str | None) -> ModelSpec:
    key = model_key or DEFAULT_MODEL_KEY
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {key!r}; available: {sorted(MODELS)}"
        ) from exc
