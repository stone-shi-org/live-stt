"""Post-call speaker diarization via pyannote.audio.

**Batch only, by design, not by current limitation.** Confirmed against the
``pyannote/speaker-diarization-community-1`` model card: the pipeline is
called as ``pipeline("audio.wav")`` or ``pipeline({"waveform": tensor,
"sample_rate": sr})`` and processes a complete clip in one pass -- there is
no incremental/streaming feed API, unlike the ASR worker's
``parakeet_capi_stream_feed``. That is a hard mismatch with this service's
per-chunk-during-the-call architecture (see CLAUDE.md), so this module is
never called from ``session.py``/``servicer.py`` while a call is live. It
runs afterward, against a recorded WAV -- which today means one produced by
some out-of-band capture, since Phase 3's audio-dump ring buffer
(``LSTT_AUDIO_DUMP``) is validated but "has nothing to act on" per
``live_stt/redaction.py``. Wiring an automatic post-call dump -> diarize
pipeline is future work, not part of this module.

**Interface shape is the house standard, not pyannote's native one.** This
house already has one diarization consumer --
``my-meeting-notes/app/services/diarize.py`` -- built against a
LocalAI-compatible ``/v1/audio/diarization`` endpoint returning::

    {"task": "diarize", "num_speakers": 2,
     "segments": [{"id": 0, "speaker": "SPEAKER_00", "label": "0",
                   "start": 0.0, "end": 1.2, "text": "..."}],
     "speakers": [{"id": "SPEAKER_00", "label": "0",
                   "total_speech_duration": 12.3, "segment_count": 4}]}

``diarize_file`` below maps pyannote's native output (a
``pyannote.core.Annotation`` of ``(Segment, track, speaker_label)``
triples -- see ``Annotation.itertracks(yield_label=True)``, and NIST's RTTM
format if you ever need pyannote's own de facto serialization instead of
this house's JSON) into exactly that shape, so a call's diarization result
is structurally interchangeable with every other diarization result already
handled in this house, rather than inventing a second, pyannote-flavored one.

**Verified end-to-end** against a real NOTSOFAR-1 meeting recording (see
CLAUDE.md): CPU-only on a 6-core dev host, 358s of audio took 312.8s
(~0.87x realtime), 5/5 speakers correctly counted, 81.3% frame-level
agreement against real ground truth. ``diarization_device`` (``"cpu"`` |
``"cuda"``) controls whether ``load_pipeline`` moves the pipeline onto a GPU
via ``pipeline.to(torch.device("cuda"))`` -- independent of the ASR worker's
own ``backend`` setting, since this runs as ordinary Python/torch in this
process, not in the C++ worker. Fails loudly at load time
(``torch.cuda.is_available()`` checked explicitly) rather than letting a
missing GPU surface as a confusing failure deep inside the first forward
pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol

from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.pb.livestt.v1 import asr_pb2

logger = get_logger("diarization")


class DiarizationError(RuntimeError):
    """Raised for any diarization failure -- missing dependency, bad/missing
    HF token, gated-model access denied, or a malformed result. Mirrors
    my-meeting-notes' DiarizationError in spirit (one exception type, a
    message good enough to act on) without importing across repos."""


class _SegmentLike(Protocol):
    start: float
    end: float


def load_pipeline(settings: Settings) -> Any:
    """Load the pyannote.audio pipeline. Imports lazily -- pyannote.audio
    pulls in torch/torchaudio (see requirements-diarization.txt), which is
    deliberately NOT a dependency of the always-on grpc.aio server process,
    only of this opt-in, offline tool.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio is not installed. It is deliberately excluded "
            "from requirements.txt (heavy torch/torchaudio dependency for an "
            "offline-only tool) -- install requirements-diarization.txt to "
            "use post-call diarization."
        ) from exc

    if not settings.diarization_hf_token:
        raise DiarizationError(
            "LSTT_DIARIZATION_HF_TOKEN is not set. "
            f"{settings.diarization_model!r} is a gated HuggingFace model "
            "(CC-BY-4.0, requires accepting pyannote's terms) and will not "
            "load without a token."
        )

    try:
        pipeline = Pipeline.from_pretrained(
            settings.diarization_model, token=settings.diarization_hf_token
        )
    except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
        raise DiarizationError(
            f"Failed to load {settings.diarization_model!r}: {exc}"
        ) from exc

    if pipeline is None:
        # pyannote.audio returns None (not an exception) when the caller has
        # not accepted the model's gated-access terms on HuggingFace, which
        # otherwise fails silently downstream on the first real call.
        raise DiarizationError(
            f"Pipeline.from_pretrained({settings.diarization_model!r}) "
            "returned None -- most likely the HF token's account has not "
            "accepted this model's user conditions on huggingface.co."
        )

    device = settings.diarization_device
    if device not in ("cpu", "cuda"):
        raise DiarizationError(f"unknown LSTT_DIARIZATION_DEVICE {device!r}; expected 'cpu' or 'cuda'")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            # Fail loudly here rather than let pyannote silently run on CPU
            # anyway (torch.device("cuda") with no GPU raises deep inside the
            # first forward pass, not at .to() time, with a much less useful
            # traceback) -- same "loud failure over silent wrong behavior"
            # standard as the rest of this module.
            raise DiarizationError(
                "LSTT_DIARIZATION_DEVICE=cuda but torch.cuda.is_available() is "
                "False -- no CUDA-capable GPU/driver visible to this process."
            )
        pipeline.to(torch.device("cuda"))
        logger.info("diarization pipeline moved to cuda")

    return pipeline


def _itertracks(annotation: Any) -> Iterable[tuple[_SegmentLike, str]]:
    """Normalize whatever pyannote's pipeline call returns into
    ``(segment, speaker_label)`` pairs.

    Handles both shapes seen in pyannote's own docs for this model: a plain
    ``Annotation`` (iterate via ``itertracks(yield_label=True)``, which
    yields ``(segment, track, label)`` triples) and the 4.x pipeline-output
    wrapper exposing ``.speaker_diarization`` as that same Annotation.
    """
    source = getattr(annotation, "speaker_diarization", annotation)
    if hasattr(source, "itertracks"):
        for segment, _track, label in source.itertracks(yield_label=True):
            yield segment, label
    else:
        # Already an iterable of (segment, label) pairs -- e.g. a test
        # double, or a future pyannote version whose top-level object is
        # directly iterable that way.
        for segment, label in source:
            yield segment, label


def annotation_to_house_json(
    annotation: Any, *, model: str
) -> dict[str, Any]:
    """Pure mapping from pyannote's diarization result to the house JSON
    shape (see module docstring). No I/O, no pyannote import at call time --
    testable against a fake ``itertracks``-shaped object with no pyannote.audio
    installed, mirroring live_stt/events.py's "pure mapping" pattern.
    """
    segments: list[dict[str, Any]] = []
    speaker_stats: dict[str, dict[str, Any]] = {}

    for i, (segment, label) in enumerate(_itertracks(annotation)):
        speaker_id = f"SPEAKER_{label}" if not str(label).startswith("SPEAKER_") else str(label)
        segments.append(
            {
                "id": i,
                "speaker": speaker_id,
                "label": str(label),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": "",
            }
        )
        stats = speaker_stats.setdefault(
            speaker_id, {"id": speaker_id, "label": str(label), "total_speech_duration": 0.0, "segment_count": 0}
        )
        stats["total_speech_duration"] += float(segment.end) - float(segment.start)
        stats["segment_count"] += 1

    # Deterministic order (by first appearance) rather than dict/set
    # iteration order -- matters for anything downstream that compares two
    # runs, e.g. a regression test.
    speakers = list(speaker_stats.values())

    if not segments:
        raise DiarizationError(
            "Diarization returned no segments -- is the audio silent, or "
            "shorter than the model's minimum window?"
        )

    return {
        "task": "diarize",
        "model": model,
        "num_speakers": len(speakers),
        "segments": segments,
        "speakers": speakers,
    }


def assign_text(house_json: dict[str, Any], words: list[asr_pb2.Word]) -> dict[str, Any]:
    """Fill each segment's ``text`` from the call's own ASR word timestamps
    (``asr_pb2.Word.start_sec``/``end_sec``, the same shape ``session.py``
    already produces), by assigning each word to the diarization segment its
    midpoint falls inside.

    This is the equivalent of the LocalAI service's ``include_text=true`` --
    but here diarization and transcription are two separate steps run by two
    separate engines (pyannote for speakers, parakeet.cpp for words) rather
    than one model doing both, so the merge has to happen on this side. A
    word whose midpoint doesn't fall in any segment (a gap between diarized
    turns) is dropped rather than guessed onto the nearest one -- silently
    misattributing a word to the wrong speaker is worse than omitting it.
    """
    segments = house_json["segments"]
    buckets: list[list[str]] = [[] for _ in segments]

    for word in words:
        midpoint = (word.start_sec + word.end_sec) / 2.0
        for i, seg in enumerate(segments):
            if seg["start"] <= midpoint < seg["end"]:
                buckets[i].append(word.text)
                break

    for seg, bucket in zip(segments, buckets):
        seg["text"] = " ".join(bucket)
    return house_json


def diarize_file(
    path: str | Path,
    *,
    settings: Settings,
    words: list[asr_pb2.Word] | None = None,
) -> dict[str, Any]:
    """Run diarization on a recorded WAV and return the house JSON shape.

    ``words``, if given, are merged in via ``assign_text`` -- pass the same
    call's ASR transcript's word list (``asr_pb2.Word``) to get per-segment
    text, the same as a LocalAI-backed diarization call would return with
    ``include_text=true``. Omit it to get speaker turns with no text, which
    is a valid partial result, not an error (unlike the LocalAI client, which
    treats all-empty text as a symptom of a misconfigured request -- here an
    empty ``words`` list is a deliberate, valid input, not a mistake).
    """
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"No such audio file: {path}")

    pipeline = load_pipeline(settings)
    kwargs: dict[str, Any] = {}
    if settings.diarization_num_speakers is not None:
        kwargs["num_speakers"] = settings.diarization_num_speakers

    try:
        result = pipeline(str(path), **kwargs)
    except Exception as exc:  # noqa: BLE001 -- one clear error type at this boundary
        raise DiarizationError(f"Diarization failed on {path}: {exc}") from exc

    house_json = annotation_to_house_json(result, model=settings.diarization_model)
    if words:
        house_json = assign_text(house_json, words)

    logger.info(
        "diarized %s: %d segments, %d speakers",
        path.name,
        len(house_json["segments"]),
        house_json["num_speakers"],
    )
    return house_json
