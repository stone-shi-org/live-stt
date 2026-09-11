"""Post-call speaker diarization -- pyannote.audio (default) or, since
LS-4, NVIDIA Sortformer v2.1, selected per-request via the
``diarization_model`` registry key (``live_stt/diarization_models.py``).

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
pipeline is future work, not part of this module. Sortformer's own engine
IS internally streaming-capable, but is run here the same fully-batched way
(one ``diarize()`` call over the whole file) for consistency -- wiring true
inline diarization into a live call is a bigger, separate change, deferred
per the LS-4 plan.

**Two engines, one house JSON shape.** ``load_pipeline``/``_diarize_file_direct``
dispatch on ``DiarizationModelSpec.engine`` (``"pyannote"`` or
``"nemo_sortformer"``); each engine's own loader/runner
(``_load_pyannote_pipeline``/``_run_pyannote`` vs.
``_load_sortformer_model``/``_run_sortformer``) normalizes its native output
into the same flat ``(start, end, speaker_label)`` shape before
``_segments_to_house_json`` builds the wire contract both share -- see that
function's docstring. This keeps ``my-meeting-notes/app/services/diarize.py``'s
``{task, model, num_speakers, segments, speakers}`` contract byte-identical
regardless of which engine actually ran.

**Subprocess isolation**: By default (``diarize_in_subprocess=True``),
``diarize_file`` spawns an isolated Python subprocess (``live_stt.diarize_worker``)
to run inference. This guarantees 100% VRAM / CUDA driver context
reclamation on process termination, preventing persistent CUDA driver context
(~300-400MB) or memory fragmentation from occupying the card after inference ends.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Protocol

from live_stt.config import Settings
from live_stt.diarization_models import resolve as resolve_diarization_model
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
    """Load the configured diarization engine's model object -- a pyannote
    ``Pipeline`` or a nemo ``SortformerEncLabelModel``, dispatched on
    ``DiarizationModelSpec.engine``.

    ``settings.diarization_model`` is a registry key
    (``live_stt/diarization_models.py``), resolved and validated FIRST --
    before either engine's heavy import is even attempted -- so an unknown
    model name fails fast and cheap regardless of whether either dependency
    is installed, the same "validate before touching the engine" order
    ASR's ``models.resolve()`` already uses.
    """
    try:
        spec = resolve_diarization_model(settings.diarization_model)
    except KeyError as exc:
        raise DiarizationError(str(exc)) from exc

    if spec.engine == "nemo_sortformer":
        return _load_sortformer_model(spec, settings)
    return _load_pyannote_pipeline(spec, settings)


def _load_pyannote_pipeline(spec: Any, settings: Settings) -> Any:
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

    if spec.gated and not settings.diarization_hf_token:
        raise DiarizationError(
            "LSTT_DIARIZATION_HF_TOKEN is not set. "
            f"{spec.hf_repo_id!r} is a gated HuggingFace model "
            "(CC-BY-4.0, requires accepting pyannote's terms) and will not "
            "load without a token."
        )

    try:
        pipeline = Pipeline.from_pretrained(spec.hf_repo_id, token=settings.diarization_hf_token)
    except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
        raise DiarizationError(f"Failed to load {spec.hf_repo_id!r}: {exc}") from exc

    if pipeline is None:
        # pyannote.audio returns None (not an exception) when the caller has
        # not accepted the model's gated-access terms on HuggingFace, which
        # otherwise fails silently downstream on the first real call.
        raise DiarizationError(
            f"Pipeline.from_pretrained({spec.hf_repo_id!r}) "
            "returned None -- most likely the HF token's account has not "
            "accepted this model's user conditions on huggingface.co."
        )

    device = _check_device(settings)
    if device == "cuda":
        import torch

        pipeline.to(torch.device("cuda"))
        logger.info("diarization pipeline moved to cuda")

    return pipeline


def _load_sortformer_model(spec: Any, settings: Settings) -> Any:
    """Load NVIDIA Sortformer v2.1 (or a future nemo-engine entry) via
    ``nemo.collections.asr.models.SortformerEncLabelModel``. Imports
    lazily -- ``nemo_toolkit[asr]`` is deliberately excluded from both
    requirements.txt and requirements-diarization.txt (see
    requirements-diarization-sortformer.txt's own comment: its dependency
    tree is meaningfully heavier than pyannote.audio's own, and this is a
    second, more niche opt-in engine on an already opt-in feature).

    Not gated (confirmed via the HF API during the LS-4 spike), so unlike
    ``_load_pyannote_pipeline`` there is no token check here -- if a future
    nemo-engine entry ever IS gated, this will need one.
    """
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        raise DiarizationError(
            "nemo_toolkit[asr] is not installed. It is deliberately excluded "
            "from requirements.txt and requirements-diarization.txt (an even "
            "heavier dependency tree than pyannote.audio's own, for a "
            "second, opt-in engine) -- install "
            "requirements-diarization-sortformer.txt to use "
            f"{spec.hf_repo_id!r}."
        ) from exc

    device = _check_device(settings)

    try:
        model = SortformerEncLabelModel.from_pretrained(spec.hf_repo_id)
    except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
        raise DiarizationError(f"Failed to load {spec.hf_repo_id!r}: {exc}") from exc

    model.eval()

    if device == "cuda":
        model = model.cuda()
        logger.info("sortformer diarization model moved to cuda")

    return model


def _check_device(settings: Settings) -> str:
    """Validate ``settings.diarization_device`` and, for "cuda", confirm a
    GPU is actually visible before either engine's loader touches torch
    any further. Shared by both engines -- same "loud failure over silent
    wrong behavior" reasoning either way: ``torch.device("cuda")`` with no
    GPU raises deep inside the first forward pass, not at load/``.to()``
    time, with a much less useful traceback.
    """
    device = settings.diarization_device
    if device not in ("cpu", "cuda"):
        raise DiarizationError(f"unknown LSTT_DIARIZATION_DEVICE {device!r}; expected 'cpu' or 'cuda'")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise DiarizationError(
                "LSTT_DIARIZATION_DEVICE=cuda but torch.cuda.is_available() is "
                "False -- no CUDA-capable GPU/driver visible to this process."
            )
    return device


def _release_cuda_memory() -> None:
    """Return this request's CUDA memory to the driver instead of letting
    PyTorch's caching allocator hold it for reuse."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _itertracks(annotation: Any) -> Iterable[tuple[_SegmentLike, str]]:
    """Normalize whatever pyannote's pipeline call returns into
    ``(segment, speaker_label)`` pairs.
    """
    source = getattr(annotation, "speaker_diarization", annotation)
    if hasattr(source, "itertracks"):
        for segment, _track, label in source.itertracks(yield_label=True):
            yield segment, label
    else:
        for segment, label in source:
            yield segment, label


def _segments_to_house_json(
    segments_in: Iterable[tuple[float, float, str]], *, model: str
) -> dict[str, Any]:
    """Pure mapping from a flat ``(start, end, speaker_label)`` sequence to
    the house JSON shape -- shared by every diarization engine.
    ``annotation_to_house_json`` normalizes pyannote's ``Annotation``-shaped
    output into this via ``_itertracks`` first; ``_run_sortformer`` parses
    Sortformer's plain ``"start end speaker_label"`` strings straight into
    it. Both land here so the wire contract
    (``my-meeting-notes/app/services/diarize.py``'s ``{task, model,
    num_speakers, segments, speakers}`` shape) stays identical regardless
    of which engine actually produced the segments.
    """
    segments: list[dict[str, Any]] = []
    speaker_stats: dict[str, dict[str, Any]] = {}

    for i, (start, end, label) in enumerate(segments_in):
        speaker_id = f"SPEAKER_{label}" if not str(label).startswith("SPEAKER_") else str(label)
        segments.append(
            {
                "id": i,
                "speaker": speaker_id,
                "label": str(label),
                "start": float(start),
                "end": float(end),
                "text": "",
            }
        )
        stats = speaker_stats.setdefault(
            speaker_id, {"id": speaker_id, "label": str(label), "total_speech_duration": 0.0, "segment_count": 0}
        )
        stats["total_speech_duration"] += float(end) - float(start)
        stats["segment_count"] += 1

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


def annotation_to_house_json(
    annotation: Any, *, model: str
) -> dict[str, Any]:
    """Pure mapping from pyannote's diarization result to the house JSON
    shape.
    """
    return _segments_to_house_json(
        (
            (float(segment.start), float(segment.end), label)
            for segment, label in _itertracks(annotation)
        ),
        model=model,
    )


def assign_text(house_json: dict[str, Any], words: list[asr_pb2.Word]) -> dict[str, Any]:
    """Fill each segment's ``text`` from the call's own ASR word timestamps."""
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


def _run_pyannote(pipeline: Any, path: Path, *, settings: Settings, spec: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if spec.supports_num_speakers_hint:
        if settings.diarization_num_speakers is not None:
            kwargs["num_speakers"] = settings.diarization_num_speakers
        else:
            if settings.diarization_min_speakers is not None:
                kwargs["min_speakers"] = settings.diarization_min_speakers
            if settings.diarization_max_speakers is not None:
                kwargs["max_speakers"] = settings.diarization_max_speakers

    try:
        result = pipeline(str(path), **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(f"Diarization failed on {path}: {exc}") from exc

    return annotation_to_house_json(result, model=spec.hf_repo_id)


def _parse_sortformer_segments(raw_segments: list[str]) -> list[tuple[float, float, str]]:
    """Parse Sortformer's ``"start_sec end_sec speaker_label"`` string
    format -- e.g. ``"6.720 8.400 speaker_0"``, confirmed against a real
    run of ``nvidia/diar_streaming_sortformer_4spk-v2.1`` during the LS-4
    spike. ``diarize()``'s public API returns ``List[List[str]]``, one
    inner list per input audio file -- callers pass a single-file batch
    and index ``result[0]``.
    """
    parsed: list[tuple[float, float, str]] = []
    for line in raw_segments:
        parts = line.split()
        if len(parts) != 3:
            raise DiarizationError(f"unexpected Sortformer segment format: {line!r}")
        start_str, end_str, label = parts
        try:
            parsed.append((float(start_str), float(end_str), label))
        except ValueError as exc:
            raise DiarizationError(f"unexpected Sortformer segment format: {line!r}") from exc
    return parsed


def _run_sortformer(model: Any, path: Path, *, spec: Any) -> dict[str, Any]:
    try:
        result = model.diarize(audio=[str(path)], batch_size=1)
    except Exception as exc:  # noqa: BLE001
        raise DiarizationError(f"Diarization failed on {path}: {exc}") from exc

    if not result or not result[0]:
        raise DiarizationError(
            "Diarization returned no segments -- is the audio silent, or "
            "shorter than the model's minimum window?"
        )

    segments = _parse_sortformer_segments(result[0])
    house_json = _segments_to_house_json(segments, model=spec.hf_repo_id)

    # Sortformer hard-caps at spec.max_speakers_supported and fails
    # SILENTLY when the real audio has more speakers than that -- no error,
    # no warning of its own (confirmed during the LS-4 spike: one real
    # speaker's speech gets absorbed into another's cluster instead of
    # surfacing as a 5th cluster or an exception). detected_count >= cap is
    # only a heuristic -- a legitimate call with exactly `cap` speakers
    # looks identical -- but it's the best signal available without
    # engine-level ground truth, and an honest "might be capped" warning
    # beats silence.
    if spec.max_speakers_supported is not None and house_json["num_speakers"] >= spec.max_speakers_supported:
        house_json.setdefault("warnings", []).append(
            f"detected speaker count ({house_json['num_speakers']}) is at or above this "
            f"model's supported maximum ({spec.max_speakers_supported}); if the real audio "
            "has more speakers, some may have been silently merged into another speaker's "
            "segments rather than reported separately -- see CLAUDE.md's LS-4 Sortformer entry"
        )

    return house_json


def _diarize_file_direct(
    path: str | Path,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Run in-process diarization inference directly (called by the
    subprocess worker), dispatching to the configured engine's runner.
    """
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"No such audio file: {path}")

    engine_obj = load_pipeline(settings)
    spec = resolve_diarization_model(settings.diarization_model)

    try:
        if spec.engine == "nemo_sortformer":
            return _run_sortformer(engine_obj, path, spec=spec)
        return _run_pyannote(engine_obj, path, settings=settings, spec=spec)
    finally:
        if settings.diarization_device == "cuda":
            _release_cuda_memory()


def _diarize_file_subprocess(
    path: str | Path,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Spawn isolated python subprocess to execute diarization for full CUDA VRAM reclamation."""
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"No such audio file: {path}")

    with tempfile.TemporaryDirectory(prefix="diarize_worker_") as tmpdir:
        req_file = Path(tmpdir) / "request.json"
        res_file = Path(tmpdir) / "response.json"

        req_payload = {
            "wav_path": str(path.resolve()),
            "settings": settings.model_dump(),
        }
        req_file.write_text(json.dumps(req_payload), encoding="utf-8")

        cmd = [sys.executable, "-m", "live_stt.diarize_worker", str(req_file), str(res_file)]
        env = dict(os.environ)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
        except Exception as exc:
            raise DiarizationError(f"Failed to spawn diarization subprocess: {exc}") from exc

        if not res_file.is_file():
            stderr_snippet = proc.stderr.strip() or proc.stdout.strip()
            raise DiarizationError(
                f"Diarization worker crashed or exited with code {proc.returncode}: {stderr_snippet}"
            )

        try:
            res_data = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DiarizationError(f"Failed to parse diarization worker response: {exc}") from exc

        if not res_data.get("ok"):
            err_msg = res_data.get("error", "Unknown diarization error")
            raise DiarizationError(err_msg)

        return res_data["result"]


def diarize_file(
    path: str | Path,
    *,
    settings: Settings,
    words: list[asr_pb2.Word] | None = None,
    diarize_in_subprocess: bool | None = None,
) -> dict[str, Any]:
    """Run diarization on a recorded WAV and return the house JSON shape.

    By default, uses the ``settings.diarize_in_subprocess`` configuration value
    (defaulting to True), isolating pyannote and PyTorch into a short-lived child
    process so all CUDA / VRAM memory is 100% freed upon completion.
    """
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"No such audio file: {path}")

    use_subprocess = (
        diarize_in_subprocess
        if diarize_in_subprocess is not None
        else getattr(settings, "diarize_in_subprocess", True)
    )

    if use_subprocess:
        house_json = _diarize_file_subprocess(path, settings=settings)
    else:
        house_json = _diarize_file_direct(path, settings=settings)

    if words:
        house_json = assign_text(house_json, words)

    logger.info(
        "diarized %s: %d segments, %d speakers",
        path.name,
        len(house_json["segments"]),
        house_json["num_speakers"],
    )
    return house_json
