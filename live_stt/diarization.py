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

**Subprocess isolation**: By default (``diarize_in_subprocess=True``),
``diarize_file`` spawns an isolated Python subprocess (``live_stt.diarize_worker``)
to run pyannote inference. This guarantees 100% VRAM / CUDA driver context
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
    """Load the pyannote.audio pipeline. Imports lazily -- pyannote.audio
    pulls in torch/torchaudio (see requirements-diarization.txt), which is
    deliberately NOT a dependency of the always-on grpc.aio server process,
    only of this opt-in, offline tool.

    ``settings.diarization_model`` is a registry key
    (``live_stt/diarization_models.py``), resolved and validated FIRST --
    before even attempting the pyannote.audio import -- so an unknown model
    name fails fast and cheap regardless of whether the heavy dependency is
    installed, the same "validate before touching the engine" order ASR's
    ``models.resolve()`` already uses.
    """
    try:
        spec = resolve_diarization_model(settings.diarization_model)
    except KeyError as exc:
        raise DiarizationError(str(exc)) from exc

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


def annotation_to_house_json(
    annotation: Any, *, model: str
) -> dict[str, Any]:
    """Pure mapping from pyannote's diarization result to the house JSON
    shape.
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


def _diarize_file_direct(
    path: str | Path,
    *,
    settings: Settings,
) -> dict[str, Any]:
    """Run in-process pyannote inference directly (called by subprocess worker)."""
    path = Path(path)
    if not path.is_file():
        raise DiarizationError(f"No such audio file: {path}")

    pipeline = load_pipeline(settings)
    spec = resolve_diarization_model(settings.diarization_model)
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
    finally:
        if settings.diarization_device == "cuda":
            _release_cuda_memory()

    return annotation_to_house_json(result, model=spec.hf_repo_id)


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
