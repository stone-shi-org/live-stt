"""HTTP surface for one-shot batch ASR: ``POST /v1/audio/transcriptions``.

Wired into ``live_stt/admin_http.py``'s existing ``ThreadingHTTPServer`` --
not a new server, not the gRPC path. Unlike diarization (a genuinely
separate, offline-only Python/torch engine -- see ``live_stt/diarization.py``),
this endpoint reuses the REAL production ASR path: ``live_stt/session.py``'s
``CallSession`` over a real spawned ``live_stt/worker.py``'s ``WorkerHandle``,
the exact same code ``servicer.Transcribe`` drives for a gRPC call. A single
HTTP request is treated as one short-lived "call" -- admitted through the
same shared ``WorkerBudget`` a gRPC call uses (see the note in
``live_stt/admission.py`` about why that object needed a lock added for
this), so a flood of HTTP transcription requests cannot spawn worker
processes past what this box was actually sized for, independent of how
busy the gRPC side is at the same moment. Uploaded audio longer than
``rotate_after_sec`` even gets ``CallSession``'s normal worker-rotation
safety net for free, for the same reason.

**Request/response shape**: the real consumer this was built for is
``my-meeting-notes/app/routers/live_caption.py``'s
``channel_worker_transcriptions`` (the "reinstated stateless per-chunk POST
backend" for a deployment with no realtime pipeline model) -- confirmed
against its actual code, not guessed: multipart fields ``file`` (a WAV),
``model``, ``stream`` (that client always sends ``"true"``), and optional
``language``; on ``stream=true`` it expects a Server-Sent-Events response
where every ``data:`` line is JSON, only ``{"type": "transcript.text.done",
"text": ...}`` is read, and a literal ``data: [DONE]`` line ends the stream.
This endpoint's ``stream=true`` path emits exactly that -- one
``transcript.text.done`` event carrying the FULL transcript, then
``[DONE]`` -- not genuine low-latency incremental streaming (no
``transcript.text.delta`` events mid-request): the whole uploaded file is
processed via ``CallSession`` first, and only then is the one SSE event
written. Sufficient for that consumer (which only ever reads the ``.done``
event) and far simpler to implement/verify than real incremental streaming,
but do not build an interface expecting partial captions mid-request on top
of this -- it does not do that. ``stream`` unset/``"false"`` returns a plain
JSON body instead (OpenAI Whisper-API-shaped: ``{"text": ...}``, or with
``response_format=verbose_json``: ``{"task", "language", "duration", "text",
"words"}``).

**Audio format**: only 16kHz mono 16-bit PCM WAV is accepted -- the same
restriction ``servicer.Transcribe`` enforces on the gRPC path ("unsupported
encoding ... decode/resample client-side, see live_stt.client.telephony"),
kept consistent here rather than adding a resampling dependency to this
admin-adjacent HTTP surface.

**Not verified**: this module has not been exercised against a real
``httpx``-generated request from an actual my-meeting-notes checkout, nor
against the real worker binary/model at all (no ``LSTT_MODEL_PATH``/model
file available while writing it) -- only unit-tested with a fake
``CallSession``. Confirm against a real worker and a real my-meeting-notes
client before trusting this on real traffic, the same standard the rest of
this repo's phases were held to.
"""

from __future__ import annotations

import io
import json
import uuid
import wave
from typing import Any

from live_stt import gpu, metrics, models
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.multipart import MultipartError, parse_multipart_form
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.session import CallSession
from live_stt.transcribe_sessions import TranscribeSessionTracker
from live_stt.worker import WorkerError

logger = get_logger("transcribe_http")

TRANSCRIBE_PATH = "/v1/audio/transcriptions"


class TranscribeBadRequestError(RuntimeError):
    """The request itself was invalid -- bad WAV, unknown model. Maps to 400."""


class TranscribeError(RuntimeError):
    """The engine failed partway through transcription. Maps to 500."""


class TranscribeUnavailableError(TranscribeError):
    """The engine could not be started/used for this request at all --
    spawn failure. Maps to 503, not 500: the request itself was fine, the
    service just couldn't serve it right now."""


def _pcm16_mono_16k_from_wav(wav_bytes: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            rate, channels, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
            if (rate, channels, width) != (16000, 1, 2):
                raise TranscribeBadRequestError(
                    f"unsupported WAV format: {rate}Hz, {channels}ch, {width * 8}-bit -- "
                    "only 16kHz mono 16-bit PCM is accepted (decode/resample client-side, "
                    "see live_stt.client.telephony)"
                )
            return wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise TranscribeBadRequestError(f"could not parse WAV file: {exc}") from exc


def _accumulate(
    event: asr_pb2.TranscriptionEvent, text_parts: list[str], words: list[asr_pb2.Word]
) -> tuple[float, int] | None:
    """Mirrors servicer.py's _count_words: each delta/final event's `text`
    is an INCREMENTAL delta, not the whole-call transcript -- see
    CallSession.finalize()'s docstring. Returns (total_audio_sec,
    worker_generations) when a `final` event is seen, else None.
    """
    kind = event.WhichOneof("event")
    if kind == "delta":
        text_parts.append(event.delta.text)
        words.extend(event.delta.words)
        return None
    if kind == "final":
        text_parts.append(event.final.text)
        words.extend(event.final.words)
        return event.final.total_audio_sec, event.final.worker_generations
    return None


async def _run_transcription(
    pcm: bytes,
    *,
    settings: Settings,
    spec: models.ModelSpec,
    language: str | None,
    budget: WorkerBudget,
) -> tuple[str, list[asr_pb2.Word], float, int]:
    config = asr_pb2.StreamConfig(
        call_id=f"http-transcribe-{uuid.uuid4().hex[:12]}",
        encoding=asr_pb2.AUDIO_ENCODING_LINEAR16,
        sample_rate_hz=16000,
        model=spec.key,
        language=language or "",
    )
    session = CallSession(settings, spec, config, budget)
    try:
        await session.start()
    except WorkerError as exc:
        raise TranscribeUnavailableError(f"engine failed to start: {exc}") from exc

    text_parts: list[str] = []
    words: list[asr_pb2.Word] = []
    total_audio_sec = 0.0
    worker_generations = 1
    try:
        for event in await session.feed_audio(pcm):
            result = _accumulate(event, text_parts, words)
            if result:
                total_audio_sec, worker_generations = result
        for event in await session.finalize():
            result = _accumulate(event, text_parts, words)
            if result:
                total_audio_sec, worker_generations = result
    except WorkerError as exc:
        raise TranscribeError(f"engine error during transcription: {exc}") from exc
    finally:
        await session.close()

    return "".join(text_parts), words, total_audio_sec, worker_generations


def _err(message: str) -> bytes:
    return json.dumps({"error": {"message": message}}).encode("utf-8")


async def handle_transcribe_request(
    *,
    content_type: str,
    body: bytes,
    settings: Settings,
    budget: WorkerBudget,
    draining: bool,
    tracker: TranscribeSessionTracker,
) -> tuple[int, bytes, str]:
    """Pure(ish) request handler -- no socket coupling beyond what's already
    unavoidable (spawning a real worker process). Returns
    ``(http_status, response_body, content_type)``; never raises.
    """
    if draining:
        return 503, _err("server is draining, not accepting new work"), "application/json"

    try:
        fields = parse_multipart_form(content_type, body)
    except MultipartError as exc:
        return 400, _err(str(exc)), "application/json"

    if not fields.get("file"):
        return 400, _err("missing required multipart field 'file'"), "application/json"

    try:
        pcm = _pcm16_mono_16k_from_wav(fields["file"])
    except TranscribeBadRequestError as exc:
        return 400, _err(str(exc)), "application/json"

    model_field = fields.get("model")
    model_key = model_field.decode("utf-8", "replace") if model_field else None
    try:
        spec = models.resolve(model_key)
    except KeyError as exc:
        return 400, _err(str(exc)), "application/json"

    response_format = fields.get("response_format", b"json").decode("utf-8", "replace")
    if response_format not in ("json", "verbose_json"):
        return (
            400,
            _err(f"unsupported response_format {response_format!r}; expected 'json' or 'verbose_json'"),
            "application/json",
        )

    stream = fields.get("stream", b"false").decode("utf-8", "replace").strip().lower() == "true"
    language_field = fields.get("language")
    language = language_field.decode("utf-8", "replace") if language_field else None

    # A pre-existing gap, closed here: live_stt/servicer.py's gRPC Transcribe
    # has always checked free VRAM before admitting a call on a cuda
    # backend (a CUDA allocation failure is an abort(), not a catchable
    # exception -- see CLAUDE.md), but this HTTP endpoint never did, even
    # though it spawns the exact same kind of worker process through the
    # exact same CallSession/WorkerHandle path. Harmless while every
    # registered model was parakeet-family and CPU-heavy HTTP traffic was
    # presumably rare, but now that the whisper family (Phase 6's CUDA
    # addition) is reachable ONLY through this endpoint, 100% of a
    # whisper-on-GPU deployment's traffic would otherwise bypass this gate
    # entirely. Mirrors servicer.py's check exactly, including running it
    # before the cheaper call-slot check below.
    if settings.backend == "cuda":
        free = gpu.free_vram_mb()
        if free is not None:
            metrics.gpu_free_vram_mb.set(free)
        required = settings.vram_per_worker_mb + settings.vram_reserve_mb
        if free is not None and free < required:
            tracker.record_rejected_vram()
            metrics.transcribe_requests_total.labels(outcome="rejected_vram").inc()
            return (
                503,
                _err(f"insufficient VRAM: {free}MB free, {required}MB required"),
                "application/json",
            )

    if not budget.try_admit_call():
        tracker.record_rejected_capacity()
        metrics.transcribe_requests_total.labels(outcome="rejected_capacity").inc()
        return 503, _err("at capacity, try again shortly"), "application/json"

    # Dashboard/metrics visibility only (see live_stt/transcribe_sessions.py) --
    # separate from budget's own admission slot, which this request already
    # holds regardless of this tracker's bookkeeping.
    request_id = tracker.start(model=spec.key)
    metrics.transcribe_sessions_active.inc()
    try:
        text, words, total_audio_sec, worker_generations = await _run_transcription(
            pcm, settings=settings, spec=spec, language=language, budget=budget
        )
    except TranscribeUnavailableError as exc:
        tracker.finish(request_id, ok=False)
        metrics.transcribe_sessions_active.dec()
        metrics.transcribe_requests_total.labels(outcome="failed").inc()
        return 503, _err(str(exc)), "application/json"
    except TranscribeError as exc:
        tracker.finish(request_id, ok=False)
        metrics.transcribe_sessions_active.dec()
        metrics.transcribe_requests_total.labels(outcome="failed").inc()
        logger.exception("transcription failed")
        return 500, _err(str(exc)), "application/json"
    except Exception as exc:  # noqa: BLE001 -- last-resort boundary; never crash the admin thread
        tracker.finish(request_id, ok=False)
        metrics.transcribe_sessions_active.dec()
        metrics.transcribe_requests_total.labels(outcome="failed").inc()
        logger.exception("unexpected transcription failure")
        return 500, _err(f"internal error: {exc}"), "application/json"
    else:
        tracker.finish(request_id, ok=True)
        metrics.transcribe_sessions_active.dec()
        metrics.transcribe_requests_total.labels(outcome="ok").inc()
    finally:
        budget.release_call()

    logger.info(
        "http-transcribe model=%s worker_generations=%d audio_sec=%.1f words=%d",
        spec.key,
        worker_generations,
        total_audio_sec,
        len(words),
    )

    if stream:
        # Exactly what my-meeting-notes' channel_worker_transcriptions reads
        # (see module docstring): one transcript.text.done event with the
        # full text, then a literal [DONE] line. Not incremental -- the
        # whole file was already fully transcribed above.
        payload = (
            f"data: {json.dumps({'type': 'transcript.text.done', 'text': text})}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        return 200, payload, "text/event-stream"

    doc: dict[str, Any]
    if response_format == "verbose_json":
        doc = {
            "task": "transcribe",
            "language": language or "",
            "duration": total_audio_sec,
            "text": text,
            "words": [{"word": w.text, "start": w.start_sec, "end": w.end_sec} for w in words],
        }
    else:
        doc = {"text": text}
    return 200, json.dumps(doc).encode("utf-8"), "application/json"
