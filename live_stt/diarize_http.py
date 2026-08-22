"""HTTP surface for post-call diarization: ``POST /v1/audio/diarization``.

Wired into ``live_stt/admin_http.py``'s existing ``ThreadingHTTPServer``
(``admin_host``/``admin_port``) -- not a new server. This is a low-traffic,
opt-in, admin-adjacent surface, not part of the hot gRPC call path. A
pyannote run can take seconds to minutes; ``ThreadingHTTPServer`` already
gives each request its own thread, so one slow diarize request only ties up
its own thread, never the health/stats handlers or (more importantly) the
gRPC event loop, which lives in a different process/thread space entirely
(see ``admin_http.py``'s own module docstring: the admin surface runs on a
daemon thread specifically so nothing here can ever stall the stream pump).

**Path and request shape deliberately match
``my-meeting-notes/app/services/diarize.py``'s LocalAI-compatible client
exactly** (POST multipart/form-data to ``.../v1/audio/diarization``, fields
``file``/``model``/``include_text``/``response_format``) so that existing
client can point ``MMN_DIARIZATION_URL`` at a live-stt instance's admin port
and just work, with no changes on that side. Two fields are live-stt-specific
extensions, not part of that client today: ``words`` (a JSON array of
``{"text", "start_sec", "end_sec"}`` -- the call's own ASR transcript word
timestamps, same shape as ``asr_pb2.Word``) and ``num_speakers``.

**Known, deliberate incompatibility**: my-meeting-notes' client always sends
``include_text=true`` and expects real per-segment transcript text, because
its other backends (LocalAI, vibevoice-cpp-asr) are single models that
diarize AND transcribe in one pass. Here diarization (pyannote) and
transcription (parakeet.cpp) are two separate engines -- this endpoint
cannot produce transcript text from audio alone. A caller that doesn't
supply ``words`` gets back real speaker segments with EMPTY text, which
``diarize_sync()`` on the other end already treats as
``DiarizationError("...ignored include_text=true or the model does not
support transcription.")`` -- an honest, correctly-typed failure, not a
silent wrong answer, but a real caller wanting text-per-segment through this
endpoint MUST also pass ``words``.

**Multipart parsing lives in ``live_stt/multipart.py``** (shared with
``live_stt/transcribe_http.py``) -- see that module's docstring for why this
doesn't use ``cgi.FieldStorage`` and how it was verified binary-safe.
``MultipartError``/``parse_multipart_form`` are re-exported here (imported
into this module's namespace below) purely so existing call sites/tests that
say ``diarize_http.parse_multipart_form`` keep working unchanged.

**Not verified**: this whole module has not been exercised against a real
``httpx``/browser multipart request from an actual my-meeting-notes
checkout (only a hand-built one and a synthetic binary-safety check) --
though the real end-to-end run recorded in CLAUDE.md DID confirm the full
request -> response path against a real pyannote model over a real socket.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

from live_stt import gpu, metrics
from live_stt.config import Settings
from live_stt.diarization import DiarizationError, diarize_file
from live_stt.diarize_sessions import DiarizationSessionTracker
from live_stt.logging_config import get_logger
from live_stt.multipart import MultipartError, parse_multipart_form
from live_stt.pb.livestt.v1 import asr_pb2

logger = get_logger("diarize_http")

DIARIZE_PATH = "/v1/audio/diarization"

__all__ = ["DIARIZE_PATH", "MultipartError", "handle_diarize_request", "parse_multipart_form"]


def _parse_words(raw: bytes | None) -> list[asr_pb2.Word]:
    if not raw:
        return []
    try:
        items = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise MultipartError(f"'words' field is not valid JSON: {exc}") from exc
    try:
        return [
            asr_pb2.Word(text=w["text"], start_sec=float(w["start_sec"]), end_sec=float(w["end_sec"]))
            for w in items
        ]
    except (KeyError, TypeError) as exc:
        raise MultipartError(f"'words' field entries must each have text/start_sec/end_sec: {exc}") from exc


def handle_diarize_request(
    *, content_type: str, body: bytes, settings: Settings, tracker: DiarizationSessionTracker
) -> tuple[int, dict[str, Any]]:
    """Pure(ish) request handler -- no socket/HTTP-server coupling, so this
    is directly unit-testable (mirrors live_stt/events.py's pure-mapping
    philosophy: the actual I/O-touching bits -- reading the socket,
    invoking pyannote -- are kept to the thinnest possible edges around
    this). Returns ``(http_status, json_body)``; never raises.

    ``tracker`` (``live_stt.state.ServerState.diarization_sessions`` in
    production) records concurrent-session counts AND a small live list of
    in-flight requests (id/elapsed/device) for ``/api/stats`` and the admin
    dashboard -- see ``live_stt/diarize_sessions.py``. When
    ``effective_settings.diarization_device == "cuda"``, this also gates
    admission on real free VRAM (``live_stt.gpu.free_vram_mb()``) against
    ``diarization_vram_mb`` -- see that setting's docstring for the real
    measurement (~12.3GB) behind its default. A CUDA allocation failure
    inside pyannote/torch is not guaranteed to be a catchable Python
    exception any more than parakeet.cpp's is (see CLAUDE.md), so this check
    exists for the same reason the ASR admission path already has one.
    """
    try:
        fields = parse_multipart_form(content_type, body)
    except MultipartError as exc:
        return 400, {"error": {"message": str(exc)}}

    if not fields.get("file"):
        return 400, {"error": {"message": "missing required multipart field 'file'"}}

    response_format = fields.get("response_format", b"verbose_json").decode("utf-8", "replace")
    if response_format != "verbose_json":
        return 400, {
            "error": {
                "message": (
                    f"unsupported response_format {response_format!r}; "
                    "only 'verbose_json' is implemented"
                )
            }
        }

    try:
        words = _parse_words(fields.get("words"))
    except MultipartError as exc:
        return 400, {"error": {"message": str(exc)}}

    overrides: dict[str, Any] = {}
    model = fields.get("model")
    if model:
        overrides["diarization_model"] = model.decode("utf-8", "replace")
    num_speakers = fields.get("num_speakers")
    if num_speakers:
        try:
            overrides["diarization_num_speakers"] = int(num_speakers)
        except ValueError:
            return 400, {"error": {"message": f"'num_speakers' must be an integer, got {num_speakers!r}"}}
    effective_settings = settings.model_copy(update=overrides) if overrides else settings

    include_text = fields.get("include_text", b"false").decode("utf-8", "replace").strip().lower() == "true"
    if include_text and not words:
        logger.info(
            "include_text=true but no 'words' field supplied -- segments will "
            "have empty text (see diarize_http.py's 'known, deliberate "
            "incompatibility' note)"
        )

    if effective_settings.diarization_device == "cuda":
        free = gpu.free_vram_mb()
        if free is not None:
            metrics.gpu_free_vram_mb.set(free)
        required = effective_settings.diarization_vram_mb
        if free is not None and free < required:
            tracker.record_rejected_vram()
            metrics.diarization_requests_total.labels(outcome="rejected_vram").inc()
            return 503, {
                "error": {
                    "message": f"insufficient VRAM for diarization: {free}MB free, {required}MB required"
                }
            }

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(fields["file"])
        tmp.flush()
        request_id = tracker.start(device=effective_settings.diarization_device)
        metrics.diarization_sessions_active.inc()
        try:
            result = diarize_file(tmp.name, settings=effective_settings, words=words)
        except DiarizationError as exc:
            tracker.finish(request_id, ok=False)
            metrics.diarization_sessions_active.dec()
            metrics.diarization_requests_total.labels(outcome="failed").inc()
            # A missing dependency or missing/bad token is a deployment
            # problem, not a bad request -- 503, mirroring admin_http.py's
            # own convention of reserving non-200s for structural failure.
            msg = str(exc)
            status = 503 if ("not installed" in msg or "HF_TOKEN" in msg or "Failed to load" in msg) else 400
            return status, {"error": {"message": msg}}
        except Exception as exc:  # noqa: BLE001 -- last-resort boundary; never crash the admin thread
            tracker.finish(request_id, ok=False)
            metrics.diarization_sessions_active.dec()
            metrics.diarization_requests_total.labels(outcome="failed").inc()
            logger.exception("unexpected diarization failure")
            return 500, {"error": {"message": f"internal error: {exc}"}}
        else:
            tracker.finish(request_id, ok=True)
            metrics.diarization_sessions_active.dec()
            metrics.diarization_requests_total.labels(outcome="ok").inc()

    return 200, result
