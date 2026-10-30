"""HTTP surface for Text-to-Speech (TTS): ``POST /v1/audio/speech``.

OpenAI Audio Speech API compatible shape, wired into ``live_stt/admin_http.py``'s
existing ``ThreadingHTTPServer``.
"""

from __future__ import annotations

import json
import time
from typing import Iterator
from urllib.parse import parse_qs

from live_stt import gpu, metrics, tts, tts_models, vram
from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.recent_requests import RecentRequests, bounded_voice, get_recent_requests
from live_stt.tts import TTSBadRequestError, TTSError, TTSUnavailableError
from live_stt.tts_sessions import TTSSessionTracker

logger = get_logger("tts_http")

SPEECH_PATH = "/v1/audio/speech"


def _parse_request_payload(content_type: str, body: bytes) -> dict:
    """Parse JSON or form-encoded request body."""
    if not body:
        return {}

    ct = content_type.lower()
    if "application/json" in ct or not ct:
        try:
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise TTSBadRequestError(f"invalid JSON request body: {exc}") from exc
    elif "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8"))
            return {k: v[0] if len(v) == 1 else v for k, v in parsed.items()}
        except Exception as exc:
            raise TTSBadRequestError(f"invalid form request body: {exc}") from exc
    else:
        # Default attempt as JSON
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            raise TTSBadRequestError(f"unsupported content type: {content_type!r}")


def handle_speech_request(
    *,
    content_type: str,
    body: bytes,
    settings: Settings,
    tracker: TTSSessionTracker,
    draining: bool = False,
    recent: RecentRequests | None = None,
) -> tuple[int, bytes | Iterator[bytes], str, dict[str, str]]:
    """Handle POST /v1/audio/speech.

    Returns (status_code, response_body_or_generator, content_type, headers_dict).

    ``recent`` gets one usage-only entry (live_stt/recent_requests.py) per
    request rejected at admission or that reached synthesis -- input LENGTH
    only, never the text itself.
    """
    recent = recent or get_recent_requests()
    started_at = time.time()
    if draining:
        recent.record(kind="tts", outcome="draining", started_at=started_at)
        return 503, json.dumps({"error": "server is draining"}).encode(), "application/json", {}

    try:
        payload = _parse_request_payload(content_type, body)
    except TTSBadRequestError as exc:
        return 400, json.dumps({"error": str(exc)}).encode(), "application/json", {}

    text = payload.get("input") or payload.get("text")
    if not text or not str(text).strip():
        return 400, json.dumps({"error": "missing or empty required field 'input'"}).encode(), "application/json", {}

    text_str = str(text)
    model_key = payload.get("model") or settings.tts_model
    response_format = payload.get("response_format", "mp3")
    stream = payload.get("stream", False)
    if isinstance(stream, str):
        stream = stream.lower() in ("true", "1", "yes")

    try:
        speed = float(payload.get("speed", 1.0))
    except (ValueError, TypeError):
        return 400, json.dumps({"error": "speed must be a float"}).encode(), "application/json", {}

    # Validate model
    try:
        model_spec = tts_models.resolve(model_key)
    except KeyError as exc:
        return 400, json.dumps({"error": str(exc)}).encode(), "application/json", {}

    # Real bug, found for real: this used to be `payload.get("voice") or
    # settings.tts_voice`, computed BEFORE model_spec existed -- so
    # Settings.tts_voice's global default ("af_heart", a kokoro voice name)
    # got passed straight through as an explicit voice= to tts.synthesize()
    # even when the request asked for a DIFFERENT engine (e.g.
    # model=piper-en-us), silently pointing piper's voice-model lookup at
    # "af_heart.onnx" -- a file that was never going to exist -- instead of
    # piper's own real default en_US-lessac-medium. tts.resolve_voice()
    # already carries the correct "only honor the global default if it's
    # actually valid for THIS model" logic; reused here instead of
    # duplicating (and re-breaking) it.
    voice = tts.resolve_voice(payload.get("voice"), settings, model_spec)

    def _record(outcome: str, *, ran: bool, stats: dict[str, Any] | None = None, processing_s: float | None = None) -> None:
        recent.record(
            kind="tts",
            outcome=outcome,
            started_at=started_at,
            model=model_spec.key,
            voice=bounded_voice(voice, model_spec.supported_voices),
            device=settings.tts_device,
            audio_seconds=(stats or {}).get("audio_seconds") if outcome == "ok" else None,
            input_chars=len(text_str) if ran else None,
            processing_s=processing_s,
        )

    # VRAM Gate for CUDA: only enforce VRAM budget for models requiring GPU VRAM.
    # Per-model requirement + reclaim of idle warm workers (live_stt/vram.py).
    # Skipped when the warm worker already holds THIS model on the GPU: the
    # request reuses that VRAM rather than allocating more, and reclaiming would
    # only kill the very worker it is about to use.
    admission: vram.Admission | None = None
    if settings.tts_device == "cuda" and (model_spec.measured_peak_vram_mb or 0) > 0:
        warm = tts.get_tts_worker_manager().snapshot()
        already_warm = bool(warm) and warm.get("model") == model_spec.key and warm.get("device") == "cuda"
        admission = (
            None
            if already_warm
            else vram.admit(
                settings,
                model_key=model_spec.key,
                model_mb=model_spec.measured_peak_vram_mb,
                fallback_mb=settings.tts_vram_mb,
            )
        )
        if admission is not None and not admission.ok:
            tracker.record_rejected_vram()
            metrics.tts_requests_total.labels(outcome="rejected_vram").inc()
            _record("rejected_vram", ran=False)
            logger.warning(
                "TTS rejected: insufficient free VRAM (%s MiB free < %d MiB required)",
                admission.free_mb,
                admission.required_mb,
            )
            return (
                503,
                json.dumps(
                    {
                        "error": (
                            f"insufficient free VRAM for TTS ({admission.free_mb} MiB free, "
                            f"{admission.required_mb} MiB required)"
                        )
                    }
                ).encode(),
                "application/json",
                {},
            )

    request_id = tracker.start(model=model_spec.key, voice=voice, device=settings.tts_device)
    metrics.tts_sessions_active.inc()

    if stream:
        def stream_generator() -> Iterator[bytes]:
            stats: dict[str, Any] = {}
            synth_s = 0.0
            try:
                chunks = iter(
                    tts.synthesize_stream(
                        text_str,
                        model=model_spec.key,
                        voice=voice,
                        response_format=response_format,
                        speed=speed,
                        settings=settings,
                        stats=stats,
                    )
                )
                # processing_s counts only time spent producing chunks, not
                # time blocked writing them to a slow client.
                while True:
                    t0 = time.monotonic()
                    try:
                        chunk = next(chunks)
                    except StopIteration:
                        synth_s += time.monotonic() - t0
                        break
                    synth_s += time.monotonic() - t0
                    yield chunk
                tracker.finish(request_id, ok=True)
                metrics.tts_sessions_active.dec()
                metrics.tts_requests_total.labels(outcome="ok").inc()
                metrics.tts_chars_total.inc(len(text_str))
                _record("ok", ran=True, stats=stats, processing_s=synth_s)
            except GeneratorExit:
                # Client went away mid-stream: still one finished request.
                _record("failed", ran=True, processing_s=synth_s)
                raise
            except Exception as exc:
                tracker.finish(request_id, ok=False)
                metrics.tts_sessions_active.dec()
                metrics.tts_requests_total.labels(outcome="failed").inc()
                _record("failed", ran=True, processing_s=synth_s)
                logger.error("TTS stream synthesis failed: %s", exc)
                raise
            finally:
                if admission is not None:
                    admission.release()

        ct = "audio/mpeg" if response_format.lower() == "mp3" else f"audio/{response_format.lower()}"
        return 200, stream_generator(), ct, {}

    stats: dict[str, Any] = {}
    synth_start = time.monotonic()
    recent_outcome = "failed"
    try:
        audio_bytes, ct = tts.synthesize(
            text_str,
            model=model_spec.key,
            voice=voice,
            response_format=response_format,
            speed=speed,
            settings=settings,
            stats=stats,
        )
        tracker.finish(request_id, ok=True)
        metrics.tts_sessions_active.dec()
        metrics.tts_requests_total.labels(outcome="ok").inc()
        metrics.tts_chars_total.inc(len(text_str))
        recent_outcome = "ok"
        return 200, audio_bytes, ct, {}
    except TTSBadRequestError as exc:
        tracker.finish(request_id, ok=False)
        metrics.tts_sessions_active.dec()
        metrics.tts_requests_total.labels(outcome="failed").inc()
        return 400, json.dumps({"error": str(exc)}).encode(), "application/json", {}
    except TTSUnavailableError as exc:
        tracker.finish(request_id, ok=False)
        metrics.tts_sessions_active.dec()
        metrics.tts_requests_total.labels(outcome="failed").inc()
        return 503, json.dumps({"error": str(exc)}).encode(), "application/json", {}
    except Exception as exc:
        tracker.finish(request_id, ok=False)
        metrics.tts_sessions_active.dec()
        metrics.tts_requests_total.labels(outcome="failed").inc()
        logger.error("TTS synthesis failed: %s", exc)
        return 500, json.dumps({"error": f"TTS synthesis failed: {exc}"}).encode(), "application/json", {}
    finally:
        if admission is not None:
            admission.release()
        _record(recent_outcome, ran=True, stats=stats, processing_s=time.monotonic() - synth_start)
