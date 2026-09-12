"""HTTP surface for Text-to-Speech (TTS): ``POST /v1/audio/speech``.

OpenAI Audio Speech API compatible shape, wired into ``live_stt/admin_http.py``'s
existing ``ThreadingHTTPServer``.
"""

from __future__ import annotations

import json
from typing import Iterator
from urllib.parse import parse_qs

from live_stt import gpu, metrics, tts, tts_models
from live_stt.config import Settings
from live_stt.logging_config import get_logger
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
) -> tuple[int, bytes | Iterator[bytes], str, dict[str, str]]:
    """Handle POST /v1/audio/speech.

    Returns (status_code, response_body_or_generator, content_type, headers_dict).
    """
    if draining:
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
    voice = payload.get("voice") or settings.tts_voice
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

    # VRAM Gate for CUDA
    if settings.tts_device == "cuda":
        free_vram = gpu.free_vram_mb()
        if free_vram is not None and free_vram < settings.tts_vram_mb:
            tracker.record_rejected_vram()
            metrics.tts_requests_total.labels(outcome="rejected_vram").inc()
            logger.warning(
                "TTS rejected: insufficient free VRAM (%d MiB free < %d MiB budget)",
                free_vram,
                settings.tts_vram_mb,
            )
            return (
                503,
                json.dumps(
                    {
                        "error": (
                            f"insufficient free VRAM for TTS ({free_vram} MiB free, "
                            f"{settings.tts_vram_mb} MiB required)"
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
            try:
                for chunk in tts.synthesize_stream(
                    text_str,
                    model=model_spec.key,
                    voice=voice,
                    response_format=response_format,
                    speed=speed,
                    settings=settings,
                ):
                    yield chunk
                tracker.finish(request_id, ok=True)
                metrics.tts_sessions_active.dec()
                metrics.tts_requests_total.labels(outcome="ok").inc()
                metrics.tts_chars_total.inc(len(text_str))
            except Exception as exc:
                tracker.finish(request_id, ok=False)
                metrics.tts_sessions_active.dec()
                metrics.tts_requests_total.labels(outcome="failed").inc()
                logger.error("TTS stream synthesis failed: %s", exc)
                raise

        ct = "audio/mpeg" if response_format.lower() == "mp3" else f"audio/{response_format.lower()}"
        return 200, stream_generator(), ct, {"Transfer-Encoding": "chunked"}

    try:
        audio_bytes, ct = tts.synthesize(
            text_str,
            model=model_spec.key,
            voice=voice,
            response_format=response_format,
            speed=speed,
            settings=settings,
        )
        tracker.finish(request_id, ok=True)
        metrics.tts_sessions_active.dec()
        metrics.tts_requests_total.labels(outcome="ok").inc()
        metrics.tts_chars_total.inc(len(text_str))
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
