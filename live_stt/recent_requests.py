"""Bounded in-memory log of recently FINISHED requests, usage numbers only --
backs ``GET /api/requests/recent`` (live_stt/admin_http.py).

Consumer: the separate gpu-monitor project polls this every ~10s and stores
one row per request, to attribute GPU busy time to speech work. It dedups by
``id`` and detects restarts via ``boot_id``; ``since=<seq>`` lets it fetch
only what it hasn't seen.

**Privacy (deliberate, narrow exception to "not a session registry", same
category as diarize_sessions.py / transcribe_sessions.py):** entries carry NO
content and NO caller identity -- no transcript text, no TTS input text, no
audio, no client-supplied ids (gRPC ``call_id``, HTTP fields) or metadata.
``id`` is server-generated random hex, unrelated to any client value. Every
string field is a closed set: ``kind``/``outcome`` are the same enums the
Prometheus counters use, ``model`` is a registry key (never a raw
client-supplied string -- unknown models record ``None``), ``voice`` is a
model's registered voice or ``"other"``, ``device`` is cuda/cpu. Everything
else is a count or a duration. ``ALLOWED_FIELDS`` is the full schema;
``record()`` drops anything else, so a future caller cannot leak a new field
by accident.

Recording must never raise into or slow the request path: ``record()``
swallows (and logs) every exception, and holds the lock only for an append.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from typing import Any

from live_stt.logging_config import get_logger

logger = get_logger("recent_requests")

RETENTION = 500
KINDS = ("stream", "transcribe", "tts", "diarize")

# The full entry schema, in contract order. Nothing outside this set is ever stored.
ALLOWED_FIELDS = (
    "id",
    "seq",
    "kind",
    "outcome",
    "model",
    "voice",
    "device",
    "started_at",
    "finished_at",
    "duration_s",
    "audio_seconds",
    "input_chars",
    "words",
    "transcript_chars",
    "processing_s",
)


class RecentRequests:
    def __init__(self, maxlen: int = RETENTION) -> None:
        self._lock = threading.Lock()
        self._ring: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0
        self.retention = maxlen
        self.boot_id = secrets.token_hex(16)

    def record(
        self,
        *,
        kind: str,
        outcome: str,
        started_at: float,
        finished_at: float | None = None,
        model: str | None = None,
        voice: str | None = None,
        device: str | None = None,
        audio_seconds: float | None = None,
        input_chars: int | None = None,
        words: int | None = None,
        transcript_chars: int | None = None,
        processing_s: float | None = None,
    ) -> None:
        """Append one finished request. ``started_at``/``finished_at`` are
        epoch seconds (``time.time()``). Never raises."""
        try:
            if kind not in KINDS:
                raise ValueError(f"unknown kind {kind!r}")
            if device not in (None, "cuda", "cpu"):
                device = None
            end = time.time() if finished_at is None else finished_at
            entry: dict[str, Any] = {
                "id": secrets.token_hex(8),
                "seq": 0,  # assigned under the lock
                "kind": kind,
                "outcome": str(outcome),
                "model": model,
                "voice": voice if kind == "tts" else None,
                "device": device,
                "started_at": round(float(started_at), 3),
                "finished_at": round(float(end), 3),
                "duration_s": round(max(0.0, float(end) - float(started_at)), 3),
                "audio_seconds": audio_seconds,
                "input_chars": input_chars,
                "words": words,
                "transcript_chars": transcript_chars,
                "processing_s": processing_s,
            }
            for key in ("audio_seconds", "processing_s"):
                if entry[key] is not None:
                    entry[key] = round(float(entry[key]), 3)
            for key in ("input_chars", "words", "transcript_chars"):
                if entry[key] is not None:
                    entry[key] = int(entry[key])
            with self._lock:
                self._seq += 1
                entry["seq"] = self._seq
                self._ring.append(entry)
        except Exception:  # noqa: BLE001 -- bookkeeping must never break a request
            logger.exception("failed to record recent request (kind=%s outcome=%s)", kind, outcome)

    def snapshot(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            items = [dict(e) for e in self._ring if e["seq"] > since]
            next_seq = self._seq
        return {
            "boot_id": self.boot_id,
            "next_seq": next_seq,
            "retention": self.retention,
            "requests": items,
        }


# One per process: boot_id must be stable for the process lifetime, and the
# HTTP handlers / servicer all record into the same ring.
_RECENT = RecentRequests()


def get_recent_requests() -> RecentRequests:
    return _RECENT


def bounded_voice(voice: str | None, supported: tuple[str, ...] | None) -> str | None:
    """A TTS ``voice`` is a client-supplied string; only record it when it is
    one of the model's registered voices, so this log can never carry
    arbitrary caller-chosen text."""
    if not voice:
        return None
    if supported and voice in supported:
        return voice
    return "other"
