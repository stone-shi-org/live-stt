"""Pure mapping from the worker's feed_json/finalize_json documents to
TranscriptionEvent protobuf messages.

Kept pure and side-effect-free, mirroring
my-meeting-notes/app/routers/live_caption.py's _handle_realtime_event: no
socket, no state, so this is testable against captured JSON fixtures in
tests/fixtures/feed_json_samples.json with no worker process at all.

Expected input shape (parakeet.cpp's stream_feed_json / stream_finalize_json,
verified against include/parakeet_capi.h):

    {"text": "...", "eou": 0, "eob": 0, "frame_sec": 0.08,
     "events": [{"type": "eou"|"eob", "frame": 31, "t": 2.48}, ...],
     "words": [{"w": "...", "start": 0.48, "end": 0.64, "conf": 0.91}, ...]}
"""

from __future__ import annotations

from typing import Any

from live_stt.models import strip_language_tag
from live_stt.pb.livestt.v1 import asr_pb2


def _word_to_proto(word: dict[str, Any], *, time_offset_sec: float) -> asr_pb2.Word:
    return asr_pb2.Word(
        text=word["w"],
        start_sec=word["start"] + time_offset_sec,
        end_sec=word["end"] + time_offset_sec,
        confidence=word.get("conf", 0.0),
    )


def worker_json_to_events(
    doc: dict[str, Any],
    *,
    time_offset_sec: float = 0.0,
    audio_offset_sec: float = 0.0,
    strip_tag: bool = False,
) -> list[asr_pb2.TranscriptionEvent]:
    """Convert one feed_json/finalize_json document into zero or more events.

    ``time_offset_sec`` rebases session-relative word/event timestamps to
    call-absolute ones across worker rotations (see live_stt/boundary.py and
    CLAUDE.md's rotation section). ``audio_offset_sec`` is the call-absolute
    audio position for TranscriptDelta.audio_offset_sec -- distinct from word
    timestamps, it tracks total audio the CALL has consumed, not where in it
    these particular words fall.
    """
    events: list[asr_pb2.TranscriptionEvent] = []

    text = doc.get("text", "")
    words_raw = doc.get("words", [])
    if text or words_raw:
        if strip_tag:
            text = strip_language_tag(text)
        words = [_word_to_proto(w, time_offset_sec=time_offset_sec) for w in words_raw]
        events.append(
            asr_pb2.TranscriptionEvent(
                delta=asr_pb2.TranscriptDelta(
                    text=text, words=words, audio_offset_sec=audio_offset_sec
                )
            )
        )

    for ev in doc.get("events", []):
        at_sec = ev.get("t", 0.0) + time_offset_sec
        ev_type = ev.get("type")
        if ev_type == "eou":
            events.append(asr_pb2.TranscriptionEvent(eou=asr_pb2.EndOfUtterance(at_sec=at_sec)))
        elif ev_type == "eob":
            events.append(asr_pb2.TranscriptionEvent(eob=asr_pb2.EndOfBoundary(at_sec=at_sec)))

    return events
