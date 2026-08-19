"""Pure helpers for the worker-rotation seam.

See CLAUDE.md's "Rotation: overlapped dual-feed" section. Kill-and-replay was
rejected because replaying the overlap while live audio keeps arriving never
converges at RTF near 1; instead the incoming worker is fed the same audio as
the outgoing one for ``rotation_overlap_sec`` before the cut, and its output
during that window is discarded rather than merged by fuzzy text alignment.

Kept pure (no socket, no process, no clock) so live_stt/session.py's rotation
state machine is the only thing that needs a real worker pair to test end to
end -- these two functions are covered directly.
"""

from __future__ import annotations

from live_stt.pb.livestt.v1 import asr_pb2

# Small slack so a word priming exactly at the cut isn't dropped by
# floating-point jitter between the outgoing and incoming workers' clocks.
_DEDUP_SLACK_SEC = 0.05


def rebase_time_offset(t_cut_sec: float, overlap_sec: float) -> float:
    """The time_offset_sec to pass to events.worker_json_to_events() for the
    INCOMING worker of a rotation.

    The incoming worker's session-relative clock starts at 0 when dual-feeding
    began -- i.e. overlap_sec before the cut -- so its timestamps must be
    shifted forward by exactly that amount to land in call-absolute time.
    """
    return t_cut_sec - overlap_sec


def dedup_incoming_words(words: list[asr_pb2.Word], *, t_cut_sec: float) -> list[asr_pb2.Word]:
    """Drop incoming-worker words (already rebased to call-absolute time) that
    fall before the cut point -- they are already covered by the outgoing
    worker's finalize() tail, and re-emitting them would duplicate text.

    With an <EOU>-bearing model this is exact: the cut point IS an <EOU>
    boundary, so there is no fuzzy text alignment to do, just a timestamp
    comparison. Without <EOU> (nemotron) this is the only seam-dedup
    mechanism available -- there is no revisable hypothesis to fall back on.
    """
    return [w for w in words if w.start_sec >= t_cut_sec - _DEDUP_SLACK_SEC]
