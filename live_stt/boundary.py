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

import re

from live_stt.pb.livestt.v1 import asr_pb2

# Small slack so a word priming exactly at the cut isn't dropped by
# floating-point jitter between the outgoing and incoming workers' clocks.
_DEDUP_SLACK_SEC = 0.05

# Maximum timestamp drift (seconds) allowed when aligning matching words
# across adjacent batch-inference windows in the overlap region.
_OVERLAP_TIME_TOLERANCE_SEC = 1.5

_PUNCT_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


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


def iter_batch_windows(
    total_samples: int, window_samples: int, overlap_samples: int
) -> list[tuple[int, int]]:
    """Slice ``total_samples`` into ``(start_sample, end_sample)`` windows of
    at most ``window_samples`` with ``overlap_samples`` overlap between
    consecutive windows.

    When ``window_samples <= 0`` or ``total_samples <= window_samples``, returns
    a single window ``[(0, max(0, total_samples))]`` so recordings that fit in
    one window take the exact single-pass path.
    """
    if total_samples <= 0:
        return [(0, 0)]
    if window_samples <= 0 or total_samples <= window_samples:
        return [(0, total_samples)]

    overlap = max(0, min(overlap_samples, window_samples - 1))
    stride = max(1, window_samples - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        windows.append((start, end))
        if end >= total_samples:
            break
        start += stride
    return windows


def _norm_word(text: str) -> str:
    cleaned = _PUNCT_STRIP_RE.sub("", text.strip().lower())
    return cleaned or text.strip().lower()


def _word_center(w: asr_pb2.Word) -> float:
    return (w.start_sec + w.end_sec) * 0.5


def merge_batch_window_words(
    accumulated: list[asr_pb2.Word],
    incoming: list[asr_pb2.Word],
    *,
    win_start_sec: float,
    prev_end_sec: float,
) -> list[asr_pb2.Word]:
    """Merge ``incoming`` words from a batch window starting at ``win_start_sec``
    (already rebased to call-absolute timestamps) onto ``accumulated`` words
    whose previous window ended at ``prev_end_sec``.

    In the overlap seam ``[win_start_sec, prev_end_sec]``:
    1. First searches for the longest contiguous lexical + timestamp match
       between the overlap suffix of ``accumulated`` and the overlap prefix of
       ``incoming``, splicing at the matched word closest to the overlap
       midpoint so neither boundary's edge-clipped words survive.
    2. If no lexical match exists in the overlap (e.g. silence or disfluency at
       the boundary), cuts cleanly at the overlap midpoint ``mid_sec`` and
       drops any boundary-straddling duplicate word.
    """
    if not accumulated:
        return list(incoming)
    if not incoming:
        return list(accumulated)
    if prev_end_sec <= win_start_sec:
        return list(accumulated) + list(incoming)

    mid_sec = (win_start_sec + prev_end_sec) * 0.5

    # Find indices of accumulated words that touch the overlap [win_start_sec, prev_end_sec]
    prev_overlap_start = len(accumulated)
    for idx in range(len(accumulated) - 1, -1, -1):
        if accumulated[idx].end_sec >= win_start_sec - _DEDUP_SLACK_SEC:
            prev_overlap_start = idx
        else:
            break

    # Find indices of incoming words that touch the overlap [win_start_sec, prev_end_sec]
    curr_overlap_end = 0
    for idx, w in enumerate(incoming):
        if w.start_sec <= prev_end_sec + _DEDUP_SLACK_SEC:
            curr_overlap_end = idx + 1
        else:
            break

    prev_slice = accumulated[prev_overlap_start:]
    curr_slice = incoming[:curr_overlap_end]

    if prev_slice and curr_slice:
        prev_norm = [_norm_word(w.text) for w in prev_slice]
        curr_norm = [_norm_word(w.text) for w in curr_slice]

        best_match: tuple[int, float, int, int, int] | None = None
        for i in range(len(prev_slice)):
            if not prev_norm[i]:
                continue
            for j in range(len(curr_slice)):
                if prev_norm[i] != curr_norm[j]:
                    continue
                if abs(prev_slice[i].start_sec - curr_slice[j].start_sec) > _OVERLAP_TIME_TOLERANCE_SEC:
                    continue
                length = 0
                while (
                    i + length < len(prev_slice)
                    and j + length < len(curr_slice)
                    and prev_norm[i + length] == curr_norm[j + length]
                    and abs(
                        prev_slice[i + length].start_sec - curr_slice[j + length].start_sec
                    )
                    <= _OVERLAP_TIME_TOLERANCE_SEC
                ):
                    length += 1
                # Pick the splice offset within the matched run closest to mid_sec
                best_k = min(
                    range(length),
                    key=lambda k: abs(_word_center(curr_slice[j + k]) - mid_sec),
                )
                dist = abs(_word_center(curr_slice[j + best_k]) - mid_sec)
                candidate = (length, -dist, i, j, best_k)
                if best_match is None or candidate > best_match:
                    best_match = candidate

        if best_match is not None:
            _, _, i_prev, j_curr, k_off = best_match
            kept_prev = accumulated[: prev_overlap_start + i_prev + k_off]
            kept_curr = incoming[j_curr + k_off :]
            return kept_prev + kept_curr

    # Fallback: cut at the midpoint of the overlap window.
    kept_prev = [
        w
        for idx, w in enumerate(accumulated)
        if idx < prev_overlap_start or _word_center(w) < mid_sec
    ]
    kept_curr = [
        w
        for idx, w in enumerate(incoming)
        if idx >= curr_overlap_end or _word_center(w) >= mid_sec
    ]
    if kept_prev and kept_curr:
        last_prev = kept_prev[-1]
        first_curr = kept_curr[0]
        if (
            _norm_word(last_prev.text) == _norm_word(first_curr.text)
            and first_curr.start_sec <= last_prev.end_sec + 0.4
        ):
            kept_curr = kept_curr[1:]
    return kept_prev + kept_curr

