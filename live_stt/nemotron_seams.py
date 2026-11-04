"""Pure helpers for Nemotron-3.5-Transcribe speaker-attributed output (LS-19).

No sockets, processes or clocks here: ``live_stt/nemotron.py`` owns the
sidecar and the queue, and this module turns what the sidecar returned into
the house diarization JSON. Kept separate so every piece of the
over-45-minute windowing (option C on LS-19) is directly unit-testable.

**Words.** A ``/v1/chat/completions`` response with
``timestamp_granularities: ["word"]`` carries ``transcription.words``: one
entry per word with ``word``/``start``/``end``/``speaker`` (``"spk:0"``...)
-- the shape seen in every spike response on homesrv3.
``extract_words`` reads that and refuses a response whose word timestamps
did not align (the 55-minute spike run returned HTTP 200, full text and
inline speaker tags but ``word timestamps unavailable`` -- no per-word
speakers or times at all). Treating that as success would silently hand back
a transcript with no speaker attribution.

**Windows over 45 minutes: overlap vote.** Each window is transcribed
independently, so its speaker labels are local to that window. For each
pair of adjacent windows, words that both windows transcribed in the
overlap are paired (same normalized text, centres within
``PAIR_TOLERANCE_SEC``) and every pair is one vote for "local label X in the
new window is global speaker Y". Labels are then mapped one-to-one, strongest
vote first; a label with no votes becomes a new global speaker. The number of
paired words is the seam's confidence: on the scaled-down LS-19 seam test,
seams with >= 100 matched words cost 0.011 cpWER and seams below that cost
0.136, so a weak seam is reported in ``warnings`` rather than trusted.

Text is joined at the overlap midpoint (the previous window's words whose
centre is before it, the next window's from it on) -- the same cut
``live_stt/boundary.py``'s batch-window merge falls back to. Diarization
turns are cut at the same midpoint (clipped, not dropped).

**Segments come from the model's own diarization turns**
(``transcription.diarization_timestamps``), with each word's text attached to
the turn of the same speaker that covers it (or the nearest one). Building
segments from runs of words instead was measured on the 20 benchmark
meetings and roughly TRIPLES DER (0.186 vs 0.054): CTC word spans are tight,
so word-built segments miss the speech between words. About 23% of turns
carry no same-speaker word (backchannels and overlapped speech whose words
went to the other speaker); they are kept with empty text because dropping
them doubles DER (0.107). my-meeting-notes only rejects a response whose
segments ALL lack text. When a response has no turns at all, segments fall
back to runs of words.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from live_stt.boundary import _norm_word, iter_batch_windows

# Centres of the "same" word in two windows' transcripts must agree within
# this to count as one vote (LS-19 seam test).
PAIR_TOLERANCE_SEC = 0.6

# A same-speaker run of words is split into two segments when the silence
# between two consecutive words exceeds this, so a segment never claims a
# long pause as speech.
SEGMENT_MAX_GAP_SEC = 1.5


class NemotronResponseError(Exception):
    """The sidecar answered, but not with usable speaker-attributed words."""


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str

    @property
    def centre(self) -> float:
        return (self.start + self.end) * 0.5


@dataclass(frozen=True, slots=True)
class Turn:
    speaker: str
    start: float
    end: float


@dataclass(slots=True)
class SeamReport:
    index: int  # seam between window index and index + 1
    overlap_start: float
    overlap_end: float
    matched_words: int
    mapping: dict[str, str] = field(default_factory=dict)  # local label -> global label
    new_speakers: list[str] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return (self.overlap_start + self.overlap_end) * 0.5


def plan_windows(duration_sec: float, window_sec: float, overlap_sec: float) -> list[tuple[float, float]]:
    """``(start_sec, end_sec)`` windows covering ``duration_sec``, each at most
    ``window_sec`` long, consecutive windows overlapping by ``overlap_sec``.
    Reuses ``boundary.iter_batch_windows`` at millisecond resolution."""
    ms = iter_batch_windows(
        int(round(duration_sec * 1000)), int(round(window_sec * 1000)), int(round(overlap_sec * 1000))
    )
    return [(s / 1000.0, e / 1000.0) for s, e in ms]


def _find_transcription(doc: dict[str, Any]) -> dict[str, Any]:
    tr = doc.get("transcription")
    if isinstance(tr, dict):
        return tr
    # Older/alternate placement: on the message itself.
    try:
        msg = doc["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        msg = {}
    tr = msg.get("transcription") if isinstance(msg, dict) else None
    return tr if isinstance(tr, dict) else {}


def response_text(doc: dict[str, Any]) -> str:
    try:
        content = doc["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        content = ""
    if not content:
        content = str(_find_transcription(doc).get("text") or "")
    return content


def extract_words(doc: dict[str, Any], *, offset_sec: float = 0.0) -> list[Word]:
    """Speaker-attributed words from one chat-completion response, shifted
    by ``offset_sec`` (a window's start in the whole recording).

    Raises ``NemotronResponseError`` when the response has text but no
    aligned word timestamps -- the silent half-failure described in this
    module's docstring. An empty transcript with no words (silent audio) is
    returned as an empty list; the caller decides what that means.
    """
    tr = _find_transcription(doc)
    alignment = tr.get("word_timestamp_alignment") or {}
    raw_words = tr.get("words") or []
    text = response_text(doc).strip()
    status = alignment.get("status") if isinstance(alignment, dict) else None
    if status not in (None, "aligned") or (text and not raw_words):
        reason = alignment.get("failure_reason") if isinstance(alignment, dict) else None
        raise NemotronResponseError(
            "Nemotron returned a transcript without aligned word timestamps "
            f"(status={status!r}, reason={reason!r}); refusing to return a transcript "
            "with no per-word speakers or times"
        )
    words: list[Word] = []
    for w in raw_words:
        try:
            token = str(w["word"]).strip()
            start = float(w["start"]) + offset_sec
            end = float(w["end"]) + offset_sec
        except (KeyError, TypeError, ValueError):
            continue
        if not token:
            continue
        words.append(Word(token, start, max(start, end), str(w.get("speaker") or "unk")))
    words.sort(key=lambda x: (x.start, x.end))
    return words


def extract_turns(doc: dict[str, Any], *, offset_sec: float = 0.0) -> list[Turn]:
    """The model's own speaker turns, labelled like the words (``spk:<n>``)."""
    out: list[Turn] = []
    for t in _find_transcription(doc).get("diarization_timestamps") or []:
        try:
            spk = t["speaker"]
            start = float(t["start"]) + offset_sec
            end = float(t["end"]) + offset_sec
        except (KeyError, TypeError, ValueError):
            continue
        label = f"spk:{spk}" if isinstance(spk, int) or str(spk).isdigit() else str(spk)
        if end > start:
            out.append(Turn(label, start, end))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def match_seam(
    prev_words: list[Word],
    next_words: list[Word],
    *,
    overlap_start: float,
    overlap_end: float,
    tolerance_sec: float = PAIR_TOLERANCE_SEC,
) -> tuple[dict[str, str], int]:
    """Overlap vote. ``prev_words`` carry GLOBAL labels, ``next_words`` the
    next window's LOCAL labels. Returns ``(local -> global mapping, matched
    word count)``; local labels with no votes are absent from the mapping."""

    def in_overlap(w: Word) -> bool:
        return overlap_start <= w.centre <= overlap_end

    by_text: dict[str, list[list[Any]]] = {}
    for w in prev_words:
        if not in_overlap(w):
            continue
        key = _norm_word(w.text)
        if key:
            by_text.setdefault(key, []).append([w.centre, w.speaker, False])

    votes: Counter[tuple[str, str]] = Counter()
    matched = 0
    for w in sorted((x for x in next_words if in_overlap(x)), key=lambda x: x.centre):
        key = _norm_word(w.text)
        best = None
        best_dt = tolerance_sec
        for cand in by_text.get(key, ()):
            if cand[2]:
                continue
            dt = abs(cand[0] - w.centre)
            if dt <= best_dt:
                best, best_dt = cand, dt
        if best is None:
            continue
        best[2] = True
        votes[(w.speaker, best[1])] += 1
        matched += 1

    mapping: dict[str, str] = {}
    used_global: set[str] = set()
    # Strongest vote first; ties broken deterministically by label text.
    for (local, glob), _count in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])):
        if local in mapping or glob in used_global:
            continue
        mapping[local] = glob
        used_global.add(glob)
    return mapping, matched


def _global_label(n: int) -> str:
    return f"SPEAKER_{n:02d}"


def stitch_windows(
    windows: list[tuple[float, float, list[Word], list[Turn]]],
) -> tuple[list[Word], list[Turn], list[SeamReport]]:
    """Join per-window words and turns (absolute times, local labels) into
    one globally labelled transcript. ``windows`` is
    ``[(start, end, words, turns)]`` in time order."""
    if not windows:
        return [], [], []

    next_id = 0
    seams: list[SeamReport] = []

    def fresh() -> str:
        nonlocal next_id
        label = _global_label(next_id)
        next_id += 1
        return label

    def labels(words: list[Word], turns: list[Turn]) -> list[str]:
        seen: dict[str, float] = {}
        for x in [*words, *turns]:
            seen[x.speaker] = min(seen.get(x.speaker, x.start), x.start)
        return sorted(seen, key=lambda k: seen[k])

    first_map: dict[str, str] = {}
    for lab in labels(windows[0][2], windows[0][3]):
        first_map[lab] = fresh()
    prev_global = [Word(w.text, w.start, w.end, first_map[w.speaker]) for w in windows[0][2]]
    merged_words = list(prev_global)
    merged_turns = [Turn(first_map[t.speaker], t.start, t.end) for t in windows[0][3]]

    for i in range(1, len(windows)):
        prev_end = windows[i - 1][1]
        start, _end, local_words, local_turns = windows[i]
        overlap_start, overlap_end = start, prev_end
        mapping, matched = match_seam(
            prev_global, local_words, overlap_start=overlap_start, overlap_end=overlap_end
        )
        report = SeamReport(
            index=i - 1,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            matched_words=matched,
            mapping=dict(mapping),
        )
        for lab in labels(local_words, local_turns):
            if lab not in mapping:
                mapping[lab] = fresh()
                report.new_speakers.append(mapping[lab])
        seams.append(report)
        cur_global = [Word(w.text, w.start, w.end, mapping[w.speaker]) for w in local_words]
        mid = report.midpoint if overlap_end > overlap_start else start
        merged_words = [w for w in merged_words if w.centre < mid] + [w for w in cur_global if w.centre >= mid]
        kept = [Turn(t.speaker, t.start, min(t.end, mid)) for t in merged_turns if t.start < mid]
        incoming = [Turn(mapping[t.speaker], max(t.start, mid), t.end) for t in local_turns if t.end > mid]
        merged_turns = [t for t in kept + incoming if t.end > t.start]
        prev_global = cur_global
    return merged_words, merged_turns, seams


def relabel_in_order(words: list[Word], turns: list[Turn] | None = None) -> tuple[list[Word], list[Turn]]:
    """Renumber speakers SPEAKER_00, SPEAKER_01, ... by first appearance
    (words and turns together), so labels that vanished at a seam cut do not
    leave gaps."""
    first: dict[str, float] = {}
    for x in [*words, *(turns or [])]:
        first[x.speaker] = min(first.get(x.speaker, x.start), x.start)
    order = {lab: _global_label(n) for n, lab in enumerate(sorted(first, key=lambda k: (first[k], k)))}
    return (
        [Word(w.text, w.start, w.end, order[w.speaker]) for w in words],
        [Turn(order[t.speaker], t.start, t.end) for t in (turns or [])],
    )


def repair_collapsed_word_speakers(words: list[Word], turns: list[Turn]) -> tuple[list[Word], bool]:
    """Relabel words from the model's turns when its inline speaker tags collapsed.

    Occasionally the model's diarization turns name several speakers while its inline
    ``<spk:N>`` tags put every word on one of them (NOTSOFAR MTG_32256: 4 speakers in the
    turns, all 877 words on ``spk:0``). ``_assign_words`` then piles the whole transcript
    into that speaker's turns: DER stays fine, but per-speaker text is wrong (cpWER 1.14
    on a meeting with WER 0.02). Only that unambiguous case is repaired -- one word label,
    two or more turn labels -- so ordinary word/turn disagreement (overlap, backchannels)
    is left alone. Each word takes the speaker of the turn it overlaps most, else of the
    nearest turn. Returns ``(words, repaired)``.

    This is a best effort, not a cure: on MTG_32256 the collapse came with broken word
    timestamps too (no words placed between 90 s and 300 s, the rest crowded around them;
    deterministic on re-run, and still present in a 0-200 s window), so relabelling only
    took cpWER from 1.14 to 1.09. The caller therefore always reports it in ``warnings``.
    """
    if not words or not turns:
        return words, False
    if len({w.speaker for w in words}) != 1 or len({t.speaker for t in turns}) < 2:
        return words, False

    def pick(w: Word) -> str:
        best: Turn | None = None
        best_key: tuple[float, float] | None = None
        for t in turns:
            overlap = max(0.0, min(w.end, t.end) - max(w.start, t.start))
            dist = 0.0 if t.start <= w.centre <= t.end else min(abs(w.centre - t.start), abs(w.centre - t.end))
            key = (-overlap, dist)
            if best_key is None or key < best_key:
                best, best_key = t, key
        assert best is not None
        return best.speaker

    return [Word(w.text, w.start, w.end, pick(w)) for w in words], True


def _word_runs(words: list[Word]) -> list[Turn]:
    runs: list[Turn] = []
    cur: Turn | None = None
    for w in words:
        if cur is not None and w.speaker == cur.speaker and w.start - cur.end <= SEGMENT_MAX_GAP_SEC:
            cur = Turn(cur.speaker, cur.start, max(cur.end, w.end))
            continue
        if cur is not None:
            runs.append(cur)
        cur = Turn(w.speaker, w.start, w.end)
    if cur is not None:
        runs.append(cur)
    return runs


def _assign_words(words: list[Word], turns: list[Turn]) -> list[list[Word]]:
    """Each word's TEXT goes to the same-speaker turn covering its centre,
    else the nearest same-speaker turn, however far. Turn timing is never
    changed: on the 20 benchmark meetings, stretching turns to reach their
    words gave DER 0.499 and giving far words their own segments 0.119,
    against 0.054 for the turns as-is. A word whose speaker has no turn at
    all is returned in a trailing bucket (index len(turns))."""
    by_spk: dict[str, list[int]] = {}
    for idx, t in enumerate(turns):
        by_spk.setdefault(t.speaker, []).append(idx)
    buckets: list[list[Word]] = [[] for _ in range(len(turns) + 1)]
    for w in words:
        cands = by_spk.get(w.speaker)
        if not cands:
            buckets[-1].append(w)
            continue
        c = w.centre

        def dist(k: int) -> float:
            t = turns[k]
            return 0.0 if t.start <= c <= t.end else min(abs(c - t.start), abs(c - t.end))

        buckets[min(cands, key=dist)].append(w)
    return buckets


def words_to_house_json(
    words: list[Word],
    *,
    model: str,
    turns: list[Turn] | None = None,
    duration_sec: float | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """House diarization JSON (``{task, model, num_speakers, segments,
    speakers}``, the shape ``my-meeting-notes/app/services/diarize.py``
    reads). Segment timing comes from ``turns`` (the model's own
    diarization) with word text attached; see this module's docstring for
    the measurement behind that. Without turns, segments are runs of
    consecutive same-speaker words, split at pauses over
    ``SEGMENT_MAX_GAP_SEC``. ``words`` is an extension field: the
    word-level speaker attribution."""
    words = sorted(words, key=lambda w: (w.start, w.end))
    words, turns = relabel_in_order(words, sorted(turns or [], key=lambda t: (t.start, t.end)))
    if not turns:
        turns = _word_runs(words)
    buckets = _assign_words(words, turns)
    seg_src: list[tuple[Turn, list[Word]]] = list(zip(turns, buckets[:-1]))
    # Words whose speaker never got a turn: one segment per run of them.
    for run in _word_runs(buckets[-1]):
        seg_src.append((run, [w for w in buckets[-1] if w.speaker == run.speaker and run.start <= w.start <= run.end]))
    seg_src.sort(key=lambda x: (x[0].start, x[0].end))

    segments: list[dict[str, Any]] = []
    speakers: dict[str, dict[str, Any]] = {}
    for turn, ws in seg_src:
        start, end = turn.start, turn.end
        segments.append(
            {
                "id": len(segments),
                "speaker": turn.speaker,
                "label": turn.speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": " ".join(w.text for w in sorted(ws, key=lambda w: w.start)),
            }
        )
        st = speakers.setdefault(
            turn.speaker, {"id": turn.speaker, "label": turn.speaker, "total_speech_duration": 0.0, "segment_count": 0}
        )
        st["total_speech_duration"] = round(st["total_speech_duration"] + (end - start), 3)
        st["segment_count"] += 1

    doc: dict[str, Any] = {
        "task": "diarize",
        "model": model,
        "num_speakers": len(speakers),
        "segments": segments,
        "speakers": sorted(speakers.values(), key=lambda x: x["id"]),
        "words": [
            {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3), "speaker": w.speaker}
            for w in words
        ],
    }
    if duration_sec is not None:
        doc["duration"] = round(float(duration_sec), 3)
    if warnings:
        doc["warnings"] = list(warnings)
    return doc


def seam_warnings(seams: list[SeamReport], *, min_votes: int) -> list[str]:
    out = []
    for s in seams:
        if s.matched_words < min_votes:
            out.append(
                f"speaker matching at the seam around {s.midpoint:.0f}s (windows {s.index}->{s.index + 1}, "
                f"overlap {s.overlap_start:.0f}-{s.overlap_end:.0f}s) rests on only {s.matched_words} "
                f"matched words (threshold {min_votes}); speaker labels after this point may be "
                "split into new speakers or swapped"
            )
    return out
