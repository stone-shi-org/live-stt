from live_stt.boundary import (
    dedup_incoming_words,
    iter_batch_windows,
    merge_batch_window_words,
    rebase_time_offset,
)
from live_stt.pb.livestt.v1 import asr_pb2


def _word(start: float, end: float, text: str = "w") -> asr_pb2.Word:
    return asr_pb2.Word(text=text, start_sec=start, end_sec=end, confidence=0.9)


def test_rebase_time_offset_is_cut_minus_overlap() -> None:
    assert rebase_time_offset(t_cut_sec=100.0, overlap_sec=10.0) == 90.0


def test_dedup_drops_words_entirely_before_cut() -> None:
    words = [_word(80.0, 80.5), _word(90.0, 90.5), _word(100.0, 100.5)]
    kept = dedup_incoming_words(words, t_cut_sec=95.0)
    assert [w.start_sec for w in kept] == [100.0]


def test_dedup_keeps_words_within_slack_of_the_cut() -> None:
    # 0.03s before the cut is within the 0.05s slack -- kept, not dropped.
    words = [_word(94.97, 95.1)]
    kept = dedup_incoming_words(words, t_cut_sec=95.0)
    assert len(kept) == 1


def test_dedup_keeps_all_words_when_cut_is_before_everything() -> None:
    words = [_word(10.0, 10.5), _word(20.0, 20.5)]
    kept = dedup_incoming_words(words, t_cut_sec=0.0)
    assert len(kept) == 2


def test_dedup_drops_all_words_when_cut_is_after_everything() -> None:
    words = [_word(10.0, 10.5), _word(20.0, 20.5)]
    kept = dedup_incoming_words(words, t_cut_sec=1000.0)
    assert kept == []


def test_iter_batch_windows_single_pass_when_short_or_disabled() -> None:
    assert iter_batch_windows(1000, 2000, 100) == [(0, 1000)]
    assert iter_batch_windows(5000, 0, 100) == [(0, 5000)]
    assert iter_batch_windows(0, 1000, 100) == [(0, 0)]


def test_iter_batch_windows_slides_with_overlap_and_bounds_every_window() -> None:
    # 461s at 16kHz with 90s windows and 5s overlap (stride 85s):
    total = int(460.9 * 16000)
    win = 90 * 16000
    ovl = 5 * 16000
    windows = iter_batch_windows(total, win, ovl)
    assert len(windows) == 6
    assert windows[0] == (0, win)
    assert windows[1] == (85 * 16000, 175 * 16000)
    assert windows[-1][1] == total
    assert all(end - start <= win for start, end in windows)
    for (_, prev_end), (cur_start, _) in zip(windows, windows[1:]):
        assert prev_end - cur_start == ovl


def test_merge_batch_window_words_aligns_matching_overlap_phrases() -> None:
    acc = [
        _word(83.0, 83.8, "the"),
        _word(84.0, 84.8, "quick"),
        _word(85.2, 85.9, "brown"),
        _word(86.1, 86.8, "fox"),
        _word(87.0, 87.8, "jumps"),
        _word(88.0, 88.8, "over"),
    ]
    inc = [
        _word(85.25, 85.95, "brown,"),
        _word(86.12, 86.82, "Fox"),
        _word(87.05, 87.82, "jumps"),
        _word(88.02, 88.85, "over"),
        _word(89.1, 89.7, "the"),
        _word(90.0, 90.6, "lazy"),
        _word(91.0, 91.6, "dog"),
    ]
    merged = merge_batch_window_words(acc, inc, win_start_sec=85.0, prev_end_sec=90.0)
    assert [w.text.lower().strip(",") for w in merged] == [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "the",
        "lazy",
        "dog",
    ]
    starts = [w.start_sec for w in merged]
    assert starts == sorted(starts)


def test_merge_batch_window_words_falls_back_to_midpoint_when_no_lexical_match() -> None:
    acc = [
        _word(84.0, 84.8, "alpha"),
        _word(86.0, 86.8, "beta"),
        _word(89.0, 89.8, "gamma"),
    ]
    inc = [
        _word(85.5, 86.2, "delta"),
        _word(88.0, 88.7, "epsilon"),
        _word(91.0, 91.7, "zeta"),
    ]
    # Midpoint of [85.0, 90.0] is 87.5s: keeps alpha, beta (<87.5) and epsilon, zeta (>=87.5)
    merged = merge_batch_window_words(acc, inc, win_start_sec=85.0, prev_end_sec=90.0)
    assert [w.text for w in merged] == ["alpha", "beta", "epsilon", "zeta"]


def test_merge_batch_window_words_suppresses_boundary_duplicate_word() -> None:
    acc = [_word(86.0, 87.4, "meeting")]
    inc = [_word(87.6, 88.2, "Meeting"), _word(88.5, 89.0, "notes")]
    merged = merge_batch_window_words(acc, inc, win_start_sec=85.0, prev_end_sec=90.0)
    assert [w.text.lower() for w in merged] == ["meeting", "notes"]

