from live_stt.boundary import dedup_incoming_words, rebase_time_offset
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
