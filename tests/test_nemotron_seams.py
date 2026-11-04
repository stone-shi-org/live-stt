"""live_stt/nemotron_seams.py + the pure parts of live_stt/nemotron.py
(tier table, duration plan, WAV slicing, request payload) -- LS-19."""

from __future__ import annotations

import io
import wave

import pytest

from live_stt import nemotron
from live_stt import nemotron_seams as seams
from live_stt.config import Settings
from live_stt.nemotron_seams import Word


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --------------------------------------------------------------------------- #
# extract_words


def _doc(words, status="aligned", text=None):
    return {
        "choices": [{"message": {"content": text if text is not None else " ".join(w["word"] for w in words)}}],
        "transcription": {"words": words, "word_timestamp_alignment": {"status": status}},
    }


def test_extract_words_offsets_and_sorts():
    doc = _doc(
        [
            {"word": "b", "start": 2.0, "end": 2.4, "speaker": "spk:1"},
            {"word": "a", "start": 1.0, "end": 1.2, "speaker": "spk:0"},
        ]
    )
    words = seams.extract_words(doc, offset_sec=100.0)
    assert [(w.text, w.start, w.speaker) for w in words] == [("a", 101.0, "spk:0"), ("b", 102.0, "spk:1")]


def test_extract_words_refuses_text_without_aligned_timestamps():
    # The 55-minute spike run: HTTP 200, text and inline tags, no word times.
    with pytest.raises(seams.NemotronResponseError, match="aligned word timestamps"):
        seams.extract_words(_doc([], status="failed", text="<spk:0> hello there"))
    with pytest.raises(seams.NemotronResponseError):
        seams.extract_words(_doc([], status=None, text="<spk:0> hello there"))


def test_extract_words_silent_audio_is_empty_not_an_error():
    assert seams.extract_words(_doc([], text="")) == []


# --------------------------------------------------------------------------- #
# overlap vote


def _w(t, start, spk):
    return Word(t, start, start + 0.3, spk)


def test_match_seam_maps_by_vote_strength_one_to_one():
    prev = [_w(f"x{i}", 100 + i, "G0" if i < 6 else "G1") for i in range(10)]
    # Next window: same words, labels permuted; small timing drift.
    nxt = [_w(f"X{i}.", 100 + i + 0.2, "L1" if i < 6 else "L0") for i in range(10)]
    mapping, matched = seams.match_seam(prev, nxt, overlap_start=100, overlap_end=110)
    assert matched == 10
    assert mapping == {"L1": "G0", "L0": "G1"}


def test_match_seam_ignores_words_outside_tolerance_or_overlap():
    prev = [_w("hello", 100.0, "G0"), _w("world", 120.0, "G0")]
    nxt = [_w("hello", 100.7, "L0"), _w("world", 120.0, "L0")]  # 0.7s > 0.6s; second outside overlap
    mapping, matched = seams.match_seam(prev, nxt, overlap_start=99, overlap_end=110)
    assert matched == 0 and mapping == {}


def test_match_seam_conflicting_votes_resolved_by_strength():
    prev = [_w(f"a{i}", 10 + i, "G0") for i in range(5)] + [_w(f"b{i}", 20 + i, "G1") for i in range(2)]
    # L0 votes 5 for G0; L1 votes 2 for G0 (confusion) -> L1 must not also take G0.
    nxt = [_w(f"a{i}", 10 + i, "L0") for i in range(5)] + [_w(f"b{i}", 20 + i, "L1") for i in range(2)]
    nxt[0] = _w("a0", 10, "L1")
    mapping, matched = seams.match_seam(prev, nxt, overlap_start=0, overlap_end=30)
    assert mapping["L0"] == "G0"
    assert mapping["L1"] == "G1"
    assert matched == 7


def test_stitch_windows_joins_at_midpoint_and_new_label_for_unvoted_speaker():
    w0 = [_w(f"w{i}", i, "spk:0") for i in range(0, 20)]
    t0 = [seams.Turn("spk:0", 0.0, 20.0)]
    # Window 1 covers 10..30; overlap 10..20, midpoint 15. Label spk:5 is a
    # speaker that only appears after the overlap -> new global speaker.
    w1 = [_w(f"w{i}", i, "spk:3") for i in range(10, 25)] + [_w(f"z{i}", i, "spk:5") for i in range(25, 30)]
    t1 = [seams.Turn("spk:3", 10.0, 25.0), seams.Turn("spk:5", 25.0, 30.0)]
    merged, turns, reports = seams.stitch_windows([(0, 20, w0, t0), (10, 30, w1, t1)])
    assert [w.text for w in merged][:16] == [f"w{i}" for i in range(16)]
    assert len([w for w in merged if w.text == "w15"]) == 1  # no duplicate at the cut
    assert reports[0].matched_words == 10
    assert reports[0].mapping == {"spk:3": "SPEAKER_00"}
    assert reports[0].new_speakers == ["SPEAKER_01"]
    assert {w.speaker for w in merged if w.text.startswith("z")} == {"SPEAKER_01"}
    # Turns clipped (not duplicated) at the midpoint.
    assert turns == [
        seams.Turn("SPEAKER_00", 0.0, 15.0),
        seams.Turn("SPEAKER_00", 15.0, 25.0),
        seams.Turn("SPEAKER_01", 25.0, 30.0),
    ]


def test_extract_turns_labels_like_words():
    doc = {"transcription": {"diarization_timestamps": [{"speaker": 1, "start": 0.5, "end": 2.0}]}}
    assert seams.extract_turns(doc, offset_sec=10) == [seams.Turn("spk:1", 10.5, 12.0)]


def test_segments_use_turn_timing_with_word_text():
    # Tight CTC word spans; the turn covers the speech between them.
    words = [_w("Hello", 1.0, "spk:0"), _w("there.", 3.0, "spk:0"), _w("Yes.", 5.2, "spk:1"), _w("ok", 9.0, "spk:1")]
    turns = [
        seams.Turn("spk:0", 0.8, 4.0),
        seams.Turn("spk:1", 4.9, 6.0),
        seams.Turn("spk:0", 6.0, 7.0),  # backchannel whose words went elsewhere
        seams.Turn("spk:1", 7.5, 8.5),  # "ok" lands just after it -> nearest same-speaker turn
    ]
    words.append(_w("far", 40.0, "spk:0"))  # far from any spk:0 turn: text joins the nearest one
    words.append(_w("solo", 50.0, "spk:9"))  # a speaker with no turn at all: its own segment
    doc = seams.words_to_house_json(words, model="m", turns=turns)
    # Segment timing is exactly the model's turns -- never stretched to reach words.
    assert [(s["speaker"], s["start"], s["end"], s["text"]) for s in doc["segments"]] == [
        ("SPEAKER_00", 0.8, 4.0, "Hello there."),
        ("SPEAKER_01", 4.9, 6.0, "Yes."),
        ("SPEAKER_00", 6.0, 7.0, "far"),
        ("SPEAKER_01", 7.5, 8.5, "ok"),
        ("SPEAKER_02", 50.0, 50.3, "solo"),
    ]
    assert doc["num_speakers"] == 3


def test_collapsed_word_speakers_are_relabelled_from_turns():
    # MTG_32256 shape: turns name 3 speakers, every word tagged spk:0.
    words = [_w("hello", 1.0, "spk:0"), _w("hi", 5.1, "spk:0"), _w("yes", 9.0, "spk:0"), _w("gap", 12.0, "spk:0")]
    turns = [seams.Turn("spk:0", 0.5, 3.0), seams.Turn("spk:1", 4.0, 7.0), seams.Turn("spk:2", 8.0, 10.0)]
    fixed, repaired = seams.repair_collapsed_word_speakers(words, turns)
    assert repaired
    # Covering turn wins; a word outside every turn takes the nearest one.
    assert [w.speaker for w in fixed] == ["spk:0", "spk:1", "spk:2", "spk:2"]
    assert [(w.text, w.start, w.end) for w in fixed] == [(w.text, w.start, w.end) for w in words]
    doc = seams.words_to_house_json(fixed, model="m", turns=turns)
    assert [(s["speaker"], s["text"]) for s in doc["segments"]] == [
        ("SPEAKER_00", "hello"), ("SPEAKER_01", "hi"), ("SPEAKER_02", "yes gap"),
    ]


def test_collapsed_word_speaker_repair_prefers_larger_overlap():
    words = [Word("both", 2.6, 3.4, "spk:0")]  # 0.4 s inside spk:0's turn, 0.5 s inside spk:1's
    turns = [seams.Turn("spk:0", 0.0, 3.0), seams.Turn("spk:1", 2.9, 6.0)]
    fixed, repaired = seams.repair_collapsed_word_speakers(words, turns)
    assert repaired and fixed[0].speaker == "spk:1"


@pytest.mark.parametrize(
    "words, turns",
    [
        # Words already carry several speakers: ordinary disagreement is left alone.
        ([_w("a", 1.0, "spk:0"), _w("b", 5.0, "spk:0"), _w("c", 5.5, "spk:1")],
         [seams.Turn("spk:0", 0.0, 2.0), seams.Turn("spk:1", 4.0, 7.0)]),
        # A genuine single-speaker recording.
        ([_w("a", 1.0, "spk:0"), _w("b", 5.0, "spk:0")], [seams.Turn("spk:0", 0.0, 7.0)]),
        # No turns at all.
        ([_w("a", 1.0, "spk:0")], []),
    ],
)
def test_word_speaker_repair_only_fires_on_full_collapse(words, turns):
    fixed, repaired = seams.repair_collapsed_word_speakers(words, turns)
    assert not repaired and fixed is words


def test_seam_warnings_flag_low_vote_seams_only():
    reports = [
        seams.SeamReport(0, 2100, 2400, matched_words=250),
        seams.SeamReport(1, 4200, 4500, matched_words=42),
    ]
    out = seams.seam_warnings(reports, min_votes=100)
    assert len(out) == 1
    assert "4350s" in out[0] and "42 matched words" in out[0]


def test_without_turns_segments_are_word_runs_split_on_speaker_and_pause():
    words = [
        _w("Hi", 0.0, "spk:1"),
        _w("there.", 0.4, "spk:1"),
        _w("Hello.", 1.0, "spk:0"),
        _w("Again", 5.0, "spk:0"),  # > SEGMENT_MAX_GAP_SEC after previous
    ]
    doc = seams.words_to_house_json(words, model="m", duration_sec=6.0)
    assert doc["task"] == "diarize" and doc["model"] == "m"
    assert doc["num_speakers"] == 2
    assert [(s["speaker"], s["text"]) for s in doc["segments"]] == [
        ("SPEAKER_00", "Hi there."),
        ("SPEAKER_01", "Hello."),
        ("SPEAKER_01", "Again"),
    ]
    assert all(s["text"] for s in doc["segments"])
    assert {sp["id"] for sp in doc["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}
    assert "warnings" not in doc


# --------------------------------------------------------------------------- #
# tiers / plan


def test_select_tier_24gb_rtx3090():
    tier = nemotron.select_tier(24576, 8.6)
    assert tier.verified and tier.decoder_filename.endswith("AD-IQ4_NL.gguf")
    assert set(tier.profiles) == {"short", "long"}
    assert tier.profiles["short"].salm["max_sequences"] == 4
    assert tier.profiles["long"].salm["max_context"] == 73728


@pytest.mark.parametrize(
    "total,cc,match",
    [
        (None, None, "CUDA GPU"),
        (24576, 7.5, "compute capability"),
        (4096, 8.6, "unsupported"),
        (49152, 8.9, "untested"),  # Q8_0 tier: declared, not configured
        (16384, 8.6, "untested"),
        (32768, 12.0, "untested"),  # Blackwell NVFP4
    ],
)
def test_select_tier_refuses_unverified_or_unsupported(total, cc, match):
    with pytest.raises(nemotron.NemotronUnavailable, match=match):
        nemotron.select_tier(total, cc)


def test_plan_job_by_duration():
    s = _settings()
    assert nemotron.plan_job(s, 25 * 60) == ("short", [(0.0, 1500.0)])
    assert nemotron.plan_job(s, 40 * 60) == ("long", [(0.0, 2400.0)])
    assert nemotron.plan_job(s, 45 * 60)[0] == "long"
    prof, windows = nemotron.plan_job(s, 65 * 60)
    assert prof == "long"
    assert windows == [(0.0, 2400.0), (2100.0, 3900.0)]
    # Nothing over 45 min is ever attempted in one pass.
    for minutes in (45.01, 50, 55, 89, 149):
        _p, ws = nemotron.plan_job(s, minutes * 60)
        assert max(e - st for st, e in ws) <= s.nemotron_long_max_sec


def test_plan_job_refuses_over_max_call_sec_and_empty():
    with pytest.raises(nemotron.NemotronBadRequest, match="max_call_sec"):
        nemotron.plan_job(_settings(), 9001)
    with pytest.raises(nemotron.NemotronBadRequest):
        nemotron.plan_job(_settings(), 0)


def test_settings_reject_window_over_long_limit():
    with pytest.raises(ValueError, match="nemotron_window_sec"):
        _settings(nemotron_window_sec=3000)
    with pytest.raises(ValueError, match="overlap"):
        _settings(nemotron_window_overlap_sec=2400)


# --------------------------------------------------------------------------- #
# audio + payload


def _wav(seconds: int, rate: int = 100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(int(k + 1).to_bytes(2, "little") * rate for k in range(seconds)))
    return buf.getvalue()


def test_parse_wav_and_window_slices():
    audio = nemotron.parse_wav(_wav(30))
    assert audio.duration_sec == 30
    assert audio.window_bytes(0, 30) is audio.raw
    piece = audio.window_bytes(10, 20)
    with wave.open(io.BytesIO(piece)) as wf:
        assert wf.getnframes() == 1000
        first = int.from_bytes(wf.readframes(1), "little")
    assert first == 11  # absolute second 10 -> sample value 11


def test_parse_wav_rejects_non_wav():
    with pytest.raises(nemotron.NemotronBadRequest, match="PCM WAV"):
        nemotron.parse_wav(b"ID3 not a wav at all")


def test_chat_payload_always_carries_explicit_speaker_instruction():
    payload = nemotron.build_chat_payload(b"RIFF....")
    assert payload["timestamp_granularities"] == ["word"]
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": nemotron.SPEAKER_INSTRUCTION}
    assert "<spk:7>" in nemotron.SPEAKER_INSTRUCTION
    assert content[1]["type"] == "input_audio"


def test_salm_config_yaml():
    y = nemotron.salm_config_yaml(nemotron.PROFILES_24GB["long"])
    assert "asr:\n  salm:\n" in y
    assert "    max_sequences: 1\n" in y and "    encoder_ahead_seconds: 600\n" in y
