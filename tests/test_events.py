from live_stt.events import worker_json_to_events


def test_empty_doc_produces_no_events() -> None:
    assert worker_json_to_events({}) == []


def test_text_only_produces_a_delta() -> None:
    events = worker_json_to_events({"text": "hello world"})
    assert len(events) == 1
    assert events[0].WhichOneof("event") == "delta"
    assert events[0].delta.text == "hello world"
    assert list(events[0].delta.words) == []


def test_words_are_converted_with_timestamps_and_confidence() -> None:
    doc = {
        "text": "hi there",
        "words": [
            {"w": "hi", "start": 0.48, "end": 0.64, "conf": 0.91},
            {"w": "there", "start": 0.64, "end": 0.9, "conf": 0.8},
        ],
    }
    events = worker_json_to_events(doc)
    delta = events[0].delta
    assert [w.text for w in delta.words] == ["hi", "there"]
    assert delta.words[0].start_sec == 0.48
    assert abs(delta.words[0].confidence - 0.91) < 1e-6  # proto float32 rounding


def test_time_offset_rebases_word_and_event_timestamps() -> None:
    doc = {
        "text": "hi",
        "words": [{"w": "hi", "start": 1.0, "end": 1.2, "conf": 0.9}],
        "events": [{"type": "eou", "t": 1.2}],
    }
    events = worker_json_to_events(doc, time_offset_sec=100.0)
    delta = events[0]
    assert delta.delta.words[0].start_sec == 101.0
    assert delta.delta.words[0].end_sec == 101.2
    eou_event = events[1]
    assert eou_event.WhichOneof("event") == "eou"
    assert eou_event.eou.at_sec == 101.2


def test_eou_and_eob_events_are_mapped_distinctly() -> None:
    doc = {"events": [{"type": "eou", "t": 1.0}, {"type": "eob", "t": 2.0}]}
    events = worker_json_to_events(doc)
    assert [e.WhichOneof("event") for e in events] == ["eou", "eob"]
    assert events[0].eou.at_sec == 1.0
    assert events[1].eob.at_sec == 2.0


def test_strip_tag_removes_language_marker_from_delta_text() -> None:
    doc = {"text": "her eyes. <en-US> It is certainly"}
    events = worker_json_to_events(doc, strip_tag=True)
    assert events[0].delta.text == "her eyes. It is certainly"


def test_strip_tag_false_leaves_tag_in_place() -> None:
    doc = {"text": "her eyes. <en-US> It is certainly"}
    events = worker_json_to_events(doc, strip_tag=False)
    assert "<en-US>" in events[0].delta.text


def test_audio_offset_sec_is_set_on_the_delta() -> None:
    events = worker_json_to_events({"text": "hi"}, audio_offset_sec=42.5)
    assert events[0].delta.audio_offset_sec == 42.5


def test_words_without_text_still_produce_a_delta() -> None:
    # words can arrive with an empty "text" if only word-finalization happened
    doc = {"words": [{"w": "hi", "start": 0.0, "end": 0.1, "conf": 0.5}]}
    events = worker_json_to_events(doc)
    assert len(events) == 1
    assert events[0].delta.words[0].text == "hi"
