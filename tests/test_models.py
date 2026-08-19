import pytest

from live_stt.models import DEFAULT_MODEL_KEY, MODELS, resolve, strip_language_tag


def test_default_model_is_the_eou_model() -> None:
    assert DEFAULT_MODEL_KEY == "realtime_eou_120m-v1"


def test_resolve_none_returns_default() -> None:
    assert resolve(None).key == DEFAULT_MODEL_KEY


def test_resolve_empty_string_returns_default() -> None:
    assert resolve("").key == DEFAULT_MODEL_KEY


def test_resolve_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        resolve("not-a-real-model")


def test_eou_model_has_eou_and_no_punctuation() -> None:
    spec = resolve("realtime_eou_120m-v1")
    assert spec.has_eou is True
    assert spec.has_punctuation is False
    assert spec.multilingual is False
    assert spec.model_chunk_ms == 160


def test_nemotron_has_no_eou_and_has_punctuation() -> None:
    spec = resolve("nemotron-3.5-asr-streaming-0.6b")
    assert spec.has_eou is False
    assert spec.has_punctuation is True
    assert spec.multilingual is True
    assert spec.model_chunk_ms == 320
    assert spec.strip_language_tag is True


def test_all_registered_models_have_a_gguf_filename() -> None:
    for spec in MODELS.values():
        assert spec.gguf_filename.endswith(".gguf")


def test_strip_language_tag_removes_locale_marker() -> None:
    assert strip_language_tag("her eyes. <en-US> It is certainly") == "her eyes. It is certainly"


def test_strip_language_tag_is_noop_without_a_tag() -> None:
    assert strip_language_tag("hello world") == "hello world"
