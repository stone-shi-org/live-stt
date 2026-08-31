import pytest

from live_stt.models import (
    DEFAULT_MODEL_KEY,
    DEFAULT_WHISPER_MODEL_KEY,
    MODELS,
    resolve,
    strip_language_tag,
)


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


def test_all_parakeet_models_have_a_gguf_filename() -> None:
    # Only the parakeet family uses real GGUF files -- whisper's use its own
    # legacy ggml *.bin format (see live_stt/models.py's docstring), so this
    # is scoped to engine="parakeet" rather than every registered model.
    for spec in MODELS.values():
        if spec.engine == "parakeet":
            assert spec.gguf_filename.endswith(".gguf")


def test_all_whisper_models_have_a_bin_filename_and_are_batch_only() -> None:
    for spec in MODELS.values():
        if spec.engine == "whisper":
            assert spec.gguf_filename.endswith(".bin")
            assert spec.streaming_capable is False
            assert spec.has_eou is False


def test_all_registered_models_have_a_nonempty_filename() -> None:
    for spec in MODELS.values():
        assert spec.gguf_filename


def test_existing_two_models_default_to_parakeet_and_streaming_capable() -> None:
    # Additive-field regression check: ModelSpec grew engine/streaming_capable
    # for the whisper family, and the two pre-existing entries must be
    # unaffected by that (defaults apply, no explicit change needed there).
    for key in ("realtime_eou_120m-v1", "nemotron-3.5-asr-streaming-0.6b"):
        spec = resolve(key)
        assert spec.engine == "parakeet"
        assert spec.streaming_capable is True


def test_default_whisper_model_key_resolves_to_a_whisper_engine_spec() -> None:
    spec = resolve(DEFAULT_WHISPER_MODEL_KEY)
    assert spec.engine == "whisper"
    assert spec.streaming_capable is False
    assert spec.key == "whisper-large-v3-turbo-q8_0"


def test_whisper_family_keys_all_resolve() -> None:
    for key in (
        "whisper-base.en-q8_0",
        "whisper-small-q8_0",
        "whisper-medium-q8_0",
        "whisper-large-v3-q5_0",
        "whisper-large-v3-turbo-q8_0",
    ):
        spec = resolve(key)
        assert spec.key == key
        assert spec.engine == "whisper"


def test_strip_language_tag_removes_locale_marker() -> None:
    assert strip_language_tag("her eyes. <en-US> It is certainly") == "her eyes. It is certainly"


def test_strip_language_tag_is_noop_without_a_tag() -> None:
    assert strip_language_tag("hello world") == "hello world"
