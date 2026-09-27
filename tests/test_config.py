import pytest

from live_stt.config import Settings


def test_defaults_load_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(__import__("os").environ):
        if key.startswith("LSTT_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.parakeet_backend == "cpu"
    assert settings.whisper_backend == "cpu"
    assert settings.parakeet_tdt_backend == "cpu"
    assert settings.default_model == "realtime_eou_120m-v1"
    assert settings.reserve_slots == 1
    assert settings.tts_model == "kokoro-82m-v1"
    assert settings.tts_voice == "af_heart"
    assert settings.tts_device == "cpu"
    assert settings.tts_vram_mb == 1500
    assert settings.tts_idle_timeout_sec == 600.0
    assert settings.stt_idle_timeout_sec == 600.0


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # LS-5: per-engine backend fields, set/overridable independently -- the
    # whole point of replacing the old single LSTT_BACKEND switch.
    monkeypatch.setenv("LSTT_PARAKEET_BACKEND", "cuda")
    monkeypatch.setenv("LSTT_MAX_CONCURRENT_CALLS", "8")
    monkeypatch.setenv("LSTT_TTS_IDLE_TIMEOUT_SEC", "120.0")
    monkeypatch.setenv("LSTT_STT_IDLE_TIMEOUT_SEC", "180.0")
    settings = Settings(_env_file=None)
    assert settings.parakeet_backend == "cuda"
    assert settings.whisper_backend == "cpu"  # untouched, independent of parakeet_backend
    assert settings.max_concurrent_calls == 8
    assert settings.tts_idle_timeout_sec == 120.0
    assert settings.stt_idle_timeout_sec == 180.0


def test_tts_idle_timeout_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="tts_idle_timeout_sec must be non-negative"):
        Settings(_env_file=None, tts_idle_timeout_sec=-1.0)


def test_stt_idle_timeout_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="stt_idle_timeout_sec must be non-negative"):
        Settings(_env_file=None, stt_idle_timeout_sec=-1.0)


def test_finalize_timeout_may_not_exceed_drain_timeout() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, finalize_timeout_sec=400.0, drain_timeout_sec=300.0)


def test_finalize_timeout_equal_to_drain_timeout_is_allowed() -> None:
    settings = Settings(_env_file=None, finalize_timeout_sec=300.0, drain_timeout_sec=300.0)
    assert settings.finalize_timeout_sec == 300.0

