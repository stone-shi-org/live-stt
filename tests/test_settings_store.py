"""SQLite-backed runtime settings store (LS-5). Uses ":memory:" throughout --
live_stt/db.py explicitly supports it and it keeps these tests fast/offline,
consistent with CLAUDE.md's unit-test philosophy (no real file I/O needed to
exercise the store's actual logic).
"""

from __future__ import annotations

import pytest

from live_stt.config import Settings
from live_stt.settings_store import HOT_RELOAD_FIELDS, SettingsStore, apply_update


def _store() -> SettingsStore:
    return SettingsStore(":memory:")


def test_migrate_if_empty_seeds_every_field_and_only_runs_once() -> None:
    store = _store()
    defaults = Settings(_env_file=None, parakeet_backend="cuda", max_concurrent_calls=7)

    ran = store.migrate_if_empty(defaults)
    assert ran is True

    overrides = store.load_overrides()
    assert overrides["parakeet_backend"] == "cuda"
    assert overrides["max_concurrent_calls"] == 7
    # Every Settings field got a row, not just the non-default ones.
    assert set(overrides) == set(defaults.model_dump())

    # Second call is a no-op even with different defaults -- migration is
    # "seed once," not "keep resyncing from env on every boot."
    ran_again = store.migrate_if_empty(Settings(_env_file=None, max_concurrent_calls=99))
    assert ran_again is False
    assert store.load_overrides()["max_concurrent_calls"] == 7


def test_set_overwrites_a_single_key_without_touching_others() -> None:
    store = _store()
    store.migrate_if_empty(Settings(_env_file=None))
    store.set("whisper_backend", "cuda")
    overrides = store.load_overrides()
    assert overrides["whisper_backend"] == "cuda"
    assert overrides["parakeet_backend"] == "cpu"  # untouched


def test_apply_update_rejects_unknown_key() -> None:
    store = _store()
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="unknown setting"):
        apply_update(store, settings, {"not_a_real_field": 1})


def test_apply_update_rejects_a_value_that_fails_cross_field_validation() -> None:
    # _check_grace_period: finalize_timeout_sec must not exceed
    # drain_timeout_sec -- must still be enforced when the change comes
    # through the settings store, not just at direct Settings() construction.
    store = _store()
    settings = Settings(_env_file=None, finalize_timeout_sec=30.0, drain_timeout_sec=300.0)
    with pytest.raises(ValueError):
        apply_update(store, settings, {"finalize_timeout_sec": 400.0})


def test_apply_update_persists_only_the_changed_key() -> None:
    store = _store()
    settings = Settings(_env_file=None)
    validated = apply_update(store, settings, {"tts_voice": "af_sky"})
    assert validated.tts_voice == "af_sky"
    assert store.load_overrides() == {"tts_voice": "af_sky"}


def test_apply_update_rejects_invalid_audio_dump_allow_pii_combination() -> None:
    # live_stt/redaction.py's validate() is the invariant this re-checks --
    # a hot change to audio_dump must not be able to produce a combination
    # the startup-only check exists specifically to prevent.
    store = _store()
    settings = Settings(_env_file=None, audio_dump="off", allow_pii=False)
    with pytest.raises(ValueError, match="LSTT_ALLOW_PII"):
        apply_update(store, settings, {"audio_dump": "always"})


def test_apply_update_allows_audio_dump_when_allow_pii_already_true() -> None:
    store = _store()
    settings = Settings(_env_file=None, audio_dump="off", allow_pii=True)
    validated = apply_update(store, settings, {"audio_dump": "always"})
    assert validated.audio_dump == "always"


def test_hot_reload_fields_includes_tts_idle_timeout_sec() -> None:
    assert "tts_idle_timeout_sec" in HOT_RELOAD_FIELDS


def test_hot_reload_fields_includes_stt_idle_timeout_sec() -> None:
    assert "stt_idle_timeout_sec" in HOT_RELOAD_FIELDS


def test_apply_update_updates_tts_idle_timeout_sec() -> None:
    store = _store()
    settings = Settings(_env_file=None)
    validated = apply_update(store, settings, {"tts_idle_timeout_sec": 300.0})
    assert validated.tts_idle_timeout_sec == 300.0
    assert store.load_overrides() == {"tts_idle_timeout_sec": 300.0}

    with pytest.raises(ValueError, match="tts_idle_timeout_sec must be non-negative"):
        apply_update(store, settings, {"tts_idle_timeout_sec": -5.0})


def test_apply_update_updates_stt_idle_timeout_sec() -> None:
    store = _store()
    settings = Settings(_env_file=None)
    validated = apply_update(store, settings, {"stt_idle_timeout_sec": 450.0})
    assert validated.stt_idle_timeout_sec == 450.0
    assert store.load_overrides() == {"stt_idle_timeout_sec": 450.0}

    with pytest.raises(ValueError, match="stt_idle_timeout_sec must be non-negative"):
        apply_update(store, settings, {"stt_idle_timeout_sec": -5.0})


def test_hot_reload_fields_excludes_allow_pii() -> None:
    # Deliberate: see settings_store.py's HOT_RELOAD_FIELDS docstring --
    # flipping the PII-logging gate live is excluded on purpose.
    assert "allow_pii" not in HOT_RELOAD_FIELDS


def test_hot_reload_fields_excludes_process_lifetime_fields() -> None:
    # A representative sample of fields live_stt/config.py's Settings
    # docstring has always documented as unsafe to change under a live
    # worker pool -- must stay restart-required.
    for field in ("max_concurrent_calls", "worker_bin", "rotate_after_sec", "vram_per_worker_mb"):
        assert field not in HOT_RELOAD_FIELDS

