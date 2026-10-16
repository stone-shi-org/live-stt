"""VRAM admission (live_stt/vram.py): per-model requirement, card-relative
reserve, reclaim-before-reject, fail-open. No GPU: gpu.* is monkeypatched."""

from __future__ import annotations

import threading

import pytest

from live_stt import gpu, models, tts, vram
from live_stt.config import Settings


@pytest.fixture(autouse=True)
def _no_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vram, "_total_mb_cache", None)
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: None)
    monkeypatch.setattr(vram, "RECLAIM_POLL_SEC", 0.0)


def _s(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_reserve_is_card_relative_with_an_absolute_floor() -> None:
    s = _s(vram_reserve_mb=256, vram_reserve_pct=10.0)
    assert vram.reserve_mb(s, total_mb=4096) == 409  # P1000
    assert vram.reserve_mb(s, total_mb=24576) == 2457  # RTX 3090
    assert vram.reserve_mb(s, total_mb=1024) == 256  # floor wins on a tiny card
    assert vram.reserve_mb(s) == 256  # no card visible -> floor


def test_need_prefers_override_then_model_figure_then_fallback() -> None:
    s = _s(vram_overrides={"m": 777})
    assert vram.need_mb(s, "m", 1000, 3000) == 777
    assert vram.need_mb(s, "other", 1000, 3000) == 1000
    assert vram.need_mb(s, "other", None, 3000) == 3000


def test_every_registered_asr_model_has_its_own_figure() -> None:
    # The whole point: no ASR model falls back to the flat per-worker guess.
    assert all(spec.vram_mb for spec in models.MODELS.values())


def test_small_model_fits_a_4gb_card_that_the_old_flat_gate_rejected(monkeypatch) -> None:
    # Old gate: 3000 + 2000 = 5000 MB for ANY model -> rejected with 4004 free.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 4004)
    spec = models.MODELS["whisper-large-v3-turbo-q8_0"]
    a = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000)
    assert a.ok and a.required_mb == spec.vram_mb + 409


def test_fails_open_when_nvidia_smi_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: None)
    assert vram.admit(_s(), model_key="m", model_mb=10**6, fallback_mb=0).ok


def test_short_with_nothing_to_reclaim_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda: 0)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert not a.ok and a.free_mb == 500 and a.required_mb == 1256
    assert a.message("diarization") == "insufficient VRAM for diarization: 500MB free, 1256MB required"


def test_reclaims_idle_workers_then_admits_once_vram_is_actually_free(monkeypatch) -> None:
    readings = iter([500, 600, 2000])  # driver frees the memory a moment after the kill
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: next(readings))
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda: 2)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert a.ok and a.reclaimed == 2 and a.free_mb == 2000


def test_reclaim_that_does_not_free_enough_still_rejects(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda: 1)
    monkeypatch.setattr(vram, "RECLAIM_WAIT_SEC", 0.05)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert not a.ok and a.reclaimed == 1


def test_reclaim_disabled_never_kills_anything(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda: pytest.fail("must not reclaim"))
    assert not vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0, reclaim=False).ok


def test_tts_stop_if_idle_never_blocks_on_a_busy_worker() -> None:
    mgr = tts.TTSWorkerManager()
    held, release = threading.Event(), threading.Event()

    def synthesizing() -> None:  # stands in for synthesize(), which holds the lock throughout
        with mgr._lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=synthesizing)
    t.start()
    held.wait(5)
    try:
        assert mgr.stop_if_idle() is False  # returned at once, did not wait for the lock
    finally:
        release.set()
        t.join()


def test_tts_stop_if_idle_is_a_noop_with_no_worker() -> None:
    assert tts.TTSWorkerManager().stop_if_idle() is False
