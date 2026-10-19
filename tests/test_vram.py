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
    vram.clear_reservations()


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
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 0)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert not a.ok and a.free_mb == 500 and a.required_mb == 1256
    assert a.message("diarization") == "insufficient VRAM for diarization: 500MB free, 1256MB required"


def test_reclaims_idle_workers_then_admits_once_vram_is_actually_free(monkeypatch) -> None:
    readings = iter([500, 600, 2000])  # driver frees the memory a moment after the kill
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: next(readings))
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 2)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert a.ok and a.reclaimed == 2 and a.free_mb == 2000


def test_reclaim_that_does_not_free_enough_still_rejects(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 1)
    monkeypatch.setattr(vram, "RECLAIM_WAIT_SEC", 0.05)
    a = vram.admit(_s(), model_key="m", model_mb=1000, fallback_mb=0)
    assert not a.ok and a.reclaimed == 1


def test_reclaim_disabled_never_kills_anything(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: pytest.fail("must not reclaim"))
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


def test_parakeet_tdt_vram_budget_includes_window_compute_buffer(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    spec = models.MODELS["parakeet-tdt-0.6b-v3"]
    # 90s default window * 7 MB/s = 630 MB compute + 1100 MB weights = 1730 MB
    assert vram.compute_mb(_s(), spec.key) == 630
    assert vram.need_mb(_s(), spec.key, spec.vram_mb, 3000) == 1100 + 630
    # On a 4 GiB Quadro P1000 (reserve 409 MB), required_mb = 2139 MB <= 4004 MB free
    assert vram.required_mb(_s(), spec.key, spec.vram_mb, 3000) == 1730 + 409

    # Overriding batch_window_sec scales the compute allowance
    assert vram.compute_mb(_s(batch_window_sec=60.0), spec.key) == 420
    # Short audio (< window) caps compute allowance to audio_sec
    assert vram.compute_mb(_s(), spec.key, audio_sec=10.0) == 70


def test_admit_already_warm_only_charges_compute_buffer_and_keeps_same_model_standby(monkeypatch) -> None:
    # Quadro P1000 (4096 MB total, 409 MB reserve) with a warm TDT standby
    # worker already holding 1100 MB of weights in VRAM -> ~1700 MB free.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 1700)
    reclaimed_excludes: list[str | None] = []
    monkeypatch.setattr(
        vram,
        "reclaim_idle_workers",
        lambda *, exclude_model_key=None: reclaimed_excludes.append(exclude_model_key) or 0,
    )
    spec = models.MODELS["parakeet-tdt-0.6b-v3"]

    # Cold admission would need 1100 + 630 + 409 = 2139 MB > 1700 MB -> rejected
    cold = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000, already_warm=False)
    assert not cold.ok

    # Warm admission only needs compute (630 MB) + reserve (409 MB) = 1039 MB <= 1700 MB -> admitted!
    warm = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000, already_warm=True)
    assert warm.ok
    assert warm.required_mb == 630 + 409
    warm.release()


def test_can_prewarm_standby_blocks_duplicate_worker_when_active_compute_is_reserved(monkeypatch) -> None:
    # Reproduces the exact LS-14 Quadro P1000 scenario: an active TDT worker is
    # running (weights in VRAM, 1700 MB currently free before finalize() allocates
    # its 630 MB compute buffer). Pre-warming a second standby TDT worker (1100 MB)
    # alongside the active 630 MB compute reservation + 409 MB reserve requires
    # 2139 MB > 1700 MB free, so can_prewarm_standby must return False.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 1700)
    vram.reserve_active_compute("session-1", 630)
    try:
        assert vram.active_compute_mb() == 630
        assert vram.can_prewarm_standby(_s(), model_key="parakeet-tdt-0.6b-v3") is False
    finally:
        vram.release_active_compute("session-1")
    assert vram.active_compute_mb() == 0


def test_concurrent_admit_calls_admit_only_what_fits_before_workers_load(monkeypatch) -> None:
    # LS-17: 22 concurrent admit() calls arrive before any worker has allocated
    # CUDA memory (gpu.free_vram_mb() still reports the pre-load 1800 MB).
    # With total=4096 (reserve=409), usable budget = 1800 - 409 = 1391 MB.
    # For realtime_eou_120m-v1 (vram_mb=260), floor(1391 / 260) = 5 fit.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    gpu_state = {"free": 1800}
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: gpu_state["free"])
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 0)

    spec = models.MODELS["realtime_eou_120m-v1"]
    assert spec.vram_mb == 260

    barrier = threading.Barrier(22)
    results: list[vram.Admission] = []
    results_lock = threading.Lock()

    def _caller() -> None:
        barrier.wait(timeout=5)
        adm = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000)
        with results_lock:
            results.append(adm)

    threads = [threading.Thread(target=_caller) for _ in range(22)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    admitted = [r for r in results if r.ok]
    rejected = [r for r in results if not r.ok]
    assert len(admitted) == 5  # floor((1800 - 409) / 260) = 5
    assert len(rejected) == 17
    assert vram.reserved_mb() == 5 * 260
    assert vram.active_reservations_count() == 5

    # Even after the 5 admitted workers actually load on the GPU (dropping
    # nvidia-smi free VRAM by 5 * 260 = 1300 MB -> 500 MB), they are not
    # double-counted, and a 6th request still does not fit (500 < 260 + 409).
    gpu_state["free"] = 1800 - 5 * 260  # 500 MB
    sixth = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000)
    assert not sixth.ok

    # Releasing one worker's reservation (and freeing its GPU memory) allows
    # exactly one new request to be admitted.
    admitted[0].release()
    gpu_state["free"] += 260  # 760 MB
    next_adm = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000)
    assert next_adm.ok

    # Release all remaining reservations
    next_adm.release()
    for r in admitted[1:]:
        r.release()
    assert vram.reserved_mb() == 0
    assert vram.active_reservations_count() == 0


def test_only_one_concurrent_request_can_claim_warm_standby_discount(monkeypatch) -> None:
    # LS-17: If 1 warm standby worker is in the pool when a burst arrives,
    # multiple callers may pass already_warm=True concurrently. Only the first
    # may claim the 0-weights discount; subsequent concurrent callers must
    # reserve cold weights because they will spawn fresh workers.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    # 500 MB free: enough for 1 warm worker (0 weights + 409 reserve = 409 <= 500),
    # not enough for a cold worker (260 weights + 409 reserve = 669 > 500).
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 500)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 0)
    spec = models.MODELS["realtime_eou_120m-v1"]

    first = vram.admit(
        _s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000, already_warm=True
    )
    assert first.ok
    assert first.reserved_mb == 0
    assert first.required_mb == 409

    # Second concurrent caller passing already_warm=True cannot reuse the same
    # single standby worker and must be gated on cold weights (669 > 500).
    second = vram.admit(
        _s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000, already_warm=True
    )
    assert not second.ok
    assert second.required_mb == 260 + 409

    # Once the first request finishes and releases its claim, a new warm request fits again.
    first.release()
    third = vram.admit(
        _s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000, already_warm=True
    )
    assert third.ok
    third.release()


def test_can_prewarm_standby_accounts_for_inflight_admission_reservations(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 900)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 0)
    spec = models.MODELS["realtime_eou_120m-v1"]

    # Before any reservation: 900 MB >= 260 + 409 = 669 MB -> standby pre-warm allowed
    assert vram.can_prewarm_standby(_s(), model_key=spec.key) is True

    # After 1 cold admission reserves 260 MB (effective free = 640 MB < 669 MB):
    # standby pre-warm is blocked until the active request releases its reservation.
    adm = vram.admit(_s(), model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=3000)
    assert adm.ok
    assert vram.can_prewarm_standby(_s(), model_key=spec.key) is False

    adm.release()
    assert vram.can_prewarm_standby(_s(), model_key=spec.key) is True


def test_warn_if_concurrency_exceeds_gpu_logs_warning(monkeypatch, caplog) -> None:
    import logging
    from live_stt.server import _warn_if_concurrency_exceeds_gpu

    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    # 4096 - 409 reserve = 3687 MB usable -> floor(3687 / 260) = 14 workers
    with caplog.at_level(logging.WARNING):
        _warn_if_concurrency_exceeds_gpu(_s(parakeet_backend="cuda", max_concurrent_calls=20))
    assert any("max_concurrent_calls=20 exceeds estimated GPU capacity" in r.message for r in caplog.records)
    assert any(
        "max_concurrent_calls=20 but only 14 realtime_eou_120m-v1 calls fit in 3687MB VRAM" in r.message
        for r in caplog.records
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_if_concurrency_exceeds_gpu(_s(parakeet_backend="cuda", max_concurrent_calls=14))
        _warn_if_concurrency_exceeds_gpu(_s(parakeet_backend="cpu", max_concurrent_calls=20))
    assert not caplog.records


def test_gpu_capacity_reports_expected_figures_and_agrees_with_admit_burst(monkeypatch) -> None:
    # LS-18: On a 4096 MB card (reserve 409 MB -> budget 3687 MB):
    # - realtime_eou_120m-v1: per_call_mb = 260, max_calls = 14, effective_max_calls = 14 (with max_concurrent_calls=20)
    # - parakeet-tdt-0.6b-v3: per_call_mb = 1100 + 630 = 1730, max_calls = 2, effective_max_calls = 2
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 4096)
    monkeypatch.setattr(vram, "reclaim_idle_workers", lambda *a, **kw: 0)

    s = _s(
        parakeet_backend="cuda",
        parakeet_tdt_backend="cuda",
        whisper_backend="cpu",
        max_concurrent_calls=20,
    )
    cap = vram.gpu_capacity(s)
    assert cap is not None
    assert cap["budget_mb"] == 3687
    assert cap["reserved_mb"] == 0
    # Only GPU-backed models are included (whisper_backend="cpu" is omitted).
    assert "realtime_eou_120m-v1" in cap["models"]
    assert "parakeet-tdt-0.6b-v3" in cap["models"]
    assert "whisper-base.en-q8_0" not in cap["models"]

    eou_cap = cap["models"]["realtime_eou_120m-v1"]
    assert eou_cap == {
        "per_call_mb": 260,
        "max_calls": 14,
        "effective_max_calls": 14,
    }
    tdt_cap = cap["models"]["parakeet-tdt-0.6b-v3"]
    assert tdt_cap == {
        "per_call_mb": 1730,
        "max_calls": 2,
        "effective_max_calls": 2,
    }
    assert vram.effective_max_calls(s) == 14

    # When max_concurrent_calls < max_calls, effective_max_calls is clamped to max_concurrent_calls.
    s_capped = _s(parakeet_backend="cuda", max_concurrent_calls=8)
    cap_capped = vram.gpu_capacity(s_capped)
    assert cap_capped is not None
    assert cap_capped["models"]["realtime_eou_120m-v1"]["max_calls"] == 14
    assert cap_capped["models"]["realtime_eou_120m-v1"]["effective_max_calls"] == 8
    assert vram.effective_max_calls(s_capped) == 8

    # Verify agreement with vram.admit() in a concurrent burst of N + 1 (15 calls):
    spec = models.MODELS["realtime_eou_120m-v1"]
    n_burst = eou_cap["max_calls"] + 1  # 15
    barrier = threading.Barrier(n_burst)
    results: list[vram.Admission] = []
    results_lock = threading.Lock()

    def _caller() -> None:
        barrier.wait(timeout=5)
        adm = vram.admit(s, model_key=spec.key, model_mb=spec.vram_mb, fallback_mb=s.vram_per_worker_mb)
        with results_lock:
            results.append(adm)

    threads = [threading.Thread(target=_caller) for _ in range(n_burst)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    admitted = [r for r in results if r.ok]
    rejected = [r for r in results if not r.ok]
    assert len(admitted) == eou_cap["max_calls"] == 14
    assert len(rejected) == 1

    active_cap = vram.gpu_capacity(s)
    assert active_cap is not None
    assert active_cap["reserved_mb"] == 14 * eou_cap["per_call_mb"]

    for r in admitted:
        r.release()
    assert vram.gpu_capacity(s)["reserved_mb"] == 0


def test_gpu_capacity_with_custom_model_specs_and_cpu_none(monkeypatch) -> None:
    # On CPU backends or when no GPU is visible, gpu_capacity and effective_max_calls return None.
    assert vram.gpu_capacity(_s(parakeet_backend="cuda")) is None  # total_vram_mb is None
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    assert vram.gpu_capacity(_s(parakeet_backend="cpu", whisper_backend="cpu", parakeet_tdt_backend="cpu")) is None
    assert vram.effective_max_calls(_s(parakeet_backend="cpu")) is None

    # Custom ModelSpec list with fake figures (250 MB streaming, 900 MB batch with compute)
    fake_specs = [
        models.ModelSpec(
            key="fake-stream",
            vram_mb=250,
            gguf_filename="fake-stream.gguf",
            model_chunk_ms=160,
            has_eou=True,
            has_punctuation=False,
            multilingual=False,
            assumed_leak_mb_per_audio_sec=0.0,
            engine="parakeet",
        ),
        models.ModelSpec(
            key="fake-batch",
            vram_mb=600,
            gguf_filename="fake-batch.gguf",
            model_chunk_ms=1000,
            has_eou=False,
            has_punctuation=True,
            multilingual=True,
            assumed_leak_mb_per_audio_sec=0.0,
            engine="parakeet_tdt",
            streaming_capable=False,
            batch_window_sec=60.0,
            vram_compute_mb_per_sec=5.0,  # +300 MB compute -> 900 MB total
        ),
    ]
    cap = vram.gpu_capacity(
        _s(parakeet_backend="cuda", parakeet_tdt_backend="cuda", max_concurrent_calls=20),
        model_specs=fake_specs,
    )
    assert cap == {
        "budget_mb": 3687,
        "reserved_mb": 0,
        "models": {
            "fake-stream": {"per_call_mb": 250, "max_calls": 14, "effective_max_calls": 14},
            "fake-batch": {"per_call_mb": 900, "max_calls": 4, "effective_max_calls": 4},
        },
    }



