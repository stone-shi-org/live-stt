from __future__ import annotations

from live_stt.diarize_sessions import DiarizationSessionTracker


def test_starts_at_zero():
    t = DiarizationSessionTracker()
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.rejected_vram_total == 0


def test_start_increments_active():
    t = DiarizationSessionTracker()
    t.start()
    t.start()
    assert t.active == 2


def test_finish_ok_decrements_active_and_counts_completed():
    t = DiarizationSessionTracker()
    t.start()
    t.finish(ok=True)
    assert t.active == 0
    assert t.completed_total == 1
    assert t.failed_total == 0


def test_finish_not_ok_decrements_active_and_counts_failed():
    t = DiarizationSessionTracker()
    t.start()
    t.finish(ok=False)
    assert t.active == 0
    assert t.failed_total == 1
    assert t.completed_total == 0


def test_finish_never_goes_negative():
    t = DiarizationSessionTracker()
    t.finish(ok=True)  # finish without a matching start -- must not happen in practice, but must not corrupt state
    assert t.active == 0


def test_record_rejected_vram_does_not_touch_active():
    t = DiarizationSessionTracker()
    t.record_rejected_vram()
    assert t.rejected_vram_total == 1
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0


def test_concurrent_start_finish_is_thread_safe():
    import threading

    t = DiarizationSessionTracker()

    def worker():
        for _ in range(200):
            t.start()
            t.finish(ok=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.active == 0
    assert t.completed_total == 8 * 200
