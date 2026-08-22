from __future__ import annotations

from live_stt.diarize_sessions import DiarizationSessionTracker


def test_starts_at_zero():
    t = DiarizationSessionTracker()
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.rejected_vram_total == 0
    assert t.snapshot_active() == []


def test_start_increments_active():
    t = DiarizationSessionTracker()
    t.start()
    t.start()
    assert t.active == 2


def test_finish_ok_decrements_active_and_counts_completed():
    t = DiarizationSessionTracker()
    rid = t.start()
    t.finish(rid, ok=True)
    assert t.active == 0
    assert t.completed_total == 1
    assert t.failed_total == 0


def test_finish_not_ok_decrements_active_and_counts_failed():
    t = DiarizationSessionTracker()
    rid = t.start()
    t.finish(rid, ok=False)
    assert t.active == 0
    assert t.failed_total == 1
    assert t.completed_total == 0


def test_finish_never_goes_negative():
    t = DiarizationSessionTracker()
    t.finish(1, ok=True)  # finish without a matching start -- must not happen in practice, but must not corrupt state
    assert t.active == 0


def test_record_rejected_vram_does_not_touch_active():
    t = DiarizationSessionTracker()
    t.record_rejected_vram()
    assert t.rejected_vram_total == 1
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.snapshot_active() == []


def test_concurrent_start_finish_is_thread_safe():
    import threading

    t = DiarizationSessionTracker()

    def worker():
        for _ in range(200):
            rid = t.start()
            t.finish(rid, ok=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.active == 0
    assert t.completed_total == 8 * 200
    assert t.snapshot_active() == []  # every entry must have been popped, not leaked


class TestActiveRequestSnapshot:
    """`start()`/`finish()` also track a small live list -- id/elapsed/device
    -- for the admin dashboard's "Active Diarization Requests" table (see
    diarize_sessions.py's docstring on why this is a deliberate, narrow
    exception to "not a session registry").
    """

    def test_start_returns_an_id_and_snapshot_reflects_it(self):
        t = DiarizationSessionTracker()
        rid = t.start(device="cuda")
        snap = t.snapshot_active()
        assert len(snap) == 1
        assert snap[0]["id"] == rid
        assert snap[0]["device"] == "cuda"
        assert snap[0]["elapsed_sec"] >= 0

    def test_finish_removes_its_own_entry_only(self):
        t = DiarizationSessionTracker()
        rid1 = t.start(device="cpu")
        rid2 = t.start(device="cuda")
        t.finish(rid1, ok=True)
        snap = t.snapshot_active()
        assert len(snap) == 1
        assert snap[0]["id"] == rid2
        assert snap[0]["device"] == "cuda"

    def test_default_device_is_unknown_when_unspecified(self):
        t = DiarizationSessionTracker()
        t.start()
        assert t.snapshot_active()[0]["device"] == "unknown"

    def test_snapshot_active_is_oldest_first(self):
        t = DiarizationSessionTracker()
        rid1 = t.start()
        rid2 = t.start()
        ids = [entry["id"] for entry in t.snapshot_active()]
        assert ids == sorted([rid1, rid2])

    def test_finish_with_unknown_id_does_not_crash_or_touch_other_entries(self):
        t = DiarizationSessionTracker()
        rid = t.start()
        t.finish(rid + 999, ok=True)  # never started -- must not raise, must not remove rid's entry
        assert len(t.snapshot_active()) == 1
