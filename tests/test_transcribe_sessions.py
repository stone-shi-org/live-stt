from __future__ import annotations

from live_stt.transcribe_sessions import TranscribeSessionTracker


def test_starts_at_zero():
    t = TranscribeSessionTracker()
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.rejected_vram_total == 0
    assert t.rejected_capacity_total == 0
    assert t.snapshot_active() == []


def test_start_increments_active():
    t = TranscribeSessionTracker()
    t.start(model="whisper-base.en-q8_0")
    t.start(model="realtime_eou_120m-v1")
    assert t.active == 2


def test_finish_ok_decrements_active_and_counts_completed():
    t = TranscribeSessionTracker()
    rid = t.start(model="realtime_eou_120m-v1")
    t.finish(rid, ok=True)
    assert t.active == 0
    assert t.completed_total == 1
    assert t.failed_total == 0


def test_finish_not_ok_decrements_active_and_counts_failed():
    t = TranscribeSessionTracker()
    rid = t.start(model="realtime_eou_120m-v1")
    t.finish(rid, ok=False)
    assert t.active == 0
    assert t.failed_total == 1
    assert t.completed_total == 0


def test_finish_never_goes_negative():
    t = TranscribeSessionTracker()
    t.finish(1, ok=True)  # finish without a matching start -- must not happen in practice, but must not corrupt state
    assert t.active == 0


def test_record_rejected_vram_does_not_touch_active():
    t = TranscribeSessionTracker()
    t.record_rejected_vram()
    assert t.rejected_vram_total == 1
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.snapshot_active() == []


def test_record_rejected_capacity_does_not_touch_active_or_vram():
    # The one real difference from DiarizationSessionTracker -- transcribe
    # has TWO pre-start rejection reasons (VRAM and WorkerBudget capacity),
    # tracked as two independent counters.
    t = TranscribeSessionTracker()
    t.record_rejected_capacity()
    assert t.rejected_capacity_total == 1
    assert t.rejected_vram_total == 0
    assert t.active == 0
    assert t.snapshot_active() == []


def test_concurrent_start_finish_is_thread_safe():
    import threading

    t = TranscribeSessionTracker()

    def worker():
        for _ in range(200):
            rid = t.start(model="realtime_eou_120m-v1")
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
    """`start()`/`finish()` also track a small live list -- id/elapsed/model
    -- for the admin dashboard's "Active Transcript Requests" table (see
    transcribe_sessions.py's docstring on why this is a deliberate, narrow
    exception to "not a session registry", mirroring diarize_sessions.py).
    """

    def test_start_returns_an_id_and_snapshot_reflects_it(self):
        t = TranscribeSessionTracker()
        rid = t.start(model="whisper-large-v3-turbo-q8_0")
        snap = t.snapshot_active()
        assert len(snap) == 1
        assert snap[0]["id"] == rid
        assert snap[0]["model"] == "whisper-large-v3-turbo-q8_0"
        assert snap[0]["elapsed_sec"] >= 0

    def test_finish_removes_its_own_entry_only(self):
        t = TranscribeSessionTracker()
        rid1 = t.start(model="realtime_eou_120m-v1")
        rid2 = t.start(model="whisper-base.en-q8_0")
        t.finish(rid1, ok=True)
        snap = t.snapshot_active()
        assert len(snap) == 1
        assert snap[0]["id"] == rid2
        assert snap[0]["model"] == "whisper-base.en-q8_0"

    def test_snapshot_active_is_oldest_first(self):
        t = TranscribeSessionTracker()
        rid1 = t.start(model="realtime_eou_120m-v1")
        rid2 = t.start(model="realtime_eou_120m-v1")
        ids = [entry["id"] for entry in t.snapshot_active()]
        assert ids == sorted([rid1, rid2])

    def test_finish_with_unknown_id_does_not_crash_or_touch_other_entries(self):
        t = TranscribeSessionTracker()
        rid = t.start(model="realtime_eou_120m-v1")
        t.finish(rid + 999, ok=True)  # never started -- must not raise, must not remove rid's entry
        assert len(t.snapshot_active()) == 1
