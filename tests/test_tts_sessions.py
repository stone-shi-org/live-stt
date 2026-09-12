from __future__ import annotations

import threading

from live_stt.tts_sessions import TTSSessionTracker


def test_starts_at_zero() -> None:
    t = TTSSessionTracker()
    assert t.active == 0
    assert t.completed_total == 0
    assert t.failed_total == 0
    assert t.rejected_vram_total == 0
    assert t.rejected_capacity_total == 0
    assert t.snapshot_active() == []


def test_start_increments_active() -> None:
    t = TTSSessionTracker()
    rid1 = t.start(model="kokoro-82m-v1", voice="af_heart", device="cpu")
    rid2 = t.start(model="piper-en-us", voice="en_US-lessac-medium", device="cpu")
    assert t.active == 2
    assert rid1 != rid2


def test_finish_ok_decrements_active_and_counts_completed() -> None:
    t = TTSSessionTracker()
    rid = t.start(model="kokoro-82m-v1")
    t.finish(rid, ok=True)
    assert t.active == 0
    assert t.completed_total == 1
    assert t.failed_total == 0


def test_finish_not_ok_decrements_active_and_counts_failed() -> None:
    t = TTSSessionTracker()
    rid = t.start(model="kokoro-82m-v1")
    t.finish(rid, ok=False)
    assert t.active == 0
    assert t.failed_total == 1
    assert t.completed_total == 0


def test_record_rejected_vram_and_capacity() -> None:
    t = TTSSessionTracker()
    t.record_rejected_vram()
    assert t.rejected_vram_total == 1
    t.record_rejected_capacity()
    assert t.rejected_capacity_total == 1
    assert t.active == 0


def test_concurrent_start_finish_is_thread_safe() -> None:
    t = TTSSessionTracker()

    def worker() -> None:
        for _ in range(100):
            rid = t.start(model="kokoro-82m-v1")
            t.finish(rid, ok=True)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.active == 0
    assert t.completed_total == 800
    assert t.snapshot_active() == []


def test_snapshot_active() -> None:
    t = TTSSessionTracker()
    rid1 = t.start(model="kokoro-82m-v1", voice="af_heart", device="cuda")
    rid2 = t.start(model="piper-en-us", voice="en_US-lessac-medium", device="cpu")

    snap = t.snapshot_active()
    assert len(snap) == 2
    assert snap[0]["id"] == rid1
    assert snap[0]["model"] == "kokoro-82m-v1"
    assert snap[0]["voice"] == "af_heart"
    assert snap[0]["device"] == "cuda"
    assert snap[0]["elapsed_sec"] >= 0

    t.finish(rid1, ok=True)
    snap2 = t.snapshot_active()
    assert len(snap2) == 1
    assert snap2[0]["id"] == rid2
