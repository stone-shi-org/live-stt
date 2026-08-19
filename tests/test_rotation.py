"""CallSession's rotation state machine against tests/fakes/fake_worker_main.py.

Real process spawn/kill and real signals throughout -- crash recovery is
tested by actually killing a real subprocess, not by mocking an exception.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.models import resolve
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.session import CallSession

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"
ONE_CHUNK = b"\x00\x00" * 2560  # exactly one 160ms model chunk


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **overrides)


def _session(settings: Settings, budget: WorkerBudget | None = None) -> tuple[CallSession, WorkerBudget]:
    spec = resolve("realtime_eou_120m-v1")
    config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
    budget = budget or WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
    assert budget.try_admit_call()
    return CallSession(settings, spec, config, budget), budget


def _kinds(events: list[asr_pb2.TranscriptionEvent]) -> list[str]:
    return [e.WhichOneof("event") for e in events]


@pytest.fixture(autouse=True)
def _clean_fake_worker_env():
    keys = [
        "FAKE_CRASH_AFTER_SEC",
        "FAKE_ABORT_AFTER_SEC",
        "FAKE_HANG_AFTER_SEC",
        "FAKE_EOU_EVERY_SEC",
        "FAKE_WORDS_PER_SEC",
        "FAKE_CONFIG_ERROR",
        "FAKE_LOAD_DELAY_SEC",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.mark.asyncio
async def test_audio_cap_triggers_a_warm_rotation_with_no_duplicate_words() -> None:
    # rotate_after_sec (0.5s) is short relative to the 3.2s fed below, so
    # this will legitimately complete SEVERAL rotations, not exactly one --
    # each new generation re-crosses the same short threshold. That's fine;
    # the claim under test is per-rotation correctness (warm, no gap, no
    # duplicated/dropped words at ANY seam), which holds regardless of count.
    os.environ["FAKE_WORDS_PER_SEC"] = "10"  # ~1.6 words per 160ms chunk
    settings = _settings(rotate_after_sec=0.5, rotation_overlap_sec=1.0, reserve_slots=1)
    session, budget = _session(settings)
    try:
        all_events: list[asr_pb2.TranscriptionEvent] = []
        all_events.append(await session.start())
        for _ in range(20):  # 20 * 160ms = 3.2s
            all_events.extend(await session.feed_audio(ONE_CHUNK))
        all_events.extend(await session.finalize())
    finally:
        await session.close()

    kinds = _kinds(all_events)
    assert "recycled" in kinds
    recycled = [e.recycled for e in all_events if e.WhichOneof("event") == "recycled"]
    assert len(recycled) >= 1
    for r in recycled:
        assert r.reason == asr_pb2.RECYCLE_REASON_AUDIO_CAP
        assert r.warm is True
        assert r.gap_sec == 0.0

    final = [e for e in all_events if e.WhichOneof("event") == "final"][0]
    assert final.final.worker_generations == 1 + len(recycled)

    # No duplicated or dropped words across the rotation seam: word
    # start_sec must be strictly non-decreasing across the WHOLE call.
    all_words = [w for e in all_events if e.WhichOneof("event") == "delta" for w in e.delta.words]
    starts = [w.start_sec for w in all_words]
    assert starts == sorted(starts)
    # And no exact duplicate (word, start_sec) pairs.
    seen = set()
    for w in all_words:
        key = (w.text, round(w.start_sec, 3))
        assert key not in seen, f"duplicate word across rotation seam: {key}"
        seen.add(key)

    assert budget.active_workers == budget.active_calls  # rotation shadow slot was released


@pytest.mark.asyncio
async def test_rss_threshold_triggers_rotation_immediately() -> None:
    # The fake worker reports its OWN real RSS (via /proc/self/status), so a
    # threshold far below any real process's RSS forces a rotation on the
    # very first feed -- no leak simulation needed for this trigger.
    settings = _settings(worker_rss_soft_kb=1, rotate_after_sec=10_000, rotation_overlap_sec=0.32)
    session, _ = _session(settings)
    try:
        events = [await session.start()]
        for _ in range(5):
            events.extend(await session.feed_audio(ONE_CHUNK))
        events.extend(await session.finalize())
    finally:
        await session.close()

    recycled = [e.recycled for e in events if e.WhichOneof("event") == "recycled"]
    assert len(recycled) == 1
    assert recycled[0].reason == asr_pb2.RECYCLE_REASON_RSS_THRESHOLD


@pytest.mark.asyncio
async def test_eou_during_overlap_cuts_over_early() -> None:
    # The overlap window is deliberately huge (10s) relative to the total
    # audio fed (3.2s). Without early-EOU cutover, at most ONE rotation
    # could ever START within this test -- once dual-feeding begins, it
    # would need the full 10s to cut over, far longer than the test runs,
    # so no SECOND rotation could ever start either (there's nowhere for a
    # new one to begin from). Observing several completed rotations well
    # inside that 3.2s therefore proves cutover is happening at <EOU>, not
    # at the deadline.
    os.environ["FAKE_EOU_EVERY_SEC"] = "0.32"
    settings = _settings(rotate_after_sec=0.16, rotation_overlap_sec=10.0)
    session, _ = _session(settings)
    try:
        events = [await session.start()]
        for _ in range(20):  # 20 * 160ms = 3.2s
            events.extend(await session.feed_audio(ONE_CHUNK))
        events.extend(await session.finalize())
    finally:
        await session.close()

    recycled = [e.recycled for e in events if e.WhichOneof("event") == "recycled"]
    assert len(recycled) >= 3, (
        f"only {len(recycled)} rotation(s) completed in 3.2s of audio with a 10s overlap "
        "window -- early-EOU cutover does not appear to be firing"
    )
    for r in recycled:
        assert r.at_audio_sec < 3.2


@pytest.mark.asyncio
async def test_reserve_exhausted_defers_rotation_instead_of_forcing_a_gap() -> None:
    # Two calls sharing one budget with zero reserve: the second call's
    # rotation attempt can never acquire a shadow slot, so it should just
    # keep retrying (never rotating) rather than doing anything destructive.
    budget = WorkerBudget(max_concurrent_calls=2, reserve_slots=0)
    settings = _settings(rotate_after_sec=0.16, rotation_overlap_sec=0.5)

    session_a, _ = _session(settings, budget=budget)
    session_b, _ = _session(settings, budget=budget)
    try:
        await session_a.start()
        await session_b.start()
        assert budget.active_workers == 2 == budget.max_workers

        events_b: list[asr_pb2.TranscriptionEvent] = []
        for _ in range(10):
            events_b.extend(await session_b.feed_audio(ONE_CHUNK))

        # No rotation ever completed for session_b -- no spare slot existed.
        assert "recycled" not in _kinds(events_b)
        assert budget.active_workers == 2  # never grew past the hard cap
    finally:
        await session_a.close()
        await session_b.close()


@pytest.mark.asyncio
async def test_worker_crash_recovers_with_a_cold_recycled_event() -> None:
    # FAKE_CRASH_AFTER_SEC="0.16" means EVERY freshly spawned worker --
    # including each replacement, since it captures the env var at ITS OWN
    # spawn time and starts counting from 0 again -- crashes on its own
    # first chunk, until the env var is popped. Popping it only takes
    # effect for the NEXT spawn, so this legitimately chains: the original
    # crashes on chunk 1 (spawning replacement A with the same doomed
    # setting), replacement A crashes on chunk 2 (spawning replacement B,
    # AFTER the pop below -- clean), and B survives from then on. The
    # claim under test is that recovery converges and the call still
    # completes normally, not an exact recovery count.
    os.environ["FAKE_CRASH_AFTER_SEC"] = "0.16"
    settings = _settings()
    session, budget = _session(settings)
    try:
        events = [await session.start()]
        events.extend(await session.feed_audio(ONE_CHUNK))
        os.environ.pop("FAKE_CRASH_AFTER_SEC", None)
        for _ in range(3):
            events.extend(await session.feed_audio(ONE_CHUNK))
        events.extend(await session.finalize())
    finally:
        await session.close()

    recycled = [e.recycled for e in events if e.WhichOneof("event") == "recycled"]
    assert len(recycled) >= 1
    for r in recycled:
        assert r.reason == asr_pb2.RECYCLE_REASON_CRASH
        assert r.warm is False
        assert r.gap_sec > 0.0

    final = [e for e in events if e.WhichOneof("event") == "final"][0]
    assert final.final.worker_generations == 1 + len(recycled)
    assert budget.active_workers == budget.active_calls  # no leaked shadow slot


@pytest.mark.asyncio
async def test_shadow_death_during_dual_feed_abandons_rotation_without_crashing_the_call() -> None:
    os.environ["FAKE_WORDS_PER_SEC"] = "0"
    settings = _settings(rotate_after_sec=0.16, rotation_overlap_sec=5.0)
    session, budget = _session(settings)
    try:
        events = [await session.start()]
        # Trigger the rotation (crosses rotate_after_sec on the first chunk).
        events.extend(await session.feed_audio(ONE_CHUNK))
        assert session._incoming is not None  # rotation is underway

        # Make every NEWLY spawned worker crash immediately, then force the
        # shadow specifically to die on its next feed by killing it directly
        # (simulating the "shadow died mid dual-feed" path without needing
        # exact timing against FAKE_CRASH_AFTER_SEC).
        session._incoming.kill()
        await session._incoming.wait_closed()

        events.extend(await session.feed_audio(ONE_CHUNK))  # should abandon, not raise
        assert session._incoming is None
        events.extend(await session.finalize())
    finally:
        await session.close()

    # The call completed cleanly (a Final exists) despite the shadow's death.
    assert any(e.WhichOneof("event") == "final" for e in events)
    assert "recycled" not in _kinds(events)  # the abandoned attempt never cut over
    assert budget.active_workers == budget.active_calls  # shadow slot was released
