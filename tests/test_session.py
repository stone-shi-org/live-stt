"""CallSession against the real IPC protocol, real process spawn/kill, and
real signals -- via tests/fakes/fake_worker_main.py instead of the real
binary/model. See that module's docstring for why a fake process beats a
mock here: the behavior under test is process/fd/signal semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.models import resolve
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.session import CallSession
from live_stt.worker import WorkerError

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **overrides)


def _session(
    settings: Settings | None = None,
    model_key: str = "realtime_eou_120m-v1",
    budget: WorkerBudget | None = None,
) -> CallSession:
    settings = settings or _settings()
    spec = resolve(model_key)
    config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
    budget = budget or WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
    budget.try_admit_call()  # a real caller always admits before constructing a CallSession
    return CallSession(settings, spec, config, budget)


@pytest.mark.asyncio
async def test_start_returns_a_ready_event_matching_the_model_spec() -> None:
    session = _session(model_key="realtime_eou_120m-v1")
    try:
        ready_event = await session.start()
    finally:
        await session.close()

    assert ready_event.WhichOneof("event") == "ready"
    ready = ready_event.ready
    assert ready.model == "realtime_eou_120m-v1"
    assert ready.supports_turn_detection is True
    assert ready.model_chunk_ms == 160
    assert ready.accepted_sample_rate_hz == 16000


@pytest.mark.asyncio
async def test_start_propagates_worker_config_error() -> None:
    session = _session()
    import os

    os.environ["FAKE_CONFIG_ERROR"] = "1"
    try:
        with pytest.raises(WorkerError):
            await session.start()
    finally:
        os.environ.pop("FAKE_CONFIG_ERROR", None)
        await session.close()


@pytest.mark.asyncio
async def test_feed_audio_coalesces_to_model_chunk_size() -> None:
    # The 120m model's chunk is 160ms = 2560 samples = 5120 bytes. Feed it in
    # much smaller pieces and confirm the session buffers rather than
    # forwarding every small write to the worker -- exercised indirectly via
    # audio_offset_sec, which only advances once a full chunk has actually
    # been fed to (and acked by) the worker.
    session = _session()
    try:
        await session.start()
        small_frame = b"\x00\x00" * 160  # 20ms, far smaller than one model chunk
        events = await session.feed_audio(small_frame)
        assert events == []  # not enough buffered yet for one chunk
        assert session.audio_offset_sec == 0.0

        # Feed enough small frames to cross one full chunk boundary.
        for _ in range(15):  # 15 * 20ms = 300ms > one 160ms chunk
            events = await session.feed_audio(small_frame)
        assert session.audio_offset_sec > 0.0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_finalize_flushes_partial_buffer_and_appends_final_event() -> None:
    session = _session()
    try:
        await session.start()
        await session.feed_audio(b"\x00\x00" * 100)  # well under one chunk
        events = await session.finalize()
    finally:
        await session.close()

    assert events[-1].WhichOneof("event") == "final"
    assert events[-1].final.worker_generations == 1


@pytest.mark.asyncio
async def test_words_per_sec_fake_worker_produces_deltas() -> None:
    import os

    os.environ["FAKE_WORDS_PER_SEC"] = "50"
    try:
        session = _session()
        try:
            await session.start()
            one_chunk = b"\x00\x00" * 2560  # exactly one 160ms chunk
            events = await session.feed_audio(one_chunk)
        finally:
            await session.close()
    finally:
        os.environ.pop("FAKE_WORDS_PER_SEC", None)

    deltas = [e for e in events if e.WhichOneof("event") == "delta"]
    assert len(deltas) == 1
    assert deltas[0].delta.text  # non-empty -- the fake worker emitted synthetic words
    assert len(deltas[0].delta.words) >= 1


@pytest.mark.asyncio
async def test_close_kills_the_worker_process() -> None:
    session = _session()
    await session.start()
    worker_proc = session._worker.proc  # noqa: SLF001 -- test white-box access
    await session.close()
    assert worker_proc.returncode is not None
    assert worker_proc.returncode < 0  # killed by a signal, not a clean exit


@pytest.mark.asyncio
async def test_close_without_finalize_does_not_hang_or_raise() -> None:
    # The client-disconnect path: never call finalize(), just close().
    session = _session()
    await session.start()
    await session.feed_audio(b"\x00\x00" * 100)
    await session.close()  # must not raise, must not hang


@pytest.mark.asyncio
async def test_close_is_idempotent() -> None:
    session = _session()
    await session.start()
    await session.close()
    await session.close()  # must not raise
