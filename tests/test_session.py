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


@pytest.mark.asyncio
async def test_whisper_engine_spawns_the_whisper_binary_not_the_parakeet_one() -> None:
    # worker_bin (parakeet's default) is deliberately left pointing at a
    # nonexistent path here -- only worker_bin_whisper points at the real
    # fake. If _spawn_worker's engine dispatch (live_stt/session.py) picked
    # the wrong binary for a whisper-engine ModelSpec, start() would fail
    # with a WorkerError (spawning a nonexistent executable), not succeed.
    settings = Settings(
        _env_file=None,
        worker_bin="/nonexistent/live_stt_worker",
        worker_bin_whisper=str(FAKE_WORKER),
        models_dir="/fake",
    )
    session = _session(settings=settings, model_key="whisper-base.en-q8_0")
    try:
        ready_event = await session.start()
    finally:
        await session.close()

    assert ready_event.WhichOneof("event") == "ready"
    assert ready_event.ready.model == "whisper-base.en-q8_0"


@pytest.mark.asyncio
async def test_use_gpu_is_derived_from_backend_setting_not_hardcoded() -> None:
    # Settings.backend == "cuda" -- CallSession._spawn_worker (session.py)
    # must derive use_gpu=True from this and thread it through
    # WorkerHandle.spawn()'s CONFIG frame (see worker/main_whisper.cpp's
    # "use_gpu" field). Checked via the fake worker's use_gpu_received echo
    # (tests/fakes/fake_worker_main.py) rather than mocking spawn() directly,
    # so this exercises the real CONFIG JSON that actually goes out over the
    # real IPC socket, not just the Python call arguments.
    cuda_settings = Settings(
        _env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", backend="cuda"
    )
    session = _session(settings=cuda_settings)
    try:
        ready_event = await session.start()
    finally:
        await session.close()
    assert ready_event.ready is not None  # sanity: start() actually completed
    assert session._worker.ready.get("use_gpu_received") is True  # noqa: SLF001

    cpu_settings = Settings(
        _env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", backend="cpu"
    )
    session = _session(settings=cpu_settings)
    try:
        await session.start()
    finally:
        await session.close()
    assert session._worker.ready.get("use_gpu_received") is False  # noqa: SLF001


def test_rotation_never_triggers_for_a_batch_only_model_even_past_every_threshold() -> None:
    # The correctness fix this whisper addition surfaced: a rotation
    # SIGKILLs the active worker mid-call, which is safe for a streaming
    # engine (words already emitted progressively survive) but would
    # silently drop unfinalized audio for a batch-only one (nothing has
    # been transcribed yet at that point -- see live_stt/models.py's
    # docstring). Call _should_start_rotation directly with every threshold
    # blown way past, on a streaming_capable=False spec, and confirm it
    # still refuses to rotate.
    settings = _settings(worker_rss_soft_kb=1, rotate_after_sec=0.0)
    session = _session(settings=settings, model_key="whisper-base.en-q8_0")
    session._last_rss_kb = 10_000_000  # noqa: SLF001 -- test white-box access, way past worker_rss_soft_kb
    session._active_generation_start_sec = 0.0  # noqa: SLF001
    session._fed_samples = 16000 * 100_000  # noqa: SLF001 -- huge audio_offset_sec, way past rotate_after_sec

    reason = session._should_start_rotation({"eou": True})  # noqa: SLF001
    assert reason is None


def test_rotation_still_triggers_normally_for_a_streaming_model_with_the_same_thresholds() -> None:
    # Contrast case for the test above -- proves the gate is specific to
    # streaming_capable, not a change in the underlying threshold logic
    # itself (which existing tests/test_rotation.py already covers in more
    # depth via the full dual-feed machinery).
    settings = _settings(worker_rss_soft_kb=1, rotate_after_sec=0.0)
    session = _session(settings=settings, model_key="realtime_eou_120m-v1")
    session._last_rss_kb = 10_000_000  # noqa: SLF001

    reason = session._should_start_rotation({})  # noqa: SLF001
    assert reason is not None
