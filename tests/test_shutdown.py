"""Graceful drain: state.draining flips admission shut immediately, active
calls are unaffected and complete normally. See live_stt/server.py's SIGTERM
handler for the real wiring (health service + state.draining together);
this exercises the servicer-level contract that handler depends on,
against the fake worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import grpc
import pytest

from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.servicer import StreamingASRServicer
from live_stt.state import ServerState

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **overrides)


class _Server:
    def __init__(self, settings: Settings) -> None:
        self.state = ServerState(
            settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
        )
        self.servicer = StreamingASRServicer(self.state)
        self._server: grpc.aio.Server | None = None
        self.port: int | None = None

    async def __aenter__(self) -> "_Server":
        self._server = grpc.aio.server()
        asr_pb2_grpc.add_StreamingASRServicer_to_server(self.servicer, self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        await self._server.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._server.stop(grace=1.0)
        await self._server.wait_for_termination()

    def channel(self) -> grpc.aio.Channel:
        return grpc.aio.insecure_channel(f"127.0.0.1:{self.port}")


async def _config_then_audio(n_chunks: int = 3) -> AsyncIterator[asr_pb2.TranscriptionRequest]:
    yield asr_pb2.TranscriptionRequest(
        config=asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
    )
    for _ in range(n_chunks):
        yield asr_pb2.TranscriptionRequest(audio=b"\x00\x00" * 2560)


@pytest.mark.asyncio
async def test_new_calls_rejected_the_instant_draining_is_set() -> None:
    async with _Server(_settings()) as server:
        server.state.draining = True
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(_config_then_audio(n_chunks=0)):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE


@pytest.mark.asyncio
async def test_draining_does_not_affect_a_call_already_in_progress() -> None:
    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)

            async def slow_call() -> list[str]:
                kinds = []
                async for event in stub.Transcribe(_config_then_audio(n_chunks=1)):
                    kinds.append(event.WhichOneof("event"))
                return kinds

            call = asyncio.ensure_future(slow_call())
            await asyncio.sleep(0.1)  # let it start and pass admission
            server.state.draining = True  # drain kicks in mid-call

            kinds = await call
    assert "ready" in kinds
    assert "final" in kinds  # the in-flight call completed normally, not aborted


@pytest.mark.asyncio
async def test_draining_flag_is_visible_before_any_grpc_activity() -> None:
    # The contract server.py's SIGTERM handler relies on: setting
    # state.draining is a plain attribute set, synchronous and immediate --
    # no await, no delay -- so a handler can flip it and know every
    # subsequent Transcribe() admission check sees it right away.
    settings = _settings()
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    assert state.draining is False
    state.draining = True
    assert state.draining is True
    assert state.health_status() == "draining"
