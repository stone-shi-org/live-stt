"""StreamingASRServicer against a REAL grpc.aio server on a loopback port,
using tests/fakes/fake_worker_main.py in place of the real binary/model.
Real gRPC end to end (not a bare-function call into the servicer) is what
actually exercises half-close, cancellation, and the admission counter --
those are properties of the RPC lifecycle, not of the servicer's Python code
in isolation.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import grpc
import pytest

from live_stt.config import Settings
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.servicer import StreamingASRServicer

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **overrides)


class _Server:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.servicer = StreamingASRServicer(settings)
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


async def _config_then_audio(
    n_chunks: int = 3, config: asr_pb2.StreamConfig | None = None
) -> AsyncIterator[asr_pb2.TranscriptionRequest]:
    yield asr_pb2.TranscriptionRequest(
        config=config
        or asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
    )
    for _ in range(n_chunks):
        yield asr_pb2.TranscriptionRequest(audio=b"\x00\x00" * 2560)


@pytest.mark.asyncio
async def test_get_server_info_reports_configured_defaults() -> None:
    async with _Server(_settings(max_concurrent_calls=5)) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            info = await stub.GetServerInfo(asr_pb2.ServerInfoRequest())
    assert info.default_model == "realtime_eou_120m-v1"
    assert info.max_concurrent_calls == 5
    assert info.active_calls == 0


@pytest.mark.asyncio
async def test_transcribe_happy_path_yields_ready_then_final() -> None:
    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            events = [e async for e in stub.Transcribe(_config_then_audio())]

    kinds = [e.WhichOneof("event") for e in events]
    assert kinds[0] == "ready"
    assert kinds[-1] == "final"


@pytest.mark.asyncio
async def test_transcribe_rejects_missing_config_first_message() -> None:
    async def audio_only() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
        yield asr_pb2.TranscriptionRequest(audio=b"\x00\x00")

    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(audio_only()):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_transcribe_rejects_unsupported_sample_rate() -> None:
    async def bad_config() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
        yield asr_pb2.TranscriptionRequest(
            config=asr_pb2.StreamConfig(
                encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=8000
            )
        )

    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(bad_config()):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_transcribe_rejects_unsupported_encoding() -> None:
    async def bad_config() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
        yield asr_pb2.TranscriptionRequest(
            config=asr_pb2.StreamConfig(
                encoding=asr_pb2.AUDIO_ENCODING_MULAW, sample_rate_hz=8000
            )
        )

    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(bad_config()):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_transcribe_rejects_unknown_model() -> None:
    async def bad_config() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
        yield asr_pb2.TranscriptionRequest(
            config=asr_pb2.StreamConfig(
                encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000, model="nonexistent"
            )
        )

    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(bad_config()):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_transcribe_rejects_second_config_message() -> None:
    async def two_configs() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
        cfg = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
        yield asr_pb2.TranscriptionRequest(config=cfg)
        yield asr_pb2.TranscriptionRequest(config=cfg)

    async with _Server(_settings()) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                async for _ in stub.Transcribe(two_configs()):
                    pass
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_transcribe_at_capacity_rejects_immediately_with_resource_exhausted() -> None:
    os.environ["FAKE_HANG_AFTER_SEC"] = "0"  # keep the first call's worker occupied
    try:
        async with _Server(_settings(max_concurrent_calls=1)) as server:
            # Two separate channels, not one shared stub: a hung first stream
            # plus a second call on the SAME HTTP/2 connection was observed
            # (empirically, while writing this test) to occasionally trip a
            # grpc-core-level "Internal error from Core" on the second call's
            # SendMessageOperation -- a channel-sharing quirk, not a
            # behavior this test is trying to exercise. Independent channels
            # is also the more realistic shape (independent telephony
            # clients), and it removes the flakiness entirely.
            async with server.channel() as channel_a, server.channel() as channel_b:
                stub_a = asr_pb2_grpc.StreamingASRStub(channel_a)
                stub_b = asr_pb2_grpc.StreamingASRStub(channel_b)

                async def slow_call() -> None:
                    async for _ in stub_a.Transcribe(_config_then_audio(n_chunks=1)):
                        pass

                first_call = asyncio.ensure_future(slow_call())
                await asyncio.sleep(0.2)  # let the first call occupy the one slot

                # n_chunks=0: the admission check runs before the servicer
                # reads a single message, so the rejected call never needs
                # to send audio. Sending a second message (audio) right
                # after config was observed, empirically, to occasionally
                # race the server's near-instant abort() and surface as a
                # generic INTERNAL error instead of RESOURCE_EXHAUSTED --
                # a client-writing-while-server-aborts timing race in
                # grpc-core itself, not a servicer bug. Not sending it
                # avoids the race outright.
                with pytest.raises(grpc.aio.AioRpcError) as exc_info:
                    async for _ in stub_b.Transcribe(_config_then_audio(n_chunks=0)):
                        pass
                assert exc_info.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED

                first_call.cancel()
                try:
                    await first_call
                except (asyncio.CancelledError, grpc.aio.AioRpcError):
                    pass
    finally:
        os.environ.pop("FAKE_HANG_AFTER_SEC", None)


@pytest.mark.asyncio
async def test_client_cancellation_does_not_hang_the_server() -> None:
    # The client-disconnect path: cancel mid-stream, confirm the server
    # cleans up (frees the admission slot) instead of leaking it forever.
    async with _Server(_settings(max_concurrent_calls=1)) as server:
        async with server.channel() as channel:
            stub = asr_pb2_grpc.StreamingASRStub(channel)

            async def long_call() -> None:
                async for _ in stub.Transcribe(_config_then_audio(n_chunks=1000)):
                    pass

            call = asyncio.ensure_future(long_call())
            await asyncio.sleep(0.2)
            call.cancel()
            try:
                await call
            except (asyncio.CancelledError, grpc.aio.AioRpcError):
                pass

            # The client observing its own cancellation (the `await call`
            # above) races the SERVER's task noticing the cancellation and
            # running its own cleanup -- those are two different tasks, on
            # (in this test) the same event loop but not the same
            # scheduling turn. Poll rather than assert immediately.
            for _ in range(50):
                if server.servicer._active_calls == 0:
                    break
                await asyncio.sleep(0.05)
            assert server.servicer._active_calls == 0, "admission slot leaked after client cancellation"

            # The slot must be free now -- a second call should succeed,
            # not hit RESOURCE_EXHAUSTED from a leaked admission count.
            events = [e async for e in stub.Transcribe(_config_then_audio(n_chunks=1))]
            assert events[0].WhichOneof("event") == "ready"
