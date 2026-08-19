"""Thin async gRPC client for StreamingASR.

Used by tests, tools/fake_telephony_client.py, and importable directly by the
telephony app. Deliberately thin: retry policy, RESOURCE_EXHAUSTED backoff,
and reconnect-with-resume_of_stream_id semantics are the caller's job, not
this wrapper's -- see the failure semantics matrix in CLAUDE.md for what each
gRPC status code means and what the client should do about it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc

from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc


class ASRClient:
    def __init__(self, target: str, *, channel: grpc.aio.Channel | None = None) -> None:
        self._owns_channel = channel is None
        self._channel = channel or grpc.aio.insecure_channel(target)
        self._stub = asr_pb2_grpc.StreamingASRStub(self._channel)

    async def close(self) -> None:
        if self._owns_channel:
            await self._channel.close()

    async def get_server_info(self) -> asr_pb2.ServerInfoResponse:
        return await self._stub.GetServerInfo(asr_pb2.ServerInfoRequest())

    def transcribe(
        self,
        config: asr_pb2.StreamConfig,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[asr_pb2.TranscriptionEvent]:
        """Returns an async iterator of TranscriptionEvent for the call.

        ``audio_chunks`` should yield raw int16 LE mono 16kHz bytes as they
        become available -- send RTP-sized frames as they arrive with no
        client-side windowing; the server coalesces to its model chunk size.
        """

        async def request_iter() -> AsyncIterator[asr_pb2.TranscriptionRequest]:
            yield asr_pb2.TranscriptionRequest(config=config)
            async for chunk in audio_chunks:
                yield asr_pb2.TranscriptionRequest(audio=chunk)

        return self._stub.Transcribe(request_iter())
