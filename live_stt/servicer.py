"""StreamingASR servicer.

Phase 0/1 skeleton: GetServerInfo is real (static build/config info).
Transcribe is NOT wired to the worker pool yet -- that is
live_stt/session.py's CallSession + live_stt/pool/supervisor.py (Phase 2/3,
see CLAUDE.md's implementation phases). It responds UNIMPLEMENTED rather than
accepting a stream and hanging, so a client finds out immediately.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc

from live_stt import __about__
from live_stt.config import Settings
from live_stt.models import DEFAULT_MODEL_KEY
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc


class StreamingASRServicer(asr_pb2_grpc.StreamingASRServicer):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def GetServerInfo(  # noqa: N802 -- generated base class naming
        self, request: asr_pb2.ServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> asr_pb2.ServerInfoResponse:
        version = __about__.info()
        return asr_pb2.ServerInfoResponse(
            version=version["hash"],
            built_at=version["timestamp"],
            parakeet_ref=version["parakeet_ref"],
            default_model=self._settings.default_model or DEFAULT_MODEL_KEY,
            backend=self._settings.backend,
            max_concurrent_calls=self._settings.max_concurrent_calls,
            active_calls=0,
            warm_spares=0,
        )

    async def Transcribe(  # noqa: N802
        self,
        request_iterator: AsyncIterator[asr_pb2.TranscriptionRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[asr_pb2.TranscriptionEvent]:
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Transcribe is not wired to the worker pool yet (Phase 2/3) -- "
            "see CLAUDE.md's implementation phases",
        )
        return
        yield  # pragma: no cover -- makes this an async generator for grpc.aio
