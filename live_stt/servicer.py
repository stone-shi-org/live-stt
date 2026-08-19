"""StreamingASR servicer.

Transcribe is wired to CallSession, which owns the full rotation state
machine (Phase 3) -- see live_stt/session.py and CLAUDE.md. Admission here
is the real reserve-aware design: WorkerBudget distinguishes call slots
(gates whether a NEW call is admitted) from worker-process slots (gates
whether an in-progress call's rotation can get a shadow), so the reserve is
never handed to a new call.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc

from live_stt import __about__, models
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.session import CallSession
from live_stt.worker import WorkerError

logger = get_logger("servicer")


class StreamingASRServicer(asr_pb2_grpc.StreamingASRServicer):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)

    async def GetServerInfo(  # noqa: N802 -- generated base class naming
        self, request: asr_pb2.ServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> asr_pb2.ServerInfoResponse:
        version = __about__.info()
        return asr_pb2.ServerInfoResponse(
            version=version["hash"],
            built_at=version["timestamp"],
            parakeet_ref=version["parakeet_ref"],
            default_model=self._settings.default_model or models.DEFAULT_MODEL_KEY,
            backend=self._settings.backend,
            max_concurrent_calls=self._settings.max_concurrent_calls,
            active_calls=self._budget.active_calls,
            warm_spares=0,  # no PRE-warmed pool -- shadows are spawned on demand at rotation time
        )

    async def Transcribe(  # noqa: N802
        self,
        request_iterator: AsyncIterator[asr_pb2.TranscriptionRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[asr_pb2.TranscriptionEvent]:
        # Race-free without a lock: try_admit_call()/release_call() have no
        # `await` between reading and mutating their counters, and grpc.aio
        # servicers run on a single event loop thread, so no other
        # Transcribe call can interleave here.
        if not self._budget.try_admit_call():
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "at capacity, try again shortly")
            return
        try:
            async for event in self._transcribe(request_iterator, context):
                yield event
        finally:
            self._budget.release_call()

    async def _transcribe(
        self,
        request_iterator: AsyncIterator[asr_pb2.TranscriptionRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[asr_pb2.TranscriptionEvent]:
        try:
            first = await request_iterator.__anext__()
        except StopAsyncIteration:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "stream closed before any message")
            return

        if first.WhichOneof("payload") != "config":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be StreamConfig")
            return
        config = first.config

        if config.encoding not in (
            asr_pb2.AUDIO_ENCODING_UNSPECIFIED,
            asr_pb2.AUDIO_ENCODING_LINEAR16,
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unsupported encoding {asr_pb2.AudioEncoding.Name(config.encoding)}; "
                "only AUDIO_ENCODING_LINEAR16 is accepted -- decode/resample client-side "
                "(see live_stt.client.telephony)",
            )
            return
        if config.sample_rate_hz not in (0, 16000):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unsupported sample_rate_hz={config.sample_rate_hz}; only 16000 is accepted",
            )
            return

        try:
            spec = models.resolve(config.model or None)
        except KeyError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return

        session = CallSession(self._settings, spec, config, self._budget)
        try:
            ready_event = await session.start()
        except WorkerError as exc:
            logger.error("worker init failed: %s", exc)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "engine failed to start")
            return
        yield ready_event

        try:
            async for request in request_iterator:
                which = request.WhichOneof("payload")
                if which == "config":
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "config sent twice")
                    return
                if which != "audio":
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty request payload")
                    return
                try:
                    for event in await session.feed_audio(request.audio):
                        yield event
                except WorkerError as exc:
                    logger.error("worker feed failed: %s", exc)
                    await context.abort(grpc.StatusCode.INTERNAL, "engine error")
                    return

            # Half-close: the client is done sending, but the tail --
            # including the final word of the final utterance, which only
            # ever materializes at finalize() -- still needs to reach the
            # client. This is why iterating request_iterator directly with
            # `async for` and continuing past it in the same coroutine is
            # correct here: nothing about half-close skips this code.
            try:
                for event in await session.finalize():
                    yield event
            except WorkerError as exc:
                logger.error("worker finalize failed: %s", exc)
                await context.abort(grpc.StatusCode.INTERNAL, "engine error during finalize")
                return
        finally:
            # Always runs: normal completion, an abort() above (which raises
            # through this finally), or the client cancelling the RPC. Kills
            # the worker unconditionally -- see CallSession.close()'s
            # docstring for why that's correct even after a client
            # disconnect that never got to finalize.
            await session.close()
