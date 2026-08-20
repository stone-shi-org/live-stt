"""StreamingASR servicer.

Transcribe is wired to CallSession, which owns the full rotation state
machine (Phase 3) -- see live_stt/session.py and CLAUDE.md. Admission uses
the shared ServerState: WorkerBudget distinguishes call slots (gates
whether a NEW call is admitted) from worker-process slots (gates whether an
in-progress call's rotation can get a shadow), so the reserve is never
handed to a new call; ``state.draining`` is set once, immediately, on
SIGTERM (live_stt/server.py) and checked here before anything else.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import grpc

from live_stt import __about__, gpu, metrics, models, redaction
from live_stt.logging_config import get_logger
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.session import CallSession
from live_stt.state import ServerState
from live_stt.worker import WorkerError

logger = get_logger("servicer")


def _count_words(event: asr_pb2.TranscriptionEvent, text_parts: list[str]) -> None:
    kind = event.WhichOneof("event")
    if kind == "delta":
        metrics.words_total.inc(len(event.delta.words))
        metrics.transcript_chars_total.inc(len(event.delta.text))
        text_parts.append(event.delta.text)
    elif kind == "final":
        metrics.words_total.inc(len(event.final.words))
        metrics.transcript_chars_total.inc(len(event.final.text))
        text_parts.append(event.final.text)


class StreamingASRServicer(asr_pb2_grpc.StreamingASRServicer):
    def __init__(self, state: ServerState) -> None:
        self._state = state
        self._settings = state.settings
        self._budget = state.budget

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
        if self._state.draining:
            metrics.admission_rejections_total.labels(reason="draining").inc()
            await context.abort(grpc.StatusCode.UNAVAILABLE, "draining")
            return

        if self._settings.backend == "cuda":
            # A CUDA allocation failure is an abort(), not a catchable
            # exception (see CLAUDE.md) -- admitting a call this box can't
            # actually fit crashes a worker process, not just slows one
            # down. Checked before the (cheap, synchronous) call-slot check
            # below since this one is the more expensive shell-out.
            free = gpu.free_vram_mb()
            if free is not None:
                metrics.gpu_free_vram_mb.set(free)
            required = self._settings.vram_per_worker_mb + self._settings.vram_reserve_mb
            if free is not None and free < required:
                metrics.admission_rejections_total.labels(reason="no_vram").inc()
                await context.abort(
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    f"insufficient VRAM: {free}MB free, {required}MB required",
                )
                return

        # Race-free without a lock: try_admit_call()/release_call() have no
        # `await` between reading and mutating their counters, and grpc.aio
        # servicers run on a single event loop thread, so no other
        # Transcribe call can interleave here.
        if not self._budget.try_admit_call():
            metrics.admission_rejections_total.labels(reason="no_capacity").inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "at capacity, try again shortly")
            return

        metrics.streams_active.inc()
        call_start = time.monotonic()
        call_ref_hash = "unset"
        outcome = "internal"
        session: CallSession | None = None
        final_text_parts: list[str] = []

        try:
            try:
                first = await request_iterator.__anext__()
            except StopAsyncIteration:
                metrics.stream_init_failures_total.labels(reason="no_messages").inc()
                outcome = "invalid_argument"
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "stream closed before any message")
                return

            if first.WhichOneof("payload") != "config":
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be StreamConfig")
                return
            config = first.config
            call_ref_hash = redaction.hash_call_ref(config.call_id)

            if config.encoding not in (
                asr_pb2.AUDIO_ENCODING_UNSPECIFIED,
                asr_pb2.AUDIO_ENCODING_LINEAR16,
            ):
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unsupported encoding {asr_pb2.AudioEncoding.Name(config.encoding)}; "
                    "only AUDIO_ENCODING_LINEAR16 is accepted -- decode/resample client-side "
                    "(see live_stt.client.telephony)",
                )
                return
            if config.sample_rate_hz not in (0, 16000):
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"unsupported sample_rate_hz={config.sample_rate_hz}; only 16000 is accepted",
                )
                return

            try:
                spec = models.resolve(config.model or None)
            except KeyError as exc:
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                return

            session = CallSession(self._settings, spec, config, self._budget)
            try:
                ready_event = await session.start()
            except WorkerError as exc:
                logger.error("stream=%s worker init failed: %s", call_ref_hash, exc)
                metrics.stream_init_failures_total.labels(reason="spawn_failed").inc()
                outcome = "unavailable"
                await context.abort(grpc.StatusCode.UNAVAILABLE, "engine failed to start")
                return
            logger.info(
                "stream=%s open enc=%s sr=%s model=%s backend=%s",
                call_ref_hash,
                asr_pb2.AudioEncoding.Name(config.encoding),
                config.sample_rate_hz or 16000,
                spec.key,
                self._settings.backend,
            )
            yield ready_event

            async for request in request_iterator:
                which = request.WhichOneof("payload")
                if which == "config":
                    outcome = "invalid_argument"
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "config sent twice")
                    return
                if which != "audio":
                    outcome = "invalid_argument"
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "empty request payload")
                    return
                try:
                    for event in await session.feed_audio(request.audio):
                        _count_words(event, final_text_parts)
                        yield event
                except WorkerError as exc:
                    logger.error("stream=%s worker feed failed: %s", call_ref_hash, exc)
                    metrics.asr_errors_total.labels(code="feed_fatal").inc()
                    outcome = "internal"
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
                    _count_words(event, final_text_parts)
                    yield event
                outcome = "ok"
            except WorkerError as exc:
                logger.error("stream=%s worker finalize failed: %s", call_ref_hash, exc)
                metrics.asr_errors_total.labels(code="finalize_fatal").inc()
                outcome = "internal"
                await context.abort(grpc.StatusCode.INTERNAL, "engine error during finalize")
                return
        except asyncio.CancelledError:
            # Client disconnect. Deliberately not caught more broadly than
            # this -- grpc.aio's own abort() exceptions must keep propagating
            # so the RPC actually ends with the status already set above.
            outcome = "client_cancel"
            raise
        finally:
            # Always runs: normal completion, an abort() above, or the
            # client cancelling the RPC. session.close() SIGKILLs every
            # worker this call ever spawned unconditionally -- see
            # CallSession.close()'s docstring for why that's correct even
            # after a client disconnect that never got to finalize().
            if session is not None:
                await session.close()
            self._budget.release_call()
            metrics.streams_active.dec()
            metrics.streams_total.labels(outcome=outcome).inc()
            duration_sec = time.monotonic() - call_start
            metrics.stream_duration_seconds.observe(duration_sec)

            full_text = "".join(final_text_parts)
            fields = redaction.transcript_log_fields(full_text, call_ref_hash, self._settings)
            logger.info(
                "stream=%s close outcome=%s dur_s=%.1f %s",
                call_ref_hash,
                outcome,
                duration_sec,
                " ".join(f"{k}={v}" for k, v in fields.items()),
            )
