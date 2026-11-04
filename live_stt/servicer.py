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

from live_stt import __about__, gpu, metrics, models, nemotron, redaction, vram
from live_stt.recent_requests import RecentRequests, get_recent_requests
from live_stt.logging_config import get_logger
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.session import CallSession
from live_stt.state import ServerState
from live_stt.worker import WorkerError

logger = get_logger("servicer")


def _count_words(event: asr_pb2.TranscriptionEvent, text_parts: list[str]) -> tuple[int, int]:
    """Bumps words_total/transcript_chars_total and returns the same
    (words, chars) increment, so the per-call totals recorded in
    live_stt/recent_requests.py match the Prometheus counters exactly."""
    kind = event.WhichOneof("event")
    if kind == "delta":
        n_words, n_chars = len(event.delta.words), len(event.delta.text)
        text_parts.append(event.delta.text)
    elif kind == "final":
        n_words, n_chars = len(event.final.words), len(event.final.text)
        text_parts.append(event.final.text)
    else:
        return 0, 0
    metrics.words_total.inc(n_words)
    metrics.transcript_chars_total.inc(n_chars)
    return n_words, n_chars


class StreamingASRServicer(asr_pb2_grpc.StreamingASRServicer):
    def __init__(self, state: ServerState, recent: RecentRequests | None = None) -> None:
        self._state = state
        self._settings = state.settings
        self._budget = state.budget
        self._recent = recent or getattr(state, "recent_requests", None) or get_recent_requests()

    async def GetServerInfo(  # noqa: N802 -- generated base class naming
        self, request: asr_pb2.ServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> asr_pb2.ServerInfoResponse:
        version = __about__.info()
        default_model = self._settings.default_model or models.DEFAULT_MODEL_KEY
        eff_max = vram.effective_max_calls(self._settings, default_model)
        return asr_pb2.ServerInfoResponse(
            version=version["hash"],
            built_at=version["timestamp"],
            parakeet_ref=version["parakeet_ref"],
            default_model=default_model,
            # ServerInfoResponse.backend is a single proto field (a schema
            # change to make it per-engine wasn't warranted for LS-5) --
            # reports parakeet_backend specifically, since GetServerInfo is
            # only ever meaningful to a gRPC streaming client, and Transcribe
            # only ever spawns the parakeet engine (batch-only engines are
            # rejected before a worker exists -- see the streaming_capable
            # check below).
            backend=self._settings.parakeet_backend,
            max_concurrent_calls=self._settings.max_concurrent_calls,
            effective_max_concurrent_calls=(
                eff_max if eff_max is not None else self._settings.max_concurrent_calls
            ),
            active_calls=self._budget.active_calls,
            warm_spares=0,  # no PRE-warmed pool -- shadows are spawned on demand at rotation time
        )

    def _record_recent(
        self,
        outcome: str,
        started_at: float,
        spec: models.ModelSpec | None,
        session: CallSession | None,
        n_words: int,
        n_chars: int,
    ) -> None:
        """Usage-only entry for GET /api/requests/recent. Never raises.
        Numbers stay None when no worker ever ran (a pre-start failure)."""
        try:
            ran = session is not None and session.device is not None
            device = session.device if ran else (
                self._settings.parakeet_backend if spec is not None else None
            )
            self._recent.record(
                kind="stream",
                outcome=outcome,
                started_at=started_at,
                model=spec.key if spec is not None else None,
                device=device,
                audio_seconds=session.fed_audio_sec if ran else None,
                words=n_words if ran else None,
                transcript_chars=n_chars if ran else None,
                processing_s=session.processing_sec if ran else None,
            )
        except Exception:  # noqa: BLE001
            logger.exception("recent-requests bookkeeping failed")

    async def Transcribe(  # noqa: N802
        self,
        request_iterator: AsyncIterator[asr_pb2.TranscriptionRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[asr_pb2.TranscriptionEvent]:
        started_at = time.time()
        if self._state.draining:
            metrics.admission_rejections_total.labels(reason="draining").inc()
            self._recent.record(kind="stream", outcome="draining", started_at=started_at)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "draining")
            return

        # Race-free without a lock: try_admit_call()/release_call() have no
        # `await` between reading and mutating their counters, and grpc.aio
        # servicers run on a single event loop thread, so no other
        # Transcribe call can interleave here.
        if not self._budget.try_admit_call():
            metrics.admission_rejections_total.labels(reason="no_capacity").inc()
            self._recent.record(kind="stream", outcome="no_capacity", started_at=started_at)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "at capacity, try again shortly")
            return

        metrics.streams_active.inc()
        call_start = time.monotonic()
        call_ref_hash = "unset"
        outcome = "internal"
        session: CallSession | None = None
        admission: vram.Admission | None = None
        cpu_live_held = False
        final_text_parts: list[str] = []
        spec: models.ModelSpec | None = None
        n_words = 0
        n_chars = 0

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
                spec = models.resolve(config.model or self._settings.default_model)
            except KeyError as exc:
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
                return

            # Batch-only engines (whisper) have no genuine incremental
            # streaming state -- see live_stt/models.py's docstring -- so
            # they are rejected here, before any worker is ever spawned,
            # rather than being allowed to silently degrade into a
            # half-working "streams, but only ever emits one big delta at
            # the end" experience over this RPC. They remain reachable via
            # POST /v1/audio/transcriptions (live_stt/transcribe_http.py),
            # which is already engine-agnostic.
            if not spec.streaming_capable:
                metrics.stream_init_failures_total.labels(reason="bad_config").inc()
                outcome = "invalid_argument"
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"model {spec.key!r} is batch-only (no incremental streaming support); "
                    "use POST /v1/audio/transcriptions instead of the Transcribe RPC",
                )
                return

            if self._settings.parakeet_backend == "cuda":
                # A CUDA allocation failure is an abort(), not a catchable
                # exception (see CLAUDE.md) -- admitting a call this box can't
                # fit crashes a worker, not just slows it. Runs HERE, after the
                # model is known, because the requirement is per model
                # (live_stt/vram.py); it used to run before the first message
                # with one flat figure for every model. parakeet_backend
                # specifically: Transcribe only ever spawns streaming parakeet
                # models (the streaming_capable gate just above). In a thread:
                # admit() shells out to nvidia-smi and may poll while idle
                # workers are reclaimed -- never block the event loop on that.
                already_warm = bool(
                    self._state.worker_pool is not None
                    and self._settings.stt_idle_timeout_sec > 0
                    and self._state.worker_pool.has_warm(spec.key, use_gpu=True)
                )
                admission = await asyncio.to_thread(
                    vram.admit,
                    self._settings,
                    model_key=spec.key,
                    model_mb=spec.vram_mb,
                    fallback_mb=self._settings.vram_per_worker_mb,
                    already_warm=already_warm,
                )
                if not admission.ok:
                    # LS-19 option (b): live calls take priority over
                    # Nemotron post-call jobs. If one holds the card, stop it
                    # (the job is re-queued and reruns later) and re-check.
                    # Nothing else ever preempts a running Nemotron job.
                    def _readmit() -> vram.Admission:
                        return vram.admit(
                            self._settings,
                            model_key=spec.key,
                            model_mb=spec.vram_mb,
                            fallback_mb=self._settings.vram_per_worker_mb,
                            already_warm=already_warm,
                        )

                    retried = await asyncio.to_thread(nemotron.preempt_and_admit, self._settings, _readmit)
                    if retried is not None:
                        logger.info(
                            "stream=%s preempted a Nemotron job for VRAM: %s",
                            call_ref_hash, "admitted" if retried.ok else "still short",
                        )
                        admission = retried
                if admission.free_mb is not None:
                    metrics.gpu_free_vram_mb.set(admission.free_mb)
                if not admission.ok:
                    metrics.admission_rejections_total.labels(reason="no_vram").inc()
                    outcome = "resource_exhausted"
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, admission.message())
                    return

            # LS-20: on CPU, a single Nemotron job saturates the CPU (~14 of 16
            # cores measured on homesrv). Hold CPU Nemotron jobs while any live
            # call is active, and preempt a running CPU Nemotron job so the
            # live call gets CPU capacity immediately.
            nemotron.get_service().enter_live_call()
            cpu_live_held = True
            if nemotron.get_service().is_active_on("cpu"):
                preempted_cpu = await asyncio.to_thread(
                    nemotron.preempt_cpu_for_live_call, self._settings
                )
                if preempted_cpu:
                    logger.info("stream=%s preempted a CPU Nemotron job for live call", call_ref_hash)

            session = CallSession(
                self._settings, spec, config, self._budget, worker_pool=self._state.worker_pool
            )
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
                self._settings.parakeet_backend,
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
                        dw, dc = _count_words(event, final_text_parts)
                        n_words += dw
                        n_chars += dc
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
                    dw, dc = _count_words(event, final_text_parts)
                    n_words += dw
                    n_chars += dc
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
            if admission is not None:
                admission.release()
            if cpu_live_held:
                nemotron.get_service().leave_live_call()
            self._budget.release_call()
            metrics.streams_active.dec()
            metrics.streams_total.labels(outcome=outcome).inc()
            duration_sec = time.monotonic() - call_start
            metrics.stream_duration_seconds.observe(duration_sec)
            self._record_recent(outcome, started_at, spec, session, n_words, n_chars)

            full_text = "".join(final_text_parts)
            fields = redaction.transcript_log_fields(full_text, call_ref_hash, self._settings)
            logger.info(
                "stream=%s close outcome=%s dur_s=%.1f %s",
                call_ref_hash,
                outcome,
                duration_sec,
                " ".join(f"{k}={v}" for k, v in fields.items()),
            )
