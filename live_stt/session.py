"""CallSession: the RPC IS the session.

one phone call = one gRPC Transcribe stream = one CallSession
  = an ordered chain of parakeet_stream GENERATIONS, exactly one live at a
    time (two only during the brief, deliberate overlap of a rotation)

A CallSession is constructed inside servicer.Transcribe()'s call and lives
only in that coroutine's local scope: there is no registry, no dict, no
lookup-by-id anywhere. That absence is what makes "no session_id" true
rather than aspirational (see the proto's service-level comment).

Rotation exists because parakeet.cpp's streaming session leaks memory that
`stream_free`+`stream_begin` cannot reclaim (upstream issue #63; measured
much smaller on this repo's CPU build than the upstream CUDA report -- see
CLAUDE.md -- but the mechanism is also what recovers from a worker crash,
so it stays even though the CPU leak alone wouldn't demand it). The design
is an OVERLAPPED dual-feed, not kill-and-replay: replaying the overlap while
live audio keeps arriving never converges at RTF near 1 (see CLAUDE.md), so
instead a shadow worker is fed the same audio as the active one for
``rotation_overlap_sec``, primed and silent, and only takes over at a clean
cut point (an <EOU> if one arrives, else the hard deadline).
"""

from __future__ import annotations

import time
from pathlib import Path

from live_stt import __about__, metrics
from live_stt.admission import WorkerBudget
from live_stt.boundary import dedup_incoming_words
from live_stt.config import Settings
from live_stt.events import worker_json_to_events
from live_stt.logging_config import get_logger
from live_stt.models import ModelSpec, strip_language_tag
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.worker import WorkerError, WorkerHandle

logger = get_logger("session")


def _reason_name(reason: int) -> str:
    return asr_pb2.RecycleReason.Name(reason).removeprefix("RECYCLE_REASON_").lower()


class CallSession:
    def __init__(
        self,
        settings: Settings,
        model_spec: ModelSpec,
        config: asr_pb2.StreamConfig,
        budget: WorkerBudget,
    ) -> None:
        self._settings = settings
        self._model_spec = model_spec
        self._config = config
        self._budget = budget

        self._worker: WorkerHandle | None = None
        self._fed_samples = 0
        self._buf = bytearray()
        self._chunk_bytes = int(16000 * model_spec.model_chunk_ms / 1000) * 2

        self._active_generation_start_sec = 0.0
        self._worker_generations = 1
        self._last_rss_kb = 0

        # Rotation-in-progress state. self._incoming is the shadow worker;
        # non-None exactly while a rotation is underway.
        self._incoming: WorkerHandle | None = None
        self._incoming_generation_start_sec: float | None = None
        self._rotation_cut_deadline_sec: float | None = None
        self._rotation_reason: int | None = None
        self._incoming_pending_words: list[asr_pb2.Word] = []

    @property
    def audio_offset_sec(self) -> float:
        return self._fed_samples / 16000.0

    async def start(self) -> asr_pb2.TranscriptionEvent:
        self._worker = await self._spawn_worker()
        return asr_pb2.TranscriptionEvent(
            ready=asr_pb2.Ready(
                model=self._model_spec.key,
                supports_turn_detection=self._model_spec.has_eou,
                has_punctuation=self._model_spec.has_punctuation,
                model_chunk_ms=self._model_spec.model_chunk_ms,
                accepted_sample_rate_hz=16000,
            )
        )

    async def _spawn_worker(self) -> WorkerHandle:
        gguf_path = str(Path(self._settings.models_dir) / self._model_spec.gguf_filename)
        ggml_lib_dir = (
            Path(self._settings.worker_ggml_lib_dir) if self._settings.worker_ggml_lib_dir else None
        )
        start = time.monotonic()
        handle = await WorkerHandle.spawn(
            worker_bin=Path(self._settings.worker_bin),
            gguf_path=gguf_path,
            language=self._config.language,
            n_threads=self._settings.n_threads_per_worker,
            ggml_lib_dir=ggml_lib_dir,
        )
        metrics.model_load_duration_seconds.observe(time.monotonic() - start)
        version = __about__.info()
        metrics.set_build_info(
            version=version["hash"],
            parakeet_ref=version["parakeet_ref"],
            backend=self._settings.backend,
            model=self._model_spec.key,
            n_threads=self._settings.n_threads_per_worker,
            ggml_features=handle.ready.get("ggml_features", ""),
        )
        return handle

    async def feed_audio(self, pcm16le: bytes) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "feed_audio called before start()"
        self._buf += pcm16le
        events: list[asr_pb2.TranscriptionEvent] = []
        while len(self._buf) >= self._chunk_bytes:
            chunk = bytes(self._buf[: self._chunk_bytes])
            del self._buf[: self._chunk_bytes]
            events.extend(await self._feed_chunk(chunk))
        return events

    async def _feed_chunk(self, chunk: bytes) -> list[asr_pb2.TranscriptionEvent]:
        events: list[asr_pb2.TranscriptionEvent] = []
        n_samples = len(chunk) // 2

        feed_start = time.monotonic()
        try:
            doc = await self._worker.feed(chunk)
        except WorkerError as exc:
            self._fed_samples += n_samples
            metrics.asr_errors_total.labels(code="feed_failed").inc()
            events.extend(await self._recover_from_crash(exc))
            return events
        metrics.feed_duration_seconds.observe(time.monotonic() - feed_start)

        self._fed_samples += n_samples
        metrics.audio_seconds_total.inc(n_samples / 16000.0)
        self._last_rss_kb = doc.get("rss_kb", self._last_rss_kb)
        events.extend(self._doc_to_events(doc, time_offset_sec=self._active_generation_start_sec))

        if self._incoming is not None:
            events.extend(await self._advance_rotation(chunk))
        else:
            reason = self._should_start_rotation(doc)
            if reason is not None:
                await self._begin_rotation(reason)

        return events

    def _should_start_rotation(self, active_doc: dict) -> int | None:
        if self._last_rss_kb >= self._settings.worker_rss_soft_kb:
            return asr_pb2.RECYCLE_REASON_RSS_THRESHOLD

        generation_audio_sec = self.audio_offset_sec - self._active_generation_start_sec
        if generation_audio_sec >= self._settings.rotate_after_sec:
            return asr_pb2.RECYCLE_REASON_AUDIO_CAP

        active_eou = bool(active_doc.get("eou")) or any(
            e.get("type") == "eou" for e in active_doc.get("events", [])
        )
        if active_eou and generation_audio_sec >= 0.7 * self._settings.rotate_after_sec:
            return asr_pb2.RECYCLE_REASON_EOU_OPPORTUNISTIC

        return None

    async def _begin_rotation(self, reason: int) -> None:
        if not self._budget.try_acquire_rotation_shadow():
            # No spare worker slot right now (e.g. another call is mid-
            # rotation and has the reserve). Not fatal -- just try again on
            # a later chunk instead of forcing a gap.
            return
        try:
            incoming = await self._spawn_worker()
        except WorkerError:
            self._budget.release_rotation_shadow()
            logger.warning("rotation shadow failed to start; will retry later")
            return

        self._incoming = incoming
        self._incoming_generation_start_sec = self.audio_offset_sec
        self._rotation_cut_deadline_sec = self.audio_offset_sec + self._settings.rotation_overlap_sec
        self._rotation_reason = reason
        self._incoming_pending_words = []

    async def _advance_rotation(self, chunk: bytes) -> list[asr_pb2.TranscriptionEvent]:
        assert self._incoming is not None
        try:
            incoming_doc = await self._incoming.feed(chunk)
        except WorkerError:
            logger.warning("rotation shadow died during dual-feed; abandoning this attempt")
            await self._abandon_rotation()
            return []

        # Collected, not emitted, until cutover. Rebased using the shadow's
        # ACTUAL start point (tracked directly), not derived from the
        # eventual cut time and the configured overlap -- those only agree
        # when cutover happens exactly at the deadline; an early <EOU> cutover
        # would make that derivation wrong.
        for w in incoming_doc.get("words", []):
            self._incoming_pending_words.append(
                asr_pb2.Word(
                    text=w["w"],
                    start_sec=w["start"] + self._incoming_generation_start_sec,
                    end_sec=w["end"] + self._incoming_generation_start_sec,
                    confidence=w.get("conf", 0.0),
                )
            )

        incoming_eou = bool(incoming_doc.get("eou")) or any(
            e.get("type") == "eou" for e in incoming_doc.get("events", [])
        )
        at_deadline = self.audio_offset_sec >= self._rotation_cut_deadline_sec
        if incoming_eou or at_deadline:
            return await self._cut_over()
        return []

    async def _cut_over(self) -> list[asr_pb2.TranscriptionEvent]:
        assert self._incoming is not None and self._incoming_generation_start_sec is not None
        events: list[asr_pb2.TranscriptionEvent] = []
        t_cut_sec = self.audio_offset_sec
        outgoing = self._worker

        try:
            final_doc = await outgoing.finalize()
            events.extend(
                self._doc_to_events(final_doc, time_offset_sec=self._active_generation_start_sec)
            )
        except WorkerError as exc:
            logger.warning("outgoing worker failed to finalize during rotation cutover: %s", exc)
        outgoing.kill()
        await outgoing.wait_closed()

        kept_words = dedup_incoming_words(self._incoming_pending_words, t_cut_sec=t_cut_sec)
        if kept_words:
            # Reconstructed from words, not the shadow's own discarded "text"
            # field: simple space-joining is an approximation (real
            # detokenization may format contractions/punctuation slightly
            # differently), acceptable because this path is rare -- a tight
            # overlap window with an early <EOU> cutover leaving a
            # non-empty remainder -- not the common case.
            events.append(
                asr_pb2.TranscriptionEvent(
                    delta=asr_pb2.TranscriptDelta(
                        text=" ".join(w.text for w in kept_words),
                        words=kept_words,
                        audio_offset_sec=self.audio_offset_sec,
                    )
                )
            )

        events.append(
            asr_pb2.TranscriptionEvent(
                recycled=asr_pb2.Recycled(
                    reason=self._rotation_reason,
                    gap_sec=0.0,
                    at_audio_sec=t_cut_sec,
                    warm=True,
                )
            )
        )
        metrics.rotations_total.labels(kind="warm").inc()
        metrics.worker_restarts_total.labels(reason=_reason_name(self._rotation_reason)).inc()

        self._worker = self._incoming
        self._active_generation_start_sec = self._incoming_generation_start_sec
        self._worker_generations += 1
        self._last_rss_kb = 0

        # It's no longer a second (shadow) worker for this call -- it's now
        # THE worker, still covered by the call's own base slot. Only the
        # extra reserve unit it was borrowing is returned.
        self._budget.release_rotation_shadow()
        self._incoming = None
        self._incoming_generation_start_sec = None
        self._rotation_cut_deadline_sec = None
        self._rotation_reason = None
        self._incoming_pending_words = []

        return events

    async def _abandon_rotation(self) -> None:
        if self._incoming is not None:
            self._incoming.kill()
            try:
                await self._incoming.wait_closed()
            except Exception:
                pass
            self._budget.release_rotation_shadow()
        self._incoming = None
        self._incoming_generation_start_sec = None
        self._rotation_cut_deadline_sec = None
        self._rotation_reason = None
        self._incoming_pending_words = []

    async def _recover_from_crash(self, exc: WorkerError) -> list[asr_pb2.TranscriptionEvent]:
        """The ACTIVE worker died unexpectedly (crash/hang/connection loss),
        not during a planned rotation. Acceptable to lose the decoder cache
        on an interrupted call (see CLAUDE.md) -- replace it with a fresh
        worker and keep the call alive. Only a terminal error (WorkerError
        propagating out of this method, from the replacement's own spawn
        failing) ends the call.
        """
        logger.warning("active worker lost: %s -- attempting cold recovery", exc)
        self._worker.kill()
        try:
            await self._worker.wait_closed()
        except Exception:
            pass

        # Any in-flight rotation shadow was dual-feeding the worker that
        # just died; its own state is now inconsistent with what the client
        # has already seen. Simpler and safer to abandon it than to try to
        # promote it in the dead worker's place.
        await self._abandon_rotation()

        replacement = await self._spawn_worker()  # WorkerError here IS terminal
        self._worker = replacement
        self._active_generation_start_sec = self.audio_offset_sec
        self._worker_generations += 1
        self._last_rss_kb = 0

        metrics.rotations_total.labels(kind="cold").inc()
        metrics.worker_restarts_total.labels(reason="crash").inc()

        return [
            asr_pb2.TranscriptionEvent(
                recycled=asr_pb2.Recycled(
                    reason=asr_pb2.RECYCLE_REASON_CRASH,
                    # The chunk that triggered the crash was sent but never
                    # acknowledged, and there is no ring buffer (yet) to
                    # replay it from -- one model chunk's worth of audio is
                    # the honest estimate of what was lost, not 0.
                    gap_sec=self._model_spec.model_chunk_ms / 1000.0,
                    at_audio_sec=self.audio_offset_sec,
                    warm=False,
                )
            )
        ]

    def _doc_to_events(self, doc: dict, *, time_offset_sec: float) -> list[asr_pb2.TranscriptionEvent]:
        return worker_json_to_events(
            doc,
            time_offset_sec=time_offset_sec,
            audio_offset_sec=self.audio_offset_sec,
            strip_tag=self._model_spec.strip_language_tag,
        )

    async def finalize(self) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "finalize called before start()"
        events: list[asr_pb2.TranscriptionEvent] = []
        if self._buf:
            # n_samples == 0 is legal per parakeet_capi.h, and so is any
            # other length -- feed whatever's left, even a partial chunk.
            events.extend(await self._feed_chunk(bytes(self._buf)))
            self._buf.clear()

        if self._incoming is not None:
            # The call is ending -- no need to complete a rotation in
            # flight; the shadow's contribution would only matter for a
            # call that keeps going.
            await self._abandon_rotation()

        doc = await self._worker.finalize()
        text = doc.get("text", "")
        if self._model_spec.strip_language_tag:
            text = strip_language_tag(text)
        words = [
            asr_pb2.Word(
                text=w["w"],
                start_sec=w["start"] + self._active_generation_start_sec,
                end_sec=w["end"] + self._active_generation_start_sec,
                confidence=w.get("conf", 0.0),
            )
            for w in doc.get("words", [])
        ]
        events.append(
            asr_pb2.TranscriptionEvent(
                final=asr_pb2.Final(
                    text=text,
                    words=words,
                    total_audio_sec=self.audio_offset_sec,
                    worker_generations=self._worker_generations,
                )
            )
        )
        return events

    async def close(self) -> None:
        """Always SIGKILLs every worker this session ever spawned -- see
        CLAUDE.md: a worker process is never reused for a second call, on
        any exit path, clean or not. Safe to call after finalize()
        (idempotent) and safe to call without ever having called finalize()
        (the client-disconnect path, which must NOT finalize -- there is
        nowhere to send the output).
        """
        await self._abandon_rotation()
        if self._worker is not None:
            self._worker.kill()
            try:
                await self._worker.wait_closed()
            except Exception:
                logger.warning("worker did not report an exit code during close()")
