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

import asyncio
import time
from pathlib import Path

from live_stt import __about__, metrics, vram
from live_stt.admission import WorkerBudget
from live_stt.backends import installer
from live_stt.boundary import (
    dedup_incoming_words,
    iter_batch_windows,
    merge_batch_window_words,
)
from live_stt.config import Settings
from live_stt.events import worker_json_to_events
from live_stt.logging_config import get_logger
from live_stt.models import ModelSpec, strip_language_tag
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.worker import StandbyKey, WorkerError, WorkerHandle
from live_stt.worker_pool import STTWorkerPool

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
        worker_pool: STTWorkerPool | None = None,
    ) -> None:
        self._settings = settings
        self._model_spec = model_spec
        self._config = config
        self._budget = budget
        self._worker_pool = worker_pool

        self._worker: WorkerHandle | None = None
        self._fed_samples = 0
        self._buf = bytearray()
        self._chunk_bytes = int(16000 * model_spec.model_chunk_ms / 1000) * 2

        self._active_generation_start_sec = 0.0
        self._worker_generations = 1
        self._last_rss_kb = 0
        self._use_gpu = False
        self._vram_reservation_id = f"session-{id(self)}"
        self._reserved_compute_mb = 0
        self._last_finalize_doc: dict[str, object] = {}

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
        if self._use_gpu:
            compute_needed = vram.compute_mb(self._settings, self._model_spec.key)
            if compute_needed > 0:
                self._reserved_compute_mb = compute_needed
                vram.reserve_active_compute(self._vram_reservation_id, compute_needed)
        return asr_pb2.TranscriptionEvent(
            ready=asr_pb2.Ready(
                model=self._model_spec.key,
                supports_turn_detection=self._model_spec.has_eou,
                has_punctuation=self._model_spec.has_punctuation,
                model_chunk_ms=self._model_spec.model_chunk_ms,
                accepted_sample_rate_hz=16000,
            )
        )

    def _resolve_lib_dir(
        self, registry_engine: str, engine_backend: str, fallback_lib_dir: Path | None
    ) -> Path | None:
        """LS-5 Phase 7b/7d: prefer a backend INSTALLED+ACTIVATED via the
        registry/admin UI (live_stt/backends/installer.py) over the
        image-baked default -- this is the one place the download
        mechanism actually changes what gets spawned. `registry_engine` is
        the CANONICAL engine name (installer.get_active_backend already
        applies live_stt/backends/registry.py's ENGINE_ALIASES, so
        "parakeet_tdt" and "parakeet" share one active row).

        No cpu/gpu flavor is baked into the image anymore as of this phase
        (LS-5's explicit goal) -- so unlike the transitional version of
        this method, a cuda request with nothing installed+activated is a
        hard, loud error here, not a silent fall-through to a cpu .so
        while still claiming use_gpu=True (which would silently run on
        CPU, or crash inside a CPU-only ggml build, depending on the
        engine -- either way a real correctness failure this codebase's
        "fail loud, not silently downgrade" convention elsewhere -- see
        live_stt/redaction.py -- exists specifically to avoid). A flavor
        MISMATCH (active installed backend doesn't match the configured
        *_backend setting) is equally an error: it means an operator's
        setting and their installed/activated backend have drifted apart,
        which WorkerHandle.spawn() silently proceeding on would hide.
        """
        active = installer.get_active_backend(self._settings.db_path, registry_engine)
        if active is not None:
            if active.flavor != engine_backend:
                raise WorkerError(
                    f"{registry_engine}_backend={engine_backend!r} but the active installed "
                    f"backend for engine {registry_engine!r} is flavor={active.flavor!r} -- "
                    "install+activate a matching flavor via POST /api/backends/install and "
                    "/api/backends/activate, or change the setting to match"
                )
            return Path(active.path)
        if engine_backend == "cuda":
            raise WorkerError(
                f"{registry_engine}_backend=cuda but no cuda backend has been installed+activated "
                f"for engine {registry_engine!r} -- POST /api/backends/install "
                f"{{'engine': '{registry_engine}', 'flavor': 'cuda'}}, then /api/backends/activate, first"
            )
        return fallback_lib_dir

    def _resolve_spawn_params(self) -> tuple[Path, str, str, int, Path | None, Path | None, bool, str]:
        gguf_path = str(Path(self._settings.models_dir) / self._model_spec.gguf_filename)
        if self._model_spec.engine == "whisper":
            worker_bin = Path(self._settings.worker_bin_whisper)
            engine_backend = self._settings.whisper_backend
            fallback_lib_dir = (
                Path(self._settings.worker_ggml_lib_dir_whisper)
                if self._settings.worker_ggml_lib_dir_whisper
                else None
            )
            ggml_lib_dir = self._resolve_lib_dir("whisper", engine_backend, fallback_lib_dir)
            n_threads = self._settings.n_threads_whisper
            ggml_backend_path = (
                ggml_lib_dir / "libggml-cuda.so" if engine_backend == "cuda" and ggml_lib_dir else None
            )
            use_gpu = engine_backend == "cuda"
        elif self._model_spec.engine == "parakeet_tdt":
            worker_bin = Path(self._settings.worker_bin_parakeet_tdt)
            engine_backend = self._settings.parakeet_tdt_backend
            fallback_lib_dir = (
                Path(self._settings.worker_ggml_lib_dir_parakeet_tdt)
                if self._settings.worker_ggml_lib_dir_parakeet_tdt
                else (
                    Path(self._settings.worker_ggml_lib_dir)
                    if self._settings.worker_ggml_lib_dir
                    else None
                )
            )
            ggml_lib_dir = self._resolve_lib_dir("parakeet", engine_backend, fallback_lib_dir)
            n_threads = self._settings.n_threads_parakeet_tdt
            ggml_backend_path = None
            use_gpu = engine_backend == "cuda"
        else:
            worker_bin = Path(self._settings.worker_bin)
            engine_backend = self._settings.parakeet_backend
            fallback_lib_dir = (
                Path(self._settings.worker_ggml_lib_dir)
                if self._settings.worker_ggml_lib_dir
                else None
            )
            ggml_lib_dir = self._resolve_lib_dir("parakeet", engine_backend, fallback_lib_dir)
            n_threads = self._settings.n_threads_per_worker
            ggml_backend_path = None
            use_gpu = engine_backend == "cuda"

        return worker_bin, gguf_path, self._config.language, n_threads, ggml_lib_dir, ggml_backend_path, use_gpu, engine_backend

    def _extra_env(self) -> tuple[tuple[str, str], ...]:
        """Per-engine worker environment. Today only the batch TDT engine needs
        one: PARAKEET_ATT_CONTEXT forces local attention so memory grows
        linearly with audio length instead of quadratically (see
        Settings.parakeet_tdt_att_context). Not set for the streaming engines,
        whose chunked encoder has its own baked-in attention context."""
        if self._model_spec.engine == "parakeet_tdt" and self._settings.parakeet_tdt_att_context > 0:
            return (("PARAKEET_ATT_CONTEXT", str(self._settings.parakeet_tdt_att_context)),)
        return ()

    def _resolve_standby_key(self) -> StandbyKey:
        worker_bin, gguf_path, language, n_threads, ggml_lib_dir, ggml_backend_path, use_gpu, _ = (
            self._resolve_spawn_params()
        )
        return StandbyKey(
            worker_bin=str(worker_bin),
            gguf_path=gguf_path,
            language=language,
            n_threads=n_threads,
            ggml_lib_dir=str(ggml_lib_dir) if ggml_lib_dir else None,
            ggml_backend_path=str(ggml_backend_path) if ggml_backend_path else None,
            use_gpu=use_gpu,
            model_key=self._model_spec.key,
            extra_env=self._extra_env(),
        )

    async def _spawn_worker(self) -> WorkerHandle:
        worker_bin, gguf_path, language, n_threads, ggml_lib_dir, ggml_backend_path, use_gpu, engine_backend = (
            self._resolve_spawn_params()
        )
        self._use_gpu = use_gpu
        # LS-10: check if a warmed-up standby worker is already waiting in memory/GPU
        if self._worker_pool is not None and self._settings.stt_idle_timeout_sec > 0:
            key = StandbyKey(
                worker_bin=str(worker_bin),
                gguf_path=gguf_path,
                language=language,
                n_threads=n_threads,
                ggml_lib_dir=str(ggml_lib_dir) if ggml_lib_dir else None,
                ggml_backend_path=str(ggml_backend_path) if ggml_backend_path else None,
                use_gpu=use_gpu,
                model_key=self._model_spec.key,
                extra_env=self._extra_env(),
            )
            standby = self._worker_pool.acquire(key, wait_inflight=False)
            if standby is None:
                standby = await asyncio.to_thread(self._worker_pool.acquire, key, wait_inflight=True)
            if standby is not None:
                metrics.stt_standby_workers_active.set(self._worker_pool.count())
                handle = await WorkerHandle.from_standby(standby)
                metrics.model_load_duration_seconds.observe(0.0)
                version = __about__.info()
                metrics.set_build_info(
                    version=version["hash"],
                    parakeet_ref=version["parakeet_ref"],
                    backend=engine_backend,
                    model=self._model_spec.key,
                    n_threads=n_threads,
                    ggml_features=handle.ready.get("ggml_features", ""),
                )
                return handle

        start = time.monotonic()
        handle = await WorkerHandle.spawn(
            worker_bin=worker_bin,
            gguf_path=gguf_path,
            language=language,
            n_threads=n_threads,
            ggml_lib_dir=ggml_lib_dir,
            ggml_backend_path=ggml_backend_path,
            use_gpu=use_gpu,
            extra_env=self._extra_env(),
        )
        metrics.model_load_duration_seconds.observe(time.monotonic() - start)
        version = __about__.info()
        metrics.set_build_info(
            version=version["hash"],
            parakeet_ref=version["parakeet_ref"],
            backend=engine_backend,
            model=self._model_spec.key,
            n_threads=n_threads,
            ggml_features=handle.ready.get("ggml_features", ""),
        )
        return handle

    async def feed_audio(self, pcm16le: bytes) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "feed_audio called before start()"
        self._buf += pcm16le
        if not self._model_spec.streaming_capable:
            # LS-14: batch-only models (parakeet_tdt, whisper) buffer in Python
            # until finalize(), which feeds and finalizes one window at a time
            # so the worker's sample buffer and GPU compute graph stay bounded
            # by batch_window_sec instead of the full recording length.
            return []
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
        # Batch-only engines (whisper) only emit a transcript at finalize()
        # -- see live_stt/models.py's docstring and worker/session_whisper.hpp.
        # A rotation SIGKILLs the active worker mid-call; for a streaming
        # engine that's safe because every word up to that point was already
        # captured progressively (self._doc_to_events already ran on each
        # prior RESULT frame), but whisper's worker has emitted NOTHING yet
        # at that point -- everything buffered since start() would be
        # silently lost. So rotation is unconditionally disabled for these;
        # long HTTP uploads rely on max_call_sec/finalize_timeout_sec as the
        # outer bound instead. Checked first, before any threshold math.
        if not self._model_spec.streaming_capable:
            return None

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

    async def _finalize_batch(self) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None
        raw_pcm = bytes(self._buf)
        self._buf.clear()
        if len(raw_pcm) % 2:
            raw_pcm = raw_pcm[:-1]
        total_samples = len(raw_pcm) // 2

        window_sec = vram.effective_batch_window_sec(self._settings, self._model_spec.key)
        overlap_override = getattr(self._settings, "batch_overlap_sec", None)
        overlap_sec = (
            float(overlap_override)
            if overlap_override is not None
            else float(self._model_spec.batch_overlap_sec)
        )
        if window_sec > 0:
            overlap_sec = min(max(0.0, overlap_sec), window_sec * 0.5)
        window_samples = int(round(window_sec * 16000)) if window_sec > 0 else 0
        overlap_samples = int(round(overlap_sec * 16000)) if window_sec > 0 else 0

        windows = iter_batch_windows(total_samples, window_samples, overlap_samples)

        if self._use_gpu and self._reserved_compute_mb > 0:
            await asyncio.to_thread(
                vram.ensure_compute_headroom,
                self._settings,
                model_key=self._model_spec.key,
                compute_needed_mb=self._reserved_compute_mb,
            )

        merged_words: list[asr_pb2.Word] = []
        window_texts: list[str] = []
        prev_win_end_sec = 0.0

        for win_idx, (win_start_sample, win_end_sample) in enumerate(windows):
            win_start_sec = win_start_sample / 16000.0
            win_end_sec = win_end_sample / 16000.0
            win_pcm = raw_pcm[win_start_sample * 2 : win_end_sample * 2]

            win_words: list[asr_pb2.Word] = []
            feed_texts: list[str] = []

            for offset in range(0, len(win_pcm), self._chunk_bytes):
                chunk = win_pcm[offset : offset + self._chunk_bytes]
                feed_start = time.monotonic()
                try:
                    feed_doc = await self._worker.feed(chunk)
                except WorkerError:
                    metrics.asr_errors_total.labels(code="feed_failed").inc()
                    raise
                metrics.feed_duration_seconds.observe(time.monotonic() - feed_start)
                self._last_rss_kb = feed_doc.get("rss_kb", self._last_rss_kb)
                feed_text = feed_doc.get("text", "")
                if self._model_spec.strip_language_tag and feed_text:
                    feed_text = strip_language_tag(feed_text)
                if feed_text:
                    feed_texts.append(feed_text)
                for w in feed_doc.get("words", []):
                    win_words.append(
                        asr_pb2.Word(
                            text=w["w"],
                            start_sec=w["start"] + win_start_sec,
                            end_sec=w["end"] + win_start_sec,
                            confidence=w.get("conf", 0.0),
                        )
                    )

            doc = await self._worker.finalize()
            self._last_finalize_doc = doc
            self._last_rss_kb = doc.get("rss_kb", self._last_rss_kb)
            final_text = doc.get("text", "")
            if self._model_spec.strip_language_tag and final_text:
                final_text = strip_language_tag(final_text)
            for w in doc.get("words", []):
                win_words.append(
                    asr_pb2.Word(
                        text=w["w"],
                        start_sec=w["start"] + win_start_sec,
                        end_sec=w["end"] + win_start_sec,
                        confidence=w.get("conf", 0.0),
                    )
                )

            win_text = " ".join([*feed_texts, *([final_text] if final_text else [])])
            window_texts.append(win_text)

            if win_idx == 0:
                merged_words = win_words
            else:
                merged_words = merge_batch_window_words(
                    merged_words,
                    win_words,
                    win_start_sec=win_start_sec,
                    prev_end_sec=prev_win_end_sec,
                )
            prev_win_end_sec = win_end_sec

        self._fed_samples = total_samples
        if total_samples > 0:
            metrics.audio_seconds_total.inc(total_samples / 16000.0)

        if len(windows) == 1:
            text = window_texts[0] if window_texts else ""
        elif merged_words:
            text = " ".join(w.text for w in merged_words)
            if self._model_spec.strip_language_tag:
                text = strip_language_tag(text)
        else:
            text = " ".join(t.strip() for t in window_texts if t.strip())

        return [
            asr_pb2.TranscriptionEvent(
                final=asr_pb2.Final(
                    text=text,
                    words=merged_words,
                    total_audio_sec=self.audio_offset_sec,
                    worker_generations=self._worker_generations,
                )
            )
        ]

    async def finalize(self) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "finalize called before start()"
        if not self._model_spec.streaming_capable:
            return await self._finalize_batch()

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
        self._last_finalize_doc = doc
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
        vram.release_active_compute(self._vram_reservation_id)
        self._reserved_compute_mb = 0
        await self._abandon_rotation()
        if self._worker is not None:
            self._worker.kill()
            try:
                await self._worker.wait_closed()
            except Exception:
                logger.warning("worker did not report an exit code during close()")

        if self._worker_pool is not None and self._settings.stt_idle_timeout_sec > 0:
            try:
                key = self._resolve_standby_key()
                self._worker_pool.maintain_warm_async(
                    key=key,
                    timeout_sec=self._settings.stt_idle_timeout_sec,
                    settings=self._settings,
                )
            except Exception as exc:
                logger.debug("could not maintain warm worker for pool: %s", exc)

