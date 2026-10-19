"""Tests for STTWorkerPool and STT model idle timeout (LS-10)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.models import resolve
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.session import CallSession
from live_stt.worker import (
    StandbyKey,
    StandbyWorker,
    WorkerHandle,
    spawn_standby_sync,
)
from live_stt.worker_pool import (
    STTWorkerPool,
    clear_worker_pool,
    evict_idle_workers,
    get_worker_pool,
    stop_worker_pool,
)

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"


def _model_filename(model_key: str) -> str:
    try:
        return resolve(model_key).gguf_filename
    except KeyError:
        return f"{model_key}.gguf"


def _make_key(model_key: str = "realtime_eou_120m-v1", language: str = "") -> StandbyKey:
    return StandbyKey(
        worker_bin=str(FAKE_WORKER),
        gguf_path=f"/fake/{_model_filename(model_key)}",
        language=language,
        n_threads=1,
        ggml_lib_dir=None,
        ggml_backend_path=None,
        use_gpu=False,
        model_key=model_key,
    )


def _spawn_fake_standby(
    model_key: str = "realtime_eou_120m-v1",
    language: str = "",
    extra_env: tuple[tuple[str, str], ...] = (),
) -> StandbyWorker:
    return spawn_standby_sync(
        worker_bin=FAKE_WORKER,
        gguf_path=f"/fake/{_model_filename(model_key)}",
        language=language,
        n_threads=1,
        model_key=model_key,
        extra_env=extra_env,
    )


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if line.startswith("State:") and "Z" in line:
                    return False
    except OSError:
        return False
    return True


class TestSTTWorkerPoolCore:
    def test_put_and_acquire_roundtrip(self) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            key = worker.key
            ok = pool.put(worker, timeout_sec=600.0)
            assert ok is True
            assert pool.count() == 1

            acquired = pool.acquire(key)
            assert acquired is worker
            assert pool.count() == 0

            # Second acquire on empty pool returns None
            assert pool.acquire(key) is None
        finally:
            pool.close()

    def test_evict_idle_workers_kills_process_and_frees_slot(self) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            pool.put(worker, timeout_sec=600.0)
            proc = worker.proc
            assert proc.poll() is None

            # Simulate idle timeout expiration
            worker.idle_since = time.monotonic() - 650.0
            evicted = pool.evict_idle(timeout_sec=600.0)
            assert evicted == 1
            assert pool.count() == 0

            # Worker process must have been terminated
            proc.wait(timeout=2.0)
            assert proc.returncode is not None
        finally:
            pool.close()

    def test_put_with_zero_timeout_kills_worker_immediately(self) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            proc = worker.proc
            ok = pool.put(worker, timeout_sec=0.0)
            assert ok is False
            assert pool.count() == 0
            proc.wait(timeout=2.0)
            assert proc.returncode is not None
        finally:
            pool.close()

    def test_crashed_worker_is_discarded_on_acquire_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            key = worker.key
            dead_pid = worker.proc.pid
            pool.put(worker, timeout_sec=600.0)

            # Simulate worker death while idling in pool
            worker.proc.kill()
            worker.proc.wait(timeout=2.0)

            with caplog.at_level(logging.WARNING):
                assert pool.acquire(key) is None
            assert pool.count() == 0
            assert any(
                "discarded standby" in rec.message
                and f"pid={dead_pid}" in rec.message
                and "exit_code=" in rec.message
                for rec in caplog.records
            )
        finally:
            pool.close()

    def test_has_warm_discards_dead_standby_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            key = worker.key
            dead_pid = worker.proc.pid
            pool.put(worker, timeout_sec=600.0)

            worker.proc.kill()
            worker.proc.wait(timeout=2.0)

            with caplog.at_level(logging.WARNING):
                assert pool.has_warm(key) is False
            assert pool.count() == 0
            assert any(
                "discarded standby" in rec.message
                and f"pid={dead_pid}" in rec.message
                and "exit_code=" in rec.message
                for rec in caplog.records
            )
        finally:
            pool.close()

    def test_clear_kills_all_workers(self) -> None:
        pool = STTWorkerPool()
        try:
            w1 = _spawn_fake_standby("model-1")
            w2 = _spawn_fake_standby("model-2")
            pool.put(w1, 600.0)
            pool.put(w2, 600.0)
            assert pool.count() == 2

            cleared = pool.clear()
            assert cleared == 2
            assert pool.count() == 0

            w1.proc.wait(timeout=2.0)
            w2.proc.wait(timeout=2.0)
            assert w1.proc.returncode is not None
            assert w2.proc.returncode is not None
        finally:
            pool.close()

    def test_clear_invalidates_inflight_background_spawns(self) -> None:
        pool = STTWorkerPool()
        try:
            key = _make_key()
            pool.maintain_warm_async(key=key, timeout_sec=600.0)
            pool.clear()
            time.sleep(0.5)
            assert pool.count() == 0
        finally:
            pool.close()

    def test_snapshot_reports_worker_metadata(self) -> None:
        pool = STTWorkerPool()
        try:
            w = _spawn_fake_standby("realtime_eou_120m-v1")
            pool.put(w, 600.0)
            snap = pool.snapshot()
            assert len(snap) == 1
            assert snap[0]["model"] == "realtime_eou_120m-v1"
            assert snap[0]["pid"] == w.proc.pid
            assert snap[0]["use_gpu"] is False
            assert snap[0]["idle_sec"] >= 0.0
        finally:
            pool.close()

    @pytest.mark.asyncio
    async def test_maintain_warm_async_survives_after_spawn_completes_and_runs_feed_finalize(
        self,
    ) -> None:
        """LS-15 regression: a worker pre-warmed via maintain_warm_async must not
        be SIGKILLed by PR_SET_PDEATHSIG after the background spawn task finishes."""
        pool = STTWorkerPool()
        os.environ["FAKE_WORDS_PER_SEC"] = "10"
        try:
            key = _make_key("parakeet-tdt-0.6b-v3")
            pool.maintain_warm_async(key=key, timeout_sec=600.0)
            for _ in range(100):
                if pool.count() == 1:
                    break
                await asyncio.sleep(0.02)
            assert pool.count() == 1

            # Wait past when a short-lived spawner thread would have exited and
            # triggered the kernel's PR_SET_PDEATHSIG(SIGKILL).
            await asyncio.sleep(0.1)
            assert pool._spawner_thread.is_alive()

            acquired = pool.acquire(key, wait_inflight=False)
            assert acquired is not None
            assert acquired.proc.poll() is None

            handle = await WorkerHandle.from_standby(acquired)
            try:
                feed_doc = await handle.feed(b"\x00\x00" * 1600)
                assert feed_doc["fed_samples"] == 1600
                final_doc = await handle.finalize()
                assert final_doc["fed_samples"] == 1600
                assert acquired.proc.poll() is None
            finally:
                handle.kill()
                await handle.wait_closed()
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)
            pool.close()

    @pytest.mark.asyncio
    async def test_acquire_waits_for_inflight_background_spawn_and_runs_feed_finalize(
        self,
    ) -> None:
        """LS-15 wait-for-in-flight regression: claiming an in-flight standby via
        acquire(wait_inflight=True) must yield a worker that stays alive and
        completes feed() + finalize() without ConnectionResetError."""
        pool = STTWorkerPool()
        os.environ["FAKE_LOAD_DELAY_SEC"] = "0.15"
        os.environ["FAKE_WORDS_PER_SEC"] = "10"
        try:
            key = _make_key("parakeet-tdt-0.6b-v3")
            pool.maintain_warm_async(key=key, timeout_sec=600.0)
            for _ in range(20):
                if pool.has_warm(key):
                    break
                await asyncio.sleep(0.01)
            assert pool.has_warm(key) is True

            acquired = await asyncio.to_thread(
                pool.acquire, key, wait_inflight=True, wait_timeout_sec=3.0
            )
            assert acquired is not None
            assert pool.count() == 0

            # Give the kernel time to deliver PDEATHSIG if the spawner thread had exited
            await asyncio.sleep(0.1)
            assert acquired.proc.poll() is None

            handle = await WorkerHandle.from_standby(acquired)
            try:
                assert acquired.key == key
                feed_doc = await handle.feed(b"\x00\x00" * 1600)
                assert feed_doc["fed_samples"] == 1600
                final_doc = await handle.finalize()
                assert final_doc["fed_samples"] == 1600
                assert acquired.proc.poll() is None
            finally:
                handle.kill()
                await handle.wait_closed()
        finally:
            os.environ.pop("FAKE_LOAD_DELAY_SEC", None)
            os.environ.pop("FAKE_WORDS_PER_SEC", None)
            pool.close()

    def test_maintain_warm_async_skips_gpu_prewarm_when_vram_budget_is_tight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from live_stt import vram

        monkeypatch.setattr(vram, "can_prewarm_standby", lambda settings, *, model_key: False)
        pool = STTWorkerPool()
        try:
            gpu_key = StandbyKey(
                worker_bin=str(FAKE_WORKER),
                gguf_path=f"/fake/{_model_filename('parakeet-tdt-0.6b-v3')}",
                language="",
                n_threads=1,
                ggml_lib_dir=None,
                ggml_backend_path=None,
                use_gpu=True,
                model_key="parakeet-tdt-0.6b-v3",
            )
            pool.maintain_warm_async(
                key=gpu_key,
                timeout_sec=600.0,
                settings=Settings(_env_file=None),
            )
            time.sleep(0.2)
            assert pool.count() == 0
        finally:
            pool.close()

    def test_evict_idle_respects_exclude_model_key(self) -> None:
        pool = STTWorkerPool()
        try:
            w_tdt = _spawn_fake_standby("parakeet-tdt-0.6b-v3")
            w_eou = _spawn_fake_standby("realtime_eou_120m-v1")
            pool.put(w_tdt, 600.0)
            pool.put(w_eou, 600.0)
            assert pool.count() == 2

            evicted = pool.evict_idle(timeout_sec=0.0, exclude_model_key="parakeet-tdt-0.6b-v3")
            assert evicted == 1
            assert pool.count() == 1
            assert pool.snapshot()[0]["model"] == "parakeet-tdt-0.6b-v3"
            w_eou.proc.wait(timeout=2.0)
            assert w_eou.proc.returncode is not None
        finally:
            pool.close()

    def test_pdeathsig_kills_standby_worker_when_server_process_dies(self) -> None:
        """LS-15: PR_SET_PDEATHSIG remains active on standby workers spawned by
        STTWorkerPool so that if the parent server process dies abruptly (SIGKILL),
        its standby workers are automatically SIGKILLed by the kernel."""
        repo_root = Path(__file__).resolve().parent.parent
        script = (
            "import sys, time; "
            f"sys.path.insert(0, {str(repo_root)!r}); "
            "from pathlib import Path; "
            "from live_stt.worker import StandbyKey; "
            "from live_stt.worker_pool import STTWorkerPool; "
            f"key = StandbyKey(worker_bin={str(FAKE_WORKER)!r}, gguf_path='/fake/m.gguf', "
            "language='', n_threads=1, model_key='realtime_eou_120m-v1'); "
            "pool = STTWorkerPool(); "
            "pool.maintain_warm_async(key=key, timeout_sec=600.0); "
            "w = pool.acquire(key, wait_inflight=True, wait_timeout_sec=5.0); "
            "assert w is not None and w.proc.poll() is None; "
            "print(w.proc.pid, flush=True); "
            "time.sleep(30)"
        )
        parent = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        worker_pid: int | None = None
        try:
            assert parent.stdout is not None
            line = parent.stdout.readline().strip()
            assert line, f"child server failed to report worker pid: {parent.stderr.read() if parent.stderr else ''}"
            worker_pid = int(line)

            # While parent server is alive, the standby worker must remain alive
            time.sleep(0.1)
            assert _is_pid_alive(worker_pid) is True

            # Abruptly SIGKILL the parent server process (no atexit or Python cleanup runs)
            parent.kill()
            parent.wait(timeout=5.0)

            # Kernel PDEATHSIG must kill the standby worker
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and _is_pid_alive(worker_pid):
                time.sleep(0.02)
            assert _is_pid_alive(worker_pid) is False
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2.0)
            if worker_pid is not None and _is_pid_alive(worker_pid):
                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class TestCallSessionWorkerPoolIntegration:
    @pytest.mark.asyncio
    async def test_session_acquires_prewarmed_worker_and_reuses_replenished_standby(self) -> None:
        pool = STTWorkerPool()
        os.environ["FAKE_WORDS_PER_SEC"] = "20"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin=str(FAKE_WORKER),
                models_dir="/fake",
                stt_idle_timeout_sec=600.0,
            )
            spec = resolve("realtime_eou_120m-v1")
            config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
            budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
            budget.try_admit_call()

            # Pre-warm a worker matching this session's parameters
            standby = _spawn_fake_standby(spec.key)
            pool.put(standby, timeout_sec=600.0)
            assert pool.count() == 1

            session = CallSession(settings, spec, config, budget, worker_pool=pool)
            ready_event = await session.start()
            assert ready_event.ready.model == spec.key
            # Worker was acquired from pool
            assert pool.count() == 0
            assert session._worker.proc is standby.proc

            # Run feed + finalize
            feed_events = await session.feed_audio(b"\x00\x00" * 2560)
            final_events = await session.finalize()
            assert len(feed_events) == 1
            assert len(final_events) == 1

            proc = session._worker.proc
            await session.close()
            # Active worker must be killed on close
            assert proc.returncode is not None

            # Background thread will replenish warm standby worker in pool
            for _ in range(50):
                if pool.count() == 1:
                    break
                await asyncio.sleep(0.05)
            assert pool.count() == 1
            await asyncio.sleep(0.1)
            replenished_pid = pool.snapshot()[0]["pid"]

            # Second session must successfully reuse the replenished standby worker
            session2 = CallSession(settings, spec, config, budget, worker_pool=pool)
            await session2.start()
            assert session2._worker is not None
            assert session2._worker.proc.pid == replenished_pid
            await session2.feed_audio(b"\x00\x00" * 2560)
            final_events2 = await session2.finalize()
            assert len(final_events2) == 1
            await session2.close()
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)
            pool.close()

    @pytest.mark.asyncio
    async def test_dead_standby_in_pool_falls_back_to_fresh_worker_and_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LS-15 fallback (pool check): a standby that died while idle in the pool
        is logged as discarded and CallSession transparently spawns a fresh worker."""
        pool = STTWorkerPool()
        os.environ["FAKE_WORDS_PER_SEC"] = "20"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin=str(FAKE_WORKER),
                models_dir="/fake",
                stt_idle_timeout_sec=600.0,
            )
            spec = resolve("realtime_eou_120m-v1")
            config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
            budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
            budget.try_admit_call()

            standby = _spawn_fake_standby(spec.key)
            dead_pid = standby.proc.pid
            pool.put(standby, timeout_sec=600.0)
            standby.proc.kill()
            standby.proc.wait(timeout=2.0)

            session = CallSession(settings, spec, config, budget, worker_pool=pool)
            with caplog.at_level(logging.WARNING):
                await session.start()
                await session.feed_audio(b"\x00\x00" * 2560)
                final_events = await session.finalize()
            assert len(final_events) == 1
            assert session._worker is not None
            assert session._worker.proc.pid != dead_pid
            assert any(
                "discarded standby" in rec.message and f"pid={dead_pid}" in rec.message
                for rec in caplog.records
            )
            await session.close()
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)
            pool.close()

    @pytest.mark.asyncio
    async def test_standby_dying_after_acquire_before_attach_falls_back_to_fresh_worker(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LS-15 fallback (attach/PING check): if a standby dies in the window
        between acquire() and WorkerHandle.from_standby(), CallSession logs the
        discarded standby, kills it, and spawns a fresh worker."""
        pool = STTWorkerPool()
        os.environ["FAKE_BATCH_WORDS_PER_SEC"] = "10"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin_parakeet_tdt=str(FAKE_WORKER),
                models_dir="/fake",
                n_threads_parakeet_tdt=1,
                stt_idle_timeout_sec=600.0,
            )
            spec = resolve("parakeet-tdt-0.6b-v3")
            config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
            budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
            budget.try_admit_call()

            session = CallSession(settings, spec, config, budget, worker_pool=pool)
            standby = _spawn_fake_standby(spec.key, extra_env=session._extra_env())
            dead_pid = standby.proc.pid
            pool.put(standby, timeout_sec=600.0)

            orig_acquire = pool.acquire

            def _acquire_then_kill(*args, **kwargs):
                w = orig_acquire(*args, **kwargs)
                if w is not None:
                    w.proc.kill()
                    w.proc.wait(timeout=2.0)
                return w

            monkeypatch.setattr(pool, "acquire", _acquire_then_kill)

            with caplog.at_level(logging.WARNING):
                await session.start()
                await session.feed_audio(b"\x00\x00" * 3200)
                final_events = await session.finalize()
            assert len(final_events) == 1
            assert final_events[0].final.text
            assert session._worker is not None
            assert session._worker.proc.pid != dead_pid
            assert any(
                "discarded standby" in rec.message and f"pid={dead_pid}" in rec.message
                for rec in caplog.records
            )
            await session.close()
        finally:
            os.environ.pop("FAKE_BATCH_WORDS_PER_SEC", None)
            pool.close()

    @pytest.mark.asyncio
    async def test_standby_dying_after_start_before_first_feed_falls_back_to_fresh_worker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """LS-15 fallback (first-request check): if a reused standby passes
        start() / PING and then dies before its first feed() in _finalize_batch,
        CallSession replaces it with a fresh worker instead of failing."""
        pool = STTWorkerPool()
        os.environ["FAKE_BATCH_WORDS_PER_SEC"] = "10"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin_parakeet_tdt=str(FAKE_WORKER),
                models_dir="/fake",
                n_threads_parakeet_tdt=1,
                stt_idle_timeout_sec=600.0,
            )
            spec = resolve("parakeet-tdt-0.6b-v3")
            config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
            budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
            budget.try_admit_call()

            session = CallSession(settings, spec, config, budget, worker_pool=pool)
            standby = _spawn_fake_standby(spec.key, extra_env=session._extra_env())
            dead_pid = standby.proc.pid
            pool.put(standby, timeout_sec=600.0)

            await session.start()
            assert session._worker is not None
            assert session._worker.proc.pid == dead_pid

            # Kill the worker after start() succeeded, before finalize() feeds the first chunk
            session._worker.proc.kill()
            session._worker.proc.wait(timeout=2.0)

            await session.feed_audio(b"\x00\x00" * 3200)
            with caplog.at_level(logging.WARNING):
                final_events = await session.finalize()
            assert len(final_events) == 1
            assert final_events[0].final.text
            assert session._worker.proc.pid != dead_pid
            assert any(
                "discarded standby" in rec.message and f"pid={dead_pid}" in rec.message
                for rec in caplog.records
            )
            await session.close()
        finally:
            os.environ.pop("FAKE_BATCH_WORDS_PER_SEC", None)
            pool.close()

    @pytest.mark.asyncio
    async def test_session_with_zero_timeout_does_not_cache(self) -> None:
        pool = STTWorkerPool()
        try:
            settings = Settings(
                _env_file=None,
                worker_bin=str(FAKE_WORKER),
                models_dir="/fake",
                stt_idle_timeout_sec=0.0,  # one-shot mode
            )
            spec = resolve("realtime_eou_120m-v1")
            config = asr_pb2.StreamConfig(encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000)
            budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
            budget.try_admit_call()

            session = CallSession(settings, spec, config, budget, worker_pool=pool)
            await session.start()
            proc = session._worker.proc
            await session.close()
            assert proc.returncode is not None

            # With idle_timeout == 0, no background warm worker is spawned
            await asyncio.sleep(0.2)
            assert pool.count() == 0
        finally:
            pool.close()


class TestTranscribeHttpWorkerPoolIntegration:
    @pytest.mark.asyncio
    async def test_transcribe_http_uses_and_replenishes_worker_pool_across_consecutive_requests(
        self,
    ) -> None:
        import io
        import json
        import wave
        from live_stt import transcribe_http
        from live_stt.transcribe_sessions import TranscribeSessionTracker

        def _wav_bytes(rate: int = 16000, n_frames: int = 2560) -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(b"\x00" * (n_frames * 2))
            return buf.getvalue()

        boundary = "----pool-test"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "parakeet-tdt-0.6b-v3\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + _wav_bytes() + f"\r\n--{boundary}--\r\n".encode()

        pool = STTWorkerPool()
        os.environ["FAKE_BATCH_WORDS_PER_SEC"] = "50"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin=str(FAKE_WORKER),
                worker_bin_parakeet_tdt=str(FAKE_WORKER),
                models_dir="/fake",
                stt_idle_timeout_sec=600.0,
            )
            budget = WorkerBudget(3, 1)
            tracker = TranscribeSessionTracker()

            # Request 1 starts cold (no standby in pool yet) and triggers background pre-warm on close
            status1, resp1, _ = await transcribe_http.handle_transcribe_request(
                content_type=f"multipart/form-data; boundary={boundary}",
                body=body,
                settings=settings,
                budget=budget,
                draining=False,
                tracker=tracker,
                worker_pool=pool,
            )

            assert status1 == 200
            doc1 = json.loads(resp1)
            assert doc1["text"]

            # Pool is replenished in background with a warm standby worker
            for _ in range(50):
                if pool.count() == 1:
                    break
                await asyncio.sleep(0.05)
            assert pool.count() == 1
            await asyncio.sleep(0.1)
            snap = pool.snapshot()[0]
            assert snap["model"] == "parakeet-tdt-0.6b-v3"
            warm_pid = snap["pid"]

            # Request 2 reuses the background-spawned standby worker and must succeed (LS-15)
            status2, resp2, _ = await transcribe_http.handle_transcribe_request(
                content_type=f"multipart/form-data; boundary={boundary}",
                body=body,
                settings=settings,
                budget=budget,
                draining=False,
                tracker=tracker,
                worker_pool=pool,
            )
            assert status2 == 200
            doc2 = json.loads(resp2)
            assert doc2["text"] == doc1["text"]

            # The warm worker from Request 1's replenishment was used and then killed on close
            for _ in range(50):
                if not _is_pid_alive(warm_pid):
                    break
                await asyncio.sleep(0.02)
            assert _is_pid_alive(warm_pid) is False
        finally:
            os.environ.pop("FAKE_BATCH_WORDS_PER_SEC", None)
            pool.close()



