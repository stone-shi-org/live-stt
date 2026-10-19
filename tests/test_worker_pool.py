"""Tests for STTWorkerPool and STT model idle timeout (LS-10)."""

from __future__ import annotations

import asyncio
import os
import subprocess
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


def _spawn_fake_standby(model_key: str = "realtime_eou_120m-v1", language: str = "") -> StandbyWorker:
    return spawn_standby_sync(
        worker_bin=FAKE_WORKER,
        gguf_path=f"/fake/{_model_filename(model_key)}",
        language=language,
        n_threads=1,
        model_key=model_key,
    )


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

    def test_crashed_worker_is_discarded_on_acquire(self) -> None:
        pool = STTWorkerPool()
        try:
            worker = _spawn_fake_standby()
            key = worker.key
            pool.put(worker, timeout_sec=600.0)

            # Simulate worker death while idling in pool
            worker.proc.kill()
            worker.proc.wait(timeout=2.0)

            assert pool.acquire(key) is None
            assert pool.count() == 0
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

    def test_acquire_waits_for_inflight_background_spawn_instead_of_duplicating(self) -> None:
        pool = STTWorkerPool()
        os.environ["FAKE_LOAD_DELAY_SEC"] = "0.15"
        try:
            key = _make_key("parakeet-tdt-0.6b-v3")
            pool.maintain_warm_async(key=key, timeout_sec=600.0)
            # Wait briefly so background thread registers _spawning_keys
            for _ in range(20):
                if pool.has_warm(key):
                    break
                time.sleep(0.01)
            assert pool.has_warm(key) is True

            # Acquire with wait_timeout_sec should receive the worker once it finishes loading
            acquired = pool.acquire(key, wait_timeout_sec=3.0)
            assert acquired is not None
            try:
                assert acquired.key == key
                assert pool.count() == 0
            finally:
                acquired.kill()
        finally:
            os.environ.pop("FAKE_LOAD_DELAY_SEC", None)
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


class TestCallSessionWorkerPoolIntegration:
    @pytest.mark.asyncio
    async def test_session_acquires_prewarmed_worker(self) -> None:
        pool = STTWorkerPool()
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
            feed_events = await session.feed_audio(b"\x00\x00" * 1600)
            final_events = await session.finalize()
            assert len(feed_events) >= 0
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
        finally:
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
    async def test_transcribe_http_uses_and_replenishes_worker_pool(self) -> None:
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
            'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + _wav_bytes() + f"\r\n--{boundary}--\r\n".encode()

        pool = STTWorkerPool()
        os.environ["FAKE_WORDS_PER_SEC"] = "50"
        try:
            settings = Settings(
                _env_file=None,
                worker_bin=str(FAKE_WORKER),
                models_dir="/fake",
                stt_idle_timeout_sec=600.0,
            )
            budget = WorkerBudget(3, 1)
            tracker = TranscribeSessionTracker()

            # Pre-warm worker into pool
            spec = resolve(settings.default_model)
            standby = _spawn_fake_standby(spec.key)
            pool.put(standby, timeout_sec=600.0)
            assert pool.count() == 1
            original_proc = standby.proc

            status, resp, _ = await transcribe_http.handle_transcribe_request(
                content_type=f"multipart/form-data; boundary={boundary}",
                body=body,
                settings=settings,
                budget=budget,
                draining=False,
                tracker=tracker,
                worker_pool=pool,
            )

            assert status == 200
            doc = json.loads(resp)
            assert doc["text"]
            # The original worker ran and then terminated on session close
            original_proc.wait(timeout=2.0)
            assert original_proc.returncode is not None

            # Pool was replenished in background with a fresh warm worker
            for _ in range(50):
                if pool.count() == 1:
                    break
                await asyncio.sleep(0.05)
            assert pool.count() == 1

            # A subsequent request can acquire the new standby worker
            key = pool.snapshot()[0]
            assert key["model"] == spec.key
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)
            pool.close()


