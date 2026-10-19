"""STT Worker Pool (LS-10): Pre-warmed standby STT workers and model idle timeout.

Eliminates cold-start model weight loading latency (~0.5s-5s) across
consecutive streaming gRPC calls and batch HTTP requests.

When stt_idle_timeout_sec > 0:
- Pre-warmed standby workers sit in memory/GPU before receiving audio.
- When a request/call starts, it immediately acquires the warm worker (0ms load).
- Upon session completion, the active worker is killed (preventing Parakeet #63
  stream leaks and CUDA static destructor teardown crashes), and a replacement
  standby worker is warmed up in the background and kept warm in memory/GPU.
- When idle longer than stt_idle_timeout_sec without any traffic, the standby worker
  process is terminated (SIGKILL) to release 100% of host RAM and GPU VRAM.
- When stt_idle_timeout_sec == 0, standby caching is disabled (one-shot mode).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path
from typing import Any

from live_stt import vram
from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.models import ModelSpec
from live_stt.worker import (
    StandbyKey,
    StandbyWorker,
    spawn_standby_sync,
)

logger = get_logger("worker_pool")


class STTWorkerPool:
    """Thread-safe manager for pre-warmed standby STT worker processes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[StandbyKey, StandbyWorker] = {}
        self._spawning_keys: set[StandbyKey] = set()
        self._spawning_events: dict[StandbyKey, threading.Event] = {}
        self._spawning_waiters: dict[StandbyKey, int] = {}
        self._reaper_timer: threading.Timer | None = None
        self._closed: bool = False
        self._epoch: int = 0

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def has_warm(self, model_key: str | StandbyKey, *, use_gpu: bool | None = None) -> bool:
        """Return True if a live standby worker (or in-flight warm spawn) for ``model_key`` is in the pool."""
        target_model = model_key.model_key if isinstance(model_key, StandbyKey) else model_key
        with self._lock:
            if self._closed:
                return False
            for key, worker in list(self._workers.items()):
                if key.model_key != target_model:
                    continue
                if use_gpu is not None and key.use_gpu != use_gpu:
                    continue
                if worker.proc.poll() is not None:
                    self._workers.pop(key, None)
                    worker.kill()
                    continue
                return True
            for key in self._spawning_keys:
                if key.model_key == target_model and (use_gpu is None or key.use_gpu == use_gpu):
                    return True
            return False

    def acquire(
        self,
        key: StandbyKey,
        *,
        wait_inflight: bool = True,
        wait_timeout_sec: float = 30.0,
    ) -> StandbyWorker | None:
        """Acquires and returns a warm standby worker matching `key`, or None if absent.

        When a background warm spawn for `key` is already in flight (`key in _spawning_keys`),
        waits for that spawn to complete and claims it directly instead of returning None
        and causing the caller to spawn a duplicate worker in parallel (LS-14).
        """
        inflight_event: threading.Event | None = None
        with self._lock:
            if self._closed:
                return None
            worker = self._workers.pop(key, None)
            if worker is not None:
                # Discard if it crashed or terminated while idling
                if worker.proc.poll() is not None:
                    worker.kill()
                    return None
                logger.info(
                    "reusing warm standby worker for %s (pid=%s, idle for %.2fs)",
                    key.model_key or key.gguf_path,
                    worker.proc.pid,
                    time.monotonic() - worker.idle_since,
                )
                return worker
            if wait_inflight and key in self._spawning_events:
                inflight_event = self._spawning_events[key]
                self._spawning_waiters[key] = self._spawning_waiters.get(key, 0) + 1

        if inflight_event is not None:
            logger.info(
                "waiting for in-flight standby worker spawn for %s",
                key.model_key or key.gguf_path,
            )
            try:
                inflight_event.wait(timeout=wait_timeout_sec)
            finally:
                with self._lock:
                    rem = self._spawning_waiters.get(key, 1) - 1
                    if rem <= 0:
                        self._spawning_waiters.pop(key, None)
                    else:
                        self._spawning_waiters[key] = rem

            with self._lock:
                if self._closed:
                    return None
                worker = self._workers.pop(key, None)
                if worker is not None:
                    if worker.proc.poll() is not None:
                        worker.kill()
                        return None
                    logger.info(
                        "reusing newly warmed standby worker for %s (pid=%s)",
                        key.model_key or key.gguf_path,
                        worker.proc.pid,
                    )
                    return worker
        return None

    def put(self, worker: StandbyWorker, timeout_sec: float) -> bool:
        """Registers a newly warmed standby worker into the pool and schedules the idle reaper."""
        with self._lock:
            if self._closed or timeout_sec <= 0:
                worker.kill()
                return False
            old = self._workers.get(worker.key)
            if old is not None:
                old.kill()
            self._workers[worker.key] = worker
            self._reschedule_reaper_locked(timeout_sec)
            return True

    def maintain_warm_async(
        self,
        *,
        key: StandbyKey,
        timeout_sec: float,
        settings: Settings | None = None,
    ) -> None:
        """Schedules background pre-warming of a standby worker if none exists for this key."""
        if timeout_sec <= 0:
            return

        with self._lock:
            if self._closed:
                return
            if key in self._workers:
                # Worker is already warm, refresh its idle timestamp and reaper
                self._workers[key].idle_since = time.monotonic()
                self._reschedule_reaper_locked(timeout_sec)
                return
            if key in self._spawning_keys:
                return
            self._spawning_keys.add(key)
            done_event = threading.Event()
            self._spawning_events[key] = done_event
            spawn_epoch = self._epoch

        def _finish_inflight() -> None:
            with self._lock:
                if self._epoch == spawn_epoch:
                    self._spawning_keys.discard(key)
                    self._spawning_events.pop(key, None)
            done_event.set()

        def _do_spawn() -> None:
            if key.use_gpu and settings is not None:
                with self._lock:
                    has_waiters = self._spawning_waiters.get(key, 0) > 0
                if not has_waiters and not vram.can_prewarm_standby(
                    settings, model_key=key.model_key
                ):
                    logger.info(
                        "skipping STT standby pre-warm for %s: active GPU compute + standby weights would exceed VRAM budget",
                        key.model_key or key.gguf_path,
                    )
                    _finish_inflight()
                    return

            worker: StandbyWorker | None = None
            try:
                worker = spawn_standby_sync(
                    worker_bin=Path(key.worker_bin),
                    gguf_path=key.gguf_path,
                    language=key.language,
                    n_threads=key.n_threads,
                    ggml_lib_dir=Path(key.ggml_lib_dir) if key.ggml_lib_dir else None,
                    ggml_backend_path=Path(key.ggml_backend_path) if key.ggml_backend_path else None,
                    use_gpu=key.use_gpu,
                    model_key=key.model_key,
                    extra_env=key.extra_env,
                )
            except Exception as exc:
                logger.debug(
                    "background STT warm worker spawn for %s failed: %s",
                    key.model_key or key.gguf_path,
                    exc,
                )
                _finish_inflight()
                return

            if key.use_gpu and settings is not None:
                with self._lock:
                    has_waiters = self._spawning_waiters.get(key, 0) > 0
                if not has_waiters and not vram.can_prewarm_standby(
                    settings,
                    model_key=key.model_key,
                    weights_already_allocated=True,
                ):
                    logger.info(
                        "discarding newly warmed STT standby worker for %s (pid=%s): active GPU job needs VRAM headroom",
                        key.model_key or key.gguf_path,
                        worker.proc.pid,
                    )
                    worker.kill()
                    _finish_inflight()
                    return

            with self._lock:
                if self._epoch == spawn_epoch:
                    self._spawning_keys.discard(key)
                    self._spawning_events.pop(key, None)
                if self._closed or self._epoch != spawn_epoch or key in self._workers:
                    if worker is not None:
                        worker.kill()
                    done_event.set()
                    return
                self._workers[key] = worker
                self._reschedule_reaper_locked(timeout_sec)
                logger.info(
                    "pre-warmed STT standby worker ready for %s (pid=%s)",
                    key.model_key or key.gguf_path,
                    worker.proc.pid,
                )
            done_event.set()

        thread = threading.Thread(target=_do_spawn, daemon=True)
        thread.start()

    def evict_idle(self, timeout_sec: float, *, exclude_model_key: str | None = None) -> int:
        """Evicts and kills any standby workers that have been idle >= timeout_sec.
        Returns the number of workers evicted.
        """
        now = time.monotonic()
        to_kill: list[StandbyWorker] = []
        with self._lock:
            for key, worker in list(self._workers.items()):
                if exclude_model_key is not None and key.model_key == exclude_model_key:
                    continue
                if (now - worker.idle_since) >= timeout_sec or timeout_sec <= 0:
                    to_kill.append(self._workers.pop(key))
            if self._workers and timeout_sec > 0:
                next_check = min(
                    max(0.1, timeout_sec - (now - w.idle_since)) for w in self._workers.values()
                )
                self._reschedule_reaper_locked(next_check)

        for w in to_kill:
            logger.info(
                "evicting idle STT standby worker for %s (pid=%s, idle=%.2fs >= %.2fs); releasing VRAM/RAM",
                w.key.model_key or w.key.gguf_path,
                w.proc.pid,
                now - w.idle_since,
                timeout_sec,
            )
            w.kill()
        return len(to_kill)

    def _reschedule_reaper_locked(self, delay: float) -> None:
        if self._reaper_timer is not None:
            self._reaper_timer.cancel()
            self._reaper_timer = None
        if not self._closed and self._workers and delay > 0:
            timer = threading.Timer(delay, self._reaper_tick, args=(delay,))
            timer.daemon = True
            timer.start()
            self._reaper_timer = timer

    def _reaper_tick(self, timeout_sec: float) -> None:
        self.evict_idle(timeout_sec)

    def clear(self) -> int:
        """Kills and clears all standby workers."""
        with self._lock:
            self._epoch += 1
            if self._reaper_timer is not None:
                self._reaper_timer.cancel()
                self._reaper_timer = None
            to_kill = list(self._workers.values())
            events = list(self._spawning_events.values())
            self._workers.clear()
            self._spawning_keys.clear()
            self._spawning_events.clear()
            self._spawning_waiters.clear()
        for ev in events:
            ev.set()
        for w in to_kill:
            w.kill()
        return len(to_kill)

    def close(self) -> None:
        """Closes the pool and terminates all active standby workers."""
        with self._lock:
            self._closed = True
        self.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        """Returns diagnostic snapshots of all active standby workers."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "model": w.key.model_key or os.path.basename(w.key.gguf_path),
                    "pid": w.proc.pid,
                    "idle_sec": round(now - w.idle_since, 2),
                    "use_gpu": w.key.use_gpu,
                }
                for w in self._workers.values()
            ]

    def count(self) -> int:
        with self._lock:
            return len(self._workers)


_STT_WORKER_POOL = STTWorkerPool()


def get_worker_pool() -> STTWorkerPool:
    return _STT_WORKER_POOL


def stop_worker_pool() -> None:
    _STT_WORKER_POOL.close()


def clear_worker_pool() -> int:
    return _STT_WORKER_POOL.clear()


def evict_idle_workers(timeout_sec: float, *, exclude_model_key: str | None = None) -> int:
    return _STT_WORKER_POOL.evict_idle(timeout_sec, exclude_model_key=exclude_model_key)


atexit.register(stop_worker_pool)
