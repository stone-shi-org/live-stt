"""VRAM admission shared by every GPU entry point (gRPC Transcribe, batch
transcription, diarization, TTS).

Why this exists: each path used to compare raw free VRAM against a flat,
hand-tuned constant (3000+2000 MB for ANY ASR model, 13000 MB for
diarization) sized for a 24GB RTX 3090. Swapping the card for a 4GB Quadro
P1000 made every request fail admission, even models needing ~0.3 GB, and
retuning needed a restart. Three changes:

1. **Per-model requirement.** Each model carries its own measured working
   VRAM (``ModelSpec.vram_mb`` / ``DiarizationModelSpec.vram_mb`` /
   ``TTSModelSpec.measured_peak_vram_mb``). The old flat settings are now only
   the FALLBACK for a model with no figure, and ``Settings.vram_overrides``
   ({model_key: MB}) is the explicit per-model escape hatch. Deliberately not
   "unset setting means auto": the settings DB is seeded with every field on
   first boot, so existing deployments already have the old flat values
   stored, and an "auto" sentinel would never take effect for them.
2. **Card-relative reserve.** reserve = max(vram_reserve_mb,
   vram_reserve_pct% of total VRAM): about 2.4 GB on a 24 GB card and 0.4 GB
   on a 4 GB one, instead of a fixed 2000 MB (half of a P1000).
3. **Reclaim before rejecting.** Raw free VRAM counts IDLE warm workers
   (standby STT workers, the warm Kokoro worker) as used, though they can be
   killed in milliseconds. When a request is short, those are evicted first
   and free VRAM re-read. Busy workers and active calls are never touched.

All of it reads ``settings`` per request, so the VRAM fields are
hot-reloadable (live_stt/settings_store.py's HOT_RELOAD_FIELDS).

Fails OPEN when nvidia-smi is unavailable (``gpu.free_vram_mb()`` is None),
the same contract every gate here already had: "cannot check" is never
treated as "zero free".

Not handled (same as before this module): two requests checked at the same
moment can both pass before either allocates.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from live_stt import gpu, models

logger = logging.getLogger("stt.vram")

# How long to wait for VRAM to actually show up as free after killing idle
# workers -- the driver releases a process's allocations only once it has
# exited, which is not instantaneous.
RECLAIM_WAIT_SEC = 3.0
RECLAIM_POLL_SEC = 0.2

_total_mb_cache: int | None = None

_active_compute_lock = threading.Lock()
_active_compute_reservations: dict[str, int] = {}


def _total_mb() -> int | None:
    """Card total VRAM. Cached once known -- it cannot change under a running
    process -- to avoid a second nvidia-smi shell-out on every request."""
    global _total_mb_cache
    if _total_mb_cache is None:
        _total_mb_cache = gpu.total_vram_mb()
    return _total_mb_cache


def reserve_mb(settings: Any, total_mb: int | None = None) -> int:
    floor = int(settings.vram_reserve_mb)
    total = total_mb if total_mb is not None else _total_mb()
    if not total:
        return floor
    return max(floor, int(total * float(settings.vram_reserve_pct) / 100.0))


def effective_batch_window_sec(settings: Any, model_key: str) -> float:
    """Effective batch inference window size (seconds) for ``model_key``."""
    override = getattr(settings, "batch_window_sec", None)
    if override is not None:
        return float(override)
    spec = models.MODELS.get(model_key)
    return float(spec.batch_window_sec) if spec is not None else 0.0


def compute_mb(settings: Any, model_key: str, *, audio_sec: float | None = None) -> int:
    """Per-window CUDA compute buffer allowance for ``model_key`` (MiB).

    For batch engines whose compute graph scales with window duration
    (``parakeet-tdt``: ~6.73 MiB/s -> 7.0 MiB/s), peak compute memory is
    bounded by ``batch_window_sec`` (or ``audio_sec`` when shorter than one
    window). When windowing is disabled (``window_sec == 0``), compute memory
    scales with the full ``audio_sec``.
    """
    spec = models.MODELS.get(model_key)
    if spec is None or spec.vram_compute_mb_per_sec <= 0:
        return 0
    window_sec = effective_batch_window_sec(settings, model_key)
    if window_sec > 0:
        span_sec = min(audio_sec, window_sec) if audio_sec is not None else window_sec
    else:
        span_sec = audio_sec if audio_sec is not None else 0.0
    return math.ceil(max(0.0, span_sec) * spec.vram_compute_mb_per_sec)


def weights_mb(settings: Any, model_key: str, model_mb: int | None, fallback_mb: int) -> int:
    """Base model weights / fixed working VRAM for ``model_key``, before window compute."""
    override = (settings.vram_overrides or {}).get(model_key)
    if override is not None:
        return int(override)
    return int(model_mb) if model_mb else int(fallback_mb)


def need_mb(
    settings: Any,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    *,
    audio_sec: float | None = None,
    include_weights: bool = True,
) -> int:
    """VRAM one instance of ``model_key`` needs (weights + per-window compute), before the reserve."""
    override = (settings.vram_overrides or {}).get(model_key)
    if override is not None:
        return int(override) if include_weights else compute_mb(settings, model_key, audio_sec=audio_sec)
    base = (int(model_mb) if model_mb else int(fallback_mb)) if include_weights else 0
    return base + compute_mb(settings, model_key, audio_sec=audio_sec)


def required_mb(
    settings: Any,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    *,
    audio_sec: float | None = None,
    include_weights: bool = True,
) -> int:
    return (
        need_mb(
            settings,
            model_key,
            model_mb,
            fallback_mb,
            audio_sec=audio_sec,
            include_weights=include_weights,
        )
        + reserve_mb(settings)
    )


def reserve_active_compute(token_id: str, mb: int) -> None:
    """Register an active GPU session's pending compute buffer requirement (MiB)."""
    if mb <= 0:
        return
    with _active_compute_lock:
        _active_compute_reservations[token_id] = int(mb)


def release_active_compute(token_id: str) -> None:
    """Release an active GPU session's compute buffer reservation."""
    with _active_compute_lock:
        _active_compute_reservations.pop(token_id, None)


def active_compute_mb(exclude_token_id: str | None = None) -> int:
    """Total compute buffer VRAM (MiB) reserved by currently active GPU jobs."""
    with _active_compute_lock:
        if exclude_token_id is None:
            return sum(_active_compute_reservations.values())
        return sum(v for k, v in _active_compute_reservations.items() if k != exclude_token_id)


def can_prewarm_standby(
    settings: Any,
    *,
    model_key: str,
    weights_already_allocated: bool = False,
) -> bool:
    """Return True if pre-warming (or keeping) a standby GPU worker for
    ``model_key`` fits alongside active jobs' compute buffers and the card reserve.
    """
    free = gpu.free_vram_mb()
    if free is None:
        return True
    spec = models.MODELS.get(model_key)
    model_mb = spec.vram_mb if spec is not None else None
    standby_weights = (
        0
        if weights_already_allocated
        else weights_mb(settings, model_key, model_mb, settings.vram_per_worker_mb)
    )
    needed = standby_weights + active_compute_mb() + reserve_mb(settings)
    return free >= needed


@dataclass(frozen=True)
class Admission:
    ok: bool
    free_mb: int | None
    required_mb: int
    reclaimed: int = 0

    def message(self, what: str = "") -> str:
        suffix = f" for {what}" if what else ""
        return f"insufficient VRAM{suffix}: {self.free_mb}MB free, {self.required_mb}MB required"


def reclaim_idle_workers(*, exclude_model_key: str | None = None) -> int:
    """Kill idle warm workers to give their VRAM back. Returns how many were
    stopped. Never touches a worker serving a request: standby STT workers are
    by definition unassigned, and the TTS worker is only stopped if idle (a
    non-blocking check, so this never waits on an in-flight synthesis)."""
    # Imported lazily: both modules pull in heavier machinery, and vram is
    # imported by the request paths that those modules' callers also import.
    from live_stt import tts, worker_pool

    stopped = worker_pool.evict_idle_workers(0.0, exclude_model_key=exclude_model_key)
    if tts.get_tts_worker_manager().stop_if_idle():
        stopped += 1
    return stopped


def ensure_compute_headroom(
    settings: Any,
    *,
    model_key: str,
    compute_needed_mb: int,
) -> int:
    """Ensure active GPU job's compute buffer takes priority over warm spares.
    If free VRAM is below ``compute_needed_mb + reserve_mb(settings)``, evicts
    idle warm workers and waits briefly for the driver to release their VRAM.
    """
    if compute_needed_mb <= 0:
        return 0
    free = gpu.free_vram_mb()
    if free is None:
        return 0
    needed = compute_needed_mb + reserve_mb(settings)
    if free >= needed:
        return 0
    stopped = reclaim_idle_workers()
    if stopped <= 0:
        return 0
    logger.info(
        "active job %s needs %dMB compute headroom (%dMB free): evicted %d idle warm worker(s)",
        model_key,
        needed,
        free,
        stopped,
    )
    deadline = time.monotonic() + RECLAIM_WAIT_SEC
    while time.monotonic() < deadline:
        free = gpu.free_vram_mb()
        if free is None or free >= needed:
            break
        time.sleep(RECLAIM_POLL_SEC)
    return stopped


def admit(
    settings: Any,
    *,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    reclaim: bool = True,
    audio_sec: float | None = None,
    already_warm: bool = False,
) -> Admission:
    """Blocking (shells out to nvidia-smi and may poll for up to
    RECLAIM_WAIT_SEC). Call from an event loop via asyncio.to_thread."""
    free = gpu.free_vram_mb()
    required = (
        required_mb(
            settings,
            model_key,
            model_mb,
            fallback_mb,
            audio_sec=audio_sec,
            include_weights=not already_warm,
        )
        + active_compute_mb()
    )
    if free is None:
        return Admission(True, None, required)
    if free >= required:
        return Admission(True, free, required)
    if not reclaim:
        return Admission(False, free, required)

    stopped = (
        reclaim_idle_workers(exclude_model_key=model_key)
        if already_warm
        else reclaim_idle_workers()
    )
    if stopped == 0:
        return Admission(False, free, required)
    logger.info(
        "VRAM short for %s (%dMB free < %dMB): stopped %d idle warm worker(s), re-checking",
        model_key,
        free,
        required,
        stopped,
    )
    deadline = time.monotonic() + RECLAIM_WAIT_SEC
    while True:
        free = gpu.free_vram_mb()
        if free is None or free >= required:
            return Admission(True, free, required, stopped)
        if time.monotonic() >= deadline:
            return Admission(False, free, required, stopped)
        time.sleep(RECLAIM_POLL_SEC)

