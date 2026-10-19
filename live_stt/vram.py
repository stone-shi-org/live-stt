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

4. **In-process VRAM reservation (LS-17).** Admission checks and records a
   reservation (weights for a cold spawn + compute buffer for the active
   window) atomically under a lock before returning, and callers release it in
   ``try/finally`` when the call ends. Concurrent requests arriving in a burst
   see earlier requests' reservations immediately, before their workers have
   allocated on the GPU.
5. **Effective GPU capacity reporting (LS-18).** ``gpu_capacity()`` and
   ``effective_max_calls()`` expose how many concurrent calls of each
   GPU-backed ASR model fit in ``budget_mb = total_vram_mb - reserve_mb`` using
   the exact same cold-worker figures ``admit()`` uses. These are static
   idle-card estimates (live free VRAM on a shared card can be lower, and a
   warm standby worker reduces a single call's weight allocation); ``admit()``
   remains authoritative at request time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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

_lock = threading.Lock()
_active_compute_lock = _lock


@dataclass
class _Reservation:
    reservation_id: str
    model_key: str
    weights_mb: int
    compute_mb: int
    warm_claim: bool = False

    @property
    def total_mb(self) -> int:
        return self.weights_mb + self.compute_mb


_reservations: dict[str, _Reservation] = {}
_active_compute_reservations: dict[str, int] = {}
_baseline_free_mb: int | None = None
_reservation_seq: int = 0


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


def model_backend(settings: Any, spec: models.ModelSpec) -> str:
    """Return the configured backend ('cpu' or 'cuda') for ASR model ``spec``."""
    if spec.engine == "whisper":
        return str(getattr(settings, "whisper_backend", "cpu"))
    if spec.engine == "parakeet_tdt":
        return str(getattr(settings, "parakeet_tdt_backend", "cpu"))
    return str(getattr(settings, "parakeet_backend", "cpu"))


def effective_batch_window_sec(
    settings: Any, model_key: str, *, spec: models.ModelSpec | None = None
) -> float:
    """Effective batch inference window size (seconds) for ``model_key``."""
    override = getattr(settings, "batch_window_sec", None)
    if override is not None:
        return float(override)
    if spec is None:
        spec = models.MODELS.get(model_key)
    return float(spec.batch_window_sec) if spec is not None else 0.0


def compute_mb(
    settings: Any,
    model_key: str,
    *,
    audio_sec: float | None = None,
    spec: models.ModelSpec | None = None,
) -> int:
    """Per-window CUDA compute buffer allowance for ``model_key`` (MiB).

    For batch engines whose compute graph scales with window duration
    (``parakeet-tdt``: ~6.73 MiB/s -> 7.0 MiB/s), peak compute memory is
    bounded by ``batch_window_sec`` (or ``audio_sec`` when shorter than one
    window). When windowing is disabled (``window_sec == 0``), compute memory
    scales with the full ``audio_sec``.
    """
    if spec is None:
        spec = models.MODELS.get(model_key)
    if spec is None or spec.vram_compute_mb_per_sec <= 0:
        return 0
    window_sec = effective_batch_window_sec(settings, model_key, spec=spec)
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
    spec: models.ModelSpec | None = None,
) -> int:
    """VRAM one instance of ``model_key`` needs (weights + per-window compute), before the reserve."""
    override = (settings.vram_overrides or {}).get(model_key)
    if override is not None:
        return (
            int(override)
            if include_weights
            else compute_mb(settings, model_key, audio_sec=audio_sec, spec=spec)
        )
    base = (int(model_mb) if model_mb else int(fallback_mb)) if include_weights else 0
    return base + compute_mb(settings, model_key, audio_sec=audio_sec, spec=spec)


def required_mb(
    settings: Any,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    *,
    audio_sec: float | None = None,
    include_weights: bool = True,
    spec: models.ModelSpec | None = None,
) -> int:
    return (
        need_mb(
            settings,
            model_key,
            model_mb,
            fallback_mb,
            audio_sec=audio_sec,
            include_weights=include_weights,
            spec=spec,
        )
        + reserve_mb(settings)
    )


def _can_claim_warm_locked(model_key: str, already_warm: bool) -> bool:
    """At most one in-flight reservation per model may claim the warm-standby
    weights discount: STTWorkerPool holds at most one warm standby worker per
    key, so concurrent requests beyond the first must spawn cold workers."""
    if not already_warm:
        return False
    return not any(r.model_key == model_key and r.warm_claim for r in _reservations.values())


def _compute_requirement_locked(
    settings: Any,
    *,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    audio_sec: float | None,
    already_warm: bool,
    spec: models.ModelSpec | None = None,
) -> tuple[int, int, int]:
    """Return ``(weights_needed_mb, compute_needed_mb, required_total_mb)``."""
    c_mb = compute_mb(settings, model_key, audio_sec=audio_sec, spec=spec)
    if already_warm:
        w_mb = 0
    else:
        total_need = need_mb(
            settings,
            model_key,
            model_mb,
            fallback_mb,
            audio_sec=audio_sec,
            include_weights=True,
            spec=spec,
        )
        w_mb = max(0, total_need - c_mb)
    required = w_mb + c_mb + reserve_mb(settings)
    return w_mb, c_mb, required


def per_call_mb(
    settings: Any,
    model_key: str,
    model_mb: int | None = None,
    fallback_mb: int | None = None,
    *,
    audio_sec: float | None = None,
    spec: models.ModelSpec | None = None,
) -> int:
    """VRAM (MiB) reserved by ``admit()`` for one cold call of ``model_key``
    (weights + active-window compute allowance, excluding the card reserve).
    """
    if spec is None:
        spec = models.MODELS.get(model_key)
    if model_mb is None and spec is not None:
        model_mb = spec.vram_mb
    if fallback_mb is None:
        fallback_mb = int(getattr(settings, "vram_per_worker_mb", 3000))
    w_mb, c_mb, _ = _compute_requirement_locked(
        settings,
        model_key=model_key,
        model_mb=model_mb,
        fallback_mb=fallback_mb,
        audio_sec=audio_sec,
        already_warm=False,
        spec=spec,
    )
    return w_mb + c_mb


def _effective_free_locked(live_free: int, *, pre_reclaim_free: int | None = None) -> int:
    """Compute available VRAM (MiB) accounting for active reservations.

    Let ``F_0`` be the GPU's free VRAM before active reservations allocated,
    and ``R`` be the sum of all active reservations. When some fraction ``A``
    of ``R`` has already been allocated in ``nvidia-smi`` (so ``live_free =
    F_0 - A``) and ``R - A`` is still pending allocation, the remaining free
    VRAM is ``live_free - (R - A) = F_0 - R`` -- independent of ``A``. Taking
    ``min(F_0 - R, live_free)`` handles both unallocated reservations (bursts,
    fake workers) and external GPU allocations without double-counting workers
    that have already loaded.
    """
    global _baseline_free_mb
    admit_reserved = sum(r.total_mb for r in _reservations.values())
    compute_only_reserved = sum(_active_compute_reservations.values())
    if not _reservations or _baseline_free_mb is None:
        _baseline_free_mb = live_free
    else:
        if pre_reclaim_free is not None and live_free > pre_reclaim_free:
            _baseline_free_mb += live_free - pre_reclaim_free
        if live_free > _baseline_free_mb:
            _baseline_free_mb = live_free
    total_reserved = admit_reserved + compute_only_reserved
    return max(0, min(_baseline_free_mb - total_reserved, live_free - compute_only_reserved))


def _record_reservation_locked(
    *,
    model_key: str,
    weights_mb: int,
    compute_mb: int,
    warm_claim: bool,
) -> str:
    global _reservation_seq
    _reservation_seq += 1
    res_id = f"vram-res-{_reservation_seq}"
    _reservations[res_id] = _Reservation(
        reservation_id=res_id,
        model_key=model_key,
        weights_mb=weights_mb,
        compute_mb=compute_mb,
        warm_claim=warm_claim,
    )
    return res_id


def release(reservation_id: str | None) -> None:
    """Release an admission VRAM reservation (idempotent)."""
    if reservation_id is None:
        return
    global _baseline_free_mb
    with _lock:
        _reservations.pop(reservation_id, None)
        if not _reservations:
            _baseline_free_mb = None


def reserved_mb() -> int:
    """Total VRAM (MiB) reserved across all in-flight admissions and active compute buffers."""
    with _lock:
        return sum(r.total_mb for r in _reservations.values()) + sum(
            _active_compute_reservations.values()
        )


def active_reservations_count() -> int:
    """Number of currently active admission reservations."""
    with _lock:
        return len(_reservations)


def has_admit_reservation(model_key: str | None = None) -> bool:
    """Return True if an active admission reservation exists (for ``model_key`` if given)."""
    with _lock:
        if model_key is None:
            return bool(_reservations)
        return any(r.model_key == model_key for r in _reservations.values())


def clear_reservations() -> None:
    """Clear all active VRAM reservations (used by test fixtures)."""
    global _baseline_free_mb, _total_mb_cache
    with _lock:
        _reservations.clear()
        _active_compute_reservations.clear()
        _baseline_free_mb = None
        _total_mb_cache = None


def gpu_capacity(
    settings: Any,
    *,
    model_specs: Mapping[str, models.ModelSpec] | Iterable[models.ModelSpec] | None = None,
    total_mb: int | None = None,
) -> dict[str, Any] | None:
    """Estimate concurrent call capacity per GPU-backed ASR model on an idle card (LS-18).

    - ``budget_mb`` = ``total_vram_mb - reserve_mb``.
    - ``reserved_mb`` = currently active admission + compute reservations.
    - ``per_call_mb`` = what ``admit()`` reserves for one cold call (weights + window compute).
    - ``max_calls`` = ``floor(budget_mb / per_call_mb)``.
    - ``effective_max_calls`` = ``min(max_calls, max_concurrent_calls)``.

    Only GPU-backed models are included. Returns ``None`` when no ASR model is
    on a CUDA backend or when total GPU VRAM is unavailable.
    Estimates assume an idle card and cold workers; ``vram.admit()`` remains
    authoritative at admission time.
    """
    specs_iter: Iterable[models.ModelSpec]
    if model_specs is None:
        specs_list = [
            spec for spec in models.MODELS.values() if model_backend(settings, spec) == "cuda"
        ]
    else:
        specs_iter = (
            model_specs.values() if isinstance(model_specs, Mapping) else model_specs
        )
        any_cuda_configured = "cuda" in (
            getattr(settings, "parakeet_backend", "cpu"),
            getattr(settings, "whisper_backend", "cpu"),
            getattr(settings, "parakeet_tdt_backend", "cpu"),
        )
        if any_cuda_configured:
            specs_list = [spec for spec in specs_iter if model_backend(settings, spec) == "cuda"]
        else:
            specs_list = list(specs_iter)

    if not specs_list:
        return None

    total = total_mb if total_mb is not None else _total_mb()
    if not total or total <= 0:
        return None

    budget = max(0, int(total) - reserve_mb(settings, total_mb=int(total)))
    max_concurrent = int(getattr(settings, "max_concurrent_calls", 1))
    fallback = int(getattr(settings, "vram_per_worker_mb", 3000))

    models_doc: dict[str, dict[str, int]] = {}
    for spec in specs_list:
        call_mb = per_call_mb(
            settings,
            spec.key,
            spec.vram_mb,
            fallback,
            spec=spec,
        )
        max_calls = (budget // call_mb) if call_mb > 0 else max_concurrent
        eff_max = min(max_calls, max_concurrent)
        models_doc[spec.key] = {
            "per_call_mb": call_mb,
            "max_calls": max_calls,
            "effective_max_calls": eff_max,
        }

    return {
        "budget_mb": budget,
        "reserved_mb": reserved_mb(),
        "models": models_doc,
    }


def effective_max_calls(
    settings: Any,
    model_key: str | None = None,
    *,
    total_mb: int | None = None,
) -> int | None:
    """Return ``effective_max_calls`` for ``model_key`` (defaults to
    ``settings.default_model``) when backed by CUDA and GPU VRAM is known,
    else ``None``.
    """
    key = model_key or getattr(settings, "default_model", None) or models.DEFAULT_MODEL_KEY
    cap = gpu_capacity(settings, total_mb=total_mb)
    if cap is None:
        return None
    entry = cap["models"].get(key)
    if entry is None:
        return None
    return entry["effective_max_calls"]


def reserve_active_compute(token_id: str, mb: int) -> None:
    """Register an active GPU session's pending compute buffer requirement (MiB)."""
    if mb <= 0:
        return
    with _lock:
        _active_compute_reservations[token_id] = int(mb)


def release_active_compute(token_id: str) -> None:
    """Release an active GPU session's compute buffer reservation."""
    with _lock:
        _active_compute_reservations.pop(token_id, None)


def active_compute_mb(exclude_token_id: str | None = None) -> int:
    """Total compute buffer VRAM (MiB) reserved by currently active GPU jobs."""
    with _lock:
        standalone = (
            sum(_active_compute_reservations.values())
            if exclude_token_id is None
            else sum(v for k, v in _active_compute_reservations.items() if k != exclude_token_id)
        )
        admitted = (
            sum(r.compute_mb for r in _reservations.values())
            if exclude_token_id is None
            else sum(r.compute_mb for k, r in _reservations.items() if k != exclude_token_id)
        )
        return standalone + admitted


def can_prewarm_standby(
    settings: Any,
    *,
    model_key: str,
    weights_already_allocated: bool = False,
) -> bool:
    """Return True if pre-warming (or keeping) a standby GPU worker for
    ``model_key`` fits alongside active jobs' reservations and the card reserve.
    """
    with _lock:
        live_free = gpu.free_vram_mb()
        if live_free is None:
            return True
        free = _effective_free_locked(live_free)
        spec = models.MODELS.get(model_key)
        model_mb = spec.vram_mb if spec is not None else None
        standby_weights = (
            0
            if weights_already_allocated
            else weights_mb(settings, model_key, model_mb, settings.vram_per_worker_mb)
        )
        needed = standby_weights + reserve_mb(settings)
        return free >= needed


@dataclass(frozen=True)
class Admission:
    ok: bool
    free_mb: int | None
    required_mb: int
    reclaimed: int = 0
    reservation_id: str | None = None
    reserved_mb: int = 0

    def message(self, what: str = "") -> str:
        suffix = f" for {what}" if what else ""
        return f"insufficient VRAM{suffix}: {self.free_mb}MB free, {self.required_mb}MB required"

    def release(self) -> None:
        if self.reservation_id is not None:
            release(self.reservation_id)


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
    RECLAIM_WAIT_SEC). Call from an event loop via asyncio.to_thread.

    When admitted (``ok=True`` and ``free_mb is not None``), records an
    in-process reservation for this call's weights + compute buffer under
    ``_lock`` so concurrent callers cannot race past the check before workers
    allocate on the GPU. Callers must invoke ``admission.release()`` in a
    ``try/finally`` block when the call finishes.
    """
    with _lock:
        live_free = gpu.free_vram_mb()
        effective_warm = _can_claim_warm_locked(model_key, already_warm)
        w_mb, c_mb, required = _compute_requirement_locked(
            settings,
            model_key=model_key,
            model_mb=model_mb,
            fallback_mb=fallback_mb,
            audio_sec=audio_sec,
            already_warm=effective_warm,
        )
        if live_free is None:
            return Admission(True, None, required)
        free = _effective_free_locked(live_free)
        if free >= required:
            res_id = _record_reservation_locked(
                model_key=model_key,
                weights_mb=w_mb,
                compute_mb=c_mb,
                warm_claim=effective_warm,
            )
            return Admission(
                True,
                free,
                required,
                reservation_id=res_id,
                reserved_mb=w_mb + c_mb,
            )
        if not reclaim:
            return Admission(False, free, required)
        pre_reclaim_free = live_free

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
        with _lock:
            live_free = gpu.free_vram_mb()
            effective_warm = _can_claim_warm_locked(model_key, already_warm)
            w_mb, c_mb, required = _compute_requirement_locked(
                settings,
                model_key=model_key,
                model_mb=model_mb,
                fallback_mb=fallback_mb,
                audio_sec=audio_sec,
                already_warm=effective_warm,
            )
            if live_free is None:
                return Admission(True, None, required, stopped)
            free = _effective_free_locked(live_free, pre_reclaim_free=pre_reclaim_free)
            pre_reclaim_free = live_free
            if free >= required:
                res_id = _record_reservation_locked(
                    model_key=model_key,
                    weights_mb=w_mb,
                    compute_mb=c_mb,
                    warm_claim=effective_warm,
                )
                return Admission(
                    True,
                    free,
                    required,
                    stopped,
                    reservation_id=res_id,
                    reserved_mb=w_mb + c_mb,
                )
            if time.monotonic() >= deadline:
                return Admission(False, free, required, stopped)
        time.sleep(RECLAIM_POLL_SEC)


