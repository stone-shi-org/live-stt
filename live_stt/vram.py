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
import time
from dataclasses import dataclass
from typing import Any

from live_stt import gpu

logger = logging.getLogger("stt.vram")

# How long to wait for VRAM to actually show up as free after killing idle
# workers -- the driver releases a process's allocations only once it has
# exited, which is not instantaneous.
RECLAIM_WAIT_SEC = 3.0
RECLAIM_POLL_SEC = 0.2

_total_mb_cache: int | None = None


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


def need_mb(settings: Any, model_key: str, model_mb: int | None, fallback_mb: int) -> int:
    """VRAM one instance of `model_key` needs, before the reserve."""
    override = (settings.vram_overrides or {}).get(model_key)
    if override is not None:
        return int(override)
    return int(model_mb) if model_mb else int(fallback_mb)


def required_mb(settings: Any, model_key: str, model_mb: int | None, fallback_mb: int) -> int:
    return need_mb(settings, model_key, model_mb, fallback_mb) + reserve_mb(settings)


@dataclass(frozen=True)
class Admission:
    ok: bool
    free_mb: int | None
    required_mb: int
    reclaimed: int = 0

    def message(self, what: str = "") -> str:
        suffix = f" for {what}" if what else ""
        return f"insufficient VRAM{suffix}: {self.free_mb}MB free, {self.required_mb}MB required"


def reclaim_idle_workers() -> int:
    """Kill idle warm workers to give their VRAM back. Returns how many were
    stopped. Never touches a worker serving a request: standby STT workers are
    by definition unassigned, and the TTS worker is only stopped if idle (a
    non-blocking check, so this never waits on an in-flight synthesis)."""
    # Imported lazily: both modules pull in heavier machinery, and vram is
    # imported by the request paths that those modules' callers also import.
    from live_stt import tts, worker_pool

    stopped = worker_pool.evict_idle_workers(0.0)
    if tts.get_tts_worker_manager().stop_if_idle():
        stopped += 1
    return stopped


def admit(
    settings: Any,
    *,
    model_key: str,
    model_mb: int | None,
    fallback_mb: int,
    reclaim: bool = True,
) -> Admission:
    """Blocking (shells out to nvidia-smi and may poll for up to
    RECLAIM_WAIT_SEC). Call from an event loop via asyncio.to_thread."""
    free = gpu.free_vram_mb()
    required = required_mb(settings, model_key, model_mb, fallback_mb)
    if free is None:
        return Admission(True, None, required)
    if free >= required:
        return Admission(True, free, required)
    if not reclaim:
        return Admission(False, free, required)

    stopped = reclaim_idle_workers()
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
