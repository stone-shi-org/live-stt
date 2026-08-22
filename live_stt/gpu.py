"""VRAM-aware admission (Phase 5) and GPU visibility for the admin dashboard.
Deliberately shells out to `nvidia-smi` rather than adding a `pynvml`
dependency for a handful of queries -- `nvidia-smi` is already present in
any image built from an `nvidia/cuda:*` base and injected by the NVIDIA
container runtime, so this adds no new dependency at all. Every function
here returns None (never raises) when unavailable, e.g. on the CPU backend
or a host with no GPU -- callers must treat None as "cannot check", not
"zero"/"empty".

The GPU this service targets (10.100.0.50, verified via nvidia-smi over SSH:
one RTX 3090, 24GB) is a SHARED box -- it also runs LocalAI and other GPU
workloads. `free_vram_mb()` exists specifically so live-stt doesn't admit a
call it can't actually fit, which on a CUDA allocation failure is an
abort(), not a catchable exception (see CLAUDE.md) -- i.e. missing this
check doesn't just mean a slow call, it means a process crash. The same
check now also gates diarization admission (`live_stt/diarize_http.py`),
after a real measurement (see CLAUDE.md's "why 12GB" entry) found pyannote's
pipeline holding ~12.3GB on this exact card after a single diarization call
over a ~6-minute recording -- PyTorch's CUDA caching allocator sizing itself
to the batched sliding-window peak and keeping it reserved for reuse, not a
leak (confirmed: a second back-to-back call used ~the same VRAM and was
faster, not slower/growing).
"""

from __future__ import annotations

import subprocess

from live_stt.logging_config import get_logger

logger = get_logger("gpu")


def _query_gpu(field: str) -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
    ) as exc:
        logger.warning("could not query GPU %s: %s", field, exc)
        return None


def free_vram_mb() -> int | None:
    return _query_gpu("memory.free")


def total_vram_mb() -> int | None:
    return _query_gpu("memory.total")


def used_vram_mb() -> int | None:
    return _query_gpu("memory.used")


def utilization_pct() -> int | None:
    """GPU compute utilization, 0-100 -- the whole card, not just this
    process (nvidia-smi has no per-process utilization query, only
    per-process memory via --query-compute-apps). Useful context alongside
    VRAM: a card can be nearly full on memory but idle on compute, or vice
    versa, and the admin dashboard shows both rather than implying one from
    the other.
    """
    return _query_gpu("utilization.gpu")


def snapshot() -> dict[str, int | None]:
    """One dict for the admin dashboard/`/api/stats` -- a single call site
    rather than four separate `nvidia-smi` shell-outs per stats request.
    All fields None together when nvidia-smi is unavailable (CPU backend or
    no GPU) -- never a mix of some real, some None, since they all come from
    the same `_query_gpu` failure mode.
    """
    return {
        "free_vram_mb": free_vram_mb(),
        "total_vram_mb": total_vram_mb(),
        "used_vram_mb": used_vram_mb(),
        "utilization_pct": utilization_pct(),
    }
