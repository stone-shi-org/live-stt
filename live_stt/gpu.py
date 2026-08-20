"""VRAM-aware admission (Phase 5). Deliberately shells out to `nvidia-smi`
rather than adding a `pynvml` dependency for one query -- `nvidia-smi` is
already present in any image built from an `nvidia/cuda:*` base and injected
by the NVIDIA container runtime, so this adds no new dependency at all.
Returns None (never raises) when unavailable, e.g. on the CPU backend or a
host with no GPU -- callers must treat None as "cannot check", not "zero
VRAM free".

The GPU this service targets (10.100.0.50, verified via nvidia-smi over SSH:
one RTX 3090, 24GB) is a SHARED box -- it also runs LocalAI and other GPU
workloads. This check exists specifically so live-stt doesn't admit a call
it can't actually fit, which on a CUDA allocation failure is an abort(), not
a catchable exception (see CLAUDE.md) -- i.e. missing this check doesn't
just mean a slow call, it means a process crash.
"""

from __future__ import annotations

import subprocess

from live_stt.logging_config import get_logger

logger = get_logger("gpu")


def free_vram_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
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
        logger.warning("could not query free VRAM: %s", exc)
        return None
