#!/usr/bin/env python3
"""Phase 1 Gate B: measure real STREAMING RTFx (audio_sec / wall_sec, feeding
chunks back-to-back with no real-time pacing) at several n_threads settings,
and recommend a (n_threads, W) pair for this host.

Deliberately not derived from parakeet.cpp's own published offline/batch RTFx
sweep (its "t=8 -> 75.4 RTFx" table): that number is file/batch mode, and
mixing it with a streaming number is comparing different workloads (see
CLAUDE.md). This script measures the one that matters for this service.

Usage:
    python tools/thread_sweep.py --model models/realtime_eou_120m-v1-q8_0.gguf \
        --audio-fixture /path/to/long.wav --audio-sec 60
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.leak_curve import DEFAULT_GGML_LIB_DIR, DEFAULT_WORKER_BIN, run  # noqa: E402


def recommend(rtfx_by_threads: dict[int, float], cores: int, min_rtfx: float = 1.5) -> tuple[int, int]:
    """floor(cores * 0.8 / n_threads) workers, subject to rtfx(n_threads) >=
    min_rtfx (1.0 is a coin flip, not a capacity plan -- see CLAUDE.md),
    maximising worker count W."""
    best: tuple[int, int] | None = None
    for n_threads, rtfx in rtfx_by_threads.items():
        if rtfx < min_rtfx:
            continue
        w = max(1, int(cores * 0.8 / n_threads))
        if best is None or w > best[1]:
            best = (n_threads, w)
    return best or (max(rtfx_by_threads, key=rtfx_by_threads.get), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-fixture", required=True)
    parser.add_argument("--audio-sec", type=float, default=60.0)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument(
        "--threads", type=int, nargs="+", default=[1, 2, 3, 4, 6],
        help="n_threads values to sweep (default fits a 6-core host)",
    )
    parser.add_argument("--cores", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    print(f"# host cores: {args.cores}")
    print("n_threads,audio_sec,wall_sec,rtfx")
    rtfx_by_threads: dict[int, float] = {}
    for n_threads in args.threads:
        summary = run(
            model_path=args.model,
            condition="speech",
            audio_sec_target=args.audio_sec,
            chunk_ms=args.chunk_ms,
            sample_every_sec=args.audio_sec,  # only need the final sample
            n_threads=n_threads,
            audio_fixture=args.audio_fixture,
            worker_bin=DEFAULT_WORKER_BIN,
            ggml_lib_dir=DEFAULT_GGML_LIB_DIR,
        )
        rtfx = summary["audio_sec_fed"] / summary["wall_sec"] if summary["wall_sec"] else 0.0
        rtfx_by_threads[n_threads] = rtfx
        print(f"{n_threads},{summary['audio_sec_fed']:.1f},{summary['wall_sec']:.2f},{rtfx:.2f}")
        sys.stdout.flush()

    best_threads, best_w = recommend(rtfx_by_threads, args.cores)
    print(f"\n# recommended: n_threads={best_threads}, W={best_w} concurrent workers "
          f"(rtfx={rtfx_by_threads[best_threads]:.2f} on {args.cores} cores)")


if __name__ == "__main__":
    main()
