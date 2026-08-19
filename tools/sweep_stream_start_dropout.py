#!/usr/bin/env python3
"""Characterize the frequency of the stream-start dropout bug found in Phase
3 (see CLAUDE.md's "Serious open risk" section and
tools/repro_stream_start_dropout.py for the minimal single-offset repro).

Method: establish a CONTINUOUS baseline transcript (one worker, one stream,
fed the whole span with no restarts) with absolute word timestamps. Then, at
many closely-spaced candidate start offsets, spawn a FRESH worker that
starts its stream exactly there (mimicking what a rotation or crash-recovery
would do) and feed a window of audio. For each candidate, compute RECALL
against the baseline's words in the corresponding absolute-time window
(skipping a startup grace period): what fraction of the words the baseline
knows are present, allowing a small ONE-word gap that's expected at any
generation boundary might be reported anyway. A recall well below 1.0 over a
multi-second span is the drop signature found in the single-offset repro.

Runs candidates concurrently (bounded) since each is an independent worker
process with no shared state.

Usage:
    python tools/sweep_stream_start_dropout.py \
        --model models/realtime_eou_120m-v1-q8_0.gguf \
        --wav /path/to/audio.wav \
        --range-start 5.0 --range-end 25.0 --step 0.16 \
        --window-sec 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live_stt.worker import WorkerHandle  # noqa: E402

DEFAULT_WORKER_BIN = REPO_ROOT / "worker" / "build" / "live_stt_worker"
DEFAULT_GGML_LIB_DIR = REPO_ROOT / "worker" / "build-parakeet" / "third_party" / "ggml" / "src"


async def transcribe_window(
    model_path: str, wav_path: str, start_sec: float, duration_sec: float, n_threads: int = 4
) -> list[tuple[float, float, str]]:
    """Returns [(abs_start_sec, abs_end_sec, word), ...] for a fresh stream
    beginning exactly at start_sec."""
    handle = await WorkerHandle.spawn(
        worker_bin=DEFAULT_WORKER_BIN,
        gguf_path=model_path,
        language="",
        n_threads=n_threads,
        ggml_lib_dir=DEFAULT_GGML_LIB_DIR,
    )
    words: list[tuple[float, float, str]] = []
    try:
        wf = wave.open(wav_path, "rb")
        sr = wf.getframerate()
        wf.setpos(int(start_sec * sr))
        chunk_samples = int(sr * 0.16)
        for _ in range(int(duration_sec / 0.16)):
            pcm = wf.readframes(chunk_samples)
            if not pcm:
                break
            doc = await handle.feed(pcm)
            for w in doc.get("words", []):
                words.append((start_sec + w["start"], start_sec + w["end"], w["w"]))
        doc = await handle.finalize()
        for w in doc.get("words", []):
            words.append((start_sec + w["start"], start_sec + w["end"], w["w"]))
    finally:
        handle.kill()
        await handle.wait_closed()
    return words


def recall(
    baseline: list[tuple[float, float, str]],
    candidate: list[tuple[float, float, str]],
    window_start: float,
    window_end: float,
    time_tolerance_sec: float = 0.75,
) -> tuple[float, int, int]:
    ref = [w for w in baseline if window_start <= w[0] < window_end]
    if not ref:
        return 1.0, 0, 0
    matched = 0
    for r_start, r_end, r_text in ref:
        hit = any(
            c_text == r_text and abs(c_start - r_start) <= time_tolerance_sec
            for c_start, c_end, c_text in candidate
        )
        if hit:
            matched += 1
    return matched / len(ref), matched, len(ref)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--range-start", type=float, required=True)
    parser.add_argument("--range-end", type=float, required=True)
    parser.add_argument("--step", type=float, default=0.16)
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--startup-grace-sec", type=float, default=3.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1,
        help="per-worker thread count; default 1, matching Gate B's aggregate-throughput finding "
        "(CLAUDE.md) -- concurrency * n_threads should stay near the core count, not exceed it",
    )
    parser.add_argument("--recall-threshold", type=float, default=0.6)
    args = parser.parse_args()

    baseline_duration = args.range_end + args.window_sec
    print(f"# building baseline: continuous transcript 0 -> {baseline_duration:.1f}s", file=sys.stderr)
    baseline = await transcribe_window(args.model, args.wav, 0.0, baseline_duration, args.n_threads)
    print(f"# baseline: {len(baseline)} words", file=sys.stderr)

    offsets = []
    s = args.range_start
    while s <= args.range_end + 1e-9:
        offsets.append(round(s, 3))
        s += args.step

    sem = asyncio.Semaphore(args.concurrency)

    async def probe(offset: float) -> tuple[float, float, int, int]:
        async with sem:
            candidate = await transcribe_window(args.model, args.wav, offset, args.window_sec, args.n_threads)
        window_start = offset + args.startup_grace_sec
        window_end = offset + args.window_sec - 1.0
        r, matched, total = recall(baseline, candidate, window_start, window_end)
        return offset, r, matched, total

    print("offset,recall,matched,total", flush=True)
    results = await asyncio.gather(*(probe(o) for o in offsets))
    flagged = []
    for offset, r, matched, total in results:
        print(f"{offset:.2f},{r:.3f},{matched},{total}", flush=True)
        if total > 0 and r < args.recall_threshold:
            flagged.append((offset, r, matched, total))

    print(f"\n# swept {len(offsets)} offsets in [{args.range_start},{args.range_end}] step {args.step}")
    print(f"# flagged (recall < {args.recall_threshold}): {len(flagged)}")
    for offset, r, matched, total in flagged:
        print(f"#   offset={offset:.2f}s recall={r:.3f} ({matched}/{total})")


if __name__ == "__main__":
    asyncio.run(main())
