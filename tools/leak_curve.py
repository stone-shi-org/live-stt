#!/usr/bin/env python3
"""Phase 1 Gate A: characterize the parakeet.cpp streaming leak
(https://github.com/mudler/parakeet.cpp/issues/63) on THIS build/backend.

The upstream issue reports 19-41 MB leaked per second of audio fed, measured
on a CUDA Jetson build, reproduced with pure silence. It explicitly was not
tested on the CPU backend. This script produces the number for whatever
backend the worker binary at hand was built for, from two conditions
(silence and real speech -- if the rate differs between them, the leak
tracks emitted tokens, not fed audio, and the rotation trigger needs to be
audio-cap AND word-count-based, not audio-cap alone).

Usage:
    python tools/leak_curve.py --model models/realtime_eou_120m-v1-q8_0.gguf \
        --condition silence --audio-sec 600
    python tools/leak_curve.py --model models/realtime_eou_120m-v1-q8_0.gguf \
        --condition speech --audio-fixture /path/to/long.wav --audio-sec 600

Prints a CSV (audio_sec,rss_kb,wall_sec) to stdout as it runs and a summary
(slope_mb_per_audio_sec, r_squared, plateau_detected) at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live_stt.framing import FrameType, pack, unpack_one  # noqa: E402

DEFAULT_WORKER_BIN = REPO_ROOT / "worker" / "build" / "live_stt_worker"
DEFAULT_GGML_LIB_DIR = REPO_ROOT / "worker" / "build-parakeet" / "third_party" / "ggml" / "src"


class WorkerHandle:
    """Minimal standalone copy of tests/worker_harness.py's WorkerHandle --
    duplicated rather than imported so this script has no dependency on the
    tests/ package layout and can be run standalone, matching the other
    tools/*.py scripts in this repo."""

    def __init__(self, worker_bin: Path, ggml_lib_dir: Path) -> None:
        self._parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        self.proc = subprocess.Popen(
            [str(worker_bin)],
            # 3 must ALSO be listed here -- see CLAUDE.md's fd-passing gotcha.
            pass_fds=(child_fd, 3),
            env={**os.environ, "LD_LIBRARY_PATH": str(ggml_lib_dir)},
            preexec_fn=lambda: os.dup2(child_fd, 3),
        )
        child_sock.close()
        self._buf = b""

    def send(self, frame_type: FrameType, payload: bytes = b"") -> None:
        self._parent_sock.sendall(pack(frame_type, payload))

    def recv_json(self) -> tuple[FrameType, dict]:
        while True:
            result = unpack_one(self._buf)
            if result is not None:
                frame, self._buf = result
                return frame.type, json.loads(frame.payload.decode())
            chunk = self._parent_sock.recv(65536)
            if not chunk:
                raise EOFError("worker closed the IPC socket")
            self._buf += chunk

    def close(self, timeout: float = 5.0) -> int:
        self._parent_sock.close()
        return self.proc.wait(timeout=timeout)


def _silence_chunks(chunk_samples: int):
    silence = b"\x00\x00" * chunk_samples
    while True:
        yield silence


def _speech_chunks(fixture_path: str, chunk_samples: int):
    wf = wave.open(fixture_path, "rb")
    assert wf.getframerate() == 16000, "fixture must be 16kHz"
    assert wf.getnchannels() == 1, "fixture must be mono"
    assert wf.getsampwidth() == 2, "fixture must be int16"
    while True:
        pcm = wf.readframes(chunk_samples)
        if not pcm:
            wf.rewind()
            continue
        yield pcm


def linregress(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple least-squares linear regression, stdlib only (no numpy/scipy
    dependency for a diagnostic script). Returns (slope, intercept, r_squared).
    """
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = mean_y - slope * mean_x
    r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_xx and ss_yy else 0.0
    return slope, intercept, r_squared


def run(
    *,
    model_path: str,
    condition: str,
    audio_sec_target: float,
    chunk_ms: int,
    sample_every_sec: float,
    n_threads: int,
    audio_fixture: str | None,
    worker_bin: Path,
    ggml_lib_dir: Path,
    language: str = "",
) -> dict:
    handle = WorkerHandle(worker_bin, ggml_lib_dir)
    config = json.dumps({"gguf_path": model_path, "language": language, "n_threads": n_threads})
    handle.send(FrameType.CONFIG, config.encode())
    frame_type, doc = handle.recv_json()
    if frame_type == FrameType.ERROR:
        raise RuntimeError(f"CONFIG failed: {doc}")
    print(f"# READY {doc}", file=sys.stderr)

    chunk_samples = int(16000 * chunk_ms / 1000)
    if condition == "silence":
        chunks = _silence_chunks(chunk_samples)
    elif condition == "speech":
        if not audio_fixture:
            raise ValueError("--audio-fixture required for --condition speech")
        chunks = _speech_chunks(audio_fixture, chunk_samples)
    else:
        raise ValueError(f"unknown condition: {condition}")

    samples: list[tuple[float, int, float]] = []  # (audio_sec, rss_kb, wall_sec)
    audio_sec = 0.0
    next_sample_at = 0.0
    start = time.monotonic()

    print("audio_sec,rss_kb,wall_sec")
    while audio_sec < audio_sec_target:
        pcm = next(chunks)
        handle.send(FrameType.AUDIO, pcm)
        _, doc = handle.recv_json()
        n_samples_fed = len(pcm) // 2
        audio_sec += n_samples_fed / 16000.0

        if audio_sec >= next_sample_at:
            wall_sec = time.monotonic() - start
            rss_kb = doc["rss_kb"]
            samples.append((audio_sec, rss_kb, wall_sec))
            print(f"{audio_sec:.2f},{rss_kb},{wall_sec:.2f}")
            sys.stdout.flush()
            next_sample_at += sample_every_sec

    handle.send(FrameType.FINALIZE)
    handle.recv_json()
    exit_code = handle.close()

    xs = [s[0] for s in samples]
    ys_mb = [s[1] / 1024.0 for s in samples]
    slope_mb_per_audio_sec, intercept_mb, r_squared = linregress(xs, ys_mb)

    # Crude plateau check: compare the growth rate over the first half of the
    # run to the second half. A genuine plateau (no leak) would show the
    # second-half slope collapsing toward zero relative to the first half.
    mid = len(samples) // 2
    plateau_detected = False
    if mid > 2:
        slope_1st, _, _ = linregress(xs[:mid], ys_mb[:mid])
        slope_2nd, _, _ = linregress(xs[mid:], ys_mb[mid:])
        if slope_1st > 0.01 and slope_2nd < 0.25 * slope_1st:
            plateau_detected = True

    return {
        "condition": condition,
        "n_threads": n_threads,
        "audio_sec_fed": audio_sec,
        "wall_sec": samples[-1][2] if samples else 0.0,
        "n_samples": len(samples),
        "slope_mb_per_audio_sec": slope_mb_per_audio_sec,
        "r_squared": r_squared,
        "plateau_detected": plateau_detected,
        "worker_exit_code": exit_code,
        "rss_mb_start": ys_mb[0] if ys_mb else None,
        "rss_mb_end": ys_mb[-1] if ys_mb else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=["silence", "speech"], required=True)
    parser.add_argument("--audio-fixture", default=None)
    parser.add_argument("--audio-sec", type=float, default=600.0)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--sample-every-sec", type=float, default=5.0)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--language", default="")
    parser.add_argument("--worker-bin", type=Path, default=DEFAULT_WORKER_BIN)
    parser.add_argument("--ggml-lib-dir", type=Path, default=DEFAULT_GGML_LIB_DIR)
    args = parser.parse_args()

    summary = run(
        model_path=args.model,
        condition=args.condition,
        audio_sec_target=args.audio_sec,
        chunk_ms=args.chunk_ms,
        sample_every_sec=args.sample_every_sec,
        n_threads=args.n_threads,
        audio_fixture=args.audio_fixture,
        worker_bin=args.worker_bin,
        ggml_lib_dir=args.ggml_lib_dir,
        language=args.language,
    )
    print("# SUMMARY " + json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
