#!/usr/bin/env python3
"""Minimal reproduction of a serious, upstream-looking parakeet.cpp/engine
behavior found while testing Phase 3's worker rotation: a FRESH streaming
session, given no rotation code, no dual-feed, no promotion -- just one
worker process, one parakeet_capi_stream_begin, fed audio starting at a
specific absolute offset into a WAV file -- can silently drop several
SECONDS of clearly-present speech partway through the stream, while
starting at a position 160ms earlier or later works perfectly.

Found because a naive integration test of the rotation state machine (which
kills a worker and promotes a "shadow" that was dual-fed from the rotation's
start point, not from file position 0) showed a duplicated/missing-word
symptom. Direct investigation ruled out every piece of live_stt's own
orchestration code: the SAME drop reproduces with a single worker process,
zero rotation/dual-feed/promotion logic, differing only in which absolute
sample offset of the source file it is told to treat as sample 0 of its own
stream.

Usage:
    python tools/repro_stream_start_dropout.py \
        --model models/realtime_eou_120m-v1-q8_0.gguf \
        --wav /path/to/audio.wav --start-sec 8.00 --duration-sec 17 \
        --watch-word yes

On this repo's fixture (~/src/transcript/output.wav, verified PCM16 mono
16kHz), --start-sec 8.00 drops "yes yes absolutely ok perfect well yeah"
(7 words, ~3.6s of clear speech) entirely; --start-sec 7.84 or 8.16 (each
160ms away) transcribe it correctly. See CLAUDE.md's "Serious open risk"
section for the full writeup and a table of probed offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live_stt.framing import FrameType, pack, unpack_one  # noqa: E402

DEFAULT_WORKER_BIN = REPO_ROOT / "worker" / "build" / "live_stt_worker"
DEFAULT_GGML_LIB_DIR = REPO_ROOT / "worker" / "build-parakeet" / "third_party" / "ggml" / "src"


def _read_frame(sock: socket.socket, buf: bytearray) -> tuple[FrameType, bytes]:
    while True:
        result = unpack_one(bytes(buf))
        if result is not None:
            frame, rest = result
            buf.clear()
            buf.extend(rest)
            return frame.type, frame.payload
        chunk = sock.recv(65536)
        if not chunk:
            raise EOFError("worker closed the IPC socket")
        buf.extend(chunk)


def run(model_path: str, wav_path: str, start_sec: float, duration_sec: float) -> list[str]:
    parent_sock, child_sock = socket.socketpair()
    child_fd = child_sock.fileno()
    proc = subprocess.Popen(
        [str(DEFAULT_WORKER_BIN)],
        pass_fds=(child_fd, 3),
        env={**os.environ, "LD_LIBRARY_PATH": str(DEFAULT_GGML_LIB_DIR)},
        preexec_fn=lambda: os.dup2(child_fd, 3),
    )
    child_sock.close()
    buf = bytearray()

    config = json.dumps({"gguf_path": model_path, "language": "", "n_threads": 4})
    parent_sock.sendall(pack(FrameType.CONFIG, config.encode()))
    frame_type, payload = _read_frame(parent_sock, buf)
    assert frame_type == FrameType.READY, payload

    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    wf.setpos(int(start_sec * sr))
    chunk_samples = int(sr * 0.16)

    all_words: list[str] = []
    n_chunks = int(duration_sec / 0.16)
    for _ in range(n_chunks):
        pcm = wf.readframes(chunk_samples)
        if not pcm:
            break
        parent_sock.sendall(pack(FrameType.AUDIO, pcm))
        _, payload = _read_frame(parent_sock, buf)
        doc = json.loads(payload.decode())
        all_words.extend(w["w"] for w in doc.get("words", []))

    parent_sock.sendall(pack(FrameType.FINALIZE))
    _, payload = _read_frame(parent_sock, buf)
    doc = json.loads(payload.decode())
    all_words.extend(w["w"] for w in doc.get("words", []))

    parent_sock.close()
    proc.kill()
    proc.wait()
    return all_words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--duration-sec", type=float, default=17.0)
    parser.add_argument("--watch-word", default=None, help="report whether this word appears")
    args = parser.parse_args()

    words = run(args.model, args.wav, args.start_sec, args.duration_sec)
    print(" ".join(words))
    if args.watch_word:
        print(f"\n--watch-word {args.watch_word!r} present: {args.watch_word in words}")


if __name__ == "__main__":
    main()
