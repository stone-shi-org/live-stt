#!/usr/bin/env python3
"""Repro/diagnostic for a reported live-caption bug: single words showing up
with a spurious space in the middle, e.g. "manager" -> "manag er", "capacity"
-> "cap acity", "Expedia" -> "exp ed ia".

This talks to the real worker binary over the real IPC socket (same wire
protocol as live_stt/worker.py) and dumps the RAW per-chunk JSON exactly as
parakeet.cpp's stream_feed_json/stream_finalize_json emit it -- before
anything in this repo's Python layer (events.py/session.py/servicer.py)
touches it. live_stt's normal streaming path is a pure passthrough of the
"text" field (see events.py/session.py), and "words" are already grouped
into whole words by parakeet.cpp itself (its comment: "the same drain as the
offline pk::group_words"). So if split words like "manag"/"er" show up here,
as two separate entries in the raw "words" list or embedded with a space in
the raw "text" field, that confirms the bug is in parakeet.cpp's own
detokenization/word-grouping, not in this codebase.

Usage:
    python tools/repro_midword_space.py \
        --model models/realtime_eou_120m-v1-q8_0.gguf \
        --wav /data/homes/stoneshi/tests/speaker_01.wav \
        --start-sec 0 --duration-sec 180
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

# Heuristic flag for suspiciously short "words" (single/double letter
# fragments) that are exactly the shape the reported bug produces.
_SUSPECT_RE = re.compile(r"^[a-z]{1,2}$")


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


def run(model_path: str, wav_path: str, start_sec: float, duration_sec: float, n_threads: int) -> None:
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

    config = json.dumps({"gguf_path": model_path, "language": "", "n_threads": n_threads})
    parent_sock.sendall(pack(FrameType.CONFIG, config.encode()))
    frame_type, payload = _read_frame(parent_sock, buf)
    assert frame_type == FrameType.READY, payload

    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    wf.setpos(int(start_sec * sr))
    chunk_samples = int(sr * 0.16)

    all_text_parts: list[str] = []
    all_words: list[dict] = []
    suspects: list[dict] = []
    n_chunks = int(duration_sec / 0.16)
    chunk_idx = 0
    for chunk_idx in range(n_chunks):
        pcm = wf.readframes(chunk_samples)
        if not pcm:
            break
        parent_sock.sendall(pack(FrameType.AUDIO, pcm))
        _, payload = _read_frame(parent_sock, buf)
        doc = json.loads(payload.decode())
        text = doc.get("text", "")
        words = doc.get("words", [])
        if text:
            print(f"[chunk {chunk_idx:5d} t={start_sec + chunk_idx * 0.16:7.2f}s] RAW text={text!r}")
        for w in words:
            all_words.append(w)
            if _SUSPECT_RE.match(w.get("w", "")):
                suspects.append({**w, "chunk": chunk_idx})
        if text:
            all_text_parts.append(text)

    parent_sock.sendall(pack(FrameType.FINALIZE))
    _, payload = _read_frame(parent_sock, buf)
    doc = json.loads(payload.decode())
    text = doc.get("text", "")
    if text:
        print(f"[FINAL] RAW text={text!r}")
        all_text_parts.append(text)
    for w in doc.get("words", []):
        all_words.append(w)
        if _SUSPECT_RE.match(w.get("w", "")):
            suspects.append({**w, "chunk": "final"})

    parent_sock.close()
    proc.kill()
    proc.wait()

    full_text = " ".join(all_text_parts)
    print("\n===== FULL RAW TEXT (concatenation of each chunk's raw \"text\" field, in order) =====")
    print(full_text)

    print("\n===== WORD LIST (raw \"words\" entries from parakeet.cpp, in order) =====")
    print(" ".join(w.get("w", "") for w in all_words))

    print(f"\n===== SUSPECT SHORT-FRAGMENT WORDS ({len(suspects)} found, regex {_SUSPECT_RE.pattern!r}) =====")
    for s in suspects:
        print(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=180.0)
    parser.add_argument("--n-threads", type=int, default=4)
    args = parser.parse_args()

    run(args.model, args.wav, args.start_sec, args.duration_sec, args.n_threads)


if __name__ == "__main__":
    main()
