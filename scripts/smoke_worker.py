#!/usr/bin/env python3
"""Manual Phase-0 smoke test: spawn the real worker binary, hand it a real
model + a slice of real audio over the actual IPC protocol, and print what
comes back. Not a pytest test (that's tests/test_capi_smoke.py, which needs
the binary built and a model mounted) -- this is the fastest way to eyeball
that the whole chain works before wiring up the gRPC front door.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import wave

sys.path.insert(0, ".")
from live_stt.framing import FrameType, pack, unpack_one  # noqa: E402

WORKER_BIN = "worker/build/live_stt_worker"
GGML_LIB_DIR = "worker/build-parakeet/third_party/ggml/src"
MODEL_PATH = "models/realtime_eou_120m-v1-q8_0.gguf"
WAV_PATH = "/data/homes/stoneshi/src/transcript/output.wav"
CHUNK_SEC = 0.16  # the 120m model's chunk size


def read_frame_from_socket(sock: socket.socket) -> tuple[FrameType, bytes]:
    buf = b""
    while True:
        result = unpack_one(buf)
        if result is not None:
            frame, buf = result
            return frame.type, frame.payload
        chunk = sock.recv(65536)
        if not chunk:
            raise EOFError("worker closed the socket")
        buf += chunk


def main() -> None:
    parent_sock, child_sock = socket.socketpair()

    child_fd = child_sock.fileno()
    proc = subprocess.Popen(
        [WORKER_BIN],
        stdin=None,
        stdout=None,
        stderr=None,
        # 3 must ALSO be listed here, not just dup2'd in preexec_fn: the
        # close-fds pass runs AFTER preexec_fn, and it only spares fds in
        # pass_fds -- a dup2 target created inside preexec_fn is not
        # automatically exempted and gets closed again immediately unless
        # its number is listed too. Confirmed empirically while building this.
        pass_fds=(child_fd, 3),
        env={"LD_LIBRARY_PATH": GGML_LIB_DIR},
        preexec_fn=lambda: _dup_to_fd3(child_fd),
    )
    child_sock.close()

    config = json.dumps({"gguf_path": MODEL_PATH, "language": "", "n_threads": 4})
    parent_sock.sendall(pack(FrameType.CONFIG, config.encode()))

    frame_type, payload = read_frame_from_socket(parent_sock)
    print(f"<- {frame_type.name} {payload.decode()}")
    if frame_type == FrameType.ERROR:
        proc.wait()
        sys.exit(1)

    wf = wave.open(WAV_PATH, "rb")
    sample_rate = wf.getframerate()
    chunk_samples = int(sample_rate * CHUNK_SEC)

    n_chunks_to_feed = 100  # ~16s of audio, enough to see real partial output
    full_text = ""
    for i in range(n_chunks_to_feed):
        pcm = wf.readframes(chunk_samples)
        if not pcm:
            break
        parent_sock.sendall(pack(FrameType.AUDIO, pcm))
        frame_type, payload = read_frame_from_socket(parent_sock)
        doc = json.loads(payload.decode())
        if doc.get("text"):
            full_text += doc["text"]
            print(f"[{i}] rss_mb={doc['rss_kb'] / 1024:.0f} text={doc['text']!r}")
        if doc.get("events"):
            print(f"[{i}] events={doc['events']}")

    parent_sock.sendall(pack(FrameType.FINALIZE))
    frame_type, payload = read_frame_from_socket(parent_sock)
    doc = json.loads(payload.decode())
    print(f"<- FINAL {doc}")
    full_text += doc.get("text", "")

    print("=" * 60)
    print(f"Full transcript from {n_chunks_to_feed} chunks:")
    print(full_text)

    parent_sock.close()
    proc.wait(timeout=5)
    print(f"worker exit code: {proc.returncode}")


def _dup_to_fd3(fd: int) -> None:
    import os

    if fd != 3:
        os.dup2(fd, 3)


if __name__ == "__main__":
    main()
