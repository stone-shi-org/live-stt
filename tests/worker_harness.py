"""Minimal harness for driving the real live_stt_worker binary over its IPC
protocol from a test process. Used only by integration/model-marked tests --
unit tests never import this (tests/conftest.py's offline-safety fixture
points LSTT_MODEL_PATH at a nonexistent path so nothing accidentally does).

Encodes one non-obvious subtlety in spawning it, confirmed empirically while
building this service: subprocess.Popen's close-fds pass runs AFTER
preexec_fn, so a dup2 target created inside preexec_fn (moving the socket to
the worker's fixed fd 3) must ALSO be listed in pass_fds, or it gets closed
again immediately and the worker sees a bad file descriptor on fd 3.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from live_stt.framing import FrameType, pack, unpack_one

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_BIN = REPO_ROOT / "worker" / "build" / "live_stt_worker"
GGML_LIB_DIR = REPO_ROOT / "worker" / "build-parakeet" / "third_party" / "ggml" / "src"


class WorkerHandle:
    def __init__(self) -> None:
        self._parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        self.proc = subprocess.Popen(
            [str(WORKER_BIN)],
            pass_fds=(child_fd, 3),
            env={**os.environ, "LD_LIBRARY_PATH": str(GGML_LIB_DIR)},
            preexec_fn=lambda: os.dup2(child_fd, 3),
        )
        child_sock.close()
        self._buf = b""

    def send(self, frame_type: FrameType, payload: bytes = b"") -> None:
        self._parent_sock.sendall(pack(frame_type, payload))

    def send_json(self, frame_type: FrameType, doc: dict[str, Any]) -> None:
        self.send(frame_type, json.dumps(doc).encode())

    def recv(self) -> tuple[FrameType, bytes]:
        while True:
            result = unpack_one(self._buf)
            if result is not None:
                frame, self._buf = result
                return frame.type, frame.payload
            chunk = self._parent_sock.recv(65536)
            if not chunk:
                raise EOFError("worker closed the IPC socket")
            self._buf += chunk

    def recv_json(self) -> tuple[FrameType, dict[str, Any]]:
        frame_type, payload = self.recv()
        return frame_type, json.loads(payload.decode())

    def close(self, timeout: float = 5.0) -> int:
        self._parent_sock.close()
        return self.proc.wait(timeout=timeout)


def configure(
    handle: WorkerHandle, gguf_path: str, *, language: str = "", n_threads: int = 4
) -> dict[str, Any]:
    handle.send_json(
        FrameType.CONFIG, {"gguf_path": gguf_path, "language": language, "n_threads": n_threads}
    )
    frame_type, doc = handle.recv_json()
    if frame_type == FrameType.ERROR:
        raise RuntimeError(f"worker CONFIG failed: {doc}")
    assert frame_type == FrameType.READY
    return doc
