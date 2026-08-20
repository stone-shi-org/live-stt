"""Production WorkerHandle: spawns one C++ worker process, speaks the IPC
protocol over asyncio streams. The production analogue of
tests/worker_harness.py (and tools/leak_curve.py's copy of it) -- see
CLAUDE.md's fd-passing gotcha before touching the spawn logic.

No rotation yet (Phase 3): one WorkerHandle lives for the life of one call.
This is also, deliberately, the seam a test monkeypatches to drive
live_stt/session.py without a real binary or model -- mirroring
my-meeting-notes/app/routers/live_caption.py's `_connect_realtime()` seam.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any

from live_stt.framing import FrameType, HEADER_SIZE, pack, parse_header
from live_stt.logging_config import get_logger

logger = get_logger("worker")


class WorkerError(Exception):
    """The worker reported an ERROR frame, or its process/socket misbehaved."""


def _harden_child(child_fd: int) -> None:
    """Runs in the forked child, before exec (subprocess.Popen's
    preexec_fn). Order matters: move the IPC socket onto fd 3 FIRST.

    Only the always-safe, sizing-independent hardening lives here.
    RLIMIT_AS and oom_score_adj need the worker pool's per-slot memory
    budget to size correctly and are added when live_stt/pool/ exists
    (Phase 3) -- adding them here now with a made-up number would be worse
    than not having them.
    """
    os.dup2(child_fd, 3)

    try:
        import resource

        # ggml aborts (not raises) on a failed allocation on some backends;
        # without this, that abort's core file can fill the disk over many
        # calls. /proc/sys/kernel/core_pattern is host-global and can't be
        # set from a container, so this rlimit is the only defense available
        # here.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        logger.warning("could not set RLIMIT_CORE=0 in worker preexec hook")

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        logger.warning("could not set PR_SET_PDEATHSIG in worker preexec hook")


async def _read_frame(reader: asyncio.StreamReader) -> tuple[FrameType, bytes]:
    header = await reader.readexactly(HEADER_SIZE)
    payload_len, frame_type = parse_header(header)
    payload = await reader.readexactly(payload_len) if payload_len else b""
    return frame_type, payload


async def _read_json_frame(reader: asyncio.StreamReader) -> tuple[FrameType, dict[str, Any]]:
    frame_type, payload = await _read_frame(reader)
    doc = json.loads(payload.decode()) if payload else {}
    return frame_type, doc


class WorkerHandle:
    def __init__(
        self, proc: subprocess.Popen, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.proc = proc
        self._reader = reader
        self._writer = writer
        self.ready: dict[str, Any] = {}

    @classmethod
    async def spawn(
        cls,
        *,
        worker_bin: Path,
        gguf_path: str,
        language: str,
        n_threads: int,
        ggml_lib_dir: Path | None = None,
    ) -> "WorkerHandle":
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        env = dict(os.environ)
        if ggml_lib_dir is not None:
            env["LD_LIBRARY_PATH"] = str(ggml_lib_dir)
        try:
            proc = subprocess.Popen(
                [str(worker_bin)],
                # 3 must ALSO be listed here, not just dup2'd inside the
                # preexec hook -- close_fds runs AFTER preexec_fn and would
                # otherwise close the very fd the hook just created. See
                # CLAUDE.md's fd-passing gotcha.
                pass_fds=(child_fd, 3),
                env=env,
                preexec_fn=lambda: _harden_child(child_fd),
            )
        finally:
            child_sock.close()

        parent_sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=parent_sock)

        handle = cls(proc, reader, writer)
        config = json.dumps(
            {"gguf_path": gguf_path, "language": language, "n_threads": n_threads}
        )
        try:
            # The write+drain must be in the SAME try as the read: a worker
            # that died (or never started -- e.g. a transient fork/exec
            # hiccup) before reading anything fails right here, not at the
            # read below. An earlier version of this code only wrapped the
            # read, so a dead-on-arrival worker surfaced as an uncaught
            # ConnectionResetError instead of a clean WorkerError -- found
            # via intermittent CI flakiness that traced back to exactly
            # this gap, not a real bug in the retry/rotation logic it looked
            # like it was in.
            writer.write(pack(FrameType.CONFIG, config.encode()))
            await writer.drain()
            frame_type, doc = await _read_json_frame(reader)
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as exc:
            handle.kill()
            raise WorkerError(f"worker died before CONFIG completed: {exc}") from exc

        if frame_type != FrameType.READY:
            handle.kill()
            raise WorkerError(f"worker CONFIG failed: {doc}")
        handle.ready = doc
        return handle

    async def feed(self, pcm16le: bytes) -> dict[str, Any]:
        return await self._request(FrameType.AUDIO, pcm16le)

    async def finalize(self) -> dict[str, Any]:
        return await self._request(FrameType.FINALIZE, b"")

    async def _request(self, frame_type: FrameType, payload: bytes) -> dict[str, Any]:
        self._writer.write(pack(frame_type, payload))
        try:
            await self._writer.drain()
            response_type, doc = await _read_json_frame(self._reader)
        except (asyncio.IncompleteReadError, ConnectionError, OSError) as exc:
            raise WorkerError(f"worker connection lost: {exc}") from exc
        if response_type == FrameType.ERROR:
            raise WorkerError(f"worker reported an error: {doc}")
        return doc

    def kill(self) -> None:
        # Always SIGKILL, never SIGTERM or a graceful request -- a worker
        # process is never reused for a second call (see CLAUDE.md), and on
        # a CUDA build letting it exit normally risks the static-destructor
        # teardown abort. Safe to call more than once.
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass

    async def wait_closed(self, timeout: float = 5.0) -> int:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, self.proc.wait), timeout)
        except asyncio.TimeoutError:
            self.kill()
            return await loop.run_in_executor(None, self.proc.wait)
