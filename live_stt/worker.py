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
from collections import deque
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from live_stt.framing import FrameType, HEADER_SIZE, pack, parse_header
from live_stt.logging_config import get_logger

logger = get_logger("worker")

_CUDA_ALLOC_MIB_RE = re.compile(
    r"allocating\s+(\d+(?:\.\d+)?)\s*MiB[^\n]*?(?:cudaMalloc failed|out of memory)",
    re.IGNORECASE,
)
_GALLOCR_BYTES_RE = re.compile(
    r"failed to allocate (?:CUDA\d*|GPU) buffer of size\s+(\d+)",
    re.IGNORECASE,
)
_GENERIC_OOM_RE = re.compile(
    r"(?:cudaMalloc failed:\s*out of memory|CUDA out of memory|ggml_gallocr_alloc_graph failed|std::bad_alloc|out of memory)",
    re.IGNORECASE,
)


class WorkerError(Exception):
    """The worker reported an ERROR frame, or its process/socket misbehaved."""


class WorkerOOMError(WorkerError):
    """The worker failed to allocate GPU/host memory (e.g. CUDA OOM during graph allocation)."""

    def __init__(self, message: str, *, allocated_mib: float | None = None) -> None:
        super().__init__(message)
        self.allocated_mib = allocated_mib


def _start_stderr_drain(
    proc: subprocess.Popen, stderr_tail: deque[str]
) -> threading.Thread | None:
    stream = proc.stderr
    if stream is None:
        return None

    def _reader() -> None:
        try:
            for raw_line in iter(stream.readline, b""):
                if not raw_line:
                    break
                text = raw_line.decode("utf-8", "replace")
                stripped = text.strip()
                if stripped:
                    stderr_tail.append(stripped)
                try:
                    sys.stderr.write(text)
                    sys.stderr.flush()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t = threading.Thread(target=_reader, daemon=True, name=f"worker-stderr-{proc.pid}")
    t.start()
    return t


def _detect_oom(
    stderr_tail: deque[str], error_doc: dict[str, Any] | None = None
) -> tuple[bool, float | None, str | None]:
    candidates = list(stderr_tail)
    if error_doc and isinstance(error_doc.get("error"), str):
        candidates.append(error_doc["error"])
    if not candidates:
        return False, None, None

    joined = "\n".join(candidates)
    m_mib = _CUDA_ALLOC_MIB_RE.search(joined)
    if m_mib:
        mib = float(m_mib.group(1))
        return True, mib, m_mib.group(0)

    m_bytes = _GALLOCR_BYTES_RE.search(joined)
    if m_bytes:
        mib = int(m_bytes.group(1)) / (1024.0 * 1024.0)
        return True, mib, m_bytes.group(0)

    m_oom = _GENERIC_OOM_RE.search(joined)
    if m_oom:
        return True, None, m_oom.group(0)

    return False, None, None


def _raise_detected_worker_error(
    *,
    proc: subprocess.Popen,
    stderr_tail: deque[str],
    stderr_thread: threading.Thread | None,
    fallback_msg: str,
    error_doc: dict[str, Any] | None = None,
    cause_exc: BaseException | None = None,
) -> None:
    if stderr_thread is not None and stderr_thread.is_alive():
        stderr_thread.join(timeout=0.05)

    is_oom, alloc_mib, detail = _detect_oom(stderr_tail, error_doc=error_doc)
    if is_oom:
        if alloc_mib is not None:
            msg = f"CUDA OOM allocating {alloc_mib:g} MiB"
        else:
            msg = f"CUDA OOM ({detail})" if detail else "CUDA OOM"
        logger.error("%s (pid=%s)", msg, proc.pid)
        if cause_exc is not None:
            raise WorkerOOMError(msg, allocated_mib=alloc_mib) from cause_exc
        raise WorkerOOMError(msg, allocated_mib=alloc_mib)

    if stderr_tail and "worker connection lost" in fallback_msg:
        last_line = stderr_tail[-1]
        fallback_msg = f"{fallback_msg} (stderr: {last_line})"
    if cause_exc is not None:
        raise WorkerError(fallback_msg) from cause_exc
    raise WorkerError(fallback_msg)


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


def _read_exact_sync(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before reading requested bytes")
        buf.extend(chunk)
    return bytes(buf)


def _read_frame_sync(sock: socket.socket) -> tuple[FrameType, bytes]:
    header = _read_exact_sync(sock, HEADER_SIZE)
    payload_len, frame_type = parse_header(header)
    payload = _read_exact_sync(sock, payload_len) if payload_len else b""
    return frame_type, payload


def _read_json_frame_sync(sock: socket.socket) -> tuple[FrameType, dict[str, Any]]:
    frame_type, payload = _read_frame_sync(sock)
    doc = json.loads(payload.decode()) if payload else {}
    return frame_type, doc


@dataclass(frozen=True, slots=True)
class StandbyKey:
    worker_bin: str
    gguf_path: str
    language: str
    n_threads: int
    ggml_lib_dir: str | None = None
    ggml_backend_path: str | None = None
    use_gpu: bool = False
    model_key: str = ""
    # Extra environment for the worker process (e.g. PARAKEET_ATT_CONTEXT for
    # the TDT engine -- see live_stt/session.py's _extra_env). Part of the key:
    # a standby spawned under different settings must never be reused.
    extra_env: tuple[tuple[str, str], ...] = ()


@dataclass
class StandbyWorker:
    proc: subprocess.Popen
    sock: socket.socket
    ready: dict[str, Any]
    key: StandbyKey
    idle_since: float
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=64))
    stderr_thread: threading.Thread | None = None

    def kill(self) -> None:
        try:
            self.proc.kill()
        except ProcessLookupError:
            pass
        try:
            self.sock.close()
        except Exception:
            pass
        if self.proc.stderr is not None:
            try:
                self.proc.stderr.close()
            except Exception:
                pass


def spawn_standby_sync(
    *,
    worker_bin: Path,
    gguf_path: str,
    language: str,
    n_threads: int,
    ggml_lib_dir: Path | None = None,
    ggml_backend_path: Path | None = None,
    use_gpu: bool = False,
    model_key: str = "",
    extra_env: tuple[tuple[str, str], ...] = (),
) -> StandbyWorker:
    parent_sock, child_sock = socket.socketpair()
    child_fd = child_sock.fileno()
    env = dict(os.environ)
    env.update(dict(extra_env))
    if ggml_lib_dir is not None:
        env["LD_LIBRARY_PATH"] = str(ggml_lib_dir)
    if ggml_backend_path is not None:
        env["GGML_BACKEND_PATH"] = str(ggml_backend_path)
    try:
        proc = subprocess.Popen(
            [str(worker_bin)],
            pass_fds=(child_fd, 3),
            env=env,
            stderr=subprocess.PIPE,
            preexec_fn=lambda: _harden_child(child_fd),
        )
    finally:
        child_sock.close()

    stderr_tail: deque[str] = deque(maxlen=64)
    stderr_thread = _start_stderr_drain(proc, stderr_tail)

    config = json.dumps(
        {
            "gguf_path": gguf_path,
            "language": language,
            "n_threads": n_threads,
            "use_gpu": use_gpu,
        }
    )
    try:
        parent_sock.settimeout(30.0)
        parent_sock.sendall(pack(FrameType.CONFIG, config.encode()))
        frame_type, doc = _read_json_frame_sync(parent_sock)
    except Exception as exc:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        parent_sock.close()
        _raise_detected_worker_error(
            proc=proc,
            stderr_tail=stderr_tail,
            stderr_thread=stderr_thread,
            fallback_msg=f"worker died before CONFIG completed: {exc}",
            cause_exc=exc,
        )

    if frame_type != FrameType.READY:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        parent_sock.close()
        _raise_detected_worker_error(
            proc=proc,
            stderr_tail=stderr_tail,
            stderr_thread=stderr_thread,
            fallback_msg=f"worker CONFIG failed: {doc}",
            error_doc=doc,
        )

    key = StandbyKey(
        worker_bin=str(worker_bin),
        gguf_path=str(gguf_path),
        language=language,
        n_threads=n_threads,
        ggml_lib_dir=str(ggml_lib_dir) if ggml_lib_dir else None,
        ggml_backend_path=str(ggml_backend_path) if ggml_backend_path else None,
        use_gpu=use_gpu,
        model_key=model_key,
        extra_env=tuple(extra_env),
    )
    return StandbyWorker(
        proc=proc,
        sock=parent_sock,
        ready=doc,
        key=key,
        idle_since=time.monotonic(),
        stderr_tail=stderr_tail,
        stderr_thread=stderr_thread,
    )


class WorkerHandle:
    def __init__(
        self,
        proc: subprocess.Popen,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        stderr_tail: deque[str] | None = None,
        stderr_thread: threading.Thread | None = None,
    ) -> None:
        self.proc = proc
        self._reader = reader
        self._writer = writer
        self.ready: dict[str, Any] = {}
        self.stderr_tail: deque[str] = stderr_tail if stderr_tail is not None else deque(maxlen=64)
        self._stderr_thread = stderr_thread

    @classmethod
    async def from_standby(cls, standby: StandbyWorker) -> "WorkerHandle":
        standby.sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=standby.sock)
        handle = cls(
            standby.proc,
            reader,
            writer,
            stderr_tail=standby.stderr_tail,
            stderr_thread=standby.stderr_thread,
        )
        handle.ready = standby.ready
        return handle

    @classmethod
    async def spawn(
        cls,
        *,
        worker_bin: Path,
        gguf_path: str,
        language: str,
        n_threads: int,
        ggml_lib_dir: Path | None = None,
        ggml_backend_path: Path | None = None,
        use_gpu: bool = False,
        extra_env: tuple[tuple[str, str], ...] = (),
    ) -> "WorkerHandle":
        parent_sock, child_sock = socket.socketpair()
        child_fd = child_sock.fileno()
        env = dict(os.environ)
        env.update(dict(extra_env))
        if ggml_lib_dir is not None:
            env["LD_LIBRARY_PATH"] = str(ggml_lib_dir)
        if ggml_backend_path is not None:
            # Only meaningful for an engine whose worker calls ggml's OWN
            # ggml_backend_load_all() (today: main_whisper.cpp, see that
            # file's comment) -- a SEPARATE dynamic-backend-discovery
            # mechanism from ordinary shared-library linking, and one that
            # does NOT consult LD_LIBRARY_PATH at all. Confirmed by reading
            # ggml_backend_load_all_from_path() in the vendored
            # ggml-backend-reg.cpp: its search_paths are the compile-time
            # GGML_BACKEND_DIR, the WORKER BINARY'S OWN directory, and the
            # process's current working directory -- never LD_LIBRARY_PATH.
            # Found for real: a runtime-installed whisper:cuda backend
            # (live_stt/backends/installer.py's oci_image path) correctly
            # set LD_LIBRARY_PATH to its install directory, yet the worker
            # logged `whisper_backend_init_gpu: no GPU found` and silently
            # ran on CPU anyway -- LD_LIBRARY_PATH pointing at
            # libggml-cuda.so simply never got looked at by this loader.
            # ggml_backend_load_all_from_path()'s own last step DOES read
            # one thing outside those three search paths: the
            # GGML_BACKEND_PATH env var, loading exactly the one file it
            # names via ggml_backend_load() directly -- so pointing this at
            # the activated backend's libggml-cuda.so (a real file, not a
            # directory) makes the runtime-installed CUDA backend
            # discoverable without touching the worker's CWD or relying on
            # LD_LIBRARY_PATH at all.
            env["GGML_BACKEND_PATH"] = str(ggml_backend_path)
        try:
            proc = subprocess.Popen(
                [str(worker_bin)],
                # 3 must ALSO be listed here, not just dup2'd inside the
                # preexec hook -- close_fds runs AFTER preexec_fn and would
                # otherwise close the very fd the hook just created. See
                # CLAUDE.md's fd-passing gotcha.
                pass_fds=(child_fd, 3),
                env=env,
                stderr=subprocess.PIPE,
                preexec_fn=lambda: _harden_child(child_fd),
            )
        finally:
            child_sock.close()

        stderr_tail: deque[str] = deque(maxlen=64)
        stderr_thread = _start_stderr_drain(proc, stderr_tail)

        parent_sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=parent_sock)

        handle = cls(proc, reader, writer, stderr_tail=stderr_tail, stderr_thread=stderr_thread)
        # use_gpu is only meaningful to the whisper engine's main_whisper.cpp
        # (see session_whisper.hpp's comment on why parakeet's CPU-vs-CUDA
        # choice is a build-time/binary one, not a per-load flag) -- sent
        # unconditionally anyway, since it's a harmless extra key the
        # parakeet engine's main.cpp simply never reads. Keeps this spawn()
        # call engine-agnostic; the caller (live_stt/session.py) decides the
        # value from the per-engine backend setting (parakeet_backend/
        # whisper_backend/parakeet_tdt_backend, LS-5) matching whichever
        # binary it's about to exec.
        config = json.dumps(
            {
                "gguf_path": gguf_path,
                "language": language,
                "n_threads": n_threads,
                "use_gpu": use_gpu,
            }
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
            _raise_detected_worker_error(
                proc=proc,
                stderr_tail=stderr_tail,
                stderr_thread=stderr_thread,
                fallback_msg=f"worker died before CONFIG completed: {exc}",
                cause_exc=exc,
            )

        if frame_type != FrameType.READY:
            handle.kill()
            _raise_detected_worker_error(
                proc=proc,
                stderr_tail=stderr_tail,
                stderr_thread=stderr_thread,
                fallback_msg=f"worker CONFIG failed: {doc}",
                error_doc=doc,
            )
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
            _raise_detected_worker_error(
                proc=self.proc,
                stderr_tail=self.stderr_tail,
                stderr_thread=self._stderr_thread,
                fallback_msg=f"worker connection lost: {exc}",
                cause_exc=exc,
            )
        if response_type == FrameType.ERROR:
            _raise_detected_worker_error(
                proc=self.proc,
                stderr_tail=self.stderr_tail,
                stderr_thread=self._stderr_thread,
                fallback_msg=f"worker reported an error: {doc}",
                error_doc=doc,
            )
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
        if self.proc.stderr is not None:
            try:
                self.proc.stderr.close()
            except Exception:
                pass

    async def wait_closed(self, timeout: float = 5.0) -> int:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(loop.run_in_executor(None, self.proc.wait), timeout)
        except asyncio.TimeoutError:
            self.kill()
            return await loop.run_in_executor(None, self.proc.wait)

