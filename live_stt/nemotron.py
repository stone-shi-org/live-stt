"""Nemotron-3.5-Transcribe-30B-A4B: speaker-attributed post-call transcripts (LS-19).

Served on ``POST /v1/audio/diarization`` with
``model=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`` (see
``live_stt/diarize_http.py``). Never on gRPC ``Transcribe`` or plain
``/v1/audio/transcriptions``: as plain ASR it bought no WER over
parakeet-tdt while needing ~22 GB instead of ~1.1 GB. What it does buy, on
20 NOTSOFAR-1 meetings: cpWER 0.295 vs 0.456 and DER 0.054 vs 0.133 against
pyannote + parakeet-tdt.

**Runtime.** NeMo-Speech.cpp v0.3.0's ``nemo-speech serve`` (installed as
the ``nemo_speech:cuda`` backend, ``live_stt/backends/registry.py``) runs as
a child process on 127.0.0.1 -- one sidecar, owned by ``NemotronService``
below, modelled on ``live_stt/tts.py``'s ``TTSWorkerManager``. States:
unloaded -> loading -> ready(profile) <-> busy.

**Prompting -- never the server's default.** Every request is a
``/v1/chat/completions`` call carrying ``SPEAKER_INSTRUCTION`` (the model
card's speaker instruction) and ``timestamp_granularities: ["word"]``. The
server's built-in prompt (``/v1/audio/transcriptions``) silently invented 4
of 10 full-meeting transcripts on the spike, with HTTP 200.

**Profiles by duration** (``PROFILES``; only the 24 GB tier is verified):

- short: 4 slots, ``max_context`` 32768 tokens -- up to
  ``nemotron_short_max_sec`` (measured cap ~29.4 min); up to 4 jobs at once.
- long: 1 slot, 73728-token context, fresh process for every request --
  up to ``nemotron_long_max_sec`` (45 min measured OK; 50 min crashed, 55
  min lost word timestamps, so it is a hard limit, never attempted).
- over 45 min: ``nemotron_window_sec`` windows overlapping by
  ``nemotron_window_overlap_sec``, each through the long profile on a fresh
  process, speakers matched across windows by overlap vote
  (``live_stt/nemotron_seams.py``). Up to ``max_call_sec`` overall.

A profile switch drains the running jobs, stops the process, and loads the
other profile (~10 s).

**Sharing the GPU.** The model needs essentially the whole card, so it is
loaded on demand and unloaded when idle (``nemotron_idle_timeout_sec``).
Admission is ``vram.admit_exclusive`` (profile figure + small margin, no
card-relative reserve), only attempted while no other GPU work holds a
reservation. Any other GPU admission can evict an IDLE sidecar
(``stop_if_idle`` via ``vram.reclaim_idle_workers``). A BUSY sidecar is only
ever taken by a live gRPC call (``preempt_for_live_call``): the sidecar is
SIGTERMed (measured exit 0.7 s), the call is admitted, and every running job
goes back to the head of the queue with its original audio and reruns from
scratch -- all windows, not just the interrupted one. Post-call jobs are
retryable; live calls are not.

**Failure semantics.** A compute-buffer OOM kills the sidecar: the in-flight
request fails with a retryable 503 and the next request loads a fresh one. A
job that waits longer than ``nemotron_job_timeout_sec`` in total (queued,
re-queued, running) fails with 503 + ``Retry-After``. Starvation under
sustained live-call load is possible by design and visible in metrics.
"""

from __future__ import annotations

import atexit
import base64
import io
import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from live_stt import gpu, metrics, vram
from live_stt import nemotron_seams as seams
from live_stt.logging_config import get_logger

logger = get_logger("nemotron")

MODEL_KEY = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
ENGINE = "nemotron_transcribe"
BACKEND_ENGINE = "nemo_speech"
BACKEND_FLAVOR = "cuda"
HF_REPO = "AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF"
MAX_SPEAKERS = 8
SIDE_TIMEOUT_SEC = 3600  # nemo-speech serve max --read-timeout / --write-timeout (default is 30s)

# The model card's speaker instruction, verbatim as used for every spike,
# benchmark and seam-test measurement on LS-19.
SPEAKER_INSTRUCTION = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
    "Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> "
    "<spk:5> <spk:6> <spk:7> on the transcription."
)

CHAT_PATH = "/v1/chat/completions"
READY_PATH = "/ready"

ENCODER_FILENAME = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf"
ENCODER_SHA256 = "4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2"

# Request body cap passed to the sidecar. A 45 min 16 kHz mono WAV is ~86 MB,
# ~115 MB once base64-encoded into the JSON body.
MAX_UPLOAD_MB = 2048

# SIGTERM grace when a live call preempts a busy sidecar, before SIGKILL.
PREEMPT_TERM_GRACE_SEC = 0.5

RETRY_AFTER_CRASH_SEC = 30
RETRY_AFTER_TIMEOUT_SEC = 120
_WAIT_POLL_SEC = 1.0
_ADMIT_RETRY_SEC = 2.0


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    salm: dict[str, int]  # asr.salm.* keys written into the sidecar's --config YAML
    vram_mb: int  # exclusive-admission figure (vram.admit_exclusive)

    @property
    def max_sequences(self) -> int:
        return int(self.salm.get("max_sequences", 1))


# 24 GB tier (RTX 3090), measured on homesrv3. vram_mb: per-PID 21,662 MiB at
# ready and 22,092-22,134 MiB after 6-min requests (short); 23,970 MiB card
# peak at 45 min (long).
PROFILES_24GB: dict[str, Profile] = {
    "short": Profile(
        "short",
        {
            "max_sequences": 4,
            "n_ctx_per_sequence": 8192,
            "max_context": 32768,
            "mtp_draft": 0,
            "n_batch": 1024,
            "encoder_max_batch": 4,
        },
        vram_mb=22200,
    ),
    "long": Profile(
        "long",
        {
            "max_sequences": 1,
            "n_ctx_per_sequence": 73728,
            "max_context": 73728,
            "mtp_draft": 0,
            "n_batch": 512,
            "encoder_max_batch": 1,
            "encoder_ahead_seconds": 600,
        },
        vram_mb=24000,
    ),
}

# CPU profile (LS-20), measured on homesrv (Ryzen 9 5900XT, 16 threads):
# 0.85x-1.21x realtime, cgroup anon peak 16.9 GB, VmRSS 36.4 GB (with 21 GB
# mmap'd GGUF page cache). 1 sequence at a time; 32768-token context (~29.4 min
# cap, windowed over nemotron_cpu_max_sec).
PROFILE_CPU = Profile(
    "cpu",
    {
        "max_sequences": 1,
        "n_ctx_per_sequence": 32768,
        "max_context": 32768,
        "mtp_draft": 0,
        "n_batch": 512,
        "encoder_max_batch": 1,
    },
    vram_mb=0,
)


@dataclass(frozen=True, slots=True)
class Tier:
    name: str
    min_total_mb: int
    decoder_filename: str
    decoder_sha256: str | None
    verified: bool
    profiles: dict[str, Profile] = field(default_factory=dict)


def _decoder(quant: str) -> str:
    return f"NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-{quant}.gguf"


TIER_24GB_MIN_MB = 22000

# Size selection by total VRAM and compute capability. Only the 24 GB tier
# ships configured; the others are declared (so the choice is visible) but
# untested -- v0.3.0 also lacks asr.salm.n_cpu_moe/batched, which the larger
# quants on smaller cards would need. Ordered: first match wins.
TIER_BLACKWELL = Tier("blackwell-nvfp4", 0, _decoder("NVFP4"), None, verified=False)
TIERS: tuple[Tier, ...] = (
    Tier("48gb-q8_0", 45000, _decoder("Q8_0"), None, verified=False),
    Tier("32gb-iq4_nl_l", 30000, _decoder("IQ4_NL_L"), None, verified=False),
    Tier(
        "24gb-iq4_nl",
        TIER_24GB_MIN_MB,
        _decoder("AD-IQ4_NL"),
        "0c47d4ab6c783e7606ec03714b354fab0b2052027a7ea1991e7e2d3009c345ad",
        verified=True,
        profiles=PROFILES_24GB,
    ),
    Tier("16gb-q2_0_m", 15000, _decoder("Q2_0_M"), None, verified=False),
)
TIER_CPU = Tier(
    "cpu-iq4_nl",
    0,
    _decoder("AD-IQ4_NL"),
    "0c47d4ab6c783e7606ec03714b354fab0b2052027a7ea1991e7e2d3009c345ad",
    verified=True,
    profiles={"cpu": PROFILE_CPU},
)
MIN_COMPUTE_CAP = 8.0
BLACKWELL_COMPUTE_CAP = 10.0


# --------------------------------------------------------------------------- #
# Host CPU & RAM probes (LS-20)


def cpu_has_avx2(cpuinfo_path: str = "/proc/cpuinfo") -> bool:
    """True when /proc/cpuinfo reports the ``avx2`` flag (x86-64-v3, required
    by the prebuilt NeMo-Speech.cpp CPU release)."""
    try:
        text = Path(cpuinfo_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if sep and key.strip().lower() == "flags":
            if "avx2" in rest.lower().split():
                return True
    return False


def available_ram_mb(
    meminfo_path: str = "/proc/meminfo",
    cgroup_max_path: str = "/sys/fs/cgroup/memory.max",
    cgroup_current_path: str = "/sys/fs/cgroup/memory.current",
    cgroup_stat_path: str = "/sys/fs/cgroup/memory.stat",
) -> int | None:
    """Available RAM in MiB for admitting a CPU Nemotron job, bounded by both
    host ``MemAvailable`` and the container's cgroup v2 memory limit (excluding
    reclaimable/reusable page cache ``file`` bytes). Returns None when
    /proc/meminfo cannot be read."""
    host_avail_mb: int | None = None
    try:
        for line in Path(meminfo_path).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    host_avail_mb = int(parts[1]) // 1024
                break
    except (OSError, ValueError):
        return None
    if host_avail_mb is None:
        return None

    try:
        raw_max = Path(cgroup_max_path).read_text(encoding="utf-8").strip()
        if raw_max and raw_max != "max":
            limit_bytes = int(raw_max)
            cur_bytes = int(Path(cgroup_current_path).read_text(encoding="utf-8").strip())
            file_bytes = 0
            try:
                for line in Path(cgroup_stat_path).read_text(encoding="utf-8").splitlines():
                    if line.startswith("file "):
                        file_bytes = int(line.split()[1])
                        break
            except (OSError, ValueError, IndexError):
                file_bytes = 0
            anon_used = max(0, cur_bytes - file_bytes)
            cg_avail_mb = max(0, (limit_bytes - anon_used) // (1024 * 1024))
            return min(host_avail_mb, cg_avail_mb)
    except (OSError, ValueError):
        pass
    return host_avail_mb


def total_ram_mb(
    meminfo_path: str = "/proc/meminfo",
    cgroup_max_path: str = "/sys/fs/cgroup/memory.max",
) -> int | None:
    """Total usable RAM in MiB (host MemTotal capped at cgroup memory.max)."""
    host_total_mb: int | None = None
    try:
        for line in Path(meminfo_path).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    host_total_mb = int(parts[1]) // 1024
                break
    except (OSError, ValueError):
        return None
    if host_total_mb is None:
        return None
    try:
        raw_max = Path(cgroup_max_path).read_text(encoding="utf-8").strip()
        if raw_max and raw_max != "max":
            cg_max_mb = int(raw_max) // (1024 * 1024)
            return min(host_total_mb, cg_max_mb)
    except (OSError, ValueError):
        pass
    return host_total_mb


def effective_cpu_threads(
    requested: int,
    cgroup_cpu_max_path: str = "/sys/fs/cgroup/cpu.max",
) -> int:
    """Cap ``requested`` threads at the container's cgroup CPU quota (if set)
    and ``os.cpu_count()``."""
    cap = os.cpu_count() or max(1, int(requested))
    try:
        raw = Path(cgroup_cpu_max_path).read_text(encoding="utf-8").strip().split()
        if len(raw) >= 2 and raw[0] != "max":
            quota, period = int(raw[0]), int(raw[1])
            if quota > 0 and period > 0:
                cap = min(cap, max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return max(1, min(int(requested), cap))


# --------------------------------------------------------------------------- #
# Errors -> HTTP status (live_stt/diarize_http.py maps these)


class NemotronError(Exception):
    status = 503
    retry_after_sec: int | None = None

    def __init__(self, message: str, *, retry_after_sec: int | None = None) -> None:
        super().__init__(message)
        if retry_after_sec is not None:
            self.retry_after_sec = retry_after_sec


class NemotronUnavailable(NemotronError):
    """Not deployable on this box (no GPU, unsupported tier, binary or model
    files missing). 503 without Retry-After: retrying will not help."""


class NemotronRetryable(NemotronError):
    """Sidecar crashed or failed mid-request. 503 + Retry-After."""

    retry_after_sec = RETRY_AFTER_CRASH_SEC


class NemotronTimeout(NemotronError):
    """Job waited longer than nemotron_job_timeout_sec in total."""

    retry_after_sec = RETRY_AFTER_TIMEOUT_SEC


class NemotronBadRequest(NemotronError):
    status = 400


class NemotronBadResponse(NemotronError):
    status = 502


class _SidecarFailure(Exception):
    def __init__(self, message: str, *, crashed: bool = False, timed_out: bool = False) -> None:
        super().__init__(message)
        self.crashed = crashed
        self.timed_out = timed_out


class _Preempted(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tier / plan / audio (pure)


def select_tier(total_mb: int | None, compute_cap: float | None) -> Tier:
    """Raise NemotronUnavailable unless the card maps to a VERIFIED tier."""
    if total_mb is None or compute_cap is None:
        raise NemotronUnavailable(
            "Nemotron-3.5-Transcribe needs a CUDA GPU (nvidia-smi reported no total VRAM / compute capability)"
        )
    if compute_cap < MIN_COMPUTE_CAP:
        raise NemotronUnavailable(
            f"Nemotron-3.5-Transcribe needs compute capability >= {MIN_COMPUTE_CAP}; this GPU is {compute_cap}"
        )
    if compute_cap >= BLACKWELL_COMPUTE_CAP:
        tier: Tier | None = TIER_BLACKWELL
    else:
        tier = next((t for t in TIERS if total_mb >= t.min_total_mb), None)
    if tier is None:
        raise NemotronUnavailable(
            f"Nemotron-3.5-Transcribe is unsupported on a {total_mb} MiB GPU (smallest tier needs 16 GB)"
        )
    if not tier.verified or not tier.profiles:
        raise NemotronUnavailable(
            f"Nemotron-3.5-Transcribe tier {tier.name!r} ({tier.decoder_filename}) is declared but untested; "
            "only the 24 GB IQ4_NL tier ships configured"
        )
    return tier


def plan_job(settings: Any, duration_sec: float, *, flavor: str = "cuda") -> tuple[str, list[tuple[float, float]]]:
    """``(profile name, [(start_sec, end_sec), ...])`` for a recording."""
    if duration_sec <= 0:
        raise NemotronBadRequest("audio is empty")
    if duration_sec > float(settings.max_call_sec):
        raise NemotronBadRequest(
            f"audio is {duration_sec:.0f}s, over the {float(settings.max_call_sec):.0f}s max_call_sec limit"
        )
    if flavor == "cpu":
        cpu_max = float(getattr(settings, "nemotron_cpu_max_sec", 1620.0))
        if duration_sec <= cpu_max:
            return "cpu", [(0.0, duration_sec)]
        win_sec = min(float(settings.nemotron_window_sec), cpu_max)
        overlap_sec = float(settings.nemotron_window_overlap_sec)
        if overlap_sec >= win_sec:
            overlap_sec = max(0.0, win_sec * 0.125)
        windows = seams.plan_windows(duration_sec, win_sec, overlap_sec)
        longest = max(e - s for s, e in windows)
        if longest > cpu_max + 1e-6:
            raise NemotronBadRequest("window plan exceeds nemotron_cpu_max_sec")
        return "cpu", windows

    if duration_sec <= float(settings.nemotron_short_max_sec):
        return "short", [(0.0, duration_sec)]
    if duration_sec <= float(settings.nemotron_long_max_sec):
        return "long", [(0.0, duration_sec)]
    windows = seams.plan_windows(
        duration_sec, float(settings.nemotron_window_sec), float(settings.nemotron_window_overlap_sec)
    )
    longest = max(e - s for s, e in windows)
    if longest > float(settings.nemotron_long_max_sec) + 1e-6:  # guarded by Settings validation too
        raise NemotronBadRequest("window plan exceeds nemotron_long_max_sec")
    return "long", windows


@dataclass(frozen=True)
class WavAudio:
    raw: bytes
    nchannels: int
    sampwidth: int
    framerate: int
    nframes: int
    _data_offset: int
    _data_len: int

    @property
    def duration_sec(self) -> float:
        return self.nframes / self.framerate if self.framerate else 0.0

    def window_bytes(self, start_sec: float, end_sec: float) -> bytes:
        """A standalone WAV for ``[start_sec, end_sec)``; the whole original
        file when the window is the whole recording."""
        f0 = max(0, int(round(start_sec * self.framerate)))
        f1 = min(self.nframes, int(round(end_sec * self.framerate)))
        if f0 == 0 and f1 >= self.nframes:
            return self.raw
        frame = self.nchannels * self.sampwidth
        pcm = self.raw[self._data_offset + f0 * frame : self._data_offset + f1 * frame]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(self.nchannels)
            w.setsampwidth(self.sampwidth)
            w.setframerate(self.framerate)
            w.writeframes(pcm)
        return buf.getvalue()


def parse_wav(data: bytes) -> WavAudio:
    """PCM WAV only: windowing slices frames, and the duration decides the
    profile, so a format the stdlib cannot read is refused up front."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            nch, sw, rate, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
    except (wave.Error, EOFError) as exc:
        raise NemotronBadRequest(f"Nemotron needs a PCM WAV upload: {exc}") from exc
    if not rate or not nch or not sw:
        raise NemotronBadRequest("WAV header has no sample rate / channels / sample width")
    off, length = _find_data_chunk(data)
    n = min(n, length // (nch * sw))
    return WavAudio(data, nch, sw, rate, n, off, length)


def _find_data_chunk(data: bytes) -> tuple[int, int]:
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if cid == b"data":
            return pos + 8, min(size, len(data) - pos - 8)
        pos += 8 + size + (size & 1)
    raise NemotronBadRequest("WAV has no data chunk")


def build_chat_payload(wav_bytes: bytes) -> dict[str, Any]:
    """The one request shape ever sent to the sidecar: a chat completion with
    the explicit speaker instruction and word timestamps."""
    return {
        "model": "salm",
        "temperature": 0,
        "timestamp_granularities": ["word"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SPEAKER_INSTRUCTION},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(wav_bytes).decode("ascii"), "format": "wav"},
                    },
                ],
            }
        ],
    }


def salm_config_yaml(profile: Profile, *, threads: int | None = None) -> str:
    lines = ["# written by live_stt/nemotron.py", "asr:"]
    if threads is not None:
        lines += ["  backend:", f"    threads: {int(threads)}"]
    lines.append("  salm:")
    lines += [f"    {k}: {v}" for k, v in profile.salm.items()]
    return "\n".join(lines) + "\n"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# Service


@dataclass(eq=False)
class _Job:
    id: int
    flavor: str
    profile: str
    windows: list[tuple[float, float]]
    deadline: float
    budget_sec: float = 1800.0
    preempted: bool = False
    attempts: int = 0


@dataclass
class _Proc:
    popen: subprocess.Popen
    port: int
    flavor: str
    profile: Profile
    workdir: str
    reservation: vram.Admission | None
    threads: int | None = None


class NemotronService:
    STATES = ("unloaded", "loading", "ready", "busy")

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.RLock())
        self._queue: deque[_Job] = deque()
        self._running: dict[int, _Job] = {}
        self._ids = itertools.count(1)
        self._proc: _Proc | None = None
        self._loading = False
        self._loading_flavor: str | None = None
        self._load_preempted = False
        self._suspended = 0
        self._active_live_calls = 0
        self._retire = False
        self._served_since_load = 0
        self._idle_since: float | None = None
        self._idle_timer: threading.Timer | None = None
        self._idle_token = 0
        self._tier_cache: Tier | None = None
        self.loads_total = 0
        self.preemptions_total = 0
        self.requeues_total = 0
        self.crashes_total = 0
        self.timeouts_total = 0
        self.last_error: str | None = None

    # -- introspection ------------------------------------------------------ #

    def _alive_locked(self) -> bool:
        return self._proc is not None and self._proc.popen.poll() is None

    @property
    def state(self) -> str:
        with self._cond:
            if self._loading:
                return "loading"
            if not self._alive_locked():
                return "unloaded"
            return "busy" if self._running else "ready"

    def is_active(self) -> bool:
        """A job is running or a sidecar is loading for one -- i.e. a live
        call short on VRAM/CPU should preempt rather than be rejected."""
        with self._cond:
            return bool(self._running) or self._loading

    def is_active_on(self, flavor: str) -> bool:
        """True when a job or load is currently active on ``flavor`` ('cpu' or 'cuda')."""
        with self._cond:
            if self._loading and self._loading_flavor == flavor:
                return True
            if self._proc is not None and self._alive_locked() and self._proc.flavor == flavor and self._running:
                return True
            return any(j.flavor == flavor for j in self._running.values())

    def snapshot(self, settings: Any = None) -> dict[str, Any]:
        active_flavor: str | None = None
        if settings is not None:
            try:
                from live_stt.backends import installer

                active = installer.get_active_backend(settings.db_path, BACKEND_ENGINE)
                if active is not None:
                    active_flavor = active.flavor
                elif getattr(settings, "nemotron_bin", None):
                    active_flavor = "cuda"
            except Exception:  # noqa: BLE001
                active_flavor = None

        with self._cond:
            alive = self._alive_locked()
            proc_flavor = self._proc.flavor if self._proc is not None and alive else None
            device = proc_flavor or active_flavor
            threads = (
                self._proc.threads
                if self._proc is not None and alive and self._proc.flavor == "cpu"
                else (
                    effective_cpu_threads(int(getattr(settings, "nemotron_cpu_threads", 16)))
                    if device == "cpu"
                    else None
                )
            )
            idle_sec = (
                round(time.monotonic() - self._idle_since, 1)
                if alive and not self._running and self._idle_since is not None
                else None
            )
            tier_name = (
                TIER_CPU.name
                if device == "cpu"
                else (self._tier_cache.name if self._tier_cache is not None else None)
            )
            snap: dict[str, Any] = {
                "model": MODEL_KEY,
                "state": self.state,
                "device": device,
                "threads": threads,
                "profile": self._proc.profile.name if self._proc is not None and alive else None,
                "pid": self._proc.popen.pid if self._proc is not None and alive else None,
                "tier": tier_name,
                "queue_depth": len(self._queue),
                "running_jobs": len(self._running),
                "live_calls_active": self._active_live_calls,
                "served_since_load": self._served_since_load,
                "idle_sec": idle_sec,
                "loads_total": self.loads_total,
                "preemptions_total": self.preemptions_total,
                "requeues_total": self.requeues_total,
                "crashes_total": self.crashes_total,
                "timeouts_total": self.timeouts_total,
                "last_error": self.last_error,
            }
        if device == "cpu":
            snap["available_ram_mb"] = available_ram_mb()
            snap["total_ram_mb"] = total_ram_mb()
            snap["required_ram_mb"] = int(getattr(settings, "nemotron_cpu_ram_mb", 24000)) if settings is not None else 24000
        return snap

    def _update_metrics_locked(self) -> None:
        metrics.nemotron_queue_depth.set(len(self._queue))
        metrics.nemotron_jobs_running.set(len(self._running))
        cur = self.state
        for s in self.STATES:
            metrics.nemotron_sidecar_state.labels(state=s).set(1 if s == cur else 0)

    # -- configuration ------------------------------------------------------ #

    def resolve_tier(self) -> Tier:
        if self._tier_cache is None:
            self._tier_cache = select_tier(gpu.total_vram_mb(), gpu.compute_cap())
        return self._tier_cache

    def resolve_tier_for_flavor(self, flavor: str) -> Tier:
        if flavor == "cpu":
            if not cpu_has_avx2():
                raise NemotronUnavailable(
                    "Nemotron-3.5-Transcribe on CPU requires AVX2 instructions (x86-64-v3)"
                )
            return TIER_CPU
        return self.resolve_tier()

    @staticmethod
    def resolve_runtime(settings: Any) -> tuple[str, str]:
        """Return ``(flavor, binary_path)`` from the active ``nemo_speech``
        backend install (or ``settings.nemotron_bin`` override)."""
        from live_stt.backends import installer

        active = installer.get_active_backend(settings.db_path, BACKEND_ENGINE)
        if active is not None:
            flavor = active.flavor
            path = Path(settings.nemotron_bin) if settings.nemotron_bin else Path(active.path) / "bin" / "nemo-speech"
        elif settings.nemotron_bin:
            flavor = "cuda"
            path = Path(settings.nemotron_bin)
        else:
            raise NemotronUnavailable(
                "no active nemo_speech backend: POST /api/backends/install "
                '{"engine": "nemo_speech", "flavor": "cpu" | "cuda"}, then /api/backends/activate'
            )
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise NemotronUnavailable(f"nemo-speech binary not found or not executable: {path}")
        return flavor, str(path)

    @staticmethod
    def resolve_binary(settings: Any) -> str:
        _, binary = NemotronService.resolve_runtime(settings)
        return binary

    @staticmethod
    def active_device(settings: Any) -> str:
        """Return the active device ('cpu' or 'cuda') for Nemotron, defaulting
        to 'cuda' if only nemotron_bin is set."""
        try:
            from live_stt.backends import installer

            active = installer.get_active_backend(settings.db_path, BACKEND_ENGINE)
            if active is not None:
                return active.flavor
        except Exception:  # noqa: BLE001
            pass
        return "cuda"

    @staticmethod
    def model_paths(settings: Any, tier: Tier) -> tuple[str, str]:
        root = Path(settings.nemotron_models_dir or settings.models_dir)
        decoder, encoder = root / tier.decoder_filename, root / ENCODER_FILENAME
        missing = [str(p) for p in (decoder, encoder) if not p.is_file()]
        if missing:
            raise NemotronUnavailable(
                f"Nemotron model files missing: {', '.join(missing)} "
                f"(fetch from {HF_REPO}: scripts/fetch_model.sh nemotron-3.5-transcribe-30b-a4b nemotron-3.5-transcribe-30b-a4b-encoder)"
            )
        return str(decoder), str(encoder)

    def on_flavor_activated(self, new_flavor: str) -> bool:
        """Called when the active ``nemo_speech`` flavor changes.
        Stops an idle sidecar immediately if it was running a different flavor;
        marks a busy sidecar ``_retire = True`` so running jobs finish on the
        old flavor and the next load uses ``new_flavor``."""
        old: _Proc | None = None
        with self._cond:
            if self._proc is None or not self._alive_locked():
                return False
            if self._proc.flavor == new_flavor:
                return False
            if self._running or self._loading:
                self._retire = True
                self._cond.notify_all()
                return False
            logger.info("stopping idle nemotron sidecar on flavor switch (%s -> %s)", self._proc.flavor, new_flavor)
            old = self._detach_locked("flavor_switch")
        if old is not None:
            self._terminate(old)
            return True
        return False

    # -- public entry point ------------------------------------------------- #

    @staticmethod
    def job_timeout_budget_sec(settings: Any, duration_sec: float, *, flavor: str) -> float:
        base = float(settings.nemotron_job_timeout_sec)
        if flavor == "cpu":
            per_sec = float(getattr(settings, "nemotron_cpu_timeout_per_audio_sec", 3.0))
            return base + max(0.0, per_sec * duration_sec)
        return base

    def transcribe(self, settings: Any, audio: WavAudio, *, model_key: str = MODEL_KEY) -> dict[str, Any]:
        """Blocking. Queue the recording, run it (possibly several times if
        preempted), and return the house diarization JSON."""
        flavor, binary = self.resolve_runtime(settings)
        tier = self.resolve_tier_for_flavor(flavor)
        decoder, encoder = self.model_paths(settings, tier)
        profile_name, windows = plan_job(settings, audio.duration_sec, flavor=flavor)
        budget_sec = self.job_timeout_budget_sec(settings, audio.duration_sec, flavor=flavor)
        job = _Job(
            id=next(self._ids),
            flavor=flavor,
            profile=profile_name,
            windows=windows,
            deadline=time.monotonic() + budget_sec,
            budget_sec=budget_sec,
        )
        launch = (binary, decoder, encoder)
        with self._cond:
            self._queue.append(job)
            self._update_metrics_locked()
            self._cond.notify_all()
        logger.info(
            "nemotron job %d queued: %.0fs audio, flavor=%s, profile=%s, %d window(s)",
            job.id, audio.duration_sec, flavor, profile_name, len(windows),
        )
        try:
            results = self._run(job, settings, tier, launch, audio)
        finally:
            with self._cond:
                if job in self._queue:
                    self._queue.remove(job)
                self._update_metrics_locked()
                self._cond.notify_all()

        # Per window, before stitching: seam votes pair words by speaker label.
        repaired_windows: list[tuple[float, float]] = []
        for i, (w_start, w_end, w_words, w_turns) in enumerate(results):
            fixed, repaired = seams.repair_collapsed_word_speakers(w_words, w_turns)
            if repaired:
                results[i] = (w_start, w_end, fixed, w_turns)
                repaired_windows.append((w_start, w_end))
        if repaired_windows:
            metrics.nemotron_word_speaker_repairs_total.inc(len(repaired_windows))
            logger.warning(
                "nemotron job %d: word speaker tags collapsed to one label in %d window(s); "
                "relabelled words from the model's diarization turns",
                job.id, len(repaired_windows),
            )

        if len(results) == 1:
            words, turns, seam_reports = results[0][2], results[0][3], []
        else:
            words, turns, seam_reports = seams.stitch_windows(results)
        if not words:
            raise NemotronBadRequest("Nemotron returned no words -- is the audio silent?")
        warnings = seams.seam_warnings(seam_reports, min_votes=int(settings.nemotron_seam_min_votes))
        if warnings:
            metrics.nemotron_low_vote_seams_total.inc(len(warnings))
        doc = seams.words_to_house_json(
            words, model=model_key, turns=turns, duration_sec=audio.duration_sec, warnings=warnings
        )
        if doc["num_speakers"] >= MAX_SPEAKERS and len(results) == 1:
            doc.setdefault("warnings", []).append(
                f"detected speaker count ({doc['num_speakers']}) is at this model's labelling maximum "
                f"({MAX_SPEAKERS}); if the real audio has more speakers, some were merged"
            )
        for w_start, w_end in repaired_windows:
            doc.setdefault("warnings", []).append(
                f"word speaker tags {w_start:.0f}-{w_end:.0f}s all named one speaker while the model's "
                "diarization found several; words were relabelled from the diarization turns, but "
                "per-speaker text in this span is unreliable (word timestamps may be misaligned too)"
            )
        if seam_reports:
            doc["seams"] = [
                {"at_sec": round(s.midpoint, 1), "matched_words": s.matched_words, "new_speakers": len(s.new_speakers)}
                for s in seam_reports
            ]
        return doc

    # -- job execution ------------------------------------------------------ #

    def _refresh_job_runtime(
        self, job: _Job, settings: Any, tier: Tier, launch: tuple[str, str, str], audio: WavAudio
    ) -> tuple[Tier, Profile, tuple[str, str, str]]:
        """Re-resolve active flavor before acquiring a sidecar so a flavor
        switch while jobs are queued takes effect on the next load."""
        cur_flavor, cur_binary = self.resolve_runtime(settings)
        if cur_flavor != job.flavor:
            tier = self.resolve_tier_for_flavor(cur_flavor)
            decoder, encoder = self.model_paths(settings, tier)
            profile_name, windows = plan_job(settings, audio.duration_sec, flavor=cur_flavor)
            with self._cond:
                job.flavor = cur_flavor
                job.profile = profile_name
                job.windows = windows
            launch = (cur_binary, decoder, encoder)
        return tier, tier.profiles[job.profile], launch

    def _run(
        self, job: _Job, settings: Any, tier: Tier, launch: tuple[str, str, str], audio: WavAudio
    ) -> list[tuple[float, float, list[seams.Word], list[seams.Turn]]]:
        while True:
            tier, profile, launch = self._refresh_job_runtime(job, settings, tier, launch, audio)
            self._acquire(job, settings, profile, launch)
            results: list[tuple[float, float, list[seams.Word], list[seams.Turn]]] = []
            try:
                for i, (start, end) in enumerate(job.windows):
                    if i > 0:
                        # Every multi-window request gets a fresh process per
                        # window: the word-timestamp buffer only grows.
                        self._restart_for_job(job, settings, profile, launch)
                    payload = build_chat_payload(audio.window_bytes(start, end))
                    doc = self._post_chat(job, payload)
                    results.append(
                        (
                            start,
                            end,
                            seams.extract_words(doc, offset_sec=start),
                            seams.extract_turns(doc, offset_sec=start),
                        )
                    )
            except (_SidecarFailure, _Preempted) as exc:
                requeue = job.preempted or isinstance(exc, _Preempted)
                with self._cond:
                    self._running.pop(job.id, None)
                    if requeue:
                        job.preempted = False
                        job.attempts += 1
                        self._queue.appendleft(job)
                        self.requeues_total += 1
                        metrics.nemotron_requeues_total.inc()
                    else:
                        self.last_error = str(exc)
                    old = self._after_job_locked(settings, served=False)
                self._terminate_maybe(old)
                if requeue:
                    logger.info("nemotron job %d re-queued after preemption (attempt %d)", job.id, job.attempts + 1)
                    continue
                if isinstance(exc, _SidecarFailure) and exc.timed_out:
                    with self._cond:
                        self.timeouts_total += 1
                    metrics.nemotron_job_timeouts_total.inc()
                    raise NemotronTimeout(f"Nemotron job timed out while running: {exc}") from exc
                raise NemotronRetryable(f"Nemotron sidecar failed: {exc}") from exc
            except seams.NemotronResponseError as exc:
                with self._cond:
                    self._running.pop(job.id, None)
                    self.last_error = str(exc)
                    old = self._after_job_locked(settings, served=True)
                self._terminate_maybe(old)
                raise NemotronBadResponse(str(exc)) from exc
            except BaseException:
                with self._cond:
                    self._running.pop(job.id, None)
                    old = self._after_job_locked(settings, served=False)
                self._terminate_maybe(old)
                raise
            with self._cond:
                self._running.pop(job.id, None)
                if profile.name == "long" or len(job.windows) > 1:
                    self._retire = True
                old = self._after_job_locked(settings, served=True)
            self._terminate_maybe(old)
            return results

    def _terminate_maybe(self, old: "_Proc | None") -> None:
        if old is not None:
            self._terminate(old)

    def _after_job_locked(self, settings: Any, *, served: bool) -> "_Proc | None":
        """Book-keeping when a job leaves the running set. Returns a detached
        process the caller must ``_terminate`` once the lock is released
        (recycling), or None."""
        if served:
            self._served_since_load += 1
            n = int(settings.nemotron_recycle_after_requests)
            if n > 0 and self._served_since_load >= n:
                self._retire = True
        old = None
        if not self._running and not self._loading and self._alive_locked():
            if self._retire:
                old = self._detach_locked("recycled")
            else:
                self._idle_since = time.monotonic()
                self._schedule_idle_locked(float(settings.nemotron_idle_timeout_sec))
        self._update_metrics_locked()
        self._cond.notify_all()
        return old

    def _can_join_locked(self, job: _Job) -> bool:
        p = self._proc
        return (
            p is not None
            and self._alive_locked()
            and not self._retire
            and p.flavor == job.flavor
            and p.profile.name == job.profile
            and len(self._running) < p.profile.max_sequences
        )

    def _acquire(self, job: _Job, settings: Any, profile: Profile, launch: tuple[str, str, str]) -> None:
        """Return once ``job`` is running on a ready sidecar of its profile."""
        while True:
            with self._cond:
                while True:
                    now = time.monotonic()
                    if now >= job.deadline:
                        if job in self._queue:
                            self._queue.remove(job)
                        self.timeouts_total += 1
                        metrics.nemotron_job_timeouts_total.inc()
                        self._update_metrics_locked()
                        raise NemotronTimeout(
                            f"Nemotron job waited over {job.budget_sec:.0f}s "
                            "(queued behind other jobs or live calls holding the GPU/CPU)"
                        )
                    head = bool(self._queue) and self._queue[0] is job
                    live_holding = (
                        vram.has_other_gpu_work(exclude_model_key=MODEL_KEY)
                        if job.flavor == "cuda"
                        else self._active_live_calls > 0
                    )
                    if head and not self._suspended and not self._loading and not live_holding:
                        if self._can_join_locked(job):
                            self._queue.popleft()
                            self._running[job.id] = job
                            self._cancel_idle_locked()
                            self._idle_since = None
                            self._update_metrics_locked()
                            self._cond.notify_all()
                            return
                        if not self._running:
                            break
                    self._cond.wait(timeout=min(_WAIT_POLL_SEC, job.deadline - now))
                live_holding = (
                    vram.has_other_gpu_work(exclude_model_key=MODEL_KEY)
                    if job.flavor == "cuda"
                    else self._active_live_calls > 0
                )
                if live_holding:
                    # A live call holds VRAM or CPU priority: wait for it to finish.
                    self._cond.wait(timeout=min(_ADMIT_RETRY_SEC, max(0.0, job.deadline - time.monotonic())))
                    continue
                self._loading = True
                self._loading_flavor = job.flavor
                self._load_preempted = False
                old = self._detach_locked("profile_switch") if self._proc is not None else None
                self._update_metrics_locked()
            if old is not None:
                self._terminate(old)
            try:
                loaded = self._load(settings, job.flavor, profile, launch, job.deadline)
            finally:
                with self._cond:
                    self._loading = False
                    self._loading_flavor = None
                    self._update_metrics_locked()
                    self._cond.notify_all()
            if not loaded:
                with self._cond:
                    self._cond.wait(timeout=min(_ADMIT_RETRY_SEC, max(0.0, job.deadline - time.monotonic())))

    def _restart_for_job(self, job: _Job, settings: Any, profile: Profile, launch: tuple[str, str, str]) -> None:
        with self._cond:
            if job.preempted:
                raise _Preempted()
            self._loading = True
            self._loading_flavor = job.flavor
            self._load_preempted = False
            old = self._detach_locked("recycled")
            self._update_metrics_locked()
        if old is not None:
            self._terminate(old)
        try:
            ok = self._load(settings, job.flavor, profile, launch, job.deadline)
        finally:
            with self._cond:
                self._loading = False
                self._loading_flavor = None
                self._update_metrics_locked()
                self._cond.notify_all()
        if not ok:
            raise _Preempted()

    def _load(
        self, settings: Any, flavor: str, profile: Profile, launch: tuple[str, str, str], deadline: float
    ) -> bool:
        """Admit + spawn + wait for ready. False = VRAM/RAM not available or the
        load was preempted (caller retries); raises on a hard failure."""
        admission: vram.Admission | None = None
        threads: int | None = None
        if flavor == "cuda":
            admission = vram.admit_exclusive(
                settings,
                model_key=MODEL_KEY,
                need_mb=profile.vram_mb,
                margin_mb=int(settings.nemotron_vram_margin_mb),
            )
            if admission.free_mb is not None:
                metrics.gpu_free_vram_mb.set(admission.free_mb)
            if not admission.ok:
                logger.info("nemotron load deferred: %s", admission.message("Nemotron"))
                with self._cond:
                    self.last_error = admission.message("Nemotron")
                return False
        else:
            avail_mb = available_ram_mb()
            need_ram_mb = int(getattr(settings, "nemotron_cpu_ram_mb", 24000))
            if avail_mb is not None and avail_mb < need_ram_mb:
                msg = (
                    f"insufficient available RAM for Nemotron on CPU: "
                    f"need {need_ram_mb} MiB, available {avail_mb} MiB"
                )
                logger.info("nemotron CPU load deferred: %s", msg)
                with self._cond:
                    self.last_error = msg
                return False
            threads = effective_cpu_threads(int(getattr(settings, "nemotron_cpu_threads", 16)))

        binary, decoder, encoder = launch
        workdir = tempfile.mkdtemp(prefix="lstt-nemotron-")
        try:
            cfg = Path(workdir) / "salm.yaml"
            cfg.write_text(salm_config_yaml(profile, threads=threads))
            port = int(settings.nemotron_port) or _free_port()
            cmd = [
                binary, "serve",
                "--asr-model", decoder,
                "--salm-encoder", encoder,
                "--device", flavor,
                "--host", "127.0.0.1",
                "--port", str(port),
                "--config", str(cfg),
                "--no-ui",
                "--max-upload-mb", str(MAX_UPLOAD_MB),
                "--read-timeout", str(SIDE_TIMEOUT_SEC),
                "--write-timeout", str(SIDE_TIMEOUT_SEC),
            ]
            log = open(Path(workdir) / "server.log", "wb")  # noqa: SIM115 -- owned by the child
            try:
                popen = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            finally:
                log.close()
        except Exception as exc:
            if admission is not None:
                admission.release()
            shutil.rmtree(workdir, ignore_errors=True)
            metrics.nemotron_sidecar_events_total.labels(event="load_failed").inc()
            raise NemotronUnavailable(f"could not start nemo-speech: {exc}") from exc

        proc = _Proc(popen, port, flavor, profile, workdir, admission, threads=threads)
        with self._cond:
            self._proc = proc
            self._retire = False
            self._served_since_load = 0
        t0 = time.monotonic()
        timeout = min(float(settings.nemotron_ready_timeout_sec), max(0.0, deadline - t0))
        try:
            self._wait_ready(proc, timeout)
        except _SidecarFailure as exc:
            with self._cond:
                preempted = self._load_preempted
                old = self._detach_if_locked(proc, "load_failed" if not preempted else None)
                self.last_error = None if preempted else str(exc)
            tail = _log_tail(proc.workdir)
            if old is not None:
                self._terminate(old)
            if preempted:
                return False
            logger.error("nemotron sidecar failed to load: %s\n%s", exc, tail)
            if time.monotonic() >= deadline:
                raise NemotronTimeout(f"Nemotron sidecar did not become ready before the job deadline: {exc}") from exc
            raise NemotronRetryable(f"Nemotron sidecar failed to load: {exc}") from exc
        with self._cond:
            self.loads_total += 1
        metrics.nemotron_sidecar_events_total.labels(event="load").inc()
        logger.info(
            "nemotron sidecar ready: flavor=%s profile=%s pid=%d port=%d in %.1fs",
            flavor, profile.name, popen.pid, port, time.monotonic() - t0,
        )
        return True

    @staticmethod
    def _wait_ready(proc: _Proc, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{proc.port}{READY_PATH}"
        while True:
            rc = proc.popen.poll()
            if rc is not None:
                raise _SidecarFailure(f"nemo-speech exited rc={rc} before ready", crashed=True)
            try:
                with urllib.request.urlopen(url, timeout=2.0) as r:  # noqa: S310 -- loopback
                    if json.loads(r.read() or b"{}").get("ready"):
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise _SidecarFailure(f"nemo-speech not ready after {timeout:.0f}s", timed_out=True)
            time.sleep(0.25)

    def _post_chat(self, job: _Job, payload: dict[str, Any]) -> dict[str, Any]:
        with self._cond:
            proc = self._proc
            if job.preempted or proc is None or not self._alive_locked():
                if job.preempted:
                    raise _Preempted()
                raise _SidecarFailure("nemo-speech is not running", crashed=True)
        remaining = job.deadline - time.monotonic()
        if remaining <= 0:
            raise _SidecarFailure("job deadline reached", timed_out=True)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{proc.port}{CHAT_PATH}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=remaining) as r:  # noqa: S310 -- loopback
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace")
            if job.preempted:
                raise _Preempted() from exc
            crashed = self._check_crash(proc)
            raise _SidecarFailure(f"HTTP {exc.code}: {detail}", crashed=crashed) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if job.preempted:
                raise _Preempted() from exc
            timed_out = isinstance(getattr(exc, "reason", exc), (TimeoutError, socket.timeout))
            crashed = self._check_crash(proc)
            raise _SidecarFailure(f"{type(exc).__name__}: {exc}", crashed=crashed, timed_out=timed_out and not crashed) from exc

    def _check_crash(self, proc: _Proc) -> bool:
        """Did the sidecar process die under this request? Detach it if so,
        so the next request loads a fresh one."""
        try:
            proc.popen.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return False
        with self._cond:
            old = self._detach_if_locked(proc, "crash")
            if old is not None:
                self.crashes_total += 1
        if old is not None:
            logger.error("nemotron sidecar crashed (rc=%s):\n%s", proc.popen.returncode, _log_tail(proc.workdir))
            self._terminate(old)
        return True

    # -- process control ---------------------------------------------------- #

    def _detach_locked(self, event: str | None) -> _Proc | None:
        old, self._proc = self._proc, None
        self._cancel_idle_locked()
        self._idle_since = None
        self._retire = False
        self._served_since_load = 0
        if old is not None and event:
            metrics.nemotron_sidecar_events_total.labels(event=event).inc()
        self._update_metrics_locked()
        return old

    def _detach_if_locked(self, proc: _Proc, event: str | None) -> _Proc | None:
        if self._proc is proc:
            return self._detach_locked(event)
        return None

    @staticmethod
    def _terminate(proc: _Proc, grace_sec: float = 5.0) -> None:
        """SIGTERM, SIGKILL after ``grace_sec``; release the VRAM reservation
        and the work dir. Never called with the lock held. An idle sidecar
        exits ~0.7 s after SIGTERM; one mid-request does not exit promptly
        (measured on homesrv3: the 5 s grace ran out), so preemption uses a
        short grace -- the job reruns from scratch anyway."""
        p = proc.popen
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=grace_sec)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            logger.exception("error stopping nemotron sidecar pid=%s", p.pid)
        finally:
            if proc.reservation is not None:
                proc.reservation.release()
            shutil.rmtree(proc.workdir, ignore_errors=True)

    def _cancel_idle_locked(self) -> None:
        self._idle_token += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_locked(self, timeout: float) -> None:
        self._cancel_idle_locked()
        token = self._idle_token
        if timeout <= 0:
            if not self._queue:
                # Unload right away; the terminate runs off-lock on a thread.
                old = self._detach_locked("unload_idle")
                if old is not None:
                    threading.Thread(target=self._terminate, args=(old,), daemon=True).start()
            return
        self._idle_timer = threading.Timer(timeout, self._on_idle, args=(token,))
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle(self, token: int) -> None:
        with self._cond:
            if token != self._idle_token or self._running or self._loading or not self._alive_locked():
                return
            logger.info("nemotron sidecar idle: unloading")
            old = self._detach_locked("unload_idle")
        if old is not None:
            self._terminate(old)

    def stop_if_idle(self) -> bool:
        """Evict an idle CUDA sidecar so other GPU work can have the card. Never
        blocks and never touches a running job (vram.reclaim_idle_workers)."""
        if not self._cond.acquire(blocking=False):
            return False
        try:
            if self._running or self._loading or not self._alive_locked():
                return False
            if self._proc is not None and self._proc.flavor != "cuda":
                return False
            logger.info("evicting idle nemotron sidecar to free VRAM")
            old = self._detach_locked("evicted")
        finally:
            self._cond.release()
        if old is not None:
            self._terminate(old)
            return True
        return False

    def enter_live_call(self) -> None:
        """Mark a live gRPC call active so CPU Nemotron jobs yield the CPU."""
        with self._cond:
            self._active_live_calls += 1
            self._cond.notify_all()

    def leave_live_call(self) -> None:
        """Release a live gRPC call's CPU priority hold."""
        with self._cond:
            self._active_live_calls = max(0, self._active_live_calls - 1)
            self._cond.notify_all()

    def preempt_for_live_call(self, settings: Any) -> bool:
        """A live gRPC call needs the GPU or CPU while a job is running: stop
        the sidecar, wait for its VRAM (when on CUDA) to come back, and put
        every running job back at the head of the queue. Returns True if
        anything was preempted. The caller MUST call ``resume_after_preempt()``
        once its own admission has been re-checked (success or not) -- until
        then no queued job can grab the card/CPU back."""
        with self._cond:
            if not self._running and not self._loading:
                return False
            self._suspended += 1
            for j in self._running.values():
                j.preempted = True
            if self._loading:
                self._load_preempted = True
            old = self._detach_locked("preempted")
            need = old.profile.vram_mb if old is not None else 0
            self.preemptions_total += 1
        metrics.nemotron_preemptions_total.inc()
        try:
            free_before = gpu.free_vram_mb() if need else None
            if old is not None:
                logger.warning("preempting nemotron sidecar pid=%d for a live call", old.popen.pid)
                self._terminate(old, grace_sec=PREEMPT_TERM_GRACE_SEC)
            if free_before is not None and need:
                deadline = time.monotonic() + float(settings.nemotron_preempt_wait_sec)
                while time.monotonic() < deadline:
                    free = gpu.free_vram_mb()
                    if free is None or free >= free_before + need // 2:
                        break
                    time.sleep(0.1)
        except BaseException:
            self.resume_after_preempt()
            raise
        return True

    def resume_after_preempt(self) -> None:
        with self._cond:
            self._suspended = max(0, self._suspended - 1)
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            old = self._detach_locked(None)
        if old is not None:
            self._terminate(old)


def _log_tail(workdir: str, limit: int = 2000) -> str:
    try:
        data = (Path(workdir) / "server.log").read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", "replace")


_SERVICE = NemotronService()


def get_service() -> NemotronService:
    return _SERVICE


def preempt_and_admit(settings: Any, admit: Callable[[], vram.Admission]) -> vram.Admission | None:
    """For a live call whose VRAM admission just failed: preempt a running
    Nemotron job if there is one and re-run ``admit``. Returns the new
    admission, or None when there was nothing to preempt. Blocking -- call
    via asyncio.to_thread from the gRPC event loop."""
    svc = get_service()
    if not svc.is_active():
        return None
    if not svc.preempt_for_live_call(settings):
        return None
    try:
        return admit()
    finally:
        svc.resume_after_preempt()


def preempt_cpu_for_live_call(settings: Any) -> bool:
    """Preempt a running/loading CPU Nemotron job for an incoming live gRPC
    call. Returns True if a CPU job was preempted."""
    svc = get_service()
    if not svc.is_active_on("cpu"):
        return False
    if not svc.preempt_for_live_call(settings):
        return False
    svc.resume_after_preempt()
    return True


atexit.register(_SERVICE.stop)
