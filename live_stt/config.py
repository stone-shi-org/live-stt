"""Service settings. LSTT_-prefixed environment variables, via pydantic-settings
(the house config idiom -- see my-meeting-notes/app/config.py).

Fixed for the process lifetime -- read once at startup, not editable at
runtime like my-meeting-notes' RUNTIME_KEYS layer, since a running worker
pool cannot safely be resized or re-thresholded out from under active calls.
"""

from __future__ import annotations

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from live_stt.models import DEFAULT_MODEL_KEY


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LSTT_", env_file=".env", extra="ignore")

    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    admin_host: str = "0.0.0.0"
    admin_port: int = 8000

    backend: str = "cpu"  # "cpu" | "cuda" -- an image choice in prod, but overridable for tests
    default_model: str = DEFAULT_MODEL_KEY
    models_dir: str = "/models"

    # Matches the Dockerfile runtime stage's layout (worker binary + its
    # ggml .so's copied into the same directory so the $ORIGIN rpath baked
    # in at link time resolves them -- see CLAUDE.md). worker_ggml_lib_dir
    # is None in that case: LD_LIBRARY_PATH is not needed and not set.
    # Native/dev use (scripts/build_worker.sh's output layout) needs it,
    # since worker/build/live_stt_worker and worker/build-parakeet/.../ggml
    # aren't co-located there.
    worker_bin: str = "/app/worker/live_stt_worker"
    worker_ggml_lib_dir: str | None = None

    # tools/thread_sweep.py (Phase 1 Gate B), measured on THIS repo's 6-core
    # dev host, single-stream-at-a-time (not concurrent -- see CLAUDE.md's
    # caveat about contention): n_threads=1 gave rtfx=2.48 and, because
    # per-thread returns were sharply sub-linear (n_threads=4 only reached
    # rtfx=5.54, 2.2x for 4x the threads), the highest AGGREGATE throughput
    # (W * rtfx) came from running MORE single-threaded workers, not fewer
    # multi-threaded ones. Re-run the sweep on the actual deployment box --
    # these numbers are dev-host-only and not a production capacity signal.
    n_threads_per_worker: int = 1
    max_concurrent_calls: int = 3
    # Admission always reserves this many slots so a mid-call rotation has
    # somewhere to put its overlap shadow. Never hand the reserve to a call.
    reserve_slots: int = 1

    # Rotation triggers (see live_stt/pool/supervisor.py). RSS is primary and
    # backend-agnostic; audio_cap is the deterministic fallback.
    #
    # tools/leak_curve.py (Phase 1 Gate A), measured on THIS repo's CPU build:
    # ~0.08 MB leaked per audio-second fed (600s runs, both silence and real
    # speech, see CLAUDE.md) -- roughly 200-500x BELOW the 19-41 MB/s the
    # upstream issue (mudler/parakeet.cpp#63) reports, which was only ever
    # measured on a CUDA Jetson build. At this rate a 2-hour call leaks only
    # ~35 MB, not tens of GB, so rotate_after_sec below is sized generously
    # rather than defensively -- the RSS watchdog is the real safety net, not
    # a tightly-tuned deadline. Re-measure before a CUDA (Phase 5) rollout;
    # do not assume this number transfers to that backend.
    worker_rss_soft_kb: int = 2_400_000  # ~2.3 GB
    rotate_after_sec: float = 3600.0
    rotation_overlap_sec: float = 10.0  # > ~4.5s left context (att_context [70,1] @ 80ms/frame)

    # Phase 5, CUDA only. 10.100.0.50 (verified via nvidia-smi over SSH: one
    # RTX 3090, 24GB) is a SHARED box -- it also runs LocalAI and other GPU
    # workloads, so this must be a conscious budget, not "assume the whole
    # card". A CUDA allocation failure is an abort(), not a catchable
    # exception (see CLAUDE.md), so under-provisioning this crashes a
    # worker process, not just slows one down. Checked by live_stt/gpu.py
    # via `nvidia-smi --query-gpu=memory.free`, not pynvml -- avoids a new
    # dependency for one query, and nvidia-smi is already present in any
    # nvidia/cuda-based image.
    vram_per_worker_mb: int = 3000
    vram_reserve_mb: int = 2000  # headroom for other tenants on the shared card

    # Backpressure / drift (live_stt/session.py's AudioRing + watchdog).
    queue_max_sec: float = 8.0
    ring_history_sec: float = 60.0
    warn_behind_sec: float = 5.0
    abort_behind_sec: float = 30.0

    # Call lifecycle.
    max_call_sec: float = 9000.0  # 150 min
    idle_timeout_sec: float = 60.0
    finalize_timeout_sec: float = 30.0
    drain_timeout_sec: float = 300.0  # compose's stop_grace_period must exceed this

    # Redaction (live_stt/redaction.py). Two independent switches: logging any
    # recoverable transcript text, or persisting audio, both require
    # allow_pii=True as well -- one env var should never be a single typo away
    # from logging every phone call in the building.
    transcript_log: str = "off"  # off | hash | sample:1inN | full
    audio_dump: str = "off"  # off | on_error | always
    allow_pii: bool = False

    # Post-call speaker diarization (live_stt/diarization.py). Confirmed via
    # the model card that pyannote/speaker-diarization-community-1 processes
    # a whole clip in one pass (pipeline("audio.wav") / pipeline({"waveform":
    # ..., "sample_rate": ...})) -- there is no incremental-feed API, so this
    # cannot run inline in Transcribe() the way the ASR worker does. It only
    # ever runs AFTER a call ends, against a recorded WAV -- which itself
    # requires audio_dump above (and therefore allow_pii=True) to have
    # produced one; there is deliberately no separate PII gate here, it just
    # inherits audio_dump's. Output is mapped into the same
    # {num_speakers, segments, speakers} JSON shape as
    # my-meeting-notes/app/services/diarize.py's LocalAI-compatible client
    # (house convention), not pyannote's native Annotation/RTTM shape.
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    # Gated model on HuggingFace (CC-BY-4.0, requires accepting pyannote's
    # terms and passing a token) -- kept as its own field, never folded into
    # diarization_model, so a log line that prints the model id never leaks it.
    diarization_hf_token: str | None = None
    # Most calls through this service are one-on-one telephony, so the
    # speaker count is usually known in advance -- passing it to pyannote's
    # pipeline (num_speakers=) measurably helps clustering accuracy over
    # letting it guess. None means "let pyannote decide", the right default
    # for anything that isn't a plain two-party call.
    diarization_num_speakers: int | None = 2
    # "cpu" | "cuda". Independent of the ASR worker's own `backend` setting
    # above -- pyannote.audio runs in this Python process via torch, not in
    # the C++ worker, so a CUDA ASR deployment and a CPU diarization
    # deployment (or vice versa) are two separate choices, not one. Measured
    # for real on a 6-core CPU dev host against a 358s NOTSOFAR-1 meeting:
    # 312.8s wall time (~0.87x realtime) -- see CLAUDE.md. Defaults to "cpu"
    # since that's what's actually been exercised; opt into "cuda" only on a
    # box that actually has a CUDA-capable GPU and driver.
    diarization_device: str = "cpu"
    # VRAM required to admit a diarization request when diarization_device
    # is "cuda" -- checked against live_stt.gpu.free_vram_mb() the same way
    # servicer.py already gates ASR admission, since a CUDA allocation
    # failure is an abort(), not a catchable exception (see CLAUDE.md).
    # Measured for real on 10.100.0.50's RTX 3090 (nvidia-smi
    # --query-compute-apps, isolated to the live-stt process's own PID):
    # ~12.3GB held after a single diarization call over a ~6-minute
    # NOTSOFAR-1 recording, and a second back-to-back call used essentially
    # the same (12262 -> 12270 MiB) rather than growing further -- PyTorch's
    # CUDA caching allocator sizing itself once to the batched
    # sliding-window peak for that file length and reusing it, not a leak.
    # 13000 leaves a ~700MB margin above the one real measurement available;
    # NOT yet tested across a range of audio durations, and VRAM plausibly
    # scales with file length (bigger batched windows for longer audio) --
    # re-measure with a much longer recording before trusting this on calls
    # significantly longer than ~6 minutes.
    diarization_vram_mb: int = 13000

    @model_validator(mode="after")
    def _check_grace_period(self) -> "Settings":
        if self.finalize_timeout_sec > self.drain_timeout_sec:
            raise ValueError("finalize_timeout_sec must not exceed drain_timeout_sec")
        return self


def get_settings() -> Settings:
    return Settings()
