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

    @model_validator(mode="after")
    def _check_grace_period(self) -> "Settings":
        if self.finalize_timeout_sec > self.drain_timeout_sec:
            raise ValueError("finalize_timeout_sec must not exceed drain_timeout_sec")
        return self


def get_settings() -> Settings:
    return Settings()
