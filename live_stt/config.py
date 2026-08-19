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

    # Derived from tools/thread_sweep.py (Phase 1 Gate B) for the target
    # concurrency; these are conservative placeholders until that sweep runs.
    n_threads_per_worker: int = 4
    max_concurrent_calls: int = 3
    # Admission always reserves this many slots so a mid-call rotation has
    # somewhere to put its overlap shadow. Never hand the reserve to a call.
    reserve_slots: int = 1

    # Rotation triggers (see live_stt/pool/supervisor.py). RSS is primary and
    # backend-agnostic; audio_cap is the deterministic fallback used when Gate
    # A hasn't run or the RSS watchdog is disabled for a test.
    worker_rss_soft_kb: int = 2_400_000  # ~2.3 GB
    rotate_after_sec: float = 1800.0
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
