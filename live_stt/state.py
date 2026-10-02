"""Shared mutable state between server.py (owns it), servicer.py (reads
`draining` for admission), and admin_http.py (reads everything for
/api/health and /api/stats). Deliberately NOT a session registry -- nothing
here is keyed by call -- just process-wide flags and the one WorkerBudget
instance, so passing this around does not reintroduce "session_id" through
the back door.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.diarize_sessions import DiarizationSessionTracker
from live_stt.settings_store import SettingsStore
from live_stt.transcribe_sessions import TranscribeSessionTracker
from live_stt.tts import TTSWorkerManager, get_tts_worker_manager
from live_stt.tts_sessions import TTSSessionTracker
from live_stt.worker_pool import STTWorkerPool, get_worker_pool


@dataclass
class ServerState:
    settings: Settings
    budget: WorkerBudget

    # SQLite-backed runtime settings store (LS-5, live_stt/settings_store.py).
    # None when nothing set it up -- every existing test constructing
    # ServerState(settings=..., budget=...) directly (the vast majority)
    # keeps working completely unchanged; live_stt/server.py's real
    # entrypoint is the only thing that populates this. admin_http.py's
    # /api/settings endpoints treat None as "no store configured" (503, not
    # a crash) rather than assuming it's always present.
    settings_store: SettingsStore | None = None

    # Diarization has no worker PROCESS the way ASR does (see
    # DiarizationSessionTracker's own docstring), so it needed its own,
    # separate counter rather than reusing `budget` -- still just an
    # aggregate count, not a session registry, same as `budget` itself.
    diarization_sessions: DiarizationSessionTracker = field(default_factory=DiarizationSessionTracker)

    # Batch transcription (POST /v1/audio/transcriptions) DOES spawn a real
    # worker process through the same WorkerBudget every gRPC Transcribe
    # call uses, so it can't reuse budget.active_calls alone to show
    # "batch transcription activity" on the dashboard -- that number is
    # shared with live streaming calls and can't tell the two apart. This
    # tracker is the transcribe-side analogue of diarization_sessions above,
    # same aggregate-counters-plus-small-live-list shape.
    transcribe_sessions: TranscribeSessionTracker = field(default_factory=TranscribeSessionTracker)

    # TTS synthesis (POST /v1/audio/speech) session tracker.
    tts_sessions: TTSSessionTracker = field(default_factory=TTSSessionTracker)

    # Standby worker pool for warmed-up STT engine caching (LS-10).
    worker_pool: STTWorkerPool = field(default_factory=get_worker_pool)

    # Persistent warm worker manager for TTS (LS-9, LS-12).
    tts_worker_manager: TTSWorkerManager = field(default_factory=get_tts_worker_manager)

    # Set once, immediately, on SIGTERM -- before grpc.aio's own drain grace
    # period even starts. New calls are rejected from this instant.
    draining: bool = False

    # Set after repeated consecutive stream-init failures (not yet driven by
    # anything in this codebase -- there is no pool/supervisor to notice a
    # pattern of failures yet, so this stays False. Field exists so
    # admin_http's /api/health has a real structural-failure signal to check
    # once one exists, instead of adding it later as a breaking response
    # shape change.)
    degraded: bool = False

    def health_status(self) -> str:
        if self.degraded:
            return "degraded"
        if self.draining:
            return "draining"
        return "ok"
