"""Shared mutable state between server.py (owns it), servicer.py (reads
`draining` for admission), and admin_http.py (reads everything for
/api/health and /api/stats). Deliberately NOT a session registry -- nothing
here is keyed by call -- just process-wide flags and the one WorkerBudget
instance, so passing this around does not reintroduce "session_id" through
the back door.
"""

from __future__ import annotations

from dataclasses import dataclass

from live_stt.admission import WorkerBudget
from live_stt.config import Settings


@dataclass
class ServerState:
    settings: Settings
    budget: WorkerBudget

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
