"""Thread-safe counters for concurrent diarization HTTP requests.

Diarization (`live_stt/diarization.py`) runs as ordinary Python/torch code
inside `admin_http.py`'s process, on whichever `ThreadingHTTPServer` thread
handled the request -- there is no `WorkerHandle`/subprocess the way ASR
has, so `live_stt/admission.py`'s `WorkerBudget` doesn't apply here at all
(it counts worker PROCESSES, and diarization spawns none). This is a
separate, much simpler counter.

This is aggregate state, not a session registry: no call-identifying key
anywhere, same category as `WorkerBudget`'s own `active_calls`/
`active_workers` (see `live_stt/state.py`'s "Deliberately NOT a session
registry" note -- this doesn't reintroduce that, it just answers "how many
diarization requests are in flight right now" for `/api/stats` and the
admin dashboard). Needs its own lock for the same reason `WorkerBudget`
needed one added: multiple `ThreadingHTTPServer` request threads can call in
concurrently.
"""

from __future__ import annotations

import threading


class DiarizationSessionTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.completed_total = 0
        self.failed_total = 0
        self.rejected_vram_total = 0

    def start(self) -> None:
        """Call once the request has passed validation and the VRAM check,
        right before actually running the pipeline -- `active` reflects
        real inference in flight, not HTTP requests merely being parsed.
        """
        with self._lock:
            self.active += 1

    def finish(self, *, ok: bool) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            if ok:
                self.completed_total += 1
            else:
                self.failed_total += 1

    def record_rejected_vram(self) -> None:
        """A request that never got to `start()` at all -- rejected before
        touching the pipeline, so it never incremented `active`.
        """
        with self._lock:
            self.rejected_vram_total += 1
