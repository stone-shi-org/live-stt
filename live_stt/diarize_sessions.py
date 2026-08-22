"""Thread-safe counters (plus a small live list) for concurrent diarization
HTTP requests.

Diarization (`live_stt/diarization.py`) runs as ordinary Python/torch code
inside `admin_http.py`'s process, on whichever `ThreadingHTTPServer` thread
handled the request -- there is no `WorkerHandle`/subprocess the way ASR
has, so `live_stt/admission.py`'s `WorkerBudget` doesn't apply here at all
(it counts worker PROCESSES, and diarization spawns none). This is a
separate, much simpler counter.

The aggregate fields (`active`/`completed_total`/`failed_total`/
`rejected_vram_total`) are the same category as `WorkerBudget`'s own
`active_calls`/`active_workers` (see `live_stt/state.py`'s "Deliberately NOT
a session registry" note).

`start()`/`finish()` ALSO track a small live list of currently in-flight
requests (opaque sequential id, monotonic start time, device) so the admin
dashboard can show "what's running right now", not just a bare count -- this
IS a session-like registry, a deliberate, narrow exception to that rule,
requested specifically for dashboard visibility. It carries no
call-identifying information (no session_id/call_id, no audio, no text, no
client address) and only ever holds entries for requests that are still
`active` -- `finish()` always removes its entry, so this cannot grow past
the real concurrency of the box and never persists past a request's own
lifetime. `id` is a plain per-process incrementing counter, not a UUID/token
-- meaningless outside this process's own dashboard.
"""

from __future__ import annotations

import itertools
import threading
import time


class DiarizationSessionTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.completed_total = 0
        self.failed_total = 0
        self.rejected_vram_total = 0
        self._ids = itertools.count(1)
        # request_id -> {"started_at": monotonic seconds, "device": str}
        self._active_requests: dict[int, dict[str, object]] = {}

    def start(self, *, device: str = "unknown") -> int:
        """Call once the request has passed validation and the VRAM check,
        right before actually running the pipeline -- `active` reflects
        real inference in flight, not HTTP requests merely being parsed.

        Returns an opaque request id; pass it back to `finish()` so its live
        entry is removed (not just the aggregate counter decremented).
        """
        with self._lock:
            self.active += 1
            request_id = next(self._ids)
            self._active_requests[request_id] = {"started_at": time.monotonic(), "device": device}
        return request_id

    def finish(self, request_id: int, *, ok: bool) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self._active_requests.pop(request_id, None)
            if ok:
                self.completed_total += 1
            else:
                self.failed_total += 1

    def record_rejected_vram(self) -> None:
        """A request that never got to `start()` at all -- rejected before
        touching the pipeline, so it never incremented `active` and never
        got a live-list entry.
        """
        with self._lock:
            self.rejected_vram_total += 1

    def snapshot_active(self) -> list[dict[str, object]]:
        """Currently in-flight requests, oldest first, for `/api/stats` and
        the admin dashboard. `elapsed_sec` is computed fresh on every call
        (not stored), so it's always accurate as of the snapshot instant.
        """
        now = time.monotonic()
        with self._lock:
            items = sorted(self._active_requests.items())
            return [
                {
                    "id": request_id,
                    "elapsed_sec": round(now - info["started_at"], 1),
                    "device": info["device"],
                }
                for request_id, info in items
            ]
