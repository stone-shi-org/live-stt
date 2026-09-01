"""Thread-safe counters (plus a small live list) for concurrent batch
transcription HTTP requests (``POST /v1/audio/transcriptions``).

The direct analogue of ``live_stt/diarize_sessions.py``'s
``DiarizationSessionTracker`` -- same rationale, same shape, same
"deliberate, narrow exception to no session registry" contract (see that
module's docstring for the full argument; not repeated here).

This is a SEPARATE counter from ``live_stt/admission.py``'s ``WorkerBudget``,
which this endpoint also uses (``try_admit_call``/``release_call``) --
``WorkerBudget`` gates whether a worker PROCESS is allowed to spawn at all
(shared with every gRPC ``Transcribe`` call too, so it cannot distinguish
"a live call" from "a batch HTTP request" on its own), while this tracker
exists purely so the admin dashboard can show batch-transcription activity
as its own thing, separate from live streaming calls -- the same
distinction ``DiarizationSessionTracker`` draws relative to ``WorkerBudget``
for diarization, just for the second HTTP-only surface rather than the
first.

Unlike diarization (one rejection path, VRAM), a batch transcription
request can be turned away for two different reasons before ever calling
``start()``: insufficient VRAM (mirrors ``servicer.py``'s gRPC-side gate,
see ``live_stt/transcribe_http.py``) or no free call slot in ``WorkerBudget``
(``try_admit_call()`` returning False) -- tracked as two separate counters
rather than folding them into one, since they point at different
operational problems (a GPU sizing issue vs. a plain capacity ceiling).
"""

from __future__ import annotations

import itertools
import threading
import time


class TranscribeSessionTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.completed_total = 0
        self.failed_total = 0
        self.rejected_vram_total = 0
        self.rejected_capacity_total = 0
        self._ids = itertools.count(1)
        # request_id -> {"started_at": monotonic seconds, "model": str}
        self._active_requests: dict[int, dict[str, object]] = {}

    def start(self, *, model: str) -> int:
        """Call once the request has passed validation/admission, right
        before actually spawning the worker and transcribing -- `active`
        reflects real work in flight, not HTTP requests merely being
        parsed.

        Returns an opaque request id; pass it back to `finish()` so its
        live entry is removed (not just the aggregate counter decremented).
        """
        with self._lock:
            self.active += 1
            request_id = next(self._ids)
            self._active_requests[request_id] = {"started_at": time.monotonic(), "model": model}
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
        """A request rejected before `start()` for insufficient VRAM --
        never incremented `active`, never got a live-list entry.
        """
        with self._lock:
            self.rejected_vram_total += 1

    def record_rejected_capacity(self) -> None:
        """A request rejected before `start()` because WorkerBudget had no
        free call slot -- same non-`active` contract as the VRAM case.
        """
        with self._lock:
            self.rejected_capacity_total += 1

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
                    "model": info["model"],
                }
                for request_id, info in items
            ]
