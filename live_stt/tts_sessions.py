"""Thread-safe counters (plus a small live list) for concurrent TTS synthesis
HTTP requests (``POST /v1/audio/speech``).

Follows the exact pattern set by ``live_stt/diarize_sessions.py`` and
``live_stt/transcribe_sessions.py``.
"""

from __future__ import annotations

import itertools
import threading
import time


class TTSSessionTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.completed_total = 0
        self.failed_total = 0
        self.rejected_vram_total = 0
        self.rejected_capacity_total = 0
        self._ids = itertools.count(1)
        # request_id -> {"started_at": monotonic seconds, "model": str, "voice": str, "device": str}
        self._active_requests: dict[int, dict[str, object]] = {}

    def start(self, *, model: str = "unknown", voice: str = "unknown", device: str = "unknown") -> int:
        """Call once the request has passed validation/admission, right
        before beginning synthesis -- `active` reflects real work in flight.

        Returns an opaque request id; pass it back to `finish()` so its
        live entry is removed (not just the aggregate counter decremented).
        """
        with self._lock:
            self.active += 1
            request_id = next(self._ids)
            self._active_requests[request_id] = {
                "started_at": time.monotonic(),
                "model": model,
                "voice": voice,
                "device": device,
            }
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
        """A request rejected before `start()` for insufficient VRAM."""
        with self._lock:
            self.rejected_vram_total += 1

    def record_rejected_capacity(self) -> None:
        """A request rejected before `start()` for insufficient capacity."""
        with self._lock:
            self.rejected_capacity_total += 1

    def snapshot_active(self) -> list[dict[str, object]]:
        """Currently in-flight requests, oldest first, for `/api/stats` and
        the admin dashboard. `elapsed_sec` is computed fresh on every call.
        """
        now = time.monotonic()
        with self._lock:
            items = sorted(self._active_requests.items())
            return [
                {
                    "id": request_id,
                    "elapsed_sec": round(now - info["started_at"], 1),
                    "model": info["model"],
                    "voice": info["voice"],
                    "device": info["device"],
                }
                for request_id, info in items
            ]
