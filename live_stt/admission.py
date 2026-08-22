"""Admission: two counters, not one, because they gate two different things.

``active_calls`` gates whether a NEW call is admitted at all --
``max_concurrent_calls``. ``active_workers`` gates whether an EXISTING call's
rotation can get a shadow worker to dual-feed into -- ``max_workers =
max_concurrent_calls + reserve_slots``. The reserve exists only for the
second number: a mid-call rotation temporarily needs two worker processes at
once (the outgoing one and its shadow), and reserving capacity for that is
what makes a rotation gapless. Handing the reserve to a new CALL instead
would mean the busiest possible moment -- every slot full -- is exactly when
rotations degrade to a gap, which is backwards. See CLAUDE.md's rotation
section.

Used to be race-free without a lock on the strength of one argument: every
method here is synchronous (no ``await`` between reading and mutating the
counters), and grpc.aio servicers all ran on one event loop thread. That
argument broke the moment ``live_stt/transcribe_http.py`` started sharing
this same ``ServerState.budget`` instance from ``admin_http.py``'s
``ThreadingHTTPServer`` -- a genuinely different OS thread per HTTP request,
which CAN run concurrently with the gRPC event loop thread's own calls into
these methods. A plain ``self.active_calls += 1`` is not atomic across
threads even under the GIL (it is read-modify-write across more than one
bytecode op), so two true concurrent callers really could lose an
increment. Fixed with one ``threading.Lock`` covering every mutation --
cheap (called at most a few times per call/rotation, never in a hot loop)
and simplest-correct over trying to re-derive a lock-free scheme for two
real threads.
"""

from __future__ import annotations

import threading


class WorkerBudget:
    def __init__(self, max_concurrent_calls: int, reserve_slots: int) -> None:
        self.max_concurrent_calls = max_concurrent_calls
        self.max_workers = max_concurrent_calls + reserve_slots
        self.active_calls = 0
        self.active_workers = 0
        self._lock = threading.Lock()

    def try_admit_call(self) -> bool:
        """A new call needs one call slot AND, since it starts with exactly
        one worker, one worker slot too. Never dips into the reserve: a call
        is only admitted while active_calls < max_concurrent_calls, which by
        construction leaves at least reserve_slots workers free.
        """
        with self._lock:
            if self.active_calls >= self.max_concurrent_calls:
                return False
            self.active_calls += 1
            self.active_workers += 1
            return True

    def release_call(self) -> None:
        with self._lock:
            self.active_calls = max(0, self.active_calls - 1)
            self.active_workers = max(0, self.active_workers - 1)

    def try_acquire_rotation_shadow(self) -> bool:
        """A shadow is a SECOND worker for an already-admitted call. May use
        the reserve -- that's what it's for.
        """
        with self._lock:
            if self.active_workers >= self.max_workers:
                return False
            self.active_workers += 1
            return True

    def release_rotation_shadow(self) -> None:
        with self._lock:
            self.active_workers = max(0, self.active_workers - 1)

    @property
    def free_workers(self) -> int:
        # Advisory read, deliberately unlocked -- used today only for
        # informational /api/stats-style reporting, where a value that's
        # microseconds stale under concurrent mutation is harmless. Take the
        # lock here too if a future caller ever makes an admission DECISION
        # off this property instead of try_admit_call()/try_acquire_
        # rotation_shadow()'s own atomic check-and-increment.
        return self.max_workers - self.active_workers
