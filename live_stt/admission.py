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

Race-free without a lock: every method here is synchronous (no ``await``
between reading and mutating the counters), and grpc.aio servicers all run
on one event loop thread -- the same argument the servicer's own admission
check already relies on.
"""

from __future__ import annotations


class WorkerBudget:
    def __init__(self, max_concurrent_calls: int, reserve_slots: int) -> None:
        self.max_concurrent_calls = max_concurrent_calls
        self.max_workers = max_concurrent_calls + reserve_slots
        self.active_calls = 0
        self.active_workers = 0

    def try_admit_call(self) -> bool:
        """A new call needs one call slot AND, since it starts with exactly
        one worker, one worker slot too. Never dips into the reserve: a call
        is only admitted while active_calls < max_concurrent_calls, which by
        construction leaves at least reserve_slots workers free.
        """
        if self.active_calls >= self.max_concurrent_calls:
            return False
        self.active_calls += 1
        self.active_workers += 1
        return True

    def release_call(self) -> None:
        self.active_calls = max(0, self.active_calls - 1)
        self.active_workers = max(0, self.active_workers - 1)

    def try_acquire_rotation_shadow(self) -> bool:
        """A shadow is a SECOND worker for an already-admitted call. May use
        the reserve -- that's what it's for.
        """
        if self.active_workers >= self.max_workers:
            return False
        self.active_workers += 1
        return True

    def release_rotation_shadow(self) -> None:
        self.active_workers = max(0, self.active_workers - 1)

    @property
    def free_workers(self) -> int:
        return self.max_workers - self.active_workers
