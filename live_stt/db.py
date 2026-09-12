"""SQLite connection helper for the runtime settings store (LS-5).

One small, focused schema: a `settings` key/value table. Deliberately not a
generic ORM/migration framework -- there is exactly one table today, and
CLAUDE.md's own convention is "a reason comment on every non-obvious
choice," not speculative infrastructure for tables that don't exist yet
(a `backends` table for LS-5's later download/version-tracking phase is
intentionally NOT created here -- add it, and its own migration, when that
phase actually starts).

WAL mode is enabled explicitly: this DB is read from the admin HTTP
handler's own thread (ThreadingHTTPServer gives each request its own OS
thread, per live_stt/admin_http.py's docstring) while the settings store
may also be written from that same pool of threads -- WAL allows concurrent
readers alongside a single writer without the "database is locked" errors
the default rollback-journal mode is prone to under exactly this pattern.
SettingsStore (live_stt/settings_store.py) additionally serializes writes
with its own threading.Lock, the same class of fix live_stt/admission.py's
WorkerBudget already needed when it stopped being single-event-loop-only.
"""

from __future__ import annotations

import os
import sqlite3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Opens (creating if needed) the settings DB at db_path.

    ``:memory:`` is accepted and used by tests -- WAL mode is a no-op there
    (SQLite silently keeps the in-memory default), which is fine since
    there's no concurrent-process access to an in-memory DB anyway.
    """
    if db_path != ":memory:":
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn
