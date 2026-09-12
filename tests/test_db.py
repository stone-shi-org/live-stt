from __future__ import annotations

import os

from live_stt import db


def test_connect_creates_the_settings_table() -> None:
    conn = db.connect(":memory:")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
    assert cursor.fetchone() is not None


def test_connect_is_idempotent_against_an_existing_schema(tmp_path) -> None:
    path = str(tmp_path / "sub" / "live_stt.db")
    conn1 = db.connect(path)
    conn1.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES ('k', '1', 'now')"
    )
    conn1.commit()
    conn1.close()

    # Re-connecting (as a fresh process restart would) must not wipe the
    # existing table/rows, and must create the parent directory if it
    # didn't already exist.
    assert os.path.isdir(os.path.dirname(path))
    conn2 = db.connect(path)
    row = conn2.execute("SELECT value FROM settings WHERE key='k'").fetchone()
    assert row == ("1",)
