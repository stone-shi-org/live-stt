"""Runtime settings store (LS-5): SQLite-backed, replaces LSTT_* env vars as
the ONGOING source of truth for a running deployment, per the explicit
decision on that ticket ("fully replace (+ migration current + default
value)"). Env vars/.env remain how a fresh deployment sets its *initial*
values -- see live_stt/config.py's `db_path` field docstring and
`get_settings()`'s docstring for the full split.

Deliberately NOT a blanket "everything is now safely live-editable" layer:
live_stt/config.py's Settings docstring has always said several fields are
"Fixed for the process lifetime... since a running worker pool cannot
safely be resized or re-thresholded out from under active calls" (e.g.
max_concurrent_calls, worker_bin*, rotate_after_sec, the VRAM budgets) --
that safety reasoning is unchanged by where the value is stored. Every
setting is writable through this store, but HOT_RELOAD_FIELDS below is the
explicit, narrow allowlist of fields the running live_stt/state.py
ServerState.settings object is actually mutated in place for; everything
else is persisted (takes effect on next process start) but not applied
live. Getting this allowlist wrong in the permissive direction is a real
safety regression, not just a UX nit -- keep it short and justified.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from live_stt import db, diarization_models, tts_models
from live_stt.config import Settings, get_settings
from live_stt.redaction import validate as validate_redaction_config

# Each field here is read fresh, per-call/per-request, by the code that
# actually uses it -- confirmed by reading the call site, not assumed:
#   - diarization_num_speakers/min/max_speakers, diarization_vram_mb:
#     live_stt/diarize_http.py reads settings at request time.
#   - tts_voice: live_stt/tts_http.py/tts.py read settings at request time
#     (and a per-request "voice" field already takes priority over it).
#   - transcript_log, audio_dump: live_stt/servicer.py calls
#     redaction.transcript_log_fields(..., settings) fresh at the end of
#     every call.
# transcript_log/audio_dump/allow_pii are handled together, not
# independently: changing any of the three re-validates the full redaction
# config (see apply_update below) so a hot change can never produce the
# "LSTT_AUDIO_DUMP=... requires LSTT_ALLOW_PII=true" combination that
# live_stt/redaction.py's startup-only validate() exists specifically to
# prevent. allow_pii itself is intentionally NOT in this set -- flipping the
# one flag that gates recoverable-PII logging/persistence on a live process
# is exactly the kind of change CLAUDE.md's "one env var should never be a
# single typo away from logging every phone call in the building" warns
# about; requiring a conscious restart for that one is a feature.
HOT_RELOAD_FIELDS: frozenset[str] = frozenset(
    {
        "default_model",
        "default_batch_model",
        "diarization_model",
        "tts_model",
        "diarization_num_speakers",
        "diarization_min_speakers",
        "diarization_max_speakers",
        "diarization_vram_mb",
        # VRAM admission (live_stt/vram.py) reads these per request; nothing is
        # sized from them at startup, so a restart bought nothing.
        "vram_per_worker_mb",
        "vram_reserve_mb",
        "vram_reserve_pct",
        "vram_overrides",
        "tts_vram_mb",
        # Read at each worker spawn; StandbyKey carries it, so a standby spawned
        # under the old value is never handed out after a change.
        "parakeet_tdt_att_context",
        "tts_voice",
        "tts_idle_timeout_sec",
        "stt_idle_timeout_sec",
        "transcript_log",
        "audio_dump",
        # LS-5 Phase 7c: a policy toggle for a periodic BACKGROUND check
        # (live_stt/backends/upgrade_checker.py) -- flipping it doesn't
        # resize a live worker pool or change anything about an in-flight
        # call, unlike the worker_bin*/max_concurrent_calls-class fields
        # that stay restart-required.
        "auto_upgrade_enabled",
    }
)

# Fields whose value change requires re-running the redaction invariant
# (live_stt/redaction.py's validate()) before it's accepted, hot or not --
# see the HOT_RELOAD_FIELDS comment above.
_REDACTION_FIELDS: frozenset[str] = frozenset({"transcript_log", "audio_dump", "allow_pii"})


class SettingsStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = db.connect(db_path)
        self._lock = threading.Lock()

    @property
    def connection(self) -> sqlite3.Connection:
        """The one already-open connection this store was constructed
        with. Exists so OTHER users of the same DB file (LS-5's backend
        installer, via live_stt/admin_http.py) reuse THIS connection
        instead of independently calling live_stt.db.connect(settings.
        db_path) again -- a second, separate connect() call is not just
        wasteful, it's a real correctness gap: it re-derives the path from
        `Settings.db_path` instead of using whatever this store was
        ACTUALLY constructed with (":memory:", a test's tmp_path, or a
        real deployment path), and a test that forgets to keep those two
        in sync gets a confusing failure against the wrong (or a
        nonexistent/unwritable) path instead of the one it meant to use --
        exactly the bug this property was added to close.
        """
        return self._conn

    def migrate_if_empty(self, defaults: Settings) -> bool:
        """Seeds the store from `defaults` (an env/.env/compiled-default
        Settings instance) exactly once -- the first time this store's DB
        has no rows at all. Returns True if migration ran. Idempotent by
        construction: a DB that already has rows (from a prior boot, or a
        prior manual edit) is never touched here, so this is safe to call
        on every process start.
        """
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM settings").fetchone()
            if count > 0:
                return False
            now = _now()
            rows = [(key, json.dumps(value), now) for key, value in defaults.model_dump().items()]
            self._conn.executemany(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)", rows
            )
            self._conn.commit()
            return True

    def load_overrides(self) -> dict[str, Any]:
        with self._lock:
            cursor = self._conn.execute("SELECT key, value FROM settings")
            return {key: json.loads(value) for key, value in cursor.fetchall()}

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value), _now()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_effective_settings(db_path: str | None = None) -> tuple[Settings, SettingsStore]:
    """The real service entrypoint's replacement for a bare `get_settings()`
    call (see live_stt/server.py). `get_settings()` itself is untouched --
    its result here is used only as (a) the migration seed on first boot and
    (b) the fallback for any field the store doesn't have an opinion on
    (e.g. a brand new Settings field added after a deployment's DB was
    already migrated).
    """
    env_settings = get_settings()
    store = SettingsStore(db_path or env_settings.db_path)
    store.migrate_if_empty(env_settings)

    known_fields = set(env_settings.model_dump())
    overrides = {k: v for k, v in store.load_overrides().items() if k in known_fields}
    merged = {**env_settings.model_dump(), **overrides}
    effective = Settings(_env_file=None, **merged)
    return effective, store


def apply_update(store: SettingsStore, live_settings: Settings, updates: dict[str, Any]) -> Settings:
    """Validates `updates` against the FULL Settings model (so cross-field
    validators -- _check_grace_period, and the redaction re-check below --
    still run against the merged result, not just the changed fields in
    isolation), persists every changed key, and returns the new validated
    Settings object. Raises ValueError (never a bare pydantic
    ValidationError leaking a different exception shape) on an unknown key
    or a value that fails validation -- callers (live_stt/admin_http.py)
    turn that into a 400, not a 500.

    Applying hot-reload fields to the LIVE, in-use Settings object is
    deliberately the caller's job: only the caller (admin_http.py, via
    live_stt/state.py's ServerState) knows which object is actually shared
    with the running service.
    """
    current = live_settings.model_dump()
    unknown = sorted(set(updates) - set(current))
    if unknown:
        raise ValueError(f"unknown setting(s): {unknown}")

    merged = {**current, **updates}
    try:
        validated = Settings(_env_file=None, **merged)
    except Exception as exc:  # pydantic's ValidationError, re-raised as plain ValueError
        raise ValueError(str(exc)) from exc

    if "diarization_model" in updates:
        try:
            diarization_models.resolve(validated.diarization_model)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    if "tts_model" in updates:
        try:
            tts_models.resolve(validated.tts_model)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

    if updates.keys() & _REDACTION_FIELDS:
        try:
            validate_redaction_config(validated)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    dumped = validated.model_dump()
    for key in updates:
        store.set(key, dumped[key])
    return validated
