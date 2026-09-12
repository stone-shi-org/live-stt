"""Runtime backend installer (LS-5 Phase 7b): download+verify+install+
activate a BackendSpec (live_stt/backends/registry.py) into a persistent
install directory -- the Python-side, admin-UI-triggered analogue of
scripts/fetch_backend.sh (which exists for the EARLY Docker build layer,
before the live_stt package/its deps are even installed -- see that
script's docstring for why the two aren't unified into one implementation).

Tracked in the same SQLite DB as live_stt/settings_store.py (a new
`installed_backends` table, schema created here via live_stt/db.py's
connection) so both live behind the one process-shared DB file under the
already-mounted ./data volume -- this is what "survive docker recreate"
means concretely, same as live_stt/settings_store.py's DB itself.

What "install" actually changes at runtime today: ONLY the parakeet engine
(live_stt/session.py checks get_active_lib_dir(..., "parakeet") when
spawning a worker) -- because only parakeet's worker binary is built
against a dynamically-linked libparakeet.so
(worker/CMakeLists.txt's LSTT_PARAKEET_PREBUILT, on by default in the CPU
runtime image as of this phase). whisper's BackendSpec is real and
fetchable (registry.py's `wired=False`) but installing/activating a
whisper entry has no effect yet -- that engine's worker binary is still
built from source, statically. Being honest about this gap here rather
than implying full symmetry across engines.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sqlite3
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from live_stt import db
from live_stt.backends.registry import BackendSpec, resolve

_SCHEMA = """
CREATE TABLE IF NOT EXISTS installed_backends (
    engine TEXT NOT NULL,
    flavor TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL,
    installed_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (engine, flavor, version)
);
"""

# One process-wide lock, same reasoning as live_stt/settings_store.py's
# SettingsStore -- this DB is written from admin HTTP request threads
# (ThreadingHTTPServer gives each request its own OS thread), and sqlite3
# connections opened with check_same_thread=False (live_stt/db.py) need an
# explicit lock around writes, the same class of fix
# live_stt/admission.py's WorkerBudget already needed.
_lock = threading.Lock()


class BackendInstallError(Exception):
    """A download, checksum, extraction, or safety-probe step failed."""


@dataclass(frozen=True, slots=True)
class InstalledBackend:
    engine: str
    flavor: str
    version: str
    sha256: str
    path: str
    installed_at: str
    active: bool


def ensure_schema(conn: sqlite3.Connection) -> None:
    with _lock:
        conn.execute(_SCHEMA)
        conn.commit()


def _download_and_verify(spec: BackendSpec, dest_tar: Path) -> None:
    url = f"https://github.com/{spec.repo}/releases/download/{spec.tag}/{spec.asset_filename}"
    tmp = dest_tar.with_suffix(dest_tar.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 -- fixed https://github.com URL, not user input
    except OSError as exc:
        raise BackendInstallError(f"download failed for {url}: {exc}") from exc

    actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if actual != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise BackendInstallError(
            f"sha256 mismatch for {spec.asset_filename}: expected {spec.sha256}, got {actual}"
        )
    tmp.rename(dest_tar)


def probe_parakeet_private_symbols(so_path: Path) -> bool:
    """dlopen()s the given libparakeet.so and confirms
    pk::set_num_threads/pk::shutdown_backend still resolve as real dynamic
    symbols -- see worker/pk_private.hpp's docstring for exactly why this
    can't be taken on faith: those two symbols are an UNPUBLISHED,
    no-guarantee contract (not part of parakeet_capi.h), confirmed once via
    `nm -D` against the real v0.5.0 release but never promised to stay that
    way by upstream. Returns False (never raises) on any failure -- the
    caller (install(), below) decides that a False result blocks
    installation rather than risking a runtime crash later.
    """
    try:
        lib = ctypes.CDLL(str(so_path))
        getattr(lib, "_ZN2pk15set_num_threadsEi")
        getattr(lib, "_ZN2pk16shutdown_backendEv")
        return True
    except (OSError, AttributeError):
        return False


def install(conn: sqlite3.Connection, install_root: str, engine: str, flavor: str) -> InstalledBackend:
    """Idempotent: a second call for the same (engine, flavor) -- and
    therefore the same pinned registry tag -- is a fast no-op (checks a
    marker file, does not re-download), matching scripts/fetch_backend.sh's
    own skip-if-present behavior. Does NOT activate -- see activate() below;
    installing and activating are deliberately separate steps (LS-5's own
    design: a bad download or a bad version must not auto-take-effect).
    """
    ensure_schema(conn)
    spec = resolve(engine, flavor)
    version_dir = Path(install_root) / engine / flavor / spec.tag
    version_dir.mkdir(parents=True, exist_ok=True)

    marker = version_dir / ".installed"
    if not marker.exists():
        tar_path = version_dir / spec.asset_filename
        _download_and_verify(spec, tar_path)
        with tarfile.open(tar_path) as tf:
            members = []
            for member in tf.getmembers():
                # Every registered release tarball wraps its contents in one
                # top-level "<name>-<version>-<os>-<arch>/" directory --
                # confirmed by actually listing both tarballs' contents
                # while building this (same fact scripts/fetch_backend.sh's
                # --strip-components=1 encodes). Skip the bare directory
                # entry itself (len(parts) == 1) and strip that one
                # component from everything else.
                parts = Path(member.name).parts
                if len(parts) <= 1:
                    continue
                member.name = str(Path(*parts[1:]))
                members.append(member)
            # filter="data" (Python 3.12+): strips uid/gid/setuid-type
            # metadata on extraction -- this repo's host runs 3.14, where
            # extractall's default is about to change to exactly this, so
            # being explicit avoids both the DeprecationWarning today and a
            # silent behavior change later. Safe here regardless: these
            # release tarballs contain plain files/dirs, no symlinks/
            # devices/setuid bits worth preserving.
            tf.extractall(version_dir, members=members, filter="data")  # noqa: S202 -- fixed GitHub release tarball, sha256-verified just above
        tar_path.unlink(missing_ok=True)

        if engine == "parakeet":
            so_path = version_dir / spec.lib_filename
            if not probe_parakeet_private_symbols(so_path):
                raise BackendInstallError(
                    f"downloaded {engine}:{flavor}:{spec.tag} failed the private-symbol safety "
                    "probe (pk::set_num_threads/shutdown_backend not found in the downloaded "
                    ".so) -- refusing to install rather than risk a runtime crash later. "
                    "See worker/pk_private.hpp's docstring."
                )
        marker.write_text("ok")

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO installed_backends "
            "(engine, flavor, version, sha256, path, installed_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (engine, flavor, spec.tag, spec.sha256, str(version_dir), now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT installed_at, active FROM installed_backends "
            "WHERE engine=? AND flavor=? AND version=?",
            (engine, flavor, spec.tag),
        ).fetchone()
    return InstalledBackend(engine, flavor, spec.tag, spec.sha256, str(version_dir), row[0], bool(row[1]))


def activate(conn: sqlite3.Connection, engine: str, flavor: str, version: str) -> None:
    """Marks exactly one installed (engine, flavor, version) active,
    deactivating any other installed version of the SAME engine (a worker
    only ever links against one .so at a time, regardless of flavor) --
    does not touch other engines' active rows.
    """
    ensure_schema(conn)
    with _lock:
        row = conn.execute(
            "SELECT 1 FROM installed_backends WHERE engine=? AND flavor=? AND version=?",
            (engine, flavor, version),
        ).fetchone()
        if row is None:
            raise BackendInstallError(f"{engine}:{flavor}:{version} is not installed")
        conn.execute("UPDATE installed_backends SET active=0 WHERE engine=?", (engine,))
        conn.execute(
            "UPDATE installed_backends SET active=1 WHERE engine=? AND flavor=? AND version=?",
            (engine, flavor, version),
        )
        conn.commit()


def get_active(conn: sqlite3.Connection, engine: str) -> InstalledBackend | None:
    ensure_schema(conn)
    with _lock:
        row = conn.execute(
            "SELECT engine, flavor, version, sha256, path, installed_at, active "
            "FROM installed_backends WHERE engine=? AND active=1",
            (engine,),
        ).fetchone()
    if row is None:
        return None
    return InstalledBackend(*row[:6], bool(row[6]))


def list_installed(conn: sqlite3.Connection) -> list[InstalledBackend]:
    ensure_schema(conn)
    with _lock:
        rows = conn.execute(
            "SELECT engine, flavor, version, sha256, path, installed_at, active "
            "FROM installed_backends ORDER BY engine, flavor, installed_at"
        ).fetchall()
    return [InstalledBackend(*r[:6], bool(r[6])) for r in rows]


def get_active_lib_dir(db_path: str, engine: str) -> str | None:
    """Convenience wrapper for callers (live_stt/session.py) that don't
    already hold an open DB connection -- opens one, reads, closes. Not on
    any hot path (once per worker spawn, not per audio chunk), so a fresh
    connection per call is fine; avoids threading a DB handle through
    CallSession's constructor, which every existing test constructs
    directly without one. Returns None (never raises, never creates a DB
    file) if db_path doesn't exist yet -- the common case for a deployment
    that has never installed a backend via this mechanism, which must
    behave exactly as before (fall back to the image-baked default).
    """
    if db_path != ":memory:" and not os.path.exists(db_path):
        return None
    conn = db.connect(db_path)
    try:
        active = get_active(conn, engine)
    finally:
        conn.close()
    return active.path if active is not None else None
