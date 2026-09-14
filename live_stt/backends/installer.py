"""Runtime backend installer (LS-5 Phase 7b/7d): download+verify+install+
activate a BackendSpec (live_stt/backends/registry.py) into a persistent
install directory -- the Python-side, admin-UI-triggered analogue of
scripts/fetch_backend.sh (which exists for the EARLY Docker build layer,
before the live_stt package/its deps are even installed -- see that
script's docstring for why the two aren't unified into one implementation,
and why it only covers the two *_cpu entries used at build time).

Tracked in the same SQLite DB as live_stt/settings_store.py (a new
`installed_backends` table, schema created here via live_stt/db.py's
connection) so both live behind the one process-shared DB file under the
already-mounted ./data volume -- this is what "survive docker recreate"
means concretely, same as live_stt/settings_store.py's DB itself.

Two fetch mechanisms, matching BackendSpec.source (see registry.py's
docstring for why both exist):
- "github_release": download the whole tarball, verify its sha256 whole,
  extract (tar preserves the SONAME symlink chain already baked into the
  release by upstream).
- "oci_image": `docker create` + `docker cp` specific files out of a
  digest-pinned image, verify EACH file's own sha256, then create the
  SONAME symlink chain ourselves (individual file copies carry none).
  Requires a reachable `docker` binary talking to a real daemon (e.g. a
  mounted host socket) -- a real, narrower requirement than
  "github_release" has, and the reason whisper:cuda specifically needs it
  (see registry.py's docstring: ggerganov/whisper.cpp publishes no
  Linux+CUDA GitHub release at all).

What "install" actually changes at runtime: BOTH the "parakeet" and
"parakeet_tdt" engines (live_stt/backends/registry.py's ENGINE_ALIASES --
they share the exact same underlying library) and the "whisper" engine
(live_stt/session.py checks get_active_lib_dir(..., "parakeet"/"whisper")
when spawning a worker) -- because all three worker binaries are now built
against a dynamically-linked .so (worker/CMakeLists.txt's
LSTT_PARAKEET_PREBUILT / LSTT_WHISPER_PREBUILT). No engine is
build-time-pinned to one flavor anymore as of this phase.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import sqlite3
import subprocess
import tarfile
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from live_stt import db
from live_stt.backends.registry import BackendSpec, canonical_engine, resolve

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
    """A download, checksum, extraction, docker, or safety-probe step failed."""


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


def _install_github_release(spec: BackendSpec, version_dir: Path) -> None:
    tar_path = version_dir / spec.asset_filename
    _download_and_verify(spec, tar_path)
    with tarfile.open(tar_path) as tf:
        members = []
        for member in tf.getmembers():
            # Every registered release tarball wraps its contents in one
            # top-level "<name>-<version>-<os>-<arch>/" directory --
            # confirmed by actually listing both tarballs' contents while
            # building this (same fact scripts/fetch_backend.sh's
            # --strip-components=1 encodes). Skip the bare directory entry
            # itself (len(parts) == 1) and strip that one component from
            # everything else -- tar preserves the SONAME symlink chain
            # upstream already baked in, no extra symlink step needed here.
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            member.name = str(Path(*parts[1:]))
            members.append(member)
        # filter="data" (Python 3.12+): strips uid/gid/setuid-type metadata
        # on extraction, avoiding both a DeprecationWarning on this repo's
        # Python 3.14 host and a future behavior change -- safe here since
        # these release tarballs contain plain files/symlinks, nothing
        # relying on preserved ownership or setuid bits.
        tf.extractall(version_dir, members=members, filter="data")  # noqa: S202 -- fixed release tarball, sha256-verified just above
    tar_path.unlink(missing_ok=True)


def _run_docker(args: list[str]) -> None:
    try:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=300, check=False
        )
    except FileNotFoundError as exc:
        raise BackendInstallError(
            "docker is not available in this environment -- oci_image-sourced backends "
            "(see live_stt/backends/registry.py) need a reachable docker binary/daemon "
            "(e.g. a mounted /var/run/docker.sock), unlike github_release-sourced ones"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BackendInstallError(f"docker {' '.join(args)} timed out") from exc
    if result.returncode != 0:
        raise BackendInstallError(f"docker {' '.join(args)} failed: {result.stderr.strip()}")


def _install_oci_image(spec: BackendSpec, version_dir: Path) -> None:
    image_ref = f"{spec.image_registry}/{spec.image_repo}@{spec.image_digest}"
    # version_dir.name is spec.pinned_version, which for an oci_image source
    # IS spec.image_digest (see registry.py's pinned_version property) --
    # e.g. "sha256:6042be...". That colon is a perfectly normal filesystem
    # character (version_dir itself is fine on disk) but Docker container
    # names only allow [a-zA-Z0-9][a-zA-Z0-9_.-]* -- found for real the hard
    # way, installing whisper:cuda against a live deployment: `docker
    # create --name lstt-backend-install-57-sha256:604...` failed with
    # "Invalid container name", a DIFFERENT error from (and only reachable
    # after fixing) the earlier "docker not available"/"Cannot connect to
    # the Docker daemon" gaps. Sanitize just the container name, not
    # version_dir itself -- the directory layout is unaffected.
    container_name = f"lstt-backend-install-{os.getpid()}-{version_dir.name.replace(':', '-')}"
    _run_docker(["create", "--platform", "linux/amd64", "--name", container_name, image_ref])
    try:
        for dest_name, (path_in_image, expected_sha) in spec.image_files.items():
            dest_path = version_dir / dest_name
            _run_docker(["cp", f"{container_name}:{path_in_image}", str(dest_path)])
            actual = hashlib.sha256(dest_path.read_bytes()).hexdigest()
            if actual != expected_sha:
                raise BackendInstallError(
                    f"sha256 mismatch for {dest_name} extracted from {image_ref}: "
                    f"expected {expected_sha}, got {actual}"
                )
    finally:
        # Best-effort cleanup -- a failed rm here must not mask a real
        # install error above (or silently succeed on top of one).
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)

    # Individual `docker cp` file copies carry no symlink metadata of their
    # own -- unlike the github_release tar path, the SONAME chain has to be
    # created explicitly here. Confirmed required by actually linking
    # against a directory missing it (see registry.py's docstring).
    for link_name, target_name in spec.symlinks.items():
        link_path = version_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(target_name)


def _fetch_headers(spec: BackendSpec, version_dir: Path) -> None:
    for filename, url in spec.header_urls.items():
        dest = version_dir / filename
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310 -- fixed raw.githubusercontent.com URL
        except OSError as exc:
            raise BackendInstallError(f"header fetch failed for {url}: {exc}") from exc


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
    therefore the same pinned registry version -- is a fast no-op (checks a
    marker file, does not re-fetch). Does NOT activate -- see activate()
    below; installing and activating are deliberately separate steps
    (LS-5's own design: a bad download or a bad version must not
    auto-take-effect).
    """
    ensure_schema(conn)
    engine = canonical_engine(engine)
    spec = resolve(engine, flavor)
    version = spec.pinned_version
    # The ON-DISK directory name is sanitized separately from `version`
    # (which stays exactly spec.pinned_version -- the DB column, and what
    # the install/activate API returns/accepts, both unchanged). For an
    # oci_image source, pinned_version IS the image digest (e.g.
    # "sha256:6042be..."), and that literal colon is fatal later: this
    # directory can become session.py's ggml_lib_dir, which
    # WorkerHandle.spawn() puts straight into LD_LIBRARY_PATH -- a
    # COLON-DELIMITED path list. The dynamic linker splits on ':', turning
    # one real directory into two bogus, nonexistent ones, so every one of
    # that directory's OWN `.so` files' transitive dependencies silently
    # fails to resolve. Found for real against the actual 10.100.0.50
    # deployment: `ldd` on the installed libggml-cuda.so reported
    # libcudart.so.13/libcublas.so.13/etc as "not found" despite
    # LD_LIBRARY_PATH pointing exactly at the directory containing them.
    version_dir = Path(install_root) / engine / flavor / version.replace(":", "-")
    version_dir.mkdir(parents=True, exist_ok=True)

    marker = version_dir / ".installed"
    if not marker.exists():
        try:
            if spec.source == "github_release":
                _install_github_release(spec, version_dir)
            elif spec.source == "oci_image":
                _install_oci_image(spec, version_dir)
            else:
                raise BackendInstallError(f"unknown BackendSpec.source {spec.source!r}")

            if spec.header_urls:
                _fetch_headers(spec, version_dir)

            if engine == "parakeet":
                so_path = version_dir / spec.lib_filename
                if not probe_parakeet_private_symbols(so_path):
                    raise BackendInstallError(
                        f"downloaded {engine}:{flavor}:{version} failed the private-symbol safety "
                        "probe (pk::set_num_threads/shutdown_backend not found in the downloaded "
                        ".so) -- refusing to install rather than risk a runtime crash later. "
                        "See worker/pk_private.hpp's docstring."
                    )
        except Exception:
            # Never leave a half-extracted directory that a later
            # get_active_lib_dir() call could point a real worker spawn at
            # -- a partial install is exactly as unsafe as a corrupt one.
            shutil.rmtree(version_dir, ignore_errors=True)
            raise
        marker.write_text("ok")

    now = datetime.now(timezone.utc).isoformat()
    sha_summary = spec.sha256 or spec.image_digest
    with _lock:
        conn.execute(
            "INSERT OR IGNORE INTO installed_backends "
            "(engine, flavor, version, sha256, path, installed_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (engine, flavor, version, sha_summary, str(version_dir), now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT installed_at, active FROM installed_backends "
            "WHERE engine=? AND flavor=? AND version=?",
            (engine, flavor, version),
        ).fetchone()
    return InstalledBackend(engine, flavor, version, sha_summary, str(version_dir), row[0], bool(row[1]))


def activate(conn: sqlite3.Connection, engine: str, flavor: str, version: str) -> None:
    """Marks exactly one installed (engine, flavor, version) active,
    deactivating any other installed version of the SAME (canonical)
    engine (a worker only ever links against one .so at a time, regardless
    of flavor) -- does not touch other engines' active rows.
    """
    ensure_schema(conn)
    engine = canonical_engine(engine)
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
    engine = canonical_engine(engine)
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


def get_active_backend(db_path: str, engine: str) -> InstalledBackend | None:
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
        return get_active(conn, engine)
    finally:
        conn.close()


def get_active_lib_dir(db_path: str, engine: str) -> str | None:
    """As get_active_backend, but returns just the install directory path
    (the common case live_stt/session.py needs for LD_LIBRARY_PATH)."""
    active = get_active_backend(db_path, engine)
    return active.path if active is not None else None
