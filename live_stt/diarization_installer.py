"""Runtime installer for diarization ENGINES' heavy Python dependency
trees (pyannote.audio+torch, nemo_toolkit[asr]) -- the diarization-engine
analogue of live_stt/backends/installer.py, which does the equivalent job
for ASR's compiled C++ artifacts.

Exists to close the last piece of per-capability image-tag proliferation
LS-5 set out to eliminate: before this, getting diarization at all meant
choosing a separately-BUILT image (runtime-cuda-diarize for pyannote,
runtime-cuda-diarize-sortformer for +nemo) at deploy time -- exactly the
"no more GPU/CPU tags on the docker container" problem the rest of LS-5
already solved for ASR's cpu/cuda flavors, just not yet applied to this
corner. Now there is exactly ONE built image (see Dockerfile's single
`runtime` stage and build.sh), and a diarization engine's dependencies are
installed into it, if/when actually needed, via
`POST /api/diarization-engines/install {"engine": "pyannote"}`.

Mechanically different from live_stt/backends/installer.py's job (no
compiled binaries, no docker/oci_image dance): this shells out to `pip
install --target <engine's persistent dir> -r <requirements file>`, where
the requirements file is one of requirements-diarization.txt /
requirements-diarization-sortformer.txt -- still shipped in the image
(COPY'd, just no longer `RUN pip install`'d, at build time) so this module
has something to point pip at without needing network access to fetch the
requirements list itself, only the packages it names.

`--target <dir>` (not a venv, not `pip install --user`) is deliberate:
it installs into a plain, self-contained, `rm -rf`-removable directory
that needs no activation step -- `sys.path.insert(0, <dir>)` is all that's
needed to make it resolvable, which is exactly what
live_stt/diarization.py's `_ensure_deps_on_path()` does immediately before
each engine's lazy `import pyannote.audio` / `import nemo...`. That
function runs in EVERY process that might load an engine -- the main
server process (for status checks) and, critically, the
`live_stt.diarize_worker` subprocess `diarize_file()` spawns by default
(see that module's docstring) -- so an engine installed once survives
`docker recreate` (the persistent dir lives under Settings.
diarization_deps_dir, the same already-mounted ./data volume as
backends_dir/db_path) without needing to re-run pip on every restart.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# key -> requirements file (repo-root-relative, shipped in the image --
# see Dockerfile's runtime stage COPY). Keep in sync with
# live_stt/diarization_models.py's DiarizationModelSpec.engine values.
REQUIREMENTS_FILES: dict[str, str] = {
    "pyannote": "requirements-diarization.txt",
    "nemo_sortformer": "requirements-diarization-sortformer.txt",
}

# Real, measured: nemo_toolkit[asr]'s dependency tree (lhotse, hydra-core,
# lightning, transformers, ...) took ~85s of pure `pip install` time in
# the old Dockerfile-baked build (see Dockerfile git history) even with a
# warm pip cache and fast mirror -- a cold runtime install over a
# real network, plus resolving/downloading torch itself for the pyannote
# case (~550MB wheel), needs real headroom. 30 minutes matches
# live_stt/backends/installer.py's own docker-pull-scale patience,
# not a tight budget.
_INSTALL_TIMEOUT_SEC = 1800.0


class DiarizationInstallError(RuntimeError):
    """Raised for any install failure -- unknown engine, missing
    requirements file, or a real pip failure (network, resolver, disk)."""


@dataclass(frozen=True, slots=True)
class DiarizationInstallResult:
    engine: str
    path: str
    already_installed: bool


def _repo_root() -> Path:
    """The parent of the live_stt/ package this module lives in -- matches
    where the Dockerfile's runtime stage COPYs the requirements files to
    (repo root, same level as requirements.txt). A separate function
    (not inlined) purely so tests can monkeypatch it to point at a fake
    requirements file, since the real requirements-diarization*.txt files
    are deliberately NOT shipped in the test-unit Docker image (same
    offline-safety-by-construction convention CLAUDE.md documents for the
    worker binary/model -- an accidentally-integration-shaped "unit" test
    must fail loudly here, not silently find and use the real files).
    """
    return Path(__file__).resolve().parent.parent


def engine_deps_dir(deps_root: str, engine: str) -> Path:
    return Path(deps_root) / engine


def is_installed(deps_root: str, engine: str) -> bool:
    """True if `engine`'s marker file is present. A marker file, not an
    import probe: an import probe would need sys.path already mutated
    (see live_stt/diarization.py's _ensure_deps_on_path) and would leave a
    real, partially-initialized torch/pyannote import sitting in
    sys.modules as a side effect of a mere status check -- exactly the
    kind of surprising global state a GET-style query must not cause.
    """
    return (engine_deps_dir(deps_root, engine) / ".installed").is_file()


def install(deps_root: str, engine: str) -> DiarizationInstallResult:
    """Idempotent: a second call for an already-installed engine is a
    cheap no-op (returns already_installed=True), the same contract
    live_stt/backends/installer.py's install() has for a version already
    on disk -- so a retried/duplicate admin request never re-downloads
    hundreds of MB it doesn't need to.
    """
    if engine not in REQUIREMENTS_FILES:
        raise DiarizationInstallError(
            f"unknown diarization engine {engine!r}; known: {sorted(REQUIREMENTS_FILES)}"
        )

    target_dir = engine_deps_dir(deps_root, engine)
    marker = target_dir / ".installed"
    if marker.is_file():
        return DiarizationInstallResult(engine, str(target_dir), already_installed=True)

    requirements_path = _repo_root() / REQUIREMENTS_FILES[engine]
    if not requirements_path.is_file():
        raise DiarizationInstallError(
            f"requirements file not found in this image: {requirements_path} -- "
            "was it dropped from the Dockerfile's runtime stage?"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--target",
        str(target_dir),
        "-r",
        str(requirements_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_INSTALL_TIMEOUT_SEC, check=False
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise DiarizationInstallError(
            f"pip install timed out after {_INSTALL_TIMEOUT_SEC:.0f}s installing {engine!r}"
        ) from exc
    except OSError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise DiarizationInstallError(f"failed to run pip installing {engine!r}: {exc}") from exc

    if result.returncode != 0:
        # Roll back a half-installed directory -- same reasoning as
        # live_stt/backends/installer.py's install(): a half-populated
        # directory with no marker file must not be mistaken for a
        # successful (if odd) install on a later is_installed() check or
        # confuse a retry into thinking some subset is already there.
        shutil.rmtree(target_dir, ignore_errors=True)
        # pip's own error output can be long (a failed wheel build's full
        # traceback); the last ~4000 chars keeps the actual failure
        # (usually at the end) without unbounded growth in an HTTP
        # response body or a log line.
        raise DiarizationInstallError(
            f"pip install failed for {engine!r} (exit {result.returncode}): "
            f"{result.stderr[-4000:] or result.stdout[-4000:]}"
        )

    marker.write_text("1", encoding="utf-8")
    return DiarizationInstallResult(engine, str(target_dir), already_installed=False)


def status(deps_root: str) -> dict[str, dict[str, object]]:
    """{"pyannote": {"installed": bool, "path": str}, ...} -- every known
    engine, not just installed ones, so a caller can see what's available
    to install, mirroring live_stt/backends/registry.py's BACKENDS being
    the full catalog regardless of what's actually installed.
    """
    return {
        engine: {
            "installed": is_installed(deps_root, engine),
            "path": str(engine_deps_dir(deps_root, engine)),
            "requirements_file": req_file,
        }
        for engine, req_file in REQUIREMENTS_FILES.items()
    }
