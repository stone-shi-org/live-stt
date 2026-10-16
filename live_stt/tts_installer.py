"""Runtime installer for TTS ENGINES' Python dependency trees (kokoro,
piper-tts) -- the TTS analogue of live_stt/diarization_installer.py, which
does the equivalent job for diarization engines (and, before that,
live_stt/backends/installer.py for ASR's compiled C++ artifacts).

Exists because kokoro/piper were found to be silently unusable in
production: live_stt/tts.py used to catch ImportError and fall back to a
synthetic sine-wave tone instead of raising -- meaning `POST
/v1/audio/speech` returned a real HTTP 200 with real-looking audio bytes
that were NOT speech, on any deployment that never happened to have these
packages installed (confirmed for real: the actual 10.100.0.50 production
container returned a 200 with a pure 480Hz tone for a real request, with
no operator-visible signal anything was wrong). That silent-fallback bug
is fixed separately, in live_stt/tts.py itself (raises TTSUnavailableError
now); THIS module is what makes "go install the real engine" an actual,
one-call, no-rebuild option once that error is visible -- via
`POST /api/tts-engines/install {"engine": "kokoro" | "piper"}`, mirroring
the diarization-engines endpoint exactly.

Mechanically identical to diarization_installer.py: `pip install --target
<persistent dir> -r <requirements file>`, where the requirements file
(requirements-tts-kokoro.txt / requirements-tts-piper.txt) is still shipped
as a plain file in the image (COPY'd, never `RUN pip install`ed at build
time) so exact, repo-pinned versions install regardless of what the repo's
default branch has since changed. live_stt/tts.py's `_ensure_deps_on_path()`
prepends the installed directory to sys.path immediately before each
engine's lazy import, in every process that might load one -- the main
server process and the `live_stt.tts_worker` subprocess `synthesize()`
spawns by default.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from live_stt.torch_index import deps_marker, marker_is_current, pip_index_args, swap_in

# key -> requirements file (repo-root-relative, shipped in the image --
# see Dockerfile's runtime stage COPY). Keep in sync with
# live_stt/tts_models.py's TTSModelSpec.engine values.
REQUIREMENTS_FILES: dict[str, str] = {
    "kokoro": "requirements-tts-kokoro.txt",
    "piper": "requirements-tts-piper.txt",
}

# Same patience budget as live_stt/diarization_installer.py's own install()
# -- a cold pip resolve+download over a real network is the dominant cost,
# not anything CPU-bound here.
_INSTALL_TIMEOUT_SEC = 1800.0


class TTSInstallError(RuntimeError):
    """Raised for any install failure -- unknown engine, missing
    requirements file, or a real pip failure (network, resolver, disk)."""


@dataclass(frozen=True, slots=True)
class TTSInstallResult:
    engine: str
    path: str
    already_installed: bool
    # True when an install built against an outdated torch (see torch_index.deps_marker)
    # was replaced by a fresh one.
    replaced_stale: bool = False


def _repo_root() -> Path:
    """See live_stt/diarization_installer.py's identical helper for why
    this is a separate function (tests monkeypatch it to point at a fake
    requirements file -- the real requirements-tts-*.txt files are
    deliberately NOT shipped in the test-unit Docker image)."""
    return Path(__file__).resolve().parent.parent


def engine_deps_dir(deps_root: str, engine: str) -> Path:
    return Path(deps_root) / engine


def is_installed(deps_root: str, engine: str) -> bool:
    """Marker-file-based, not an import probe -- see
    live_stt/diarization_installer.py's is_installed() for why a status
    check must never itself trigger a real, partially-initialized import
    as a side effect."""
    return (engine_deps_dir(deps_root, engine) / ".installed").is_file()


def install(deps_root: str, engine: str) -> TTSInstallResult:
    """Idempotent: a second call for an already-installed engine is a
    cheap no-op (returns already_installed=True)."""
    if engine not in REQUIREMENTS_FILES:
        raise TTSInstallError(f"unknown TTS engine {engine!r}; known: {sorted(REQUIREMENTS_FILES)}")

    target_dir = engine_deps_dir(deps_root, engine)
    marker = target_dir / ".installed"
    stale = False
    if marker.is_file():
        if marker_is_current(marker, engine):
            return TTSInstallResult(engine, str(target_dir), already_installed=True)
        # Installed, but against an outdated torch build (e.g. CUDA 13, which has no
        # Pascal support). Build the replacement NEXT TO the live tree and swap it in
        # only on success: the old install keeps serving requests meanwhile, and a
        # failed upgrade leaves it untouched.
        stale = True
    staging = target_dir.with_name(engine + ".new") if stale else target_dir
    if stale:
        shutil.rmtree(staging, ignore_errors=True)

    requirements_path = _repo_root() / REQUIREMENTS_FILES[engine]
    if not requirements_path.is_file():
        raise TTSInstallError(
            f"requirements file not found in this image: {requirements_path} -- "
            "was it dropped from the Dockerfile's runtime stage?"
        )

    staging.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--target",
        str(staging),
        "-r",
        str(requirements_path),
        *pip_index_args(),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_INSTALL_TIMEOUT_SEC, check=False
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise TTSInstallError(
            f"pip install timed out after {_INSTALL_TIMEOUT_SEC:.0f}s installing {engine!r}"
        ) from exc
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise TTSInstallError(f"failed to run pip installing {engine!r}: {exc}") from exc

    if result.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise TTSInstallError(
            f"pip install failed for {engine!r} (exit {result.returncode}): "
            f"{result.stderr[-4000:] or result.stdout[-4000:]}"
        )

    if stale:
        swap_in(staging, target_dir)
    marker.write_text(deps_marker(engine), encoding="utf-8")
    return TTSInstallResult(engine, str(target_dir), already_installed=False, replaced_stale=stale)


def status(deps_root: str) -> dict[str, dict[str, object]]:
    """{"kokoro": {"installed": bool, "path": str}, ...} -- every known
    engine, not just installed ones."""
    return {
        engine: {
            "installed": is_installed(deps_root, engine),
            # installed but built against outdated torch -- POST install to rebuild
            "stale": is_installed(deps_root, engine)
            and not marker_is_current(engine_deps_dir(deps_root, engine) / ".installed", engine),
            "path": str(engine_deps_dir(deps_root, engine)),
            "requirements_file": req_file,
        }
        for engine, req_file in REQUIREMENTS_FILES.items()
    }
