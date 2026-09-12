"""Backend artifact registry (LS-5 Phase 7b): which upstream prebuilt
release artifact to download for each (engine, flavor) pair, replacing
"vendor + build from source" with "download and verify" -- one layer down
from live_stt/models.py's registry (model WEIGHTS) for the engine
BINARY/LIBRARY itself.

Real GitHub release artifacts, verified for real while building this (see
the LS-5 Jira ticket for the full nm -D/ldd investigation) -- not invented:

- mudler/parakeet.cpp publishes real per-(os,arch,flavor) release tarballs.
  The "lib" tarball for linux-cpu-x64 is just parakeet_capi.h +
  libparakeet.so -- ldd shows it depends on nothing but
  libgomp/libstdc++/libm/libgcc_s/libc, i.e. ggml is linked STATICALLY into
  libparakeet.so itself, unlike this repo's old from-source build (where
  ggml was always a separate trio of shared libs). `nm -D` on this exact
  .so shows pk::set_num_threads/shutdown_backend/num_threads exported as
  public dynamic symbols even though the private header that originally
  declared them (parakeet.cpp's src/ggml_graph.hpp) isn't shipped in this
  package -- see worker/pk_private.hpp, which declares them ourselves.
- ggerganov/whisper.cpp publishes a "bin" tarball (no separate lib+headers
  package for Linux) containing libwhisper.so + a SET of per-microarch
  libggml-cpu-<name>.so variants (runtime CPU dispatch). Registered here
  (real tag, real sha256, real fetch verified via
  scripts/fetch_backend.sh) but NOT yet wired into
  worker/CMakeLists.txt's build -- that engine's from-source static build
  is unchanged this pass; a real follow-up, not a placeholder pretending
  to be done.

CUDA flavors are NOT registered yet for either engine: a real cuda entry
needs a full download (500MB-900MB+ per file, verified live against the
actual release listing while researching this) and real GPU hardware to
confirm it actually runs, neither of which was in scope for this pass. Add
them following this exact pattern once that's actually been done, with a
freshly computed sha256 from a full download, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendSpec:
    engine: str  # "parakeet" | "whisper" -- matches live_stt/models.py's ModelSpec.engine values
    flavor: str  # "cpu" | "cuda"
    repo: str  # "owner/name" on github.com
    tag: str  # exact release tag, pinned -- NEVER "latest" (see this module's docstring)
    asset_filename: str
    sha256: str
    # Filename of the actual shared library the worker links against /
    # live_stt/session.py's runtime dispatch points LD_LIBRARY_PATH at,
    # relative to the extracted directory's top level.
    lib_filename: str
    # Whether the release tarball itself ships a matching header
    # (parakeet's "lib" package does -- parakeet_capi.h; whisper's "bin"
    # package does not, its headers are fetched separately at the same tag).
    ships_headers: bool
    # Wired into worker/CMakeLists.txt's build / live_stt/session.py's
    # runtime dispatch today, vs. registered-and-fetchable-but-not-yet-
    # consumed. False entries are real and fetchable (scripts/fetch_backend.sh
    # works today) but installing+activating one has no effect on which
    # binary the service actually spawns until a follow-up wires it up --
    # see live_stt/backends/installer.py's docstring.
    wired: bool = False


BACKENDS: dict[tuple[str, str], BackendSpec] = {
    ("parakeet", "cpu"): BackendSpec(
        engine="parakeet",
        flavor="cpu",
        repo="mudler/parakeet.cpp",
        tag="v0.5.0",
        asset_filename="parakeet-v0.5.0-lib-linux-cpu-x64.tar.gz",
        sha256="07cc1095924a134fe5dc7e6e44247fea4eada70ba56a663f73281b96da0624f2",
        lib_filename="libparakeet.so",
        ships_headers=True,
        wired=True,
    ),
    ("whisper", "cpu"): BackendSpec(
        engine="whisper",
        flavor="cpu",
        repo="ggerganov/whisper.cpp",
        tag="b5130",
        asset_filename="whisper-bin-ubuntu-x64.tar.gz",
        sha256="53e7fd8b5764edad916b8848dd0af6abb1ff1d3b86c899e79c78652412536c32",
        lib_filename="libwhisper.so",
        ships_headers=False,
        wired=False,
    ),
}


def resolve(engine: str, flavor: str) -> BackendSpec:
    try:
        return BACKENDS[(engine, flavor)]
    except KeyError as exc:
        raise KeyError(
            f"no registered backend for engine={engine!r} flavor={flavor!r}; "
            f"available: {sorted(BACKENDS)}"
        ) from exc


def available_flavors(engine: str) -> list[str]:
    return sorted(flavor for (e, flavor) in BACKENDS if e == engine)
