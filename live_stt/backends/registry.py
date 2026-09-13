"""Backend artifact registry (LS-5 Phase 7b/7d): which upstream prebuilt
artifact to download for each (engine, flavor) pair, replacing "vendor +
build from source" with "download and verify" -- one layer down from
live_stt/models.py's registry (model WEIGHTS) for the engine
BINARY/LIBRARY itself.

Two real, verified sources, not invented (see the LS-5 Jira ticket for the
full nm -D/ldd/docker-extraction investigation):

- "github_release": a real GitHub release asset, downloaded and
  sha256-verified whole (scripts/fetch_backend.sh mirrors this in bash for
  the early Docker build layer -- see that script's docstring for why it's
  a second implementation, not a shared one). Used for parakeet:cpu,
  parakeet:cuda, and whisper:cpu.
- "oci_image": specific files extracted from a real, digest-pinned OCI
  container image (`docker create` + `docker cp`, see
  live_stt/backends/installer.py's _install_oci_image), each individually
  sha256-verified. Used for whisper:cuda ONLY -- ggerganov/whisper.cpp
  publishes no Linux+CUDA GitHub release at all (confirmed by scanning
  every release back through 2025: only Windows cublas builds exist), but
  DOES publish `ghcr.io/ggml-org/whisper.cpp:main-cuda` for linux/amd64.
  Pinned by the image's own immutable manifest-index digest (NOT the
  floating "main-cuda" tag) -- resolved for real via the real
  ghcr.io token+manifest HTTP endpoints, not assumed. Requires a reachable
  `docker` (e.g. a mounted host socket) wherever the installer runs -- a
  real, documented limitation this source type has that "github_release"
  does not; see installer.py.

Real, load-bearing facts specific to each entry, confirmed by actually
building and linking (not assumed from either project's docs):

- parakeet's release .so's (both flavors) link ggml STATICALLY into
  themselves -- `ldd` on the cpu flavor shows nothing but
  libgomp/libstdc++/libm/libgcc_s/libc; the cuda flavor additionally
  bundles its OWN libcudart/libcublas/libcublasLt right in the tarball,
  needing only the real driver's libcuda.so.1 (injected by
  nvidia-container-toolkit's `--gpus all`) at actual runtime. `nm -D` on
  both shows pk::set_num_threads/shutdown_backend/num_threads exported as
  public dynamic symbols even though the private header that originally
  declared them (parakeet.cpp's src/ggml_graph.hpp) isn't shipped in
  either package -- see worker/pk_private.hpp, which declares them
  ourselves.
- whisper's release "bin" tarball (cpu) ships NO plain libggml-cpu.so at
  all -- only per-microarchitecture variants (libggml-cpu-x64.so,
  -icelake.so, -sapphirerapids.so, ...), loaded at RUNTIME via ggml's own
  `ggml_backend_load_all()` directory-scan/dlopen mechanism. This is NOT
  automatic inside whisper_init() -- every upstream example (cli.cpp,
  server.cpp, ...) calls it as literally the first line of main(), and our
  own worker/main_whisper.cpp now does too (LSTT_WHISPER_PREBUILT only --
  see that file's comment). Headers (whisper.h + the ggml/include/*.h it
  transitively pulls in) are not in the tarball at all and are fetched
  separately as raw source text at the SAME release tag.
- whisper:cuda's oci_image DOES ship a single already-selected
  libggml-cpu.so/libggml-cuda.so (that image was built for one specific
  target, not distributed as multiple variants) -- and its libggml-cuda.so
  calls the CUDA Driver API's newer VMM-based memory-pool functions
  directly (cuMemCreate, cuMemMap, ... -- confirmed via
  `nm -D --undefined-only`), needing worker/cuda_driver_stub.c's
  build-time-only link stub the same way the from-source CUDA build needs
  CUDA::cuda_driver, but without requiring the CUDA toolkit to get it.
  Headers are fetched from the whisper:cpu entry's SAME tag (b5130) as a
  reasonable, but not byte-for-byte git-commit-confirmed, match -- the
  ghcr.io "main-cuda" image's own libwhisper.so.1.9.4 SONAME and the real
  "v1.9.4" GitHub tag were published the same day as b5130, but no exact
  commit-identity check was possible from the image's metadata alone; flag
  this for the real-GPU verification pass (a genuine header/binary ABI
  mismatch would very likely surface loudly -- a crash or obviously wrong
  struct field -- not silently).

No cpu/gpu split baked into the Docker image anymore as of this phase
(LS-5's explicit goal) -- every flavor here is fetched, either at image
build time (the *_cpu entries, baked in as the image's initial default) or
at runtime via the admin UI/API (any flavor, including switching an
engine to cuda without rebuilding the image) -- see
live_stt/session.py's per-engine active-backend dispatch and
live_stt/backends/installer.py's flavor-consistency check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ggerganov/whisper.cpp's ggml/include/*.h + include/whisper.h, fetched as
# raw source text (never built) at the SAME tag as the whisper:cpu release
# artifact below -- whisper.h #includes ggml.h/ggml-cpu.h, which itself
# #includes ggml-backend.h and others, so the full set is fetched rather
# than chasing one #include error at a time (confirmed necessary by
# actually compiling against a partial set first and hitting exactly that).
_WHISPER_HEADER_TAG = "b5130"
_WHISPER_HEADER_BASE = f"https://raw.githubusercontent.com/ggerganov/whisper.cpp/{_WHISPER_HEADER_TAG}"
_WHISPER_GGML_HEADERS = (
    "ggml-alloc.h", "ggml-backend.h", "ggml-blas.h", "ggml-cann.h", "ggml-cpp.h",
    "ggml-cpu.h", "ggml-cuda.h", "ggml-et.h", "ggml-hexagon.h", "ggml-metal.h",
    "ggml-opencl.h", "ggml-openvino.h", "ggml-opt.h", "ggml-rpc.h", "ggml-sycl.h",
    "ggml-virtgpu.h", "ggml-vulkan.h", "ggml-webgpu.h", "ggml-zdnn.h", "ggml-zendnn.h",
    "ggml.h", "gguf.h",
)
_WHISPER_HEADER_URLS: dict[str, str] = {
    name: f"{_WHISPER_HEADER_BASE}/ggml/include/{name}" for name in _WHISPER_GGML_HEADERS
}
_WHISPER_HEADER_URLS["whisper.h"] = f"{_WHISPER_HEADER_BASE}/include/whisper.h"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    engine: str  # "parakeet" | "whisper" -- matches live_stt/models.py's ModelSpec.engine values
    flavor: str  # "cpu" | "cuda"
    source: str  # "github_release" | "oci_image" -- see this module's docstring

    # --- source == "github_release" ---
    repo: str = ""  # "owner/name" on github.com
    tag: str = ""  # exact release tag, pinned -- NEVER "latest"
    asset_filename: str = ""
    sha256: str = ""  # of the whole downloaded tarball

    # --- source == "oci_image" ---
    image_repo: str = ""  # e.g. "ggml-org/whisper.cpp"
    image_registry: str = "ghcr.io"
    image_digest: str = ""  # pinned "sha256:..." manifest-index digest -- NEVER a floating tag
    # dest filename -> (absolute path inside the image, expected sha256)
    image_files: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Shared-lib SONAME symlink chain the installer must create after
    # extraction (dest symlink name -> its target filename, both within
    # the same install directory). github_release tarballs already embed
    # the correct chain (tar preserves symlinks natively); oci_image
    # extraction is individual file copies and needs this explicitly --
    # confirmed necessary by actually linking against a directory missing
    # these (ld/the dynamic loader look for the exact SONAME-versioned
    # name, e.g. "libwhisper.so.1", not just the fully-versioned real file).
    symlinks: dict[str, str] = field(default_factory=dict)

    # Headers not shipped in the artifact itself (whisper's "bin" release
    # AND its oci_image both lack them) -- fetched separately as raw
    # source text. dest filename -> URL.
    header_urls: dict[str, str] = field(default_factory=dict)

    # Filename of the primary shared library live_stt/session.py's runtime
    # dispatch points LD_LIBRARY_PATH at, relative to the installed
    # directory's top level.
    lib_filename: str = ""

    # Wired into worker/CMakeLists.txt's build / live_stt/session.py's
    # runtime dispatch today, vs. registered-and-fetchable-but-not-yet-
    # consumed by anything. All four entries below are wired as of this
    # phase; the field is kept (rather than deleted) as a deliberate,
    # explicit place to land a genuinely-not-yet-wired future entry
    # without silently implying it works.
    wired: bool = False

    @property
    def pinned_version(self) -> str:
        """The one string identifying which pinned build this spec is --
        `tag` for github_release, `image_digest` for oci_image (which has
        no tag at all, deliberately: see this module's docstring on why
        oci_image entries pin by digest, not a floating tag). Used as the
        `version` column in installed_backends and everywhere else that
        needs "the identity of what's pinned" without caring which source
        type produced it.
        """
        return self.tag or self.image_digest


BACKENDS: dict[tuple[str, str], BackendSpec] = {
    ("parakeet", "cpu"): BackendSpec(
        engine="parakeet",
        flavor="cpu",
        source="github_release",
        repo="mudler/parakeet.cpp",
        tag="v0.5.0",
        asset_filename="parakeet-v0.5.0-lib-linux-cpu-x64.tar.gz",
        sha256="07cc1095924a134fe5dc7e6e44247fea4eada70ba56a663f73281b96da0624f2",
        lib_filename="libparakeet.so",
        wired=True,
    ),
    ("parakeet", "cuda"): BackendSpec(
        engine="parakeet",
        flavor="cuda",
        source="github_release",
        repo="mudler/parakeet.cpp",
        tag="v0.5.0",
        asset_filename="parakeet-v0.5.0-lib-linux-cuda-x64.tar.gz",
        sha256="976d55f45d04f64dc53d6d7815f48abf906f52c70db90a66ab8accff3f1028fe",
        lib_filename="libparakeet.so",
        wired=True,
    ),
    ("whisper", "cpu"): BackendSpec(
        engine="whisper",
        flavor="cpu",
        source="github_release",
        repo="ggerganov/whisper.cpp",
        tag="b5130",
        asset_filename="whisper-bin-ubuntu-x64.tar.gz",
        sha256="53e7fd8b5764edad916b8848dd0af6abb1ff1d3b86c899e79c78652412536c32",
        header_urls=dict(_WHISPER_HEADER_URLS),
        lib_filename="libwhisper.so",
        wired=True,
    ),
    ("whisper", "cuda"): BackendSpec(
        engine="whisper",
        flavor="cuda",
        source="oci_image",
        image_repo="ggml-org/whisper.cpp",
        image_registry="ghcr.io",
        # The real manifest-INDEX digest for the "main-cuda" tag, resolved
        # live via ghcr.io's token+manifest endpoints (docker pull reports
        # the same digest) -- pinning this, not the tag, is what makes this
        # a stable reference despite "main-cuda" itself being a rolling
        # build. installer.py resolves the linux/amd64 platform manifest
        # from within this index.
        image_digest="sha256:6042be448d2f11be3dfadceef9cbfd86f38a5cc5901919e7ac442fd4fd5c82e1",
        image_files={
            "libwhisper.so.1.9.4": ("/app/build/bin/libwhisper.so.1.9.4", "12443f2c9d16e0ddeeb11509349c6fb54356947f7ac8f98cb71b56a6fc53f4fd"),
            "libggml.so.0.23.0": ("/app/build/bin/libggml.so.0.23.0", "dabe4c198708c859c668bd07bb530475696d84ffb04274f1b8d8ba0fac2e0ed7"),
            "libggml-base.so.0.23.0": ("/app/build/bin/libggml-base.so.0.23.0", "ead06f24b5546aa094603d306881ad5409e97181c35e79c1f5ca977d2c965ed7"),
            "libggml-cpu.so.0.23.0": ("/app/build/bin/libggml-cpu.so.0.23.0", "311356327699459fdefe28c9882aeb05e6bdba3a7d96e05c4186541a91153f87"),
            "libggml-cuda.so.0.23.0": ("/app/build/bin/libggml-cuda.so.0.23.0", "059ad52520576a0d025a4701b108996e869d847916c0c75c250ab1b7e3bdd1f4"),
            "libcudart.so.13.0.48": ("/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so.13.0.48", "6876f4841bdbe30b187dcf8d7a7d2d6579bf12a36eeb430c7b28484ccdeed89a"),
            "libcublas.so.13.0.0.19": ("/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcublas.so.13.0.0.19", "ea443b3d282af9c82e16be974860b5ba993020bbe98988a7be201bf0c0215e88"),
            "libcublasLt.so.13.0.0.19": ("/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcublasLt.so.13.0.0.19", "c92c4c24c257d638a61130f8a5f7f31eb6d6262c24f8e19816ff56b58b73a857"),
            "libnccl.so.2.27.7": ("/usr/lib/x86_64-linux-gnu/libnccl.so.2.27.7", "616933cf29bacefcfbc7bdc0222892f4682d2d4687c191ad3557a10cdb2dd723"),
        },
        # Full 3-level SONAME chain per library (unversioned -> soname ->
        # fully-versioned) -- confirmed REQUIRED by actually linking
        # against a directory missing the middle level (the dynamic
        # loader/linker look up the exact SONAME string, e.g.
        # "libwhisper.so.1", not just any file containing the right bytes).
        symlinks={
            "libwhisper.so": "libwhisper.so.1.9.4",
            "libwhisper.so.1": "libwhisper.so.1.9.4",
            "libggml.so": "libggml.so.0.23.0",
            "libggml.so.0": "libggml.so.0.23.0",
            "libggml-base.so": "libggml-base.so.0.23.0",
            "libggml-base.so.0": "libggml-base.so.0.23.0",
            "libggml-cpu.so": "libggml-cpu.so.0.23.0",
            "libggml-cpu.so.0": "libggml-cpu.so.0.23.0",
            "libggml-cuda.so": "libggml-cuda.so.0.23.0",
            "libggml-cuda.so.0": "libggml-cuda.so.0.23.0",
            "libcudart.so": "libcudart.so.13.0.48",
            "libcudart.so.13": "libcudart.so.13.0.48",
            "libcublas.so": "libcublas.so.13.0.0.19",
            "libcublas.so.13": "libcublas.so.13.0.0.19",
            "libcublasLt.so": "libcublasLt.so.13.0.0.19",
            "libcublasLt.so.13": "libcublasLt.so.13.0.0.19",
            "libnccl.so.2": "libnccl.so.2.27.7",
        },
        header_urls=dict(_WHISPER_HEADER_URLS),
        lib_filename="libwhisper.so",
        wired=True,
    ),
}


# live_stt/models.py's ModelSpec.engine value "parakeet_tdt" (the batch
# Parakeet TDT worker, live_stt_worker_parakeet_tdt) is a DIFFERENT worker
# BINARY from the streaming "parakeet" engine but links against the exact
# SAME underlying library (worker/CMakeLists.txt: both executables link
# the same `parakeet` CMake target) -- so it deliberately has no separate
# registry entry of its own; it reuses "parakeet"'s. Resolving/installing/
# activating a backend for either engine name is meant to affect both
# worker binaries identically. See live_stt/session.py's dispatch and
# live_stt/backends/installer.py's get_active_lib_dir (both apply this
# alias) for where this actually takes effect.
ENGINE_ALIASES: dict[str, str] = {"parakeet_tdt": "parakeet"}


def canonical_engine(engine: str) -> str:
    return ENGINE_ALIASES.get(engine, engine)


def resolve(engine: str, flavor: str) -> BackendSpec:
    engine = canonical_engine(engine)
    try:
        return BACKENDS[(engine, flavor)]
    except KeyError as exc:
        raise KeyError(
            f"no registered backend for engine={engine!r} flavor={flavor!r}; "
            f"available: {sorted(BACKENDS)}"
        ) from exc


def available_flavors(engine: str) -> list[str]:
    engine = canonical_engine(engine)
    return sorted(flavor for (e, flavor) in BACKENDS if e == engine)
