# syntax=docker/dockerfile:1

# ---- stage: parakeet engine, PREBUILT (LS-5 Phase 7b/7d) -------------------
# Fetches the CPU flavor -- the one baked into the image at build time as
# the initial default. Replaces this Dockerfile's previous from-source
# parakeet-build stage (compile parakeet.cpp's vendored submodule + its
# vendored ggml via CMake) with a straight download of the real, verified
# mudler/parakeet.cpp v0.5.0 linux-cpu-x64 release artifact --
# scripts/fetch_backend.sh, the same download/sha256-verify pattern
# scripts/fetch_model.sh already uses for model weights, one layer down for
# the engine library itself. See worker/CMakeLists.txt's
# LSTT_PARAKEET_PREBUILT option and worker/pk_private.hpp's docstring for
# the real nm -D/ldd investigation this is based on (LS-5 Jira ticket):
# the prebuilt libparakeet.so exports the two private pk:: symbols this
# worker needs as public dynamic symbols, and links ggml STATICALLY into
# itself (confirmed via ldd -- no separate libggml*.so at all, unlike the
# old from-source build). Net effect: no vendored parakeet.cpp checkout
# needed in the image build at all, and a build step measured in seconds
# (a ~2MB download) instead of the several minutes a from-source compile
# took.
#
# The CUDA flavor is ALSO fully downloadable now (registered in
# live_stt/backends/registry.py, self-contained, verified for real -- see
# that module's docstring) -- but deliberately fetched at RUNTIME instead
# of baked in here, via POST /api/backends/install (live_stt/backends/
# installer.py), not at image build time. That is the whole point of this
# phase: no cpu/gpu split baked into the Docker image at all anymore (see
# the "runtime" stage's own comment further down -- there is now exactly
# ONE image tag) -- an operator picks a flavor per engine, per deployment,
# without rebuilding or even recreating the image.
FROM python:3.12-slim AS parakeet-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY scripts/fetch_backend.sh scripts/fetch_backend.sh
RUN bash scripts/fetch_backend.sh parakeet cpu /out

# ---- stage: whisper engine, PREBUILT (LS-5 Phase 7d) -----------------------
# CPU flavor. Replaces the previous from-source whisper-build stage the same
# way parakeet-fetch replaced parakeet-build above -- see
# worker/CMakeLists.txt's LSTT_WHISPER_PREBUILT option and
# live_stt/backends/registry.py's docstring for the real investigation this
# is based on: ggerganov/whisper.cpp's "bin" release ships libwhisper.so +
# a SET of per-microarchitecture libggml-cpu-<name>.so variants (no plain
# libggml-cpu.so at all -- loaded at runtime via ggml's own
# ggml_backend_load_all() directory scan, which worker/main_whisper.cpp now
# calls explicitly -- see that file's comment), plus headers fetched
# separately as raw source text (not shipped in the release tarball).
FROM python:3.12-slim AS whisper-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY scripts/fetch_backend.sh scripts/fetch_backend.sh
RUN bash scripts/fetch_backend.sh whisper cpu /out
# Headers aren't in the release tarball -- fetched separately at the same
# tag scripts/fetch_backend.sh pins (see that script's own table). A tiny,
# one-off curl loop here rather than teaching fetch_backend.sh to also
# fetch loose files -- this is the ONLY engine that needs it.
RUN set -eux; \
    for f in ggml-alloc.h ggml-backend.h ggml-blas.h ggml-cann.h ggml-cpp.h \
             ggml-cpu.h ggml-cuda.h ggml-et.h ggml-hexagon.h ggml-metal.h \
             ggml-opencl.h ggml-openvino.h ggml-opt.h ggml-rpc.h ggml-sycl.h \
             ggml-virtgpu.h ggml-vulkan.h ggml-webgpu.h ggml-zdnn.h ggml-zendnn.h \
             ggml.h gguf.h; do \
        curl -fsSL -o "/out/$f" "https://raw.githubusercontent.com/ggerganov/whisper.cpp/b5130/ggml/include/$f"; \
    done; \
    curl -fsSL -o /out/whisper.h "https://raw.githubusercontent.com/ggerganov/whisper.cpp/b5130/include/whisper.h"

# ---- stage: docker CLI (LS-5 -- real gap found and closed post-deploy) -----
# live_stt/backends/installer.py's oci_image install path (whisper:cuda --
# see registry.py's docstring) shells out to a real `docker create`/
# `docker cp` against a real ghcr.io image to extract libwhisper.so.* without
# needing this Dockerfile to vendor a CUDA toolkit at build time. That means
# the RUNNING container needs an actual `docker` binary on PATH -- found
# missing for real when an operator tried `POST /api/backends/install`
# {"engine":"whisper","flavor":"cuda"} against a deployed container and got
# installer.py's own "docker is not available in this environment" error.
# Not `apt-get install docker.io`: that metapackage drags in dockerd,
# containerd, runc, containerd-shim-runc-v2, etc -- a full engine's worth of
# binaries this container will never run (it only ever talks to the HOST's
# daemon over a mounted socket, see docker-compose.yml's own comment on
# that). Fetched instead as Docker's real, pinned, sha256-verified static-CLI
# release tarball (the same download-verify-extract shape
# scripts/fetch_backend.sh already uses for the ASR engines), keeping only
# the single ~38MB `docker` binary and discarding the tarball's other seven
# files (docker-init/ctr/containerd/containerd-shim-runc-v2/dockerd/
# docker-proxy/runc) -- none of which this container has any use for.
FROM python:3.12-slim AS docker-cli-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o docker-cli.tgz \
        https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
    && echo "9b4f6fe406e50f9085ee474c451e2bb5adb119a03591f467922d3b4e2ddf31d3  docker-cli.tgz" | sha256sum -c - \
    && tar -xzf docker-cli.tgz docker/docker \
    && mkdir -p /out && mv docker/docker /out/docker \
    && chmod +x /out/docker

# ---- stage: the C++ worker, linked against the artifacts above ------------
# Builds live_stt_worker (parakeet), live_stt_worker_parakeet_tdt, and
# live_stt_worker_whisper in one `cmake --build` -- three targets of the
# same worker/CMakeLists.txt project, ALL linked against downloaded
# prebuilt artifacts as of this phase (LSTT_PARAKEET_PREBUILT and
# LSTT_WHISPER_PREBUILT both ON) -- no vendored parakeet.cpp/whisper.cpp
# submodule checkout, no CUDA toolkit, needed anywhere in this image build
# anymore (LS-5's explicit goal: no cpu/gpu split baked into the Docker
# image -- cuda flavors are fetched at RUNTIME instead, via the admin
# UI/API; see live_stt/backends/installer.py). libgomp1 is needed here too
# (not just in runtime) because CMake's find_package(OpenMP) probes for a
# working compiler+runtime pair at CONFIGURE time, not just link time.
FROM python:3.12-slim AS worker-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/ ./worker/
COPY --from=parakeet-fetch /out/ worker/prebuilt-parakeet/
COPY --from=whisper-fetch /out/ worker/prebuilt-whisper/
RUN cd worker && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DLSTT_PARAKEET_PREBUILT=ON \
        -DPARAKEET_PREBUILT_DIR=/src/live-stt-worker/worker/prebuilt-parakeet \
        -DLSTT_WHISPER_PREBUILT=ON \
        -DWHISPER_PREBUILT_DIR=/src/live-stt-worker/worker/prebuilt-whisper \
    && cmake --build build -j"$(nproc)"

# ---- stage: Python dependencies (shared by runtime and test-unit) ---------
FROM python:3.12-slim AS py-deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # grpc-core's default epoll1 poller has a fork-safety bug: forking a
    # worker subprocess (WorkerHandle.spawn -- every call, every rotation,
    # every crash recovery) from a process that also runs a live
    # grpc.aio.server() can hit "Check failed: next_worker->state ==
    # KICKED" and crash the WHOLE PROCESS, not just the spawn attempt.
    # Reproduced under load in this repo's own containerized test suite
    # (intermittent, worse under container CPU scheduling than natively).
    # GRPC_ENABLE_FORK_SUPPORT=1 alone does NOT fix it (tested); switching
    # off epoll1 does. This must be set in every environment that runs
    # live_stt/server.py, not just tests -- it is set here (the common
    # ancestor of runtime, test-unit, and test-integration) rather than only
    # in docker-compose.yml so nothing can accidentally run without it.
    GRPC_POLL_STRATEGY=poll
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple/

# ---- stage: proto codegen (shared by runtime and test-unit) ---------------
FROM py-deps AS proto-build
COPY proto/ ./proto/
COPY scripts/gen_proto.sh ./scripts/gen_proto.sh
RUN mkdir -p live_stt && bash scripts/gen_proto.sh

# ---- stage: runtime ---------------------------------------------------------
# THE single image tag this repo ships (LS-5's full ask, extended past ASR
# cpu/cuda to diarization engines too): no runtime-cuda / runtime-cuda-diarize
# / runtime-cuda-diarize-sortformer variants exist anymore. Every capability
# beyond the always-shipped parakeet/whisper CPU baseline -- any ASR engine's
# cuda flavor (live_stt/backends/installer.py) AND any diarization engine's
# heavy Python dependency tree (live_stt/diarization_installer.py, pyannote
# or nemo_sortformer) -- is installed into this SAME running container at
# RUNTIME, on demand, via the admin API. See CLAUDE.md for the full writeup
# of why the earlier per-capability image tags existed and why they were
# removed once the download-at-runtime mechanism covered diarization too.
FROM py-deps AS runtime
# libgomp1: OpenMP runtime ggml-cpu links against.
# ffmpeg: needed only once a diarization engine is actually installed at
# runtime (pyannote.audio decodes audio via torchcodec, which dlopens
# libavutil/libavcodec/etc at IMPORT time -- confirmed the hard way,
# see CLAUDE.md's "container-specific bug" entry from when this was still
# baked into a separate runtime-cuda-diarize stage). Baked into the one
# image unconditionally now rather than made itself runtime-installable:
# it's a small system package (not the multi-hundred-MB-to-multi-GB
# Python ML stacks this file's diarization_deps_dir mechanism exists for),
# and `apt-get install` at container runtime would need network-reachable
# package indexes AND `rm -rf /var/lib/apt/lists/*` from earlier layers
# undone -- not worth reintroducing for a small, always-safe-to-have utility.
# espeak-ng: for the kokoro TTS engine (live_stt/tts_installer.py) --
# kokoro's own README lists it as a dependency of its misaki G2P library,
# used for English out-of-dictionary word fallback and non-English
# languages. Tested for real WITHOUT it first: a real runtime kokoro
# install + a real "hello world" synthesis call both succeeded with no
# espeak-ng present at all -- misaki's English path fell back to a
# spaCy model (en_core_web_sm, auto-pip-installed on first use, a real
# ~13MB one-time download misaki does entirely on its own the first time
# ANY text is synthesized post-install, not something live_stt's own code
# triggers) rather than requiring espeak-ng. Kept in the image anyway,
# proactively, as cheap (a few MB) insurance for out-of-dictionary words
# or a future non-English kokoro voice -- not because a real failure was
# observed without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*
# HF_HOME: inside the already-mounted, already-writable /app/data volume
# (docker-compose.yml/ai.yml: live-stt-data:/app/data), harmless to set
# unconditionally -- only read by pyannote.audio/huggingface_hub, which
# aren't even importable until a diarization engine is installed at
# runtime. Ensures a gated model download, once it happens, survives
# `docker recreate` using infrastructure that already exists rather than a
# container-local cache re-downloaded (and re-burning the HF token's rate
# limit) on every restart.
ENV HF_HOME=/app/data/hf-cache
# CUDA JIT cache on the persistent volume. Pascal (sm_61) runs the CUDA 12 builds'
# PTX via driver JIT, which costs ~15s on the first request after every container
# start and writes to ~/.nv/ComputeCache -- ephemeral, so it was paid on EVERY
# restart. Persisting it makes that a one-time cost. The 256MB default cap is too
# small for several engines' kernels, and an over-cap cache silently evicts.
ENV CUDA_CACHE_PATH=/app/data/cuda-cache \
    CUDA_CACHE_MAXSIZE=2147483648

# See docker-cli-fetch's own comment above for why this is a pinned static
# binary and not `apt-get install docker.io`. This alone is NOT sufficient
# for live_stt/backends/installer.py's oci_image install path to work --
# the container also needs a real docker DAEMON reachable, i.e. the deploy's
# compose file must mount the host's /var/run/docker.sock (see
# docker-compose.yml's own comment on that mount and its real security
# tradeoff). Without that mount, installer.py's _run_docker() now gets past
# the FileNotFoundError this fixes and instead fails at the daemon-connect
# step with a clear "Cannot connect to the Docker daemon" error -- a
# different, correctly-diagnosable failure, not a silent hang or crash.
COPY --from=docker-cli-fetch /out/docker /usr/local/bin/docker

COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker /app/worker/live_stt_worker
COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker_parakeet_tdt /app/worker/live_stt_worker_parakeet_tdt
# LS-5 Phase 7b: libparakeet.so itself, not a separate libggml*.so set --
# the prebuilt release links ggml STATICALLY into libparakeet.so (verified
# via ldd: only libgomp/libstdc++/libm/libgcc_s/libc show up), unlike the
# old from-source build where ggml was always a separate trio of shared
# libs. $ORIGIN rpath (worker/CMakeLists.txt) resolves it from right here,
# same mechanism as before, a different file.
COPY --from=worker-build /src/live-stt-worker/worker/prebuilt-parakeet/libparakeet.so /app/worker/libparakeet.so
COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker_whisper /app/worker/whisper/live_stt_worker_whisper
# LS-5 Phase 7d: libwhisper.so + libggml*.so (co-located in the SAME
# /app/worker/whisper/ subdirectory as the binary, for $ORIGIN to resolve --
# unlike parakeet's single self-contained .so, whisper's prebuilt release
# ships libwhisper.so/libggml.so/libggml-base.so PLUS a whole SET of
# per-microarchitecture libggml-cpu-<name>.so variants (ggml's own
# ggml_backend_load_all() picks the matching one at runtime via dlopen --
# see worker/main_whisper.cpp's comment -- not something this Dockerfile
# selects at build time). No headers copied here -- those were compile-time
# only (worker-build already used them), and shipping them in the runtime
# image would be dead weight.
COPY --from=worker-build /src/live-stt-worker/worker/prebuilt-whisper/*.so* /app/worker/whisper/
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY run.py version.txt* ./
# Shipped as plain files, deliberately NOT `pip install -r`'d here --
# live_stt/diarization_installer.py's install() reads these at RUNTIME
# (pip install --target <persistent dir> -r <this file>), on demand, via
# POST /api/diarization-engines/install. Keeping the files in the image
# (rather than fetching them from the internet at install time too) means
# the exact, repo-pinned dependency versions get installed regardless of
# what a future git push to this repo's default branch might change --
# same "pin what THIS build verified" reasoning as everywhere else in this
# codebase that resists floating on "latest".
COPY requirements-diarization.txt requirements-diarization-sortformer.txt ./
# Same reasoning, for TTS engines (live_stt/tts_installer.py, POST
# /api/tts-engines/install) -- kokoro/piper are the last remaining
# baked-in-nowhere-to-runtime-installable-somewhere gap this repo had;
# see tts_installer.py's own docstring for the real silent-fallback bug
# this closes (an uninstalled engine used to make live_stt/tts.py return
# a fake sine-wave tone instead of real speech, with no visible error).
COPY requirements-tts-kokoro.txt requirements-tts-piper.txt ./

ENV LSTT_WORKER_BIN=/app/worker/live_stt_worker \
    LSTT_WORKER_BIN_PARAKEET_TDT=/app/worker/live_stt_worker_parakeet_tdt \
    LSTT_WORKER_BIN_WHISPER=/app/worker/whisper/live_stt_worker_whisper \
    LSTT_PARAKEET_BACKEND=cpu \
    LSTT_WHISPER_BACKEND=cpu \
    LSTT_PARAKEET_TDT_BACKEND=cpu \
    LSTT_GRPC_HOST=0.0.0.0 \
    LSTT_GRPC_PORT=50051 \
    LSTT_ADMIN_HOST=0.0.0.0 \
    LSTT_ADMIN_PORT=8000 \
    LSTT_MODELS_DIR=/models

VOLUME ["/app/data"]
EXPOSE 50051 8000

# Python-urllib-based, no curl in the image (house convention). Longer
# start-period than the house default 20s: a cold model load can genuinely
# take that long, especially the first CUDA build's PTX JIT (Phase 5).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["python", "run.py"]

# ---- stage: unit tests -- deliberately ships NO worker binary and NO model.
# Offline-safety-by-construction: an accidental integration-shaped "unit"
# test fails the build here (ModuleNotFoundError / FileNotFoundError)
# instead of silently passing against the real thing.
FROM py-deps AS test-unit
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY tests/ ./tests/
COPY tools/ ./tools/
COPY pyproject.toml ./
RUN mkdir -p /app/test-reports
CMD ["python", "-m", "pytest", "tests/", "-v", "-m", "not integration", \
     "--junitxml=/app/test-reports/unit-results.xml", "--junit-prefix=unit"]

# ---- stage: integration tests -- the real worker binary, model mounted at
# runtime (docker run -v ./models:/models:ro), not baked into the image.
FROM runtime AS test-integration
COPY tests/ ./tests/
COPY tools/ ./tools/
COPY pyproject.toml ./
RUN mkdir -p /app/test-reports
CMD ["python", "-m", "pytest", "tests/", "-v", "-m", "integration and not slow and not gpu", \
     "--junitxml=/app/test-reports/integration-results.xml", "--junit-prefix=integration"]

# ---- stage: worker's own ctest suite (upstream parakeet.cpp tests) --------
FROM python:3.12-slim AS test-worker
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY worker/third_party/parakeet.cpp/ third_party/parakeet.cpp/
RUN cmake -S third_party/parakeet.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release -DPARAKEET_BUILD_TESTS=ON \
        -DPARAKEET_BUILD_CLI=OFF -DPARAKEET_BUILD_SERVER=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    && cmake --build build -j"$(nproc)"
CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
