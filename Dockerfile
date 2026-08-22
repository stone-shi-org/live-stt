# syntax=docker/dockerfile:1

# ---- stage: parakeet.cpp, built STANDALONE (not add_subdirectory'd) -------
# parakeet.cpp's own CMakeLists.txt references ${CMAKE_SOURCE_DIR}/third_party
# (dr_wav.h) and .../scripts/apply_ggml_patches.sh, both assuming it is the
# top-level project. Nesting it via add_subdirectory from another project's
# CMakeLists breaks the include path (hard failure) AND silently skips the
# ggml-patch step (the EXISTS guard just resolves to nothing and no-ops, no
# warning) -- discovered by actually trying it. So it is a separate build
# here, matching scripts/build_worker.sh's native equivalent.
FROM python:3.12-slim AS parakeet-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/third_party/parakeet.cpp/ third_party/parakeet.cpp/
# id= is required, not cosmetic: an unqualified cache mount target (the
# default before this fix) is scoped just by its target PATH, "/build" --
# generic enough that it collided with an unrelated Dockerfile elsewhere on
# this host also using "/build", leaking a stale CMakeCache.txt in (from a
# plain ubuntu:24.04 context where gcc-14 is the default) and causing a
# baffling "No rule to make target .../14/libgomp.so" failure on THIS image,
# which only ever has gcc-13. Found by actually diagnosing it, not guessed.
RUN --mount=type=cache,target=/build,id=live-stt-parakeet-build-cpu \
    cmake -S third_party/parakeet.cpp -B /build/parakeet \
        -DCMAKE_BUILD_TYPE=Release \
        -DPARAKEET_SHARED=OFF \
        -DPARAKEET_BUILD_CLI=OFF \
        -DPARAKEET_BUILD_SERVER=OFF \
        -DPARAKEET_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
    && cmake --build /build/parakeet -j"$(nproc)" \
    && mkdir -p /out/lib \
    && cp /build/parakeet/libparakeet.a /out/ \
    # -a (not a plain cp) to preserve the libNAME.so -> .so.0 -> .so.0.13.0
    # symlink chain: worker/CMakeLists.txt's find_library() needs the
    # unversioned libggml.so name at configure time, and the worker binary's
    # DT_NEEDED entries (its SONAME references, baked in at link time) need
    # libggml.so.0 to resolve at runtime -- copying only the fully-versioned
    # file (an earlier version of this Dockerfile did exactly that) satisfies
    # neither and fails as a "could not find GGML_LIB" configure error two
    # stages later, discovered by actually building this.
    && cp -a /build/parakeet/third_party/ggml/src/libggml*.so* /out/lib/

# ---- stage: parakeet.cpp, CUDA build (Phase 5) -----------------------------
# sm_86 -- Ampere, the VERIFIED GPU on 10.100.0.50 (an RTX 3090, confirmed via
# nvidia-smi over SSH; NOT the RTX 5090/Blackwell an earlier draft of this
# repo speculated from unrelated circumstantial evidence -- see CLAUDE.md's
# GPU section). Compiling here needs only the CUDA toolkit (nvcc), not a
# real device or driver -- this stage builds fine on a driverless host; only
# RUNNING the result needs the actual GPU box.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS parakeet-build-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/third_party/parakeet.cpp/ third_party/parakeet.cpp/
RUN --mount=type=cache,target=/build,id=live-stt-parakeet-build-cuda \
    cmake -S third_party/parakeet.cpp -B /build/parakeet \
        -DCMAKE_BUILD_TYPE=Release \
        -DPARAKEET_SHARED=OFF \
        -DPARAKEET_BUILD_CLI=OFF \
        -DPARAKEET_BUILD_SERVER=OFF \
        -DPARAKEET_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DPARAKEET_GGML_CUDA=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=86" \
    && cmake --build /build/parakeet -j"$(nproc)" \
    && mkdir -p /out/lib \
    && cp /build/parakeet/libparakeet.a /out/ \
    && cp -a /build/parakeet/third_party/ggml/src/libggml*.so* /out/lib/ \
    # libggml-cuda.so* is NOT alongside the others -- unlike libggml-cpu.so
    # (which sits directly in third_party/ggml/src/), the CUDA backend builds
    # one directory deeper, in third_party/ggml/src/ggml-cuda/. Missing this
    # produced a build that LOOKED complete (parakeet-build-cuda succeeded,
    # /out/lib had 3 libs) but broke the next stage: worker-build-cuda's link
    # failed with "undefined reference to `ggml_backend_cuda_reg'" because
    # libggml.so's backend-registry code needs this symbol from
    # libggml-cuda.so and it was silently never copied. Found by actually
    # building worker-build-cuda, not by reading cp's output.
    && cp -a /build/parakeet/third_party/ggml/src/ggml-cuda/libggml-cuda.so* /out/lib/

# ---- stage: the C++ worker, linked against the artifacts above ------------
FROM python:3.12-slim AS worker-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/ ./worker/
COPY --from=parakeet-build /out/libparakeet.a worker/build-parakeet/libparakeet.a
COPY --from=parakeet-build /out/lib/ worker/build-parakeet/third_party/ggml/src/
RUN cd worker && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

# ---- stage: the C++ worker, CUDA build -------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS worker-build-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/ ./worker/
COPY --from=parakeet-build-cuda /out/libparakeet.a worker/build-parakeet/libparakeet.a
COPY --from=parakeet-build-cuda /out/lib/ worker/build-parakeet/third_party/ggml/src/
RUN cd worker && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
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
FROM py-deps AS runtime
# libgomp1: OpenMP runtime ggml-cpu links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker /app/worker/live_stt_worker
COPY --from=worker-build /src/live-stt-worker/worker/build-parakeet/third_party/ggml/src/*.so* /app/worker/
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY run.py version.txt* ./

ENV LSTT_WORKER_BIN=/app/worker/live_stt_worker \
    LSTT_BACKEND=cpu \
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

# ---- stage: runtime, CUDA (Phase 5) ----------------------------------------
# Ubuntu 24.04's system python3 IS 3.12 (matches the CPU image's python:3.12-
# slim), so no deadsnakes/PPA needed -- just apt. --break-system-packages:
# Ubuntu's apt-installed python3-pip marks the environment "externally
# managed" per PEP 668; harmless here since the whole image is disposable.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRPC_POLL_STRATEGY=poll \
    # libggml.so.0 (CUDA build only) is itself built by ggml's own CMake with
    # a hardcoded RUNPATH pointing at the standalone build's cache-mount dir
    # (/build/parakeet/.../ggml/src) -- correct there, meaningless here where
    # the .so's are copied to /app/worker/. And even if that path did exist,
    # RUNPATH (unlike old-style RPATH) is NOT transitively inherited: the
    # worker binary's own $ORIGIN RUNPATH only resolves ITS direct NEEDED
    # entries, not the ones libggml.so.0 itself declares (libggml-cuda.so.0)
    # -- confirmed empirically: `ldd` showed "libggml-cuda.so.0 => not found"
    # despite the file sitting right next to the binary, until this was set.
    # Not needed on the CPU image: CPU-only libggml.so has no CUDA backend to
    # need, so this class of transitive lookup never arises there.
    LD_LIBRARY_PATH=/app/worker
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt \
        --index-url https://pypi.org/simple/

COPY --from=worker-build-cuda /src/live-stt-worker/worker/build/live_stt_worker /app/worker/live_stt_worker
COPY --from=worker-build-cuda /src/live-stt-worker/worker/build-parakeet/third_party/ggml/src/*.so* /app/worker/
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY run.py version.txt* ./

ENV LSTT_WORKER_BIN=/app/worker/live_stt_worker \
    LSTT_BACKEND=cuda \
    LSTT_GRPC_HOST=0.0.0.0 \
    LSTT_GRPC_PORT=50051 \
    LSTT_ADMIN_HOST=0.0.0.0 \
    LSTT_ADMIN_PORT=8000 \
    LSTT_MODELS_DIR=/models

VOLUME ["/app/data"]
EXPOSE 50051 8000

# Longer start-period than the CPU image's 60s: a cold CUDA model load pays
# for context/PTX setup the CPU path doesn't.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["python", "run.py"]

# ---- stage: runtime-cuda + post-call diarization baked in ------------------
# pyannote.audio/torch/torchaudio (requirements-diarization.txt) are
# deliberately NOT in runtime/runtime-cuda themselves -- diarization is an
# opt-in, offline-only tool, and every OTHER deployment of the always-on
# grpc.aio server shouldn't pay for a multi-hundred-MB torch install it never
# uses (see live_stt/diarization.py). This stage exists because a specific
# deployment (10.100.0.50) wants BOTH GPU ASR and GPU diarization in the one
# running container -- see CLAUDE.md's "Batch ASR over HTTP" /
# "Post-call speaker diarization" entries. Kept as a separate stage/tag
# (`-cuda-diarize`, not a change to what `-cuda` means) so CLAUDE.md's
# existing Phase 5 description of runtime-cuda (built and verified WITHOUT
# these deps) stays accurate rather than silently going stale.
#
# HF_HOME points inside the already-mounted, already-writable /app/data
# volume (see docker-compose.yml/ai.yml: `live-stt-data:/app/data`) rather
# than a new named volume -- the pyannote model then survives container
# restarts/recreates for free, using infrastructure that already exists,
# instead of a container-local cache that re-downloads the gated model from
# HuggingFace (and burns a token's rate limit) on every restart.
FROM runtime-cuda AS runtime-cuda-diarize
ENV HF_HOME=/app/data/hf-cache
# ffmpeg: pyannote.audio 4.x decodes audio via torchcodec, which dlopens
# libavutil/libavcodec/etc at import time (tries several ffmpeg ABI versions
# in turn) rather than linking them at build time -- found only by actually
# running diarize_file() in a real container, not by reading either
# project's docs: "OSError: libavutil.so.61: cannot open shared object
# file", repeated for every ffmpeg 4-9 ABI torchcodec tried, on this image's
# runtime-cuda base (python3.12/python3-pip/libgomp1 only, no media libs at
# all). Every dev machine this was built/tested on before this container
# already had ffmpeg installed for unrelated reasons, which is exactly how
# this stayed invisible until the real container build+run in CLAUDE.md's
# GPU deploy log.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-diarization.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements-diarization.txt \
        --index-url https://pypi.org/simple/

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
