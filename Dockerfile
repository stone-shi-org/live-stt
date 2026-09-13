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
# the runtime-cuda alias stage further down) -- an operator picks a flavor
# per engine, per deployment, without rebuilding or even recreating the
# image.
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
FROM py-deps AS runtime
# libgomp1: OpenMP runtime ggml-cpu links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

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

# ---- stage: runtime-cuda -- BACKWARD-COMPATIBILITY ALIAS (LS-5 Phase 7d) ---
# Through Phase 6, this was a SEPARATE image built from an nvidia/cuda base,
# compiling every engine from source with CUDA enabled. As of this phase
# that entire separate build path is gone: every engine's cpu AND cuda
# flavors are DOWNLOADED (github_release or oci_image -- see
# live_stt/backends/registry.py), most of them self-contained (bundling
# their own cudart/cublas/etc), needing only the real driver's libcuda.so.1
# at RUNTIME (nvidia-container-toolkit's `--gpus all`, never baked into any
# image) -- so there is no longer a build-time reason for a CUDA-specific
# base image, or for `runtime` and `runtime-cuda` to differ AT ALL. This is
# exactly LS-5's ask: "no more GPU/CPU tags on docker container."
#
# Kept as a real, buildable stage/tag anyway -- not deleted -- purely for
# backward compatibility with what already depends on the NAME
# "runtime-cuda": docker-compose.gpu.yml's `target: runtime-cuda`,
# build.sh's `--cuda` flag (tags images `-cuda`), and (most importantly) the
# real production deployment's `ai.yml`, which references image tags built
# from `runtime-cuda-diarize` below. Renaming all of those was judged
# higher-risk than keeping one clearly-commented empty alias stage.
#
# Getting an actual GPU-flavored engine running on either `runtime` or
# `runtime-cuda` now means the SAME thing either way: install+activate a
# cuda backend at runtime via POST /api/backends/install,activate (see
# live_stt/admin_http.py) -- see docker-compose.gpu.yml's own comment for
# the operational sequence.
FROM runtime AS runtime-cuda

# ---- stage: runtime + post-call diarization baked in ------------------
# pyannote.audio/torch/torchaudio (requirements-diarization.txt) are
# deliberately NOT in runtime itself -- diarization is an opt-in,
# offline-only tool, and every OTHER deployment of the always-on grpc.aio
# server shouldn't pay for a multi-hundred-MB torch install it never uses
# (see live_stt/diarization.py). This stage exists because a specific
# deployment (10.100.0.50) wants both ASR and diarization in the one
# running container -- see CLAUDE.md's "Batch ASR over HTTP" / "Post-call
# speaker diarization" entries. Named "-cuda-diarize" for the same
# backward-compatibility reason runtime-cuda above is kept: real deployment
# config already references this exact tag. torch's own PyPI wheels bundle
# their own CUDA runtime (nvidia-cublas-cu12 etc as pip dependencies) --
# confirmed by this needing no CUDA-toolkit system package here at all,
# only the real driver at container-run time, same story as the C++ engines.
#
# ffmpeg (pyannote.audio decodes audio via torchcodec/libavutil, and
# kokoro/piper TTS engines also benefit from audio utilities) is installed
# directly in THIS stage now, not inherited from a CUDA base image the way
# it used to be -- `runtime`'s own base (python:3.12-slim) doesn't carry it.
#
# HF_HOME points inside the already-mounted, already-writable /app/data
# volume (see docker-compose.yml/ai.yml: `live-stt-data:/app/data`) rather
# than a new named volume -- the pyannote model then survives container
# restarts/recreates for free, using infrastructure that already exists,
# instead of a container-local cache that re-downloads the gated model from
# HuggingFace (and burns a token's rate limit) on every restart.
FROM runtime-cuda AS runtime-cuda-diarize
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/app/data/hf-cache
COPY requirements-diarization.txt .
RUN pip install --no-cache-dir -r requirements-diarization.txt \
        --index-url https://pypi.org/simple/

CMD ["python", "run.py"]

# ---- stage: adds NVIDIA Sortformer v2.1 (live_stt/diarization.py's
# "nemo_sortformer" engine, LS-4) on top of runtime-cuda-diarize's
# pyannote.audio install. A separate stage/tag, not a change to what
# runtime-cuda-diarize means -- see requirements-diarization-sortformer.txt's
# own comment: nemo_toolkit's dependency tree is heavier, and this is a
# second, more niche opt-in engine on an already-opt-in feature. Confirmed
# during the LS-4 spike that nemo_toolkit[asr]==3.0.0 installs and imports
# cleanly under this image's actual Python 3.12 -- unlike the dev-host/
# GPU-box throwaway venvs used for the spike itself, which hit a real
# Python-3.14 packaging blocker (see CLAUDE.md's LS-4 entry for the full
# story; it does not apply to this stage's interpreter).
FROM runtime-cuda-diarize AS runtime-cuda-diarize-sortformer
COPY requirements-diarization-sortformer.txt .
RUN pip install --no-cache-dir -r requirements-diarization-sortformer.txt \
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
