# syntax=docker/dockerfile:1

# ---- stage: parakeet engine, PREBUILT (LS-5 Phase 7b) ----------------------
# CPU flavor only. Replaces this Dockerfile's previous from-source
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
# needed in the image build at all for this flavor, and a build step
# measured in seconds (a ~2MB download) instead of the several minutes a
# from-source compile took.
#
# CUDA stays on the from-source path (parakeet-build-cuda below) --
# deliberately NOT converted this pass. A real cuda prebuilt entry needs a
# full download of a 500MB-900MB+ tarball and a real GPU to confirm it
# actually runs, neither of which was in scope here; see
# scripts/fetch_backend.sh's own comment on why CUDA isn't registered yet.
FROM python:3.12-slim AS parakeet-fetch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY scripts/fetch_backend.sh scripts/fetch_backend.sh
RUN bash scripts/fetch_backend.sh parakeet cpu /out

# ---- stage: whisper.cpp, built STANDALONE (batch-only second engine) ------
# Same two-step reasoning as parakeet.cpp above -- kept symmetric even
# though whisper.cpp's own CMakeLists.txt is actually safe to nest (it
# correctly guards its CMAKE_SOURCE_DIR-relative paths, unlike parakeet.cpp's
# unguarded ones -- see worker/CMakeLists.txt's comment on this block).
# CPU only, no CUDA variant of this stage -- see CLAUDE.md, whisper on CUDA
# is a later phase.
#
# -DBUILD_SHARED_LIBS=OFF here produces something DIFFERENT from parakeet's
# stage above, verified by actually building it: whisper.cpp's vendored ggml
# (a different pin from parakeet.cpp's own vendored copy) goes fully STATIC
# (.a, no .so at all), where parakeet's ggml builds shared regardless of
# PARAKEET_SHARED. So there is no /out/lib .so-copying dance needed here --
# just the two static archives worker/CMakeLists.txt expects at
# build-whisper/src/libwhisper.a and build-whisper/ggml/src/libggml*.a.
FROM python:3.12-slim AS whisper-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/third_party/whisper.cpp/ third_party/whisper.cpp/
RUN --mount=type=cache,target=/build,id=live-stt-whisper-build-cpu \
    cmake -S third_party/whisper.cpp -B /build/whisper \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_SERVER=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
    && cmake --build /build/whisper -j"$(nproc)" \
    && mkdir -p /out/lib \
    && cp /build/whisper/src/libwhisper.a /out/ \
    && cp /build/whisper/ggml/src/libggml*.a /out/lib/

# ---- stage: cmake provider for CUDA builds (avoids running apt in devel image) ---
FROM python:3.12-slim AS cmake-provider
RUN pip install --no-cache-dir cmake \
    && cp -r /usr/local/lib/python3.12/site-packages/cmake/data/* /usr/local/

# ---- stage: whisper.cpp, CUDA build (Phase 6) ------------------------------
# Same sm_86 target as parakeet's CUDA stage below -- see that stage's
# comment for the real (not speculated) GPU identity this targets.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS whisper-build-cuda
COPY --from=cmake-provider /usr/local/ /usr/local/
WORKDIR /src/live-stt-worker
COPY worker/third_party/whisper.cpp/ third_party/whisper.cpp/
RUN --mount=type=cache,target=/build,id=live-stt-whisper-build-cuda \
    cmake -S third_party/whisper.cpp -B /build/whisper \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_SERVER=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX2=ON \
        -DGGML_FMA=ON \
        -DGGML_F16C=ON \
        -DGGML_CUDA=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=86" \
        # NCCL (multi-GPU tensor-parallel comm) is ON by default whenever
        # ggml's CMake finds it, which this devel image does -- linked into
        # ggml-cuda via ITS OWN internal target_link_libraries call, which
        # (like cudart/cublas below) doesn't transfer through
        # worker/CMakeLists.txt's plain find_library()-based import, so the
        # final executable's link fails on "undefined reference to
        # `ncclCommInitAll'" without this. 10.100.0.50 is single-GPU (see
        # CLAUDE.md) -- disabled rather than also linking NCCL for a
        # feature this never needs. Found by actually linking it.
        -DGGML_CUDA_NCCL=OFF \
    && cmake --build /build/whisper -j"$(nproc)" \
    && mkdir -p /out/lib \
    && cp /build/whisper/src/libwhisper.a /out/ \
    && cp /build/whisper/ggml/src/libggml*.a /out/lib/ \
    # ggml-cuda may land one directory deeper (ggml/src/ggml-cuda/), exactly
    # like parakeet's CUDA backend below -- and, unlike the CPU-only whisper
    # build, may not be a plain .a here (nvcc-compiled CUDA code doesn't
    # always follow BUILD_SHARED_LIBS=OFF as cleanly as pure C/C++ -- see
    # worker/CMakeLists.txt's comment on this). Copy whatever actually
    # exists in both plausible locations, .a or .so*, rather than assuming
    # one shape -- find handles "doesn't exist" gracefully where a bare cp
    # with a non-matching glob would just silently no-op (the exact failure
    # mode parakeet's own CUDA stage hit and documented below).
    && find /build/whisper/ggml/src -maxdepth 2 \( -name 'libggml-cuda.a' -o -name 'libggml-cuda.so*' \) \
        -exec cp -a {} /out/lib/ \;

# ---- stage: parakeet.cpp, CUDA build (Phase 5) -----------------------------
# sm_86 -- Ampere, the VERIFIED GPU on 10.100.0.50 (an RTX 3090, confirmed via
# nvidia-smi over SSH; NOT the RTX 5090/Blackwell an earlier draft of this
# repo speculated from unrelated circumstantial evidence -- see CLAUDE.md's
# GPU section). Compiling here needs only the CUDA toolkit (nvcc), not a
# real device or driver -- this stage builds fine on a driverless host; only
# RUNNING the result needs the actual GPU box.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS parakeet-build-cuda
COPY --from=cmake-provider /usr/local/ /usr/local/
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
# Builds BOTH live_stt_worker (parakeet) and live_stt_worker_whisper
# (whisper) in one `cmake --build` -- they are two targets of the same
# worker/CMakeLists.txt project. libgomp1 is needed here too (not just in
# runtime) because CMake's find_package(OpenMP) probes for a working
# compiler+runtime pair at CONFIGURE time, not just link time.
FROM python:3.12-slim AS worker-build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src/live-stt-worker
COPY worker/ ./worker/
# LS-5 Phase 7b: parakeet comes from the prebuilt fetch above now, not a
# from-source build -- see worker/CMakeLists.txt's LSTT_PARAKEET_PREBUILT
# option. whisper is unchanged (still from-source, still static).
COPY --from=parakeet-fetch /out/ worker/prebuilt-parakeet/
COPY --from=whisper-build /out/libwhisper.a worker/build-whisper/src/libwhisper.a
COPY --from=whisper-build /out/lib/ worker/build-whisper/ggml/src/
RUN cd worker && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DLSTT_PARAKEET_PREBUILT=ON \
        -DPARAKEET_PREBUILT_DIR=/src/live-stt-worker/worker/prebuilt-parakeet \
    && cmake --build build -j"$(nproc)"

# ---- stage: the C++ worker, CUDA build -------------------------------------
# Builds BOTH CUDA binaries now (Phase 6 added the whisper half) -- libgomp1
# added for the same find_package(OpenMP) configure-time probe reason as the
# CPU worker-build stage above.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS worker-build-cuda
COPY --from=cmake-provider /usr/local/ /usr/local/
WORKDIR /src/live-stt-worker
COPY worker/ ./worker/
COPY --from=parakeet-build-cuda /out/libparakeet.a worker/build-parakeet/libparakeet.a
COPY --from=parakeet-build-cuda /out/lib/ worker/build-parakeet/third_party/ggml/src/
COPY --from=whisper-build-cuda /out/libwhisper.a worker/build-whisper/src/libwhisper.a
COPY --from=whisper-build-cuda /out/lib/ worker/build-whisper/ggml/src/
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
COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker_parakeet_tdt /app/worker/live_stt_worker_parakeet_tdt
# LS-5 Phase 7b: libparakeet.so itself, not a separate libggml*.so set --
# the prebuilt release links ggml STATICALLY into libparakeet.so (verified
# via ldd: only libgomp/libstdc++/libm/libgcc_s/libc show up), unlike the
# old from-source build where ggml was always a separate trio of shared
# libs. $ORIGIN rpath (worker/CMakeLists.txt) resolves it from right here,
# same mechanism as before, a different file.
COPY --from=worker-build /src/live-stt-worker/worker/prebuilt-parakeet/libparakeet.so /app/worker/libparakeet.so
# live_stt_worker_whisper is a single self-contained static executable (see
# worker/CMakeLists.txt's whisper block -- its vendored ggml links fully
# static, unlike parakeet's) -- no .so's to copy alongside it. Placed in its
# own subdirectory for clarity/symmetry with the parakeet layout above, not
# because collision avoidance is load-bearing here (there is nothing to
# collide with). libgomp1 (its one real shared dependency, OpenMP) is
# already installed just below for the parakeet worker's benefit.
COPY --from=worker-build /src/live-stt-worker/worker/build/live_stt_worker_whisper /app/worker/whisper/live_stt_worker_whisper
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

# ---- stage: runtime, CUDA (Phase 5) ----------------------------------------
# Ubuntu 24.04's system python3 IS 3.12 (matches the CPU image's python:3.12-
# slim), so no deadsnakes/PPA needed -- just apt. --break-system-packages:
# Ubuntu's apt-installed python3-pip marks the environment "externally
# managed" per PEP 668; harmless here since the whole image is disposable.
# ffmpeg: pyannote.audio decodes audio via torchcodec/libavutil, and kokoro/piper
# TTS engines also benefit from audio utilities. Installed directly here in
# runtime-cuda so all downstream CUDA stages inherit it without needing an extra
# apt-get update cycle.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime-cuda
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && echo 'APT::Get::Check-Space "false";' > /etc/apt/apt.conf.d/99nodiskcheck \
    && sed -i 's/Signed-By: .*/Trusted: yes/' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated -o APT::Get::Check-Space=false \
        python3.12 python3-pip libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
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
    #
    # /app/worker/whisper is NOT needed here, unlike the comment above might
    # suggest before this was actually built -- confirmed by ldd against the
    # real compiled artifact: live_stt_worker_whisper links `_static` CUDA
    # toolkit libraries (cudart_static/cublas_static/cublasLt_static, see
    # worker/CMakeLists.txt) and its own ggml-cuda.a fully statically, so its
    # ONLY runtime dependencies are libgomp.so.1 and the real driver's
    # libcuda.so.1 -- neither co-located, both resolved via the system
    # library path already. Left out of LD_LIBRARY_PATH rather than added
    # defensively, now that this is known rather than assumed.
    LD_LIBRARY_PATH=/app/worker
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt \
        --index-url https://pypi.org/simple/

COPY --from=worker-build-cuda /src/live-stt-worker/worker/build/live_stt_worker /app/worker/live_stt_worker
COPY --from=worker-build-cuda /src/live-stt-worker/worker/build/live_stt_worker_parakeet_tdt /app/worker/live_stt_worker_parakeet_tdt
COPY --from=worker-build-cuda /src/live-stt-worker/worker/build-parakeet/third_party/ggml/src/*.so* /app/worker/
# live_stt_worker_whisper (Phase 6, CUDA): confirmed via ldd (see the
# LD_LIBRARY_PATH comment above) to be fully self-contained -- no .a/.so
# from build-whisper/ggml/src/ needs to ship alongside it at all. An
# earlier version of this copied that whole directory "just in case",
# which turned out to add ~72MB of genuinely unused static archives
# (libggml-cuda.a alone is ~70MB) to the image for nothing -- caught by
# actually inspecting the built image's contents, not assumed safe to skip.
COPY --from=worker-build-cuda /src/live-stt-worker/worker/build/live_stt_worker_whisper /app/worker/whisper/live_stt_worker_whisper
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY run.py version.txt* ./

ENV LSTT_WORKER_BIN=/app/worker/live_stt_worker \
    LSTT_WORKER_BIN_PARAKEET_TDT=/app/worker/live_stt_worker_parakeet_tdt \
    LSTT_WORKER_BIN_WHISPER=/app/worker/whisper/live_stt_worker_whisper \
    LSTT_PARAKEET_BACKEND=cuda \
    LSTT_WHISPER_BACKEND=cuda \
    LSTT_PARAKEET_TDT_BACKEND=cuda \
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
COPY requirements-diarization.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements-diarization.txt \
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
RUN pip install --no-cache-dir --break-system-packages -r requirements-diarization-sortformer.txt \
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
