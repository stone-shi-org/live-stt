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
RUN --mount=type=cache,target=/build \
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

# ---- stage: Python dependencies (shared by runtime and test-unit) ---------
FROM python:3.12-slim AS py-deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
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

# ---- stage: unit tests -- deliberately ships NO worker binary and NO model.
# Offline-safety-by-construction: an accidental integration-shaped "unit"
# test fails the build here (ModuleNotFoundError / FileNotFoundError)
# instead of silently passing against the real thing.
FROM py-deps AS test-unit
COPY --from=proto-build /app/live_stt/pb/ /app/live_stt/pb/
COPY live_stt/ ./live_stt/
COPY tests/ ./tests/
COPY pyproject.toml ./
RUN mkdir -p /app/test-reports
CMD ["python", "-m", "pytest", "tests/", "-v", "-m", "not integration", \
     "--junitxml=/app/test-reports/unit-results.xml", "--junit-prefix=unit"]

# ---- stage: integration tests -- the real worker binary, model mounted at
# runtime (docker run -v ./models:/models:ro), not baked into the image.
FROM runtime AS test-integration
COPY tests/ ./tests/
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
