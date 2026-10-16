# whisper.cpp CUDA 12 backend bundle (libwhisper + ggml + CUDA 12 runtime libs),
# built for Pascal through Ampere so ONE artifact runs on both the Quadro P1000
# (sm_61) and the RTX 3090 (sm_86).
#
# Why this exists: ghcr.io/ggml-org/whisper.cpp:main-cuda (the old whisper:cuda
# source) is CUDA 13 built for sm_75+ only -- CUDA 13 dropped Pascal, so it dies
# with "no kernel image is available for execution on the device" on a P1000.
# whisper.cpp publishes no CUDA 12 image or release, so we build it. (parakeet
# needs no build: mudler/parakeet.cpp ships a cuda12 release asset.)
#
# Build + package:   scripts/build_whisper_cuda12.sh   -> dist/whisper-<ver>-lib-linux-cuda12-x64.tar.gz
#
# CMAKE_CUDA_ARCHITECTURES "61;75;80;86" emits SASS + PTX for each. 61 = P1000,
# 86 = 3090, 75/80 = Turing/A100 for free. GGML_NATIVE=OFF + explicit AVX2/FMA/F16C
# for the same reason the parakeet build needs it (portable CPU backend; see
# CLAUDE.md "GGML_NATIVE=OFF alone silently ships a scalar build").
# GGML_CUDA_NO_VMM=ON: build host has no driver to link libcuda's VMM symbols
# against, and it matches upstream parakeet's cuda12 release. GGML_CUDA_NCCL=OFF:
# ggml turns NCCL on whenever it finds it (the devel image has it), which makes
# libggml-cuda need libnccl.so.2 at load -- dead weight on single-GPU boxes and
# not in this bundle, so the worker died at startup (found on the P1000).
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ARG WHISPER_REF=v1.9.4
ARG CUDA_ARCHS="61;75;80;86"
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git clone --depth 1 --branch "${WHISPER_REF}" https://github.com/ggml-org/whisper.cpp.git .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_NCCL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
      -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" --target whisper ggml ggml-base ggml-cpu ggml-cuda

ARG BUNDLE=whisper-${WHISPER_REF}-lib-linux-cuda12-x64
RUN set -eux; d=/out/${BUNDLE}; mkdir -p "$d"; \
    find build -name 'libwhisper.so*' -o -name 'libggml*.so*' | xargs -I{} cp -a {} "$d"/; \
    cl=/usr/local/cuda/targets/x86_64-linux/lib; \
    cp -a $cl/libcudart.so* $cl/libcublas.so* $cl/libcublasLt.so* "$d"/; \
    cp -a include/whisper.h ggml/include/*.h "$d"/; \
    # Fail the build if the bundle can't resolve by itself (libcuda comes from the host driver).
    missing=$(LD_LIBRARY_PATH="$d" ldd "$d"/libggml-cuda.so | grep 'not found' | grep -v 'libcuda.so.1' || true); \
    [ -z "$missing" ] || { echo "UNRESOLVED: $missing"; exit 1; }; \
    tar -C /out -czf /out/${BUNDLE}.tar.gz ${BUNDLE}; \
    ls -la "$d"; sha256sum /out/${BUNDLE}.tar.gz

FROM scratch AS dist
COPY --from=build /out/*.tar.gz /
