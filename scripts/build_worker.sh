#!/usr/bin/env bash
#
# Two-step native build of the C++ worker.
#
#   ./scripts/build_worker.sh          CPU
#   ./scripts/build_worker.sh --cuda   Phase 5, needs the CUDA toolkit (not the
#                                       device -- see CLAUDE.md, this builds fine
#                                       on a driverless host)
#
# Two steps because parakeet.cpp's own CMakeLists.txt assumes it is the
# top-level project (see worker/CMakeLists.txt's comment on CMAKE_SOURCE_DIR)
# -- add_subdirectory()'ing it from here silently breaks its third_party/
# include path and silently skips its own ggml-patch step. So it is
# configured and built standalone first, and worker/CMakeLists.txt just
# points at the result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../worker"

NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

CUDA_ARGS=()
WHISPER_CUDA_ARGS=()
if [[ "${1:-}" == "--cuda" ]]; then
    # sm_86 -- Ampere, the actual verified GPU on 10.100.0.50 (an RTX 3090;
    # confirmed via nvidia-smi, NOT the RTX 5090/Blackwell an earlier
    # version of this comment speculated from unrelated circumstantial
    # evidence -- see CLAUDE.md). "sm_75+" in upstream's docs does not mean
    # the default arch list covers this specific one; re-target if this
    # ever runs on different hardware. The semicolon MUST be quoted here --
    # unquoted, the shell parses it as a command separator, not part of the
    # value (a latent bug in an earlier version of this script, never
    # actually exercised until this one ran for real).
    CUDA_ARGS=(-DPARAKEET_GGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=86")
    # GGML_CUDA_NCCL=OFF: NCCL (NVIDIA Collective Comm. Library) is
    # multi-GPU tensor-parallel comm code -- ggml's own CMake default is ON
    # whenever it finds NCCL available (which the CUDA devel image does),
    # linking it into ggml-cuda internally via its own target_link_libraries
    # call. That internal link relationship does NOT transfer to
    # worker/CMakeLists.txt's plain find_library()-based import of the
    # resulting static .a (same class of gap as cudart/cublas below), so
    # without this flag the FINAL EXECUTABLE link fails on
    # "undefined reference to `ncclCommInitAll'" and similar -- found by
    # actually linking it. 10.100.0.50 is a single-GPU box (one RTX 3090,
    # see CLAUDE.md), so multi-GPU collectives are simply unneeded, not a
    # feature this ever wants: disabling it at configure time avoids the
    # dependency entirely rather than also linking NCCL just to satisfy it.
    WHISPER_CUDA_ARGS=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=86" -DGGML_CUDA_NCCL=OFF)
fi

echo "==> Configuring parakeet.cpp (standalone build)"
cmake -S third_party/parakeet.cpp -B build-parakeet \
    -DCMAKE_BUILD_TYPE=Release \
    -DPARAKEET_SHARED=OFF \
    -DPARAKEET_BUILD_CLI=OFF \
    -DPARAKEET_BUILD_SERVER=OFF \
    -DPARAKEET_BUILD_TESTS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    "${CUDA_ARGS[@]}"

echo "==> Building parakeet.cpp"
cmake --build build-parakeet -j"$NPROC"

# whisper.cpp (batch-only engine; Phase 6 added CUDA via WHISPER_CUDA_ARGS
# above). Also built standalone first, same two-step reasoning as
# parakeet.cpp above -- see worker/CMakeLists.txt's comment on this block
# for why (whisper.cpp's own CMakeLists.txt is actually nesting-safe, unlike
# parakeet.cpp's, but standalone keeps the two engines symmetric and avoids
# configuring two vendored ggml trees in one CMake project).
echo "==> Configuring whisper.cpp (standalone build)"
cmake -S third_party/whisper.cpp -B build-whisper \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DGGML_FMA=ON \
    -DGGML_F16C=ON \
    "${WHISPER_CUDA_ARGS[@]}"

echo "==> Building whisper.cpp"
cmake --build build-whisper -j"$NPROC"

echo "==> Configuring live_stt_worker / live_stt_worker_whisper"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo "==> Building live_stt_worker / live_stt_worker_whisper"
cmake --build build -j"$NPROC"

echo "==> Built: worker/build/live_stt_worker, worker/build/live_stt_worker_whisper"
