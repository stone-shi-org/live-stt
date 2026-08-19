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
if [[ "${1:-}" == "--cuda" ]]; then
    # sm_90 (Hopper) + sm_120 (Blackwell, e.g. RTX 5090) -- "sm_75+" in
    # upstream's docs does not mean the default arch list covers either.
    CUDA_ARGS=(-DPARAKEET_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90;120)
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

echo "==> Configuring live_stt_worker"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

echo "==> Building live_stt_worker"
cmake --build build -j"$NPROC"

echo "==> Built: worker/build/live_stt_worker"
