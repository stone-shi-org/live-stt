cmake_minimum_required(VERSION 3.18)
project(live_stt_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------
# parakeet.cpp is deliberately NOT add_subdirectory()'d from here.
#
# Its own CMakeLists.txt references ${CMAKE_SOURCE_DIR}/third_party (for
# dr_wav.h) and ${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh, both
# assuming it is the top-level project. CMAKE_SOURCE_DIR always resolves to
# the OUTERMOST project's source dir, so nesting it via add_subdirectory
# from this project silently breaks the dr_wav.h include path (a hard build
# failure) AND silently SKIPS the ggml-patch step (the EXISTS guard on the
# patch script just resolves to nothing and no-ops, with no warning) --
# discovered by actually trying it. Build it standalone first via
# scripts/build_worker.sh, which produces this project's two inputs:
#   ${PARAKEET_BUILD_DIR}/libparakeet.a
#   ${PARAKEET_BUILD_DIR}/third_party/ggml/src/libggml{,-base,-cpu}.so*
# --------------------------------------------------------------------------
set(PARAKEET_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/parakeet.cpp" CACHE PATH
    "parakeet.cpp checkout (pinned submodule)")
set(PARAKEET_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build-parakeet" CACHE PATH
    "standalone parakeet.cpp build output; see scripts/build_worker.sh")

if(NOT EXISTS "${PARAKEET_BUILD_DIR}/libparakeet.a")
  message(FATAL_ERROR
    "libparakeet.a not found at ${PARAKEET_BUILD_DIR}. Build parakeet.cpp "
    "standalone first -- run scripts/build_worker.sh (or see that script "
    "for the equivalent cmake invocation), then reconfigure this project.")
endif()

add_library(parakeet STATIC IMPORTED)
set_target_properties(parakeet PROPERTIES
    IMPORTED_LOCATION "${PARAKEET_BUILD_DIR}/libparakeet.a"
    INTERFACE_INCLUDE_DIRECTORIES "${PARAKEET_SRC_DIR}/include"
)

set(_GGML_LIB_DIR "${PARAKEET_BUILD_DIR}/third_party/ggml/src")
find_library(GGML_LIB ggml PATHS "${_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS "${_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_CPU_LIB ggml-cpu PATHS "${_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

add_executable(live_stt_worker
    main.cpp
    session.cpp
    framing.cpp
    json_util.cpp
)

target_link_libraries(live_stt_worker PRIVATE parakeet ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})

# pk::set_num_threads / pk::shutdown_backend live in a private header
# (parakeet.cpp's src/, not include/) -- see main.cpp's comment at the
# ggml_graph.hpp include for why this is reached anyway. ggml-cpu.h's public
# ggml_cpu_has_avx2/fma/f16c() give an accurate runtime read of what the
# loaded CPU backend actually has, unlike checking __AVX2__ etc in this
# executable's own translation units (which only reflects how main.cpp
# itself was compiled, not the separately-flagged ggml-cpu sources -- a
# mistake caught by actually running this and seeing "scalar" reported for a
# build that was, in fact, AVX2).
target_include_directories(live_stt_worker PRIVATE
    "${PARAKEET_SRC_DIR}/src"
    "${PARAKEET_SRC_DIR}/third_party/ggml/include"
)

# ggml is loaded as shared libs (ggml's own backend-registry design, not a
# PARAKEET_SHARED artifact -- true even though libparakeet.a itself is
# static). $ORIGIN rpath lets the binary find them co-located in the same
# directory with no LD_LIBRARY_PATH or ldconfig step -- see the Dockerfile's
# runtime stage, which copies them there deliberately for this reason.
set_target_properties(live_stt_worker PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
)
