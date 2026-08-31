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

# Phase 5. NOT REQUIRED: absent entirely on a CPU-only build-parakeet output.
#
# Correction to an earlier assumption here: this comment used to claim ggml's
# backend-registry dlopens libggml-cuda.so at runtime by soname and is "never
# a direct link-time symbol dependency" -- disproven by actually building
# this. ggml-backend-reg.cpp is compiled INTO libggml.so with a direct
# (non-weak, non-dlsym) reference to ggml_backend_cuda_reg() whenever the
# overall ggml build had CUDA enabled, so libggml.so itself carries a genuine
# unresolved symbol. A shared library tolerates that (a .so's own link does
# not require every symbol resolved), but our EXECUTABLE'S link does, so
# omitting this target_link_libraries call below produces
# "undefined reference to `ggml_backend_cuda_reg'" -- linking it here isn't
# just for co-location, it's required to resolve the executable at all.
find_library(GGML_CUDA_LIB ggml-cuda PATHS "${_GGML_LIB_DIR}" NO_DEFAULT_PATH)

add_executable(live_stt_worker
    main.cpp
    session.cpp
    framing.cpp
    json_util.cpp
)

target_link_libraries(live_stt_worker PRIVATE parakeet ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})
if(GGML_CUDA_LIB)
  target_link_libraries(live_stt_worker PRIVATE ${GGML_CUDA_LIB})
  message(STATUS "live_stt_worker: linking CUDA backend ${GGML_CUDA_LIB}")

  # libggml-cuda.so calls CUDA Driver API symbols (cuMemCreate,
  # cuDeviceGet, ...) that live in libcuda.so.1 -- shipped by the NVIDIA
  # DRIVER, not the CUDA toolkit, so it does not exist at all on a
  # driverless build host (this dev box, and the "-devel" image used to
  # compile this). find_package(CUDAToolkit) resolves CUDA::cuda_driver to
  # the toolkit's link-time STUB (.../targets/x86_64-linux/lib/stubs/
  # libcuda.so on the devel image used here) when no real driver is
  # present, or the real libcuda.so.1 if one is -- exactly the "compile
  # anywhere, run only where there's a GPU" split Phase 5 relies on.
  # Confirmed the stub path exists by grepping the actual devel image
  # rather than assuming a fixed path across CUDA versions.
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(live_stt_worker PRIVATE CUDA::cuda_driver)
endif()

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

# --------------------------------------------------------------------------
# Second engine: whisper.cpp, the batch-only worker binary
# (live_stt_worker_whisper -- see CLAUDE.md's "Model choice" section and
# live_stt/servicer.py's streaming_capable gate for why this is a SEPARATE
# binary rather than a runtime-selected mode of live_stt_worker above).
#
# Built standalone first, same two-step reasoning as parakeet even though
# whisper.cpp's own CMakeLists.txt is actually safe to add_subdirectory()
# nest (it correctly guards CMAKE_SOURCE_DIR-relative paths behind
# `if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)`, unlike
# parakeet.cpp's unguarded third_party/patch-script references) -- kept
# symmetric with the parakeet block above and avoids configuring two
# separately-vendored ggml trees inside one CMake project. See
# scripts/build_worker.sh for the standalone build step that produces:
#   ${WHISPER_BUILD_DIR}/src/libwhisper.a
#   ${WHISPER_BUILD_DIR}/ggml/src/libggml{,-base,-cpu}.a
#
# Verified by actually building it (not assumed from parakeet's behavior):
# with -DBUILD_SHARED_LIBS=OFF passed explicitly, whisper.cpp's vendored
# ggml (a different pin than parakeet.cpp's own vendored ggml copy) builds
# fully STATIC .a archives, unlike parakeet's ggml which builds shared .so's
# "regardless" of PARAKEET_SHARED (see the parakeet block's comment above).
# So there is no ggml .so to co-locate or $ORIGIN-resolve for this binary at
# all -- live_stt_worker_whisper is a single self-contained static
# executable, with libgomp.so.1 (OpenMP runtime) as its only real shared
# dependency, already installed by the Dockerfile's runtime stage for the
# parakeet worker's benefit. This also means the "landmine" this design
# originally guarded against (two differently-built ggml .so's, identically
# named, colliding if copied into the same runtime directory) does not
# actually apply to whisper's artifacts -- there is nothing to collide.
# --------------------------------------------------------------------------
set(WHISPER_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp" CACHE PATH
    "whisper.cpp checkout (pinned submodule)")
set(WHISPER_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build-whisper" CACHE PATH
    "standalone whisper.cpp build output; see scripts/build_worker.sh")

# NOT a FATAL_ERROR guard, unlike the parakeet block above -- deliberately.
# parakeet's libparakeet.a is REQUIRED unconditionally (live_stt_worker is
# this project's original, always-built target), but live_stt_worker_whisper
# is new and, as of this pin, CPU-only (see CLAUDE.md -- CUDA whisper is a
# later phase). The Dockerfile's worker-build-cuda stage configures this
# SAME CMakeLists.txt without ever building whisper.cpp at all, and a plain
# native dev checkout may reasonably have only run scripts/build_worker.sh's
# parakeet half. A FATAL_ERROR here would break both of those on every
# reconfigure the moment this block was added -- caught by checking, not by
# assuming a symmetric build/test matrix for a brand-new optional engine.
if(EXISTS "${WHISPER_BUILD_DIR}/src/libwhisper.a")
  add_library(whisper_lib STATIC IMPORTED)
  set_target_properties(whisper_lib PROPERTIES
      IMPORTED_LOCATION "${WHISPER_BUILD_DIR}/src/libwhisper.a"
      # whisper.h itself #includes "ggml.h" (its public struct fields
      # reference ggml types), so ggml's own include/ must also be on the
      # search path here, not just whisper.cpp's include/ -- found by
      # actually compiling session_whisper.cpp, not by reading whisper.h in
      # isolation.
      INTERFACE_INCLUDE_DIRECTORIES "${WHISPER_SRC_DIR}/include;${WHISPER_SRC_DIR}/ggml/include"
  )

  # whisper.cpp's own src/ dir also builds a "common" convenience target in
  # its examples, but session_whisper.cpp only needs the public whisper.h C
  # API, so only include/ is exposed above -- no private-header reach-in
  # needed for this engine (n_threads is a public whisper_full_params field).

  set(_WHISPER_GGML_LIB_DIR "${WHISPER_BUILD_DIR}/ggml/src")
  find_library(WHISPER_GGML_LIB ggml PATHS "${_WHISPER_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(WHISPER_GGML_BASE_LIB ggml-base PATHS "${_WHISPER_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(WHISPER_GGML_CPU_LIB ggml-cpu PATHS "${_WHISPER_GGML_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

  # Phase 6 GPU follow-up. NOT REQUIRED: absent on a CPU-only build-whisper
  # output, same optionality pattern as GGML_CUDA_LIB in the parakeet block
  # above. May be found one directory deeper (ggml/src/ggml-cuda/) instead
  # of directly in ggml/src/ -- parakeet's CUDA backend needed that deeper
  # path (see the Dockerfile's cp -a .../ggml-cuda/libggml-cuda.so* note),
  # so both locations are searched here rather than assuming whisper's ggml
  # (a different vendored copy) lays it out identically; PATH_SUFFIXES
  # covers the deeper case without needing two separate find_library calls.
  find_library(WHISPER_GGML_CUDA_LIB ggml-cuda
      PATHS "${_WHISPER_GGML_LIB_DIR}"
      PATH_SUFFIXES "ggml-cuda"
      NO_DEFAULT_PATH)

  add_executable(live_stt_worker_whisper
      main_whisper.cpp
      session_whisper.cpp
      framing.cpp
      json_util.cpp
  )

  # whisper's vendored ggml-cpu is compiled with -fopenmp (its own CMake
  # found and enabled OpenMP, unlike parakeet's ggml-cpu build here, which
  # doesn't need this) -- the GOMP_barrier/GOMP_parallel/omp_get_thread_num
  # etc symbols it calls only resolve at LINK time via the OpenMP runtime,
  # not via ggml's own libs. Found by actually linking this executable:
  # omitting this produces "undefined reference to `GOMP_barrier'" and
  # similar, since static-linking (see the comment above -- there is no .so
  # to defer this to) makes every transitive symbol the final executable's
  # link problem.
  find_package(OpenMP REQUIRED)

  target_link_libraries(live_stt_worker_whisper PRIVATE
      whisper_lib ${WHISPER_GGML_LIB} ${WHISPER_GGML_BASE_LIB} ${WHISPER_GGML_CPU_LIB} OpenMP::OpenMP_CXX
  )

  if(WHISPER_GGML_CUDA_LIB)
    target_link_libraries(live_stt_worker_whisper PRIVATE ${WHISPER_GGML_CUDA_LIB})
    message(STATUS "live_stt_worker_whisper: linking CUDA backend ${WHISPER_GGML_CUDA_LIB}")

    # Same reasoning as the parakeet block's identical CUDA::cuda_driver
    # line above -- libggml-cuda's CUDA Driver API calls (cuMemCreate,
    # cuDeviceGet, ...) resolve to the toolkit's link-time stub on a
    # driverless build host, or the real driver's libcuda.so.1 at runtime
    # on an actual GPU box. find_package(CUDAToolkit) is idempotent if the
    # parakeet block above already called it in this same configure.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(live_stt_worker_whisper PRIVATE CUDA::cuda_driver)

    # ALSO needed here, unlike the parakeet block above: found by actually
    # linking this executable, not assumed from parakeet's own CUDA block.
    # ggml-cuda's .cu-compiled kernels use the CUDA Runtime API's
    # triple-chevron launch ABI (cudaLaunchKernel, __cudaPushCallConfiguration,
    # __cudaRegisterFatBinaryEnd, cudaStreamCreateWithFlags, ...) -- a
    # DIFFERENT API surface than the Driver API cuda_driver provides.
    # parakeet's own worker never needed this explicitly because its
    # ggml-cuda is a SHARED .so (see that block's comment on PARAKEET_SHARED):
    # a .so's own build already links it against cudart internally (ggml's
    # CMake does this), and that dependency travels via the .so's own
    # DT_NEEDED metadata, resolved by the dynamic loader at process start --
    # invisible to (and not the responsibility of) whatever links AGAINST
    # that .so later. A STATIC archive carries no such transitive-dependency
    # metadata through a raw find_library()-based IMPORTED target, so
    # whisper's statically-linked ggml-cuda.a makes every one of its own
    # unresolved symbols -- cudart included -- this executable's own link
    # problem, exactly the same class of issue OpenMP was above.
    # CUDA::cudart_static (not the shared CUDA::cudart) to stay consistent
    # with this binary otherwise having no CUDA-toolkit .so dependencies.
    target_link_libraries(live_stt_worker_whisper PRIVATE CUDA::cudart_static)

    # A second, separate CUDA-toolkit library ggml-cuda needs (cublasCreate_v2,
    # cublasSgemm_v2, cublasGemmStridedBatchedEx, ...): ggml's matmul kernels
    # dispatch to cuBLAS for GEMM rather than reimplementing it, a completely
    # different library from both the Driver API and cudart. Same
    # static-linking-surfaces-every-transitive-dependency reasoning as both
    # blocks above; cublas_static additionally needs cublasLt_static (its
    # own lightweight-template-GEMM backend, split out as a separate static
    # lib in modern CUDA toolkits) and culibos (small static helper lib
    # cublas_static itself depends on for pthread/dl-adjacent glue) to
    # resolve fully -- found by iterating on real link errors one at a time,
    # each fixed library surfacing the next layer's undefined symbols,
    # exactly the "each bug hid behind the previous one" pattern CLAUDE.md
    # already documents for parakeet's own CUDA build.
    target_link_libraries(live_stt_worker_whisper PRIVATE
        CUDA::cublas_static CUDA::cublasLt_static CUDA::culibos
    )
  endif()

  # $ORIGIN rpath kept for symmetry with live_stt_worker above. Confirmed by
  # actually building AND ldd-inspecting both the CPU and CUDA variants of
  # this binary that it is a no-op in both cases, not just the CPU one: even
  # with GGML_CUDA=ON, WHISPER_GGML_CUDA_LIB comes back as a static .a here
  # (given -DBUILD_SHARED_LIBS=OFF), and with CUDA::cudart_static/
  # cublas_static/cublasLt_static also linked in (see above), the resulting
  # executable's only real runtime dependencies are libgomp.so.1 and the
  # driver's libcuda.so.1 -- both resolved via the ordinary system library
  # path, nothing $ORIGIN needs to help find. Kept anyway as free insurance
  # against a future pin reverting any of this to shared linkage.
  set_target_properties(live_stt_worker_whisper PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "$ORIGIN"
  )
else()
  message(STATUS
    "live_stt_worker_whisper: SKIPPED -- ${WHISPER_BUILD_DIR}/src/libwhisper.a "
    "not found. Run scripts/build_worker.sh to build whisper.cpp standalone "
    "first if you need this target; parakeet's live_stt_worker is unaffected.")
endif()
