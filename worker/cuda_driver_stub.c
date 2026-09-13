/* Build-time-ONLY link stub for the CUDA Driver API. Used exclusively by
 * worker/CMakeLists.txt's LSTT_WHISPER_PREBUILT CUDA branch (LS-5 Phase
 * 7b), to link against the oci_image-sourced libggml-cuda.so (extracted
 * from ghcr.io/ggml-org/whisper.cpp -- see
 * live_stt/backends/registry.py's docstring) without requiring the full
 * CUDA toolkit (which ships its own real stubs/libcuda.so, but pulling in
 * the whole toolkit just for that one file would defeat the point of this
 * phase: a build with no CUDA toolkit dependency at all).
 *
 * The exact symbol list below is not guessed -- it is every `cu*` symbol
 * `nm -D --undefined-only` reports as unresolved in the real extracted
 * libggml-cuda.so.0.23.0 (ggml's newer VMM-based CUDA memory-pool
 * allocator calls the Driver API directly for these, unlike
 * mudler/parakeet.cpp's own CUDA build, whose .so has no such direct
 * driver dependency at all -- confirmed by `ldd` showing no libcuda.so.1
 * entry there, only cudart/cublas/cublasLt).
 *
 * NEVER shipped in the runtime artifact directory or the Docker image --
 * worker/CMakeLists.txt builds this into a link-time-only helper library
 * that lives in the CMake build directory, which nothing ever COPYs out.
 * At real runtime the executable's baked-in SONAME (libcuda.so.1) resolves
 * to the REAL driver's library, injected by nvidia-container-toolkit's
 * `--gpus all` -- never this stub. Shipping this stub by accident would be
 * a real, silent correctness bug (a fake "driver" shadowing the real one
 * if it ever ended up ahead of the real one in the runtime search path),
 * which is exactly why it is built into an isolated, never-installed
 * target rather than just another file in the fetched artifact directory.
 */
int cuDeviceGet(void) { return 0; }
int cuDeviceGetAttribute(void) { return 0; }
int cuGetErrorString(void) { return 0; }
int cuMemAddressFree(void) { return 0; }
int cuMemAddressReserve(void) { return 0; }
int cuMemCreate(void) { return 0; }
int cuMemGetAllocationGranularity(void) { return 0; }
int cuMemMap(void) { return 0; }
int cuMemRelease(void) { return 0; }
int cuMemSetAccess(void) { return 0; }
int cuMemUnmap(void) { return 0; }
