#pragma once
// Declares the two parakeet.cpp `pk::` functions this worker needs that have
// no public C-API equivalent (pk::set_num_threads / pk::shutdown_backend --
// see main.cpp's comment on ggml_graph.hpp). Normally reached via
// parakeet.cpp's own PRIVATE header (src/ggml_graph.hpp, not part of the
// public include/ contract) when building parakeet.cpp from source -- the
// default path (CMake option LSTT_PARAKEET_PREBUILT=OFF).
//
// This file exists for the LSTT_PARAKEET_PREBUILT=ON alternative (LS-5
// Phase 7b): downloading a prebuilt libparakeet.so instead of
// vendoring/building parakeet.cpp's source at all means ggml_graph.hpp
// genuinely isn't available to include. Confirmed for real (not assumed)
// that this is still safe: `nm -D libparakeet.so` on the real v0.5.0
// linux-cpu-x64 release artifact (see the LS-5 Jira ticket for the full
// investigation) shows these symbols EXPORTED as public dynamic symbols --
//
//     0000000000060ee0 T _ZN2pk15set_num_threadsEi
//     0000000000060fd0 T _ZN2pk16shutdown_backendEv
//     0000000000060f00 T _ZN2pk11num_threadsEv
//
// -- even though the header that originally declared them isn't shipped in
// the public release package. Declaring the same signatures ourselves here
// and linking dynamically against that .so resolves them at load time
// exactly as if we'd included the real (private, unshipped) header.
//
// Real, unavoidable risk this file carries that ggml_graph.hpp's inclusion
// does NOT: this is an UNPUBLISHED, not-part-of-the-public-API contract by
// upstream's own design (that's exactly why it isn't in parakeet_capi.h) --
// a future parakeet.cpp release could rename, change the signature of, or
// remove these symbols with zero notice and zero bump to the public ABI
// marker (parakeet_capi_abi_version()). This header alone does not protect
// against that -- live_stt/backends/installer.py's dlopen/dlsym probe,
// run before ever activating a newly downloaded version, is the real
// safety net; see that module's docstring.
namespace pk {
void set_num_threads(int n);
void shutdown_backend();
int num_threads();
}  // namespace pk
