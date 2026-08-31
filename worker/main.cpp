// The frame loop. See CLAUDE.md's "Where the streaming state lives" section:
// this process holds exactly one live_stt::Session, which holds exactly one
// parakeet_ctx and one parakeet_stream. One worker process == one call
// generation. The front door (live_stt/worker.py) owns this process's
// lifetime and always terminates it with SIGKILL -- never SIGTERM, never a
// self-initiated exit on the normal end-of-call path -- so that no static
// destructor ever runs after a CUDA driver teardown (the upstream core-dump
// bug this avoids). This file's own exit path (falling out of the frame loop
// on socket EOF) exists for local/manual testing, not for the production
// lifecycle; it is designed to also be safe if it ever does fire.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "framing.hpp"
#include "json_util.hpp"
#include "pcm.hpp"
#include "rss.hpp"
#include "session.hpp"

extern "C" {
#include "parakeet_capi.h"
}

// Private header -- lives under parakeet.cpp's src/, not include/, so it is
// NOT part of the upstream public API contract. Reached deliberately: this
// is the ONLY way to call pk::set_num_threads / pk::shutdown_backend, which
// have no C-API equivalent (see CLAUDE.md's "why C++ for the worker"
// rationale -- it's the decisive reason this worker is C++ and not a ctypes
// or purego binding). Vendoring parakeet.cpp at a pinned git SHA (see
// worker/third_party/parakeet.cpp and .gitmodules) is what makes this an
// acceptable risk rather than a landmine: if a future upstream version moves
// or removes this header, bumping the pin is where that surfaces, as a build
// failure, not a silent runtime break.
#include "ggml_graph.hpp"

// Public ggml header, for ggml_cpu_has_avx2/fma/f16c() -- a runtime read of
// the loaded CPU backend's actual capabilities, not a compile-time check of
// this translation unit's own flags (which would report "scalar" for an
// AVX2 build, since only ggml-cpu's sources are compiled with -mavx2 etc,
// not main.cpp).
#include <ggml-cpu.h>

namespace {

constexpr int kIpcFd = 3;

// read_rss_kb()/wrap_result_json() moved to rss.hpp -- shared with
// main_whisper.cpp once a second worker binary needed the identical logic.
// wrap_result_json here relies on the documented shape in parakeet_capi.h
// always starting with '{'.

std::string ggml_feature_string() {
    std::string features;
    if (ggml_cpu_has_avx2()) features += "AVX2 ";
    if (ggml_cpu_has_fma()) features += "FMA ";
    if (ggml_cpu_has_f16c()) features += "F16C ";
    if (features.empty()) features = "scalar";
    return features;
}

bool send_error(const std::string& reason) {
    std::string doc = "{\"error\":\"" + live_stt::json::escape(reason) + "\"}";
    return live_stt::write_frame(kIpcFd, live_stt::FrameType::kError, doc);
}

}  // namespace

int main() {
    live_stt::FrameType type;
    std::vector<uint8_t> payload;

    if (!live_stt::read_frame(kIpcFd, &type, &payload) || type != live_stt::FrameType::kConfig) {
        send_error("expected CONFIG as the first frame");
        return 1;
    }

    std::string config_doc(payload.begin(), payload.end());
    auto gguf_path = live_stt::json::get_string(config_doc, "gguf_path");
    std::string language = live_stt::json::get_string(config_doc, "language").value_or("");
    long long n_threads = live_stt::json::get_int(config_doc, "n_threads").value_or(4);

    if (!gguf_path || gguf_path->empty()) {
        send_error("CONFIG missing gguf_path");
        return 1;
    }

    // The one place n_threads is applied. There is no C-API equivalent to
    // this call -- see the ggml_graph.hpp include comment above.
    pk::set_num_threads(static_cast<int>(n_threads));

    live_stt::Session session;
    std::string configure_error = session.configure(*gguf_path, language);
    if (!configure_error.empty()) {
        send_error(configure_error);
        return 1;
    }

    {
        std::string ready = "{\"abi_version\":" + std::to_string(parakeet_capi_abi_version()) +
                             ",\"n_threads\":" + std::to_string(n_threads) + ",\"ggml_features\":\"" +
                             ggml_feature_string() + "\"}";
        if (!live_stt::write_frame(kIpcFd, live_stt::FrameType::kReady, ready)) {
            pk::shutdown_backend();
            return 1;
        }
    }

    uint64_t fed_samples = 0;
    std::vector<int16_t> pcm16;
    std::vector<float> scratch;

    for (;;) {
        if (!live_stt::read_frame(kIpcFd, &type, &payload)) {
            break;  // peer closed the socket -- the normal end of a generation
        }

        if (type == live_stt::FrameType::kAudio || type == live_stt::FrameType::kPing) {
            int n_samples = 0;
            if (type == live_stt::FrameType::kAudio) {
                // Copy into an aligned int16_t buffer rather than
                // reinterpret_cast'ing the raw uint8_t payload: the latter is
                // undefined behavior if the vector's allocation happens not
                // to be 2-byte aligned, even though it's harmless in
                // practice on x86.
                n_samples = static_cast<int>(payload.size() / sizeof(int16_t));
                pcm16.resize(static_cast<size_t>(n_samples));
                if (n_samples > 0) {
                    std::memcpy(pcm16.data(), payload.data(),
                                static_cast<size_t>(n_samples) * sizeof(int16_t));
                }
            } else {
                pcm16.clear();
            }

            live_stt::int16_to_float32(n_samples > 0 ? pcm16.data() : nullptr, n_samples, &scratch);
            fed_samples += static_cast<uint64_t>(n_samples);

            std::string json_out, feed_error;
            if (!session.feed(scratch.data(), n_samples, &json_out, &feed_error)) {
                send_error(feed_error);
                break;
            }
            if (!live_stt::write_frame(kIpcFd, live_stt::FrameType::kResult,
                                        live_stt::wrap_result_json(json_out, fed_samples))) {
                break;
            }
        } else if (type == live_stt::FrameType::kFinalize) {
            std::string json_out, finalize_error;
            if (!session.finalize(&json_out, &finalize_error)) {
                send_error(finalize_error);
                break;
            }
            live_stt::write_frame(kIpcFd, live_stt::FrameType::kFinal,
                                   live_stt::wrap_result_json(json_out, fed_samples));
            // Deliberately no exit here. See the file header comment: the
            // front door owns this process's lifetime via SIGKILL, on both
            // the clean-half-close and the rotation paths alike.
        } else {
            send_error("unexpected frame type after CONFIG");
            break;
        }
    }

    // Reached only on socket EOF, not on the production SIGKILL path (which
    // by definition never reaches any C++ code, including this line) -- see
    // the file header comment.
    pk::shutdown_backend();
    return 0;
}
