// The frame loop for the whisper engine's worker binary. Deliberately
// parallel in structure to main.cpp (the parakeet engine's frame loop), but
// simpler in two ways this engine's batch nature affords:
//   - no private-header reach-in for thread count (whisper_full_params.n_threads
//     is public; parakeet needs pk::set_num_threads from a private header --
//     see main.cpp's comment on ggml_graph.hpp)
//   - AUDIO frames only buffer (WhisperSession::feed), all real inference
//     happens once in finalize() -- see session_whisper.hpp's header comment
//     and CLAUDE.md for why this engine is batch-only end to end.
//
// Same production lifecycle contract as main.cpp: live_stt/worker.py always
// terminates this process with SIGKILL, never SIGTERM, never a self-exit on
// the normal end-of-call path. This file's own fall-out-of-the-loop exit
// path exists for local/manual testing only.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "framing.hpp"
#include "json_util.hpp"
#include "pcm.hpp"
#include "rss.hpp"
#include "session_whisper.hpp"

#include "whisper.h"

namespace {

constexpr int kIpcFd = 3;

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
    // Same field name as main.cpp's CONFIG contract ("gguf_path") even
    // though whisper's files are not GGUF -- see live_stt/models.py's
    // comment on ModelSpec.gguf_filename. Kept identical on purpose so
    // live_stt/worker.py's WorkerHandle.spawn() needs no per-engine
    // branching to build the CONFIG frame; it is just "the model file path"
    // regardless of engine.
    auto model_path = live_stt::json::get_string(config_doc, "gguf_path");
    std::string language = live_stt::json::get_string(config_doc, "language").value_or("");
    long long n_threads = live_stt::json::get_int(config_doc, "n_threads").value_or(4);
    // Absent (older Python side, or the parakeet main.cpp's own CONFIG
    // shape, which never sets this key) means false -- CPU, same as before
    // this field existed. See session_whisper.hpp's comment on why this is
    // an explicit per-load flag for whisper but not for the parakeet engine.
    bool use_gpu = live_stt::json::get_bool(config_doc, "use_gpu").value_or(false);

    if (!model_path || model_path->empty()) {
        send_error("CONFIG missing gguf_path");
        return 1;
    }

    live_stt::WhisperSession session;
    std::string configure_error =
        session.configure(*model_path, language, static_cast<int>(n_threads), use_gpu);
    if (!configure_error.empty()) {
        send_error(configure_error);
        return 1;
    }

    {
        // "gpu_requested" echoes back what was asked for, not a runtime
        // probe of which backend actually ended up active -- whisper.cpp
        // silently falls back to CPU if use_gpu=true is requested on a
        // binary built without a CUDA backend registered (standard ggml
        // behavior: a backend that was never registered just isn't
        // selected), so this field alone doesn't prove GPU was really used.
        std::string ready = "{\"abi_version\":1,\"n_threads\":" + std::to_string(n_threads) +
                             ",\"gpu_requested\":" + (use_gpu ? "true" : "false") +
                             ",\"ggml_features\":\"whisper.cpp " + whisper_version() + "\"}";
        if (!live_stt::write_frame(kIpcFd, live_stt::FrameType::kReady, ready)) {
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
                // See main.cpp's identical comment: copy into an aligned
                // int16_t buffer rather than reinterpret_cast'ing the raw
                // uint8_t payload.
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
            // Deliberately no exit here -- see the file header comment.
        } else {
            send_error("unexpected frame type after CONFIG");
            break;
        }
    }

    return 0;
}
