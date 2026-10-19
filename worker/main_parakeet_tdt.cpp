// The frame loop for the parakeet-tdt engine's batch worker binary.
// Mirrors main_whisper.cpp in structure: audio frames are buffered during
// feed(), and full TDT transcription with timestamps is executed once
// in finalize() via parakeet_capi_transcribe_pcm_batch_json_lang.

#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include <unistd.h>

#include "framing.hpp"
#include "json_util.hpp"
#include "pcm.hpp"
#include "rss.hpp"
#include "session_parakeet_tdt.hpp"

extern "C" {
#include "parakeet_capi.h"
}

// Private header for pk::set_num_threads / pk::shutdown_backend -- see
// main.cpp's comment block for the LSTT_PARAKEET_PREBUILT (LS-5 Phase 7b)
// alternative and why it's safe; kept in sync here since this binary links
// against the same `parakeet` CMake target.
#ifdef LSTT_PARAKEET_PREBUILT
#include "pk_private.hpp"
#else
#include "ggml_graph.hpp"
#include <ggml-cpu.h>
#endif

namespace {

constexpr int kIpcFd = 3;

std::string ggml_feature_string() {
#ifdef LSTT_PARAKEET_PREBUILT
    return "unknown (LSTT_PARAKEET_PREBUILT: no ggml headers in the prebuilt release package)";
#else
    std::string features;
    if (ggml_cpu_has_avx2()) features += "AVX2 ";
    if (ggml_cpu_has_fma()) features += "FMA ";
    if (ggml_cpu_has_f16c()) features += "F16C ";
    if (features.empty()) features = "scalar";
    return features;
#endif
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

    pk::set_num_threads(static_cast<int>(n_threads));

    live_stt::SessionParakeetTDT session;
    std::string configure_error;
    try {
        configure_error = session.configure(*gguf_path, language, static_cast<int>(n_threads));
    } catch (const std::bad_alloc& e) {
        send_error(std::string("out of memory loading model: ") + e.what());
        return 1;
    } catch (const std::exception& e) {
        send_error(std::string("exception loading model: ") + e.what());
        return 1;
    }
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
            break;  // peer closed the socket
        }

        if (type == live_stt::FrameType::kAudio || type == live_stt::FrameType::kPing) {
            int n_samples = 0;
            try {
                if (type == live_stt::FrameType::kAudio) {
                    n_samples = static_cast<int>(payload.size() / sizeof(int16_t));
                    pcm16.resize(static_cast<size_t>(n_samples));
                    if (n_samples > 0) {
                        std::memcpy(pcm16.data(), payload.data(),
                                    static_cast<size_t>(n_samples) * sizeof(int16_t));
                    }
                } else {
                    pcm16.clear();
                }

                live_stt::int16_to_float32(n_samples > 0 ? pcm16.data() : nullptr, n_samples,
                                           &scratch);
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
            } catch (const std::bad_alloc& e) {
                send_error(std::string("out of memory buffering audio: ") + e.what());
                break;
            } catch (const std::exception& e) {
                send_error(std::string("exception in feed: ") + e.what());
                break;
            }
        } else if (type == live_stt::FrameType::kFinalize) {
            std::string json_out, finalize_error;
            bool ok = false;
            try {
                ok = session.finalize(&json_out, &finalize_error);
            } catch (const std::bad_alloc& e) {
                finalize_error = std::string("out of memory during finalize: ") + e.what();
                ok = false;
            } catch (const std::exception& e) {
                finalize_error = std::string("exception during finalize: ") + e.what();
                ok = false;
            }
            if (!ok) {
                send_error(finalize_error);
                break;
            }
            live_stt::write_frame(kIpcFd, live_stt::FrameType::kFinal,
                                   live_stt::wrap_result_json(json_out, fed_samples));
        } else {
            send_error("unexpected frame type after CONFIG");
            break;
        }
    }

    pk::shutdown_backend();
    return 0;
}
