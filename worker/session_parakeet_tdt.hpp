#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct parakeet_ctx;

namespace live_stt {

// The Parakeet TDT batch-engine analogue of Session (session.hpp).
// Parakeet-TDT-0.6B-v3 has no streaming state in parakeet.cpp's StreamingSession
// (which asserts num_durations == 0), so feed() only buffers samples in memory,
// and all inference happens in finalize() via parakeet_capi_transcribe_pcm_batch_json_lang.
class SessionParakeetTDT {
public:
    SessionParakeetTDT() = default;
    ~SessionParakeetTDT();

    SessionParakeetTDT(const SessionParakeetTDT&) = delete;
    SessionParakeetTDT& operator=(const SessionParakeetTDT&) = delete;

    std::string configure(const std::string& gguf_path, const std::string& language,
                           int n_threads);

    bool feed(const float* samples, int n_samples, std::string* json_out, std::string* error);

    bool finalize(std::string* json_out, std::string* error);

private:
    parakeet_ctx* ctx_ = nullptr;
    std::string language_;
    int n_threads_ = 4;
    std::vector<float> buffer_;
};

}  // namespace live_stt
