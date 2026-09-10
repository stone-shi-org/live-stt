#include "session_parakeet_tdt.hpp"

#include "json_util.hpp"

extern "C" {
#include "parakeet_capi.h"
}

namespace live_stt {

SessionParakeetTDT::~SessionParakeetTDT() {
    if (ctx_) parakeet_capi_free(ctx_);
}

std::string SessionParakeetTDT::configure(const std::string& gguf_path,
                                          const std::string& language,
                                          int n_threads) {
    ctx_ = parakeet_capi_load(gguf_path.c_str());
    if (!ctx_) {
        return "failed to load model: " + gguf_path;
    }
    language_ = language;
    n_threads_ = n_threads > 0 ? n_threads : 4;
    return "";
}

bool SessionParakeetTDT::feed(const float* samples, int n_samples, std::string* json_out,
                              std::string* /*error*/) {
    if (n_samples > 0 && samples != nullptr) {
        buffer_.insert(buffer_.end(), samples, samples + n_samples);
    }
    // Batch model: nothing is transcribed until finalize().
    *json_out = "{\"text\":\"\",\"words\":[]}";
    return true;
}

bool SessionParakeetTDT::finalize(std::string* json_out, std::string* error) {
    if (buffer_.empty()) {
        *json_out = "{\"text\":\"\",\"words\":[]}";
        return true;
    }

    int n_samples = static_cast<int>(buffer_.size());
    const char* lang = language_.empty() ? nullptr : language_.c_str();

    // decoder: 2 = TDT (transducer)
    char* json_res = parakeet_capi_transcribe_pcm_batch_json_lang(
        ctx_, buffer_.data(), &n_samples, 1, 16000, 2, lang);

    if (!json_res) {
        std::string err = parakeet_capi_last_error(ctx_);
        *error = err.empty() ? "parakeet_capi_transcribe_pcm_batch_json_lang failed" : err;
        return false;
    }

    std::string raw(json_res);
    parakeet_capi_free_string(json_res);

    // raw is a JSON array of 1 clip: [{"text":"...", ...}]
    // Unwrap the outer '[' and ']' if present to get the single object.
    size_t first = raw.find('{');
    size_t last = raw.rfind('}');
    if (first != std::string::npos && last != std::string::npos && last > first) {
        *json_out = raw.substr(first, last - first + 1);
    } else {
        *json_out = raw;
    }
    return true;
}

}  // namespace live_stt
