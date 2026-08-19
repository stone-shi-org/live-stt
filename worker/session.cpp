#include "session.hpp"

#include "parakeet_capi.h"

namespace live_stt {

Session::~Session() {
    if (stream_) parakeet_capi_stream_free(stream_);
    if (ctx_) parakeet_capi_free(ctx_);
}

std::string Session::configure(const std::string& gguf_path, const std::string& language) {
    ctx_ = parakeet_capi_load(gguf_path.c_str());
    if (!ctx_) {
        return "failed to load model: " + gguf_path;
    }

    stream_ = language.empty() ? parakeet_capi_stream_begin(ctx_)
                                : parakeet_capi_stream_begin_lang(ctx_, language.c_str());
    if (!stream_) {
        std::string err = parakeet_capi_last_error(ctx_);
        return err.empty() ? "failed to begin streaming session (not a cache-aware streaming model?)"
                            : err;
    }
    return "";
}

bool Session::feed(const float* samples, int n_samples, std::string* json_out, std::string* error) {
    char* text = parakeet_capi_stream_feed_json(stream_, samples, n_samples);
    if (!text) {
        *error = parakeet_capi_last_error(ctx_);
        return false;
    }
    *json_out = text;
    parakeet_capi_free_string(text);
    return true;
}

bool Session::finalize(std::string* json_out, std::string* error) {
    char* text = parakeet_capi_stream_finalize_json(stream_);
    if (!text) {
        *error = parakeet_capi_last_error(ctx_);
        return false;
    }
    *json_out = text;
    parakeet_capi_free_string(text);
    return true;
}

}  // namespace live_stt
