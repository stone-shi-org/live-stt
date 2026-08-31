#include "session_whisper.hpp"

#include "json_util.hpp"
#include "whisper.h"

namespace live_stt {

namespace {

// whisper.cpp's own convention (examples/main.cpp, examples/stream/stream.cpp
// both do the same grouping): a token's text piece starting with a leading
// space marks the start of a new word; pieces without one are a continuation
// of the previous word's BPE split. Special/control tokens (<SOT>, <EOT>,
// timestamp tokens, ...) always have id >= whisper_token_eot(ctx) and carry
// no real text -- skip them entirely rather than let a stray "[_BEG_]"-style
// piece leak into a word.
bool is_special_token(whisper_context* ctx, whisper_token id) {
    return id >= whisper_token_eot(ctx);
}

}  // namespace

WhisperSession::~WhisperSession() {
    if (ctx_) whisper_free(ctx_);
}

std::string WhisperSession::configure(const std::string& model_path, const std::string& language,
                                       int n_threads, bool use_gpu) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu;
    cparams.gpu_device = 0;  // single-GPU host assumed, same as the parakeet engine's CUDA path
    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx_) {
        return "failed to load model: " + model_path;
    }
    // "" (model default) means auto-detect at finalize() time, matching
    // whisper_full_params.language's own documented contract ("for
    // auto-detection, set to nullptr, \"\" or \"auto\"").
    language_ = language.empty() ? "auto" : language;
    n_threads_ = n_threads > 0 ? n_threads : 4;
    return "";
}

bool WhisperSession::feed(const float* samples, int n_samples, std::string* json_out,
                           std::string* /*error*/) {
    if (n_samples > 0 && samples != nullptr) {
        buffer_.insert(buffer_.end(), samples, samples + n_samples);
    }
    // Nothing has been transcribed yet -- see the header comment. This is
    // never an error: there is no engine call here that can fail.
    *json_out = "{\"text\":\"\",\"words\":[]}";
    return true;
}

bool WhisperSession::finalize(std::string* json_out, std::string* error) {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads = n_threads_;
    wparams.language = language_.c_str();
    wparams.translate = false;
    wparams.token_timestamps = true;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_special = false;
    wparams.print_timestamps = false;
    wparams.single_segment = false;

    int rc = whisper_full(ctx_, wparams, buffer_.data(), static_cast<int>(buffer_.size()));
    if (rc != 0) {
        *error = "whisper_full failed with code " + std::to_string(rc);
        return false;
    }

    std::string text;
    std::string words_json;
    bool first_word = true;

    const int n_segments = whisper_full_n_segments(ctx_);
    for (int i = 0; i < n_segments; ++i) {
        text += whisper_full_get_segment_text(ctx_, i);

        std::string current_word;
        int64_t word_t0 = 0, word_t1 = 0;
        float word_p_sum = 0.0f;
        int word_n_tokens = 0;
        bool have_word = false;

        auto flush_word = [&]() {
            if (!have_word) return;
            // trim a single leading space, if present (the word-boundary
            // marker itself, not part of the word's text)
            std::string w = current_word;
            if (!w.empty() && w.front() == ' ') w.erase(w.begin());
            if (!w.empty()) {
                if (!first_word) words_json += ",";
                first_word = false;
                float conf = word_n_tokens > 0 ? word_p_sum / word_n_tokens : 0.0f;
                words_json += "{\"w\":\"" + json::escape(w) + "\",\"start\":" +
                              std::to_string(word_t0 / 100.0) + ",\"end\":" +
                              std::to_string(word_t1 / 100.0) + ",\"conf\":" +
                              std::to_string(conf) + "}";
            }
            current_word.clear();
            word_p_sum = 0.0f;
            word_n_tokens = 0;
            have_word = false;
        };

        const int n_tokens = whisper_full_n_tokens(ctx_, i);
        for (int j = 0; j < n_tokens; ++j) {
            whisper_token_data td = whisper_full_get_token_data(ctx_, i, j);
            if (is_special_token(ctx_, td.id)) continue;

            std::string piece = whisper_full_get_token_text(ctx_, i, j);
            if (piece.empty()) continue;

            bool starts_new_word = piece.front() == ' ';
            if (starts_new_word) flush_word();

            if (!have_word) {
                word_t0 = td.t0;
                have_word = true;
            }
            word_t1 = td.t1;
            word_p_sum += td.p;
            word_n_tokens += 1;
            current_word += piece;
        }
        flush_word();
    }

    *json_out = "{\"text\":\"" + json::escape(text) + "\",\"words\":[" + words_json + "]}";
    return true;
}

}  // namespace live_stt
