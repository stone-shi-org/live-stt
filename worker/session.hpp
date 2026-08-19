#pragma once

#include <string>

struct parakeet_ctx;
struct parakeet_stream;

namespace live_stt {

// Owns exactly one parakeet_ctx and, once configured, exactly one
// parakeet_stream -- never re-begun. One Session per worker process; one
// worker process per call generation (see CLAUDE.md's "Where the streaming
// state lives" section -- this class is the bottom of that chain). There is
// no loop anywhere near stream_begin, so there is no code path that could
// call it twice for the life of this process.
class Session {
public:
    Session() = default;
    ~Session();

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Loads the model and begins the one streaming session this process will
    // ever hold. `language` empty means the model default. Returns an empty
    // string on success, or an error message on failure.
    std::string configure(const std::string& gguf_path, const std::string& language);

    // Feed samples (n_samples == 0 is legal -- used for PING/liveness). On
    // success writes the library's feed_json document to *json_out and
    // returns true. On failure returns false and writes the ctx's last error
    // to *error.
    bool feed(const float* samples, int n_samples, std::string* json_out, std::string* error);

    // Flushes the end-of-stream tail (finalize_json). Same success contract
    // as feed().
    bool finalize(std::string* json_out, std::string* error);

private:
    parakeet_ctx* ctx_ = nullptr;
    parakeet_stream* stream_ = nullptr;
};

}  // namespace live_stt
