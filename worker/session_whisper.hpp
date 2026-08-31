#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct whisper_context;

namespace live_stt {

// The whisper-engine analogue of Session (session.hpp), but batch-shaped
// rather than streaming: whisper.cpp has no incremental encoder/decoder
// state (confirmed against examples/stream/stream.cpp -- it just re-runs
// whisper_full() over a manually reconstructed sliding window, not a real
// carried state), so feed() is deliberately cheap -- it only buffers -- and
// all real inference happens once, in finalize(). This mirrors how this
// engine is exposed at the Python layer: batch-only, via
// POST /v1/audio/transcriptions, never the streaming gRPC Transcribe RPC
// (see live_stt/servicer.py's streaming_capable gate and CLAUDE.md).
//
// One WhisperSession per worker process, same one-process-per-call-
// generation invariant as the parakeet engine's Session -- see CLAUDE.md's
// "Where the streaming state lives" section, which applies to both engines
// even though only the parakeet engine's rotation ever kicks in in practice
// (live_stt/session.py disables rotation entirely for batch-only models,
// since killing this worker mid-buffer would silently drop unfinalized
// audio -- there is no progressive per-chunk transcript to fall back on
// the way there is for parakeet).
class WhisperSession {
public:
    WhisperSession() = default;
    ~WhisperSession();

    WhisperSession(const WhisperSession&) = delete;
    WhisperSession& operator=(const WhisperSession&) = delete;

    // Loads the model. `language` empty means auto-detect ("auto", passed
    // through to whisper_full_params.language at finalize() time -- unlike
    // parakeet, whisper's language selection is a per-call decode parameter,
    // not a different stream-begin entry point, so it is stored here rather
    // than applied at load time). Returns an empty string on success, or an
    // error message on failure.
    //
    // use_gpu maps directly to whisper_context_params.use_gpu -- unlike the
    // parakeet engine, where CPU-vs-CUDA is entirely a build-time/binary
    // choice (ggml's backend registry auto-selects CUDA if the linked .so
    // registers it, with no explicit per-load flag in parakeet_capi.h),
    // whisper.cpp exposes this as a real per-context parameter, so it is
    // threaded through from live_stt/config.py's Settings.backend via the
    // CONFIG frame's "use_gpu" field (see main_whisper.cpp) rather than
    // inferred. False on a CPU-only build too (harmless -- there is no CUDA
    // backend registered to select in that binary regardless of this flag).
    std::string configure(const std::string& model_path, const std::string& language,
                           int n_threads, bool use_gpu);

    // Buffers samples; does not run any inference. n_samples == 0 is legal
    // (PING/liveness). Always succeeds (returns true) -- there is no
    // engine call here that can fail. json_out is a cheap, fixed
    // "nothing to report yet" document, matching the shape finalize()
    // produces so callers don't need to special-case which method they
    // called.
    bool feed(const float* samples, int n_samples, std::string* json_out, std::string* error);

    // Runs the one and only whisper_full() over everything buffered by
    // feed(), then builds {"text": ..., "words": [{"w","start","end","conf"},
    // ...]} -- the same shape live_stt/events.py::worker_json_to_events
    // already accepts from any worker (eou/eob/events are optional there
    // and simply omitted here, since whisper has no turn-detection signal).
    // Words are token groups: whisper's BPE token pieces start a new word
    // with a leading space (upstream convention, same one whisper.cpp's own
    // examples/main.cpp uses for word-level output), and tokens whose id is
    // >= whisper_token_eot() are the model's special/timestamp control
    // tokens, skipped rather than emitted as text.
    bool finalize(std::string* json_out, std::string* error);

private:
    whisper_context* ctx_ = nullptr;
    std::string language_;
    int n_threads_ = 4;
    std::vector<float> buffer_;
};

}  // namespace live_stt
