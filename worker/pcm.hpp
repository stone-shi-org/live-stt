#pragma once

#include <cstdint>
#include <vector>

namespace live_stt {

// The single point where int16 PCM becomes the float32 that parakeet.cpp's
// streaming API expects (stream_feed_json wants 16 kHz mono float, ±1.0
// range). Called exactly once per AUDIO frame -- see CLAUDE.md's audio
// boundary section: float32 never appears on any wire in this service.
inline void int16_to_float32(const int16_t* in, int n, std::vector<float>* out) {
    out->resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        (*out)[static_cast<size_t>(i)] = static_cast<float>(in[i]) / 32768.0f;
    }
}

}  // namespace live_stt
