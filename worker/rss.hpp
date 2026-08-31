#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// Shared between main.cpp (parakeet engine) and main_whisper.cpp (whisper
// engine) -- both worker binaries report rss_kb/fed_samples on every
// RESULT/FINAL frame so the front door's RSS watchdog (live_stt/session.py's
// rotation trigger -- the primary defense against parakeet.cpp#63-style
// leaks, though whisper's worker is a one-shot batch process where the same
// watchdog matters far less) has a fresh sample without a separate IPC round
// trip. Factored out here (rather than duplicated) once a second worker
// binary needed the identical logic.

namespace live_stt {

inline std::string read_rss_kb() {
    // Linux-only (/proc), which is fine: this worker only ever runs inside
    // the service's Linux container.
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return "0";
    char line[256];
    std::string result = "0";
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmRSS:", 6) == 0) {
            long kb = 0;
            std::sscanf(line + 6, "%ld", &kb);
            result = std::to_string(kb);
            break;
        }
    }
    std::fclose(f);
    return result;
}

// Splices this worker's own rss_kb/fed_samples fields into an engine's own
// result JSON document by string surgery rather than parsing and
// re-serialising it, so the library's own JSON (parakeet's feed_json/
// finalize_json) or this worker's own hand-built JSON (whisper's session,
// which has no library-produced JSON to splice into) is passed through with
// a uniform prefix either way. Relies on `lib_json` always starting with
// '{'.
inline std::string wrap_result_json(const std::string& lib_json, uint64_t fed_samples) {
    std::string prefix =
        "{\"rss_kb\":" + read_rss_kb() + ",\"fed_samples\":" + std::to_string(fed_samples) + ",";
    return prefix + lib_json.substr(1);
}

}  // namespace live_stt
