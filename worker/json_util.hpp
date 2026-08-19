#pragma once

#include <optional>
#include <string>

// Minimal hand-rolled JSON helpers, matching parakeet.cpp's own house style
// (examples/server/openai_format.cpp's json_escape) rather than adding a
// dependency for a handful of small, fixed-shape messages: the CONFIG frame
// this worker reads, and the READY/ERROR frames it writes. Everything else
// (RESULT/FINAL) passes the library's own feed_json/finalize_json text
// through verbatim -- see main.cpp's wrap_result_json.
//
// get_string/get_int/get_bool are a top-level-field scanner, not a general
// parser: they are only ever used against JSON this codebase itself produces
// (live_stt/worker.py's CONFIG encoder), so a full recursive-descent parser
// would be dead weight.

namespace live_stt::json {

std::string escape(const std::string& s);

std::optional<std::string> get_string(const std::string& doc, const std::string& key);
std::optional<long long> get_int(const std::string& doc, const std::string& key);
std::optional<bool> get_bool(const std::string& doc, const std::string& key);

}  // namespace live_stt::json
