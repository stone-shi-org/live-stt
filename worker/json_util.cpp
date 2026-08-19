#include "json_util.hpp"

#include <cctype>
#include <cstdio>

namespace live_stt::json {

std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

namespace {

size_t find_value_start(const std::string& doc, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = doc.find(needle);
    if (pos == std::string::npos) return std::string::npos;
    pos += needle.size();
    while (pos < doc.size() && (doc[pos] == ' ' || doc[pos] == '\t')) ++pos;
    if (pos >= doc.size() || doc[pos] != ':') return std::string::npos;
    ++pos;
    while (pos < doc.size() && (doc[pos] == ' ' || doc[pos] == '\t')) ++pos;
    return pos;
}

}  // namespace

std::optional<std::string> get_string(const std::string& doc, const std::string& key) {
    size_t pos = find_value_start(doc, key);
    if (pos == std::string::npos || pos >= doc.size() || doc[pos] != '"') return std::nullopt;
    ++pos;
    std::string out;
    while (pos < doc.size() && doc[pos] != '"') {
        if (doc[pos] == '\\' && pos + 1 < doc.size()) {
            char next = doc[pos + 1];
            switch (next) {
                case '"':
                    out += '"';
                    break;
                case '\\':
                    out += '\\';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                default:
                    out += next;
                    break;
            }
            pos += 2;
        } else {
            out += doc[pos];
            ++pos;
        }
    }
    if (pos >= doc.size()) return std::nullopt;  // unterminated string
    return out;
}

std::optional<long long> get_int(const std::string& doc, const std::string& key) {
    size_t pos = find_value_start(doc, key);
    if (pos == std::string::npos) return std::nullopt;
    size_t start = pos;
    if (pos < doc.size() && (doc[pos] == '-' || doc[pos] == '+')) ++pos;
    while (pos < doc.size() && std::isdigit(static_cast<unsigned char>(doc[pos]))) ++pos;
    if (pos == start || (pos == start + 1 && !std::isdigit(static_cast<unsigned char>(doc[start]))))
        return std::nullopt;
    try {
        return std::stoll(doc.substr(start, pos - start));
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> get_bool(const std::string& doc, const std::string& key) {
    size_t pos = find_value_start(doc, key);
    if (pos == std::string::npos) return std::nullopt;
    if (doc.compare(pos, 4, "true") == 0) return true;
    if (doc.compare(pos, 5, "false") == 0) return false;
    return std::nullopt;
}

}  // namespace live_stt::json
