#include "framing.hpp"

#include <cerrno>
#include <unistd.h>

namespace live_stt {

namespace {

bool read_exact(int fd, uint8_t* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = ::read(fd, buf + got, len - got);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  // EOF
        got += static_cast<size_t>(n);
    }
    return true;
}

bool write_exact(int fd, const uint8_t* buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::write(fd, buf + sent, len - sent);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

bool read_frame(int fd, FrameType* type, std::vector<uint8_t>* payload) {
    uint8_t header[kHeaderSize];
    if (!read_exact(fd, header, kHeaderSize)) return false;

    uint32_t length = static_cast<uint32_t>(header[0]) | (static_cast<uint32_t>(header[1]) << 8) |
                       (static_cast<uint32_t>(header[2]) << 16) |
                       (static_cast<uint32_t>(header[3]) << 24);
    if (length < 1) return false;

    *type = static_cast<FrameType>(header[4]);
    size_t payload_len = static_cast<size_t>(length) - 1;
    payload->resize(payload_len);
    if (payload_len > 0 && !read_exact(fd, payload->data(), payload_len)) return false;
    return true;
}

bool write_frame(int fd, FrameType type, const uint8_t* payload, size_t len) {
    uint32_t length = static_cast<uint32_t>(len + 1);
    uint8_t header[kHeaderSize];
    header[0] = static_cast<uint8_t>(length & 0xFF);
    header[1] = static_cast<uint8_t>((length >> 8) & 0xFF);
    header[2] = static_cast<uint8_t>((length >> 16) & 0xFF);
    header[3] = static_cast<uint8_t>((length >> 24) & 0xFF);
    header[4] = static_cast<uint8_t>(type);
    if (!write_exact(fd, header, kHeaderSize)) return false;
    if (len > 0 && !write_exact(fd, payload, len)) return false;
    return true;
}

}  // namespace live_stt
