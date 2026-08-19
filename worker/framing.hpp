#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Mirrors live_stt/framing.py exactly -- see that module's docstring for the
// wire shape. Deliberately not protobuf: this worker image needs no protoc,
// no grpc++, and no generated code.

namespace live_stt {

enum class FrameType : uint8_t {
    // client -> worker
    kConfig = 0x01,
    kAudio = 0x02,
    kFinalize = 0x03,
    kPing = 0x05,
    // worker -> client
    kReady = 0x81,
    kResult = 0x82,
    kFinal = 0x83,
    kError = 0x8F,
};

constexpr size_t kHeaderSize = 5;  // u32 length_le + u8 type

// Reads exactly one frame from fd, blocking, retrying on EINTR. Returns false
// on EOF (peer closed the socket -- the normal end of a generation) or a read
// error; the caller should stop its loop, not treat this as a crash.
bool read_frame(int fd, FrameType* type, std::vector<uint8_t>* payload);

// Writes one frame to fd, blocking, retrying on EINTR and short writes.
bool write_frame(int fd, FrameType type, const uint8_t* payload, size_t len);

inline bool write_frame(int fd, FrameType type, const std::string& payload) {
    return write_frame(fd, type, reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
}

inline bool write_frame(int fd, FrameType type) {
    return write_frame(fd, type, nullptr, 0);
}

}  // namespace live_stt
