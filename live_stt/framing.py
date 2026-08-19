"""Pure pack/unpack for the worker IPC protocol.

Deliberately not protobuf: the worker image needs no protoc, no grpc++, and no
generated code, and the library's own feed_json/finalize_json output is passed
through verbatim (zero re-serialisation in C++; parsed here where it is
testable against captured fixtures).

Wire shape, one frame:

    u32 length_le | u8 type | payload[length - 1]

``length`` counts the type byte plus the payload, matching how a single
``recv``/``read`` loop naturally accumulates it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum


class FrameType(IntEnum):
    # client -> worker
    CONFIG = 0x01
    AUDIO = 0x02
    FINALIZE = 0x03
    PING = 0x05
    # worker -> client
    READY = 0x81
    RESULT = 0x82
    FINAL = 0x83
    ERROR = 0x8F


_HEADER = struct.Struct("<IB")  # length_le, type


@dataclass(frozen=True, slots=True)
class Frame:
    type: FrameType
    payload: bytes


def pack(frame_type: FrameType, payload: bytes = b"") -> bytes:
    length = len(payload) + 1  # + the type byte
    return _HEADER.pack(length, int(frame_type)) + payload


class FrameDecodeError(ValueError):
    """Raised when a buffer does not contain a well-formed frame header/type."""


def parse_header(header: bytes) -> tuple[int, FrameType]:
    """Parse the fixed 5-byte header. Returns (total_payload_length, type).

    ``total_payload_length`` is the number of bytes to read *after* the header
    (i.e. ``length - 1`` from the wire encoding, with the type byte already
    consumed) -- the caller reads exactly that many more bytes to complete the
    frame.
    """
    if len(header) != _HEADER.size:
        raise FrameDecodeError(f"header must be {_HEADER.size} bytes, got {len(header)}")
    length, type_byte = _HEADER.unpack(header)
    if length < 1:
        raise FrameDecodeError(f"frame length must be >= 1 (type byte), got {length}")
    try:
        frame_type = FrameType(type_byte)
    except ValueError as exc:
        raise FrameDecodeError(f"unknown frame type byte 0x{type_byte:02x}") from exc
    return length - 1, frame_type


HEADER_SIZE = _HEADER.size


def unpack_one(buf: bytes) -> tuple[Frame, bytes] | None:
    """Convenience one-shot decode for a buffer that may hold >= 1 frame.

    Returns (frame, remaining_buf), or None if buf does not yet contain a
    complete frame. Used by tests and by any caller not doing incremental
    socket reads (the real worker.py reads the header and payload in two
    ``asyncio.StreamReader.readexactly`` calls instead).
    """
    if len(buf) < HEADER_SIZE:
        return None
    payload_len, frame_type = parse_header(buf[:HEADER_SIZE])
    end = HEADER_SIZE + payload_len
    if len(buf) < end:
        return None
    return Frame(type=frame_type, payload=buf[HEADER_SIZE:end]), buf[end:]
