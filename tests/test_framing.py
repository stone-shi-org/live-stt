import pytest

from live_stt.framing import FrameDecodeError, FrameType, HEADER_SIZE, pack, parse_header, unpack_one


def test_pack_unpack_round_trip() -> None:
    buf = pack(FrameType.AUDIO, b"\x01\x02\x03\x04")
    result = unpack_one(buf)
    assert result is not None
    frame, rest = result
    assert frame.type == FrameType.AUDIO
    assert frame.payload == b"\x01\x02\x03\x04"
    assert rest == b""


def test_pack_empty_payload() -> None:
    buf = pack(FrameType.FINALIZE)
    frame, rest = unpack_one(buf)
    assert frame.type == FrameType.FINALIZE
    assert frame.payload == b""
    assert rest == b""


def test_unpack_one_returns_none_on_incomplete_header() -> None:
    assert unpack_one(b"\x01\x02") is None


def test_unpack_one_returns_none_on_incomplete_payload() -> None:
    buf = pack(FrameType.AUDIO, b"\x01\x02\x03\x04")
    assert unpack_one(buf[:-1]) is None


def test_unpack_one_leaves_remaining_bytes_for_next_frame() -> None:
    buf = pack(FrameType.PING) + pack(FrameType.AUDIO, b"\xaa\xbb")
    frame1, rest = unpack_one(buf)
    assert frame1.type == FrameType.PING
    frame2, rest2 = unpack_one(rest)
    assert frame2.type == FrameType.AUDIO
    assert frame2.payload == b"\xaa\xbb"
    assert rest2 == b""


def test_parse_header_rejects_wrong_size() -> None:
    with pytest.raises(FrameDecodeError):
        parse_header(b"\x01\x02\x03")


def test_parse_header_rejects_unknown_type_byte() -> None:
    header = pack(FrameType.PING)[:HEADER_SIZE]
    bad = header[:4] + bytes([0xEE])
    with pytest.raises(FrameDecodeError):
        parse_header(bad)


def test_parse_header_rejects_zero_length() -> None:
    # length must be >= 1 (the type byte itself)
    bad_header = (0).to_bytes(4, "little") + bytes([FrameType.PING])
    with pytest.raises(FrameDecodeError):
        parse_header(bad_header)


@pytest.mark.parametrize("frame_type", list(FrameType))
def test_all_frame_types_round_trip(frame_type: FrameType) -> None:
    buf = pack(frame_type, b"payload")
    frame, _ = unpack_one(buf)
    assert frame.type == frame_type
