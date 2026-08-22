"""Shared ``multipart/form-data`` parsing for this service's HTTP surface
(``live_stt/diarize_http.py``'s ``POST /v1/audio/diarization`` and
``live_stt/transcribe_http.py``'s ``POST /v1/audio/transcriptions``, both
wired into ``admin_http.py``'s ``ThreadingHTTPServer``). Extracted here
because neither endpoint's request-parsing has anything diarization- or
transcription-specific about it -- it's the same file-upload-plus-form-
fields shape either way.

**Not multipart-library-based**: ``cgi.FieldStorage`` (the traditional
stdlib answer) was removed by PEP 594 in Python 3.13 (this host runs 3.14 --
same reason ``live_stt/client/telephony.py`` doesn't use ``audioop``, see
CLAUDE.md), and this repo carries no other multipart-parsing dependency.
Standard workaround instead: wrap the raw body as a MIME message (borrowing
the client's own Content-Type header, boundary and all) and let the
``email`` package's MIME parser -- never deprecated -- do the real
splitting. Verified binary-safe before relying on it here: fed a synthetic
1KB payload covering all 256 byte values through it and confirmed a
byte-for-byte round trip (see ``live_stt/diarize_http.py``'s original
verification, done before this module existed as a separate file).
"""

from __future__ import annotations

import email


class MultipartError(ValueError):
    """Malformed request body -- always maps to a 400, never a 500."""


def parse_multipart_form(content_type: str, body: bytes) -> dict[str, bytes]:
    """Parse a ``multipart/form-data`` body into ``{field_name: raw_bytes}``.

    See the module docstring for why this doesn't use ``cgi.FieldStorage``.
    Field values are returned as raw bytes -- callers decide how to decode
    each one (UTF-8 text field vs. binary file upload).
    """
    if "multipart/form-data" not in content_type:
        raise MultipartError(f"expected multipart/form-data, got {content_type!r}")

    # Borrow the client's own Content-Type (boundary and all) as the
    # top-level header of a synthetic MIME message, then let `email` do the
    # actual multipart splitting -- see module docstring.
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii")
    msg = email.message_from_bytes(header + body)
    if not msg.is_multipart():
        raise MultipartError("body did not parse as multipart (missing/bad boundary?)")

    fields: dict[str, bytes] = {}
    for part in msg.get_payload():
        disposition = part.get("Content-Disposition", "")
        name = None
        for piece in disposition.split(";"):
            piece = piece.strip()
            if piece.startswith("name="):
                name = piece[len("name=") :].strip('"')
        if name is None:
            continue
        payload = part.get_payload(decode=True)
        fields[name] = payload if payload is not None else b""
    return fields
