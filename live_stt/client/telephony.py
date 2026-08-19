"""G.711 decode + 8kHz->16kHz resample, run in the CALLER's process.

The ASR service accepts exactly one canonical wire format (16 kHz mono int16
LE) and rejects everything else -- see StreamConfig in the proto and the audio
boundary rationale in CLAUDE.md. This module is what makes that boundary
practical: the telephony app imports it rather than reinventing mu-law/A-law
decode and resampling, so there is one implementation and one test suite
(tests/test_telephony.py), not N per telephony deployment.

Not audioop / audioop-lts: audioop was removed by PEP 594 in Python 3.13 (this
host runs 3.14, `import audioop` fails). A 256-entry numpy LUT is a few lines,
carries no packaging risk, and is faster than the ctypes round-trip on
block-sized input. mu-law and A-law each get their OWN LUT -- do not share one,
the bit layouts differ.

8kHz -> 16kHz uses soxr, not sample-doubling or zero-stuffing: an unfiltered
upsample adds imaging artifacts that measurably hurt WER, and scipy is not a
house dependency. soxr is small and built exactly for this job.
"""

from __future__ import annotations

import numpy as np
import soxr

_BIAS = 0x84  # ITU-T G.711 mu-law bias, 132


def _ulaw_decode_one(u_val: int) -> int:
    u_val = ~u_val & 0xFF
    t = ((u_val & 0x0F) << 3) + _BIAS
    t <<= (u_val & 0x70) >> 4
    return -(t - _BIAS) if u_val & 0x80 else (t - _BIAS)


def _alaw_decode_one(a_val: int) -> int:
    a_val = (a_val ^ 0x55) & 0xFF
    t = (a_val & 0x0F) << 4
    seg = (a_val & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if a_val & 0x80 else -t


ULAW_LUT: np.ndarray = np.array([_ulaw_decode_one(i) for i in range(256)], dtype=np.int16)
ALAW_LUT: np.ndarray = np.array([_alaw_decode_one(i) for i in range(256)], dtype=np.int16)


def decode_ulaw(payload: bytes) -> np.ndarray:
    """mu-law bytes -> int16 mono samples, same length as input."""
    return ULAW_LUT[np.frombuffer(payload, dtype=np.uint8)]


def decode_alaw(payload: bytes) -> np.ndarray:
    """A-law bytes -> int16 mono samples, same length as input."""
    return ALAW_LUT[np.frombuffer(payload, dtype=np.uint8)]


def resample_8k_to_16k(pcm_int16: np.ndarray) -> np.ndarray:
    """8 kHz int16 mono -> 16 kHz int16 mono via a windowed-sinc resampler."""
    return soxr.resample(pcm_int16, 8000, 16000, quality="HQ").astype(np.int16)


def g711_to_pcm16le_16k(payload: bytes, *, encoding: str) -> bytes:
    """Convenience one-shot: G.711 8kHz bytes -> canonical 16kHz int16 LE bytes,
    ready to send as a TranscriptionRequest.audio chunk.

    ``encoding`` is ``"mulaw"`` or ``"alaw"``.
    """
    if encoding == "mulaw":
        pcm8k = decode_ulaw(payload)
    elif encoding == "alaw":
        pcm8k = decode_alaw(payload)
    else:
        raise ValueError(f"unsupported G.711 encoding: {encoding!r}")
    pcm16k = resample_8k_to_16k(pcm8k)
    return pcm16k.astype("<i2").tobytes()
