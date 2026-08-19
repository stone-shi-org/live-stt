import json

import numpy as np

from live_stt.client.telephony import (
    ALAW_LUT,
    ULAW_LUT,
    decode_alaw,
    decode_ulaw,
    g711_to_pcm16le_16k,
    resample_8k_to_16k,
)


def test_ulaw_lut_matches_golden_vector(fixtures_dir: str) -> None:
    with open(f"{fixtures_dir}/ulaw_lut_golden.json") as f:
        golden = json.load(f)
    assert ULAW_LUT.tolist() == golden


def test_alaw_lut_matches_golden_vector(fixtures_dir: str) -> None:
    with open(f"{fixtures_dir}/alaw_lut_golden.json") as f:
        golden = json.load(f)
    assert ALAW_LUT.tolist() == golden


def test_ulaw_luts_are_not_shared_objects() -> None:
    # mu-law and A-law bit layouts differ; sharing a LUT would be silently wrong.
    assert ULAW_LUT is not ALAW_LUT
    assert not np.array_equal(ULAW_LUT, ALAW_LUT)


def test_decode_ulaw_preserves_length() -> None:
    payload = bytes(range(256))
    assert len(decode_ulaw(payload)) == 256


def test_decode_alaw_preserves_length() -> None:
    payload = bytes(range(256))
    assert len(decode_alaw(payload)) == 256


def test_ulaw_silence_round_trips_near_zero() -> None:
    # 0xFF and 0x7F are the canonical mu-law "silence" codes.
    silence = decode_ulaw(bytes([0xFF, 0x7F]))
    assert np.all(np.abs(silence) <= 8)


def test_resample_8k_to_16k_doubles_length() -> None:
    pcm8k = (np.sin(2 * np.pi * 1000 * np.arange(8000) / 8000) * 10000).astype(np.int16)
    pcm16k = resample_8k_to_16k(pcm8k)
    assert len(pcm16k) == 16000
    assert pcm16k.dtype == np.int16


def test_resample_preserves_tone_frequency() -> None:
    # A resampled 1kHz tone should still measure ~1kHz by its FFT peak, not
    # just have the right length -- catches a resampler that scrambles
    # content. Excludes the buffer edges (filter ringing) and uses FFT rather
    # than zero-crossing counting, which is corrupted by that same ringing.
    sr = 8000
    freq = 1000
    duration_sec = 2.0
    n = int(sr * duration_sec)
    pcm8k = (np.sin(2 * np.pi * freq * np.arange(n) / sr) * 10000).astype(np.int16)
    pcm16k = resample_8k_to_16k(pcm8k)

    trimmed = pcm16k[1000:-1000].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(trimmed))
    freqs = np.fft.rfftfreq(len(trimmed), d=1 / 16000)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - freq) < 20


def test_g711_to_pcm16le_16k_mulaw_end_to_end() -> None:
    payload = bytes([0xFF] * 160)  # 20ms of mu-law silence @ 8kHz
    out = g711_to_pcm16le_16k(payload, encoding="mulaw")
    assert len(out) == 320 * 2  # 320 samples @ 16kHz, 2 bytes/sample (int16 LE)


def test_g711_to_pcm16le_16k_rejects_unknown_encoding() -> None:
    import pytest

    with pytest.raises(ValueError):
        g711_to_pcm16le_16k(b"\x00", encoding="opus")
