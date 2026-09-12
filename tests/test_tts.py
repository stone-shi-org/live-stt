from __future__ import annotations

import io
import wave
import pytest

from live_stt.config import Settings
from live_stt.tts import (
    TTSBadRequestError,
    chunk_text,
    pcm_to_format,
    pcm_to_wav,
    synthesize,
    synthesize_stream,
)


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


def test_chunk_text_single_sentence() -> None:
    assert chunk_text("Hello world.") == ["Hello world."]
    assert chunk_text("Single sentence without punctuation") == ["Single sentence without punctuation"]


def test_chunk_text_multiple_sentences() -> None:
    text = "First sentence. Second sentence! Third sentence? Fourth sentence\nFifth sentence."
    chunks = chunk_text(text)
    assert len(chunks) == 5
    assert chunks[0] == "First sentence."
    assert chunks[1] == "Second sentence!"
    assert chunks[2] == "Third sentence?"
    assert chunks[3] == "Fourth sentence"
    assert chunks[4] == "Fifth sentence."


def test_pcm_to_wav_produces_valid_riff() -> None:
    pcm_sample = b"\x00\x00\x10\x00" * 100
    wav_bytes = pcm_to_wav(pcm_sample, sample_rate=24000)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:12]

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 200


def test_pcm_to_format() -> None:
    pcm_sample = b"\x00\x00\x10\x00" * 50
    pcm_out, ct_pcm = pcm_to_format(pcm_sample, sample_rate=24000, response_format="pcm")
    assert pcm_out == pcm_sample
    assert ct_pcm == "audio/pcm"

    wav_out, ct_wav = pcm_to_format(pcm_sample, sample_rate=24000, response_format="wav")
    assert wav_out.startswith(b"RIFF")
    assert ct_wav == "audio/wav"

    mp3_out, ct_mp3 = pcm_to_format(pcm_sample, sample_rate=24000, response_format="mp3")
    assert len(mp3_out) > 0
    assert ct_mp3 == "audio/mpeg"


def test_pcm_to_format_invalid() -> None:
    with pytest.raises(TTSBadRequestError, match="unsupported response_format"):
        pcm_to_format(b"abc", sample_rate=24000, response_format="invalid-fmt")


def test_synthesize_empty_raises() -> None:
    with pytest.raises(TTSBadRequestError, match="must not be empty"):
        synthesize("")


def test_synthesize_full() -> None:
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    audio_bytes, ct = synthesize(
        "Hello world. This is live STT testing speech synthesis.",
        model="kokoro-82m-v1",
        voice="af_heart",
        response_format="wav",
        settings=settings,
    )
    assert len(audio_bytes) > 0
    assert audio_bytes.startswith(b"RIFF")
    assert ct == "audio/wav"


def test_synthesize_stream() -> None:
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    chunks = list(
        synthesize_stream(
            "First sentence. Second sentence.",
            model="kokoro-82m-v1",
            voice="af_heart",
            response_format="pcm",
            settings=settings,
        )
    )
    assert len(chunks) == 2
    for chunk in chunks:
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0
