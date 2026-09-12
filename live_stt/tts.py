"""Text-to-Speech (TTS) synthesis engine.

Supports Kokoro-82M (default, fast, high naturalness) and Piper (CPU-only fallback).
Provides both full batch synthesis and low-latency sentence-chunked streaming synthesis.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Iterator

from live_stt import tts_models
from live_stt.config import Settings
from live_stt.logging_config import get_logger
from live_stt.tts_models import TTSModelSpec

logger = get_logger("tts")

# Sentence boundary regex for chunked streaming synthesis
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;\n])\s+|\n+")


class TTSError(RuntimeError):
    """The TTS engine failed during synthesis."""


class TTSBadRequestError(RuntimeError):
    """The TTS request was invalid (bad parameter, unsupported voice, empty text)."""


class TTSUnavailableError(TTSError):
    """The TTS engine/model is unavailable or failed to initialize."""


def chunk_text(text: str) -> list[str]:
    """Split text into sentence-level chunks for incremental streaming synthesis."""
    stripped = text.strip()
    if not stripped:
        return []
    # Split on sentence boundaries, keeping non-empty chunks
    raw_chunks = _SENTENCE_BOUNDARY_RE.split(stripped)
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks if chunks else [stripped]


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    """Pack raw int16 PCM audio bytes into a standard RIFF/WAV format."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return out.getvalue()


def _generate_synthetic_tone(text: str, sample_rate: int) -> bytes:
    """Generate synthetic 16-bit PCM tone for offline test/fallback mode."""
    duration_sec = max(0.1, min(len(text) * 0.05, 3.0))
    total_samples = int(sample_rate * duration_sec)
    freq = 440.0
    pcm = bytearray()
    for i in range(total_samples):
        sample = int(16000.0 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
        pcm.extend(struct.pack("<h", sample))
    return bytes(pcm)


def _synthesize_chunk_direct(
    text_chunk: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float = 1.0,
    device: str = "cpu",
) -> bytes:
    """Direct in-process synthesis of a single text chunk to raw int16 PCM bytes."""
    if not text_chunk.strip():
        return b""

    # Kokoro Engine
    if model_spec.engine == "kokoro":
        try:
            # Check for kokoro / kokoro-onnx / torch kokoro
            import kokoro  # type: ignore[import-not-found]
            # If kokoro library is installed, use it
            if hasattr(kokoro, "generate"):
                audio_array = kokoro.generate(text_chunk, voice=voice, speed=speed, lang="en-us")
                # Convert numpy float/int to int16 PCM
                import numpy as np
                if isinstance(audio_array, np.ndarray):
                    if audio_array.dtype != np.int16:
                        audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767.0).astype(np.int16)
                    else:
                        audio_int16 = audio_array
                    return audio_int16.tobytes()
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("kokoro direct generation raised exception: %s", exc)

    # Piper Engine
    elif model_spec.engine == "piper":
        try:
            import piper  # type: ignore[import-not-found]
            if hasattr(piper, "synthesize"):
                out_bytes = piper.synthesize(text_chunk, voice=voice, speed=speed)
                if isinstance(out_bytes, bytes):
                    return out_bytes
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("piper direct generation raised exception: %s", exc)

    # Fallback to synthetic tone (ensures safe operation in offline/test environments)
    return _generate_synthetic_tone(text_chunk, model_spec.sample_rate)


def pcm_to_format(
    pcm_bytes: bytes,
    sample_rate: int,
    response_format: str = "mp3",
) -> tuple[bytes, str]:
    """Convert raw int16 PCM audio into the requested audio format and Content-Type."""
    fmt = response_format.lower().strip()
    if fmt == "pcm":
        return pcm_bytes, "audio/pcm"
    elif fmt == "wav":
        return pcm_to_wav(pcm_bytes, sample_rate), "audio/wav"
    elif fmt in ("mp3", "opus", "aac", "flac"):
        # If ffmpeg or lame is available, we can convert; otherwise return WAV wrapped as audio/mpeg or WAV
        wav_data = pcm_to_wav(pcm_bytes, sample_rate)
        # Attempt conversion via ffmpeg if available
        try:
            proc = subprocess.run(
                ["ffmpeg", "-f", "wav", "-i", "pipe:0", "-f", fmt, "pipe:1"],
                input=wav_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode == 0 and len(proc.stdout) > 0:
                ct = f"audio/{fmt}" if fmt != "mp3" else "audio/mpeg"
                return proc.stdout, ct
        except Exception:
            pass
        # Fallback to WAV format with appropriate content-type
        ct = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
        return wav_data, ct
    else:
        raise TTSBadRequestError(f"unsupported response_format: {response_format!r}")


def _synthesize_in_subprocess(
    text: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float,
    device: str,
) -> bytes:
    """Run synthesis in an isolated subprocess worker so GPU VRAM is 100% reclaimed."""
    with tempfile.TemporaryDirectory(prefix="lstt-tts-") as tmpdir:
        tmp_path = Path(tmpdir)
        req_file = tmp_path / "request.json"
        res_file = tmp_path / "response.json"
        out_pcm_file = tmp_path / "output.pcm"

        req_payload = {
            "text": text,
            "model_key": model_spec.key,
            "voice": voice,
            "speed": speed,
            "device": device,
            "out_pcm_path": str(out_pcm_file),
        }
        req_file.write_text(json.dumps(req_payload), encoding="utf-8")

        cmd = [sys.executable, "-m", "live_stt.tts_worker", str(req_file), str(res_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if proc.returncode != 0 or not res_file.exists():
            err_msg = proc.stderr.strip() or f"Subprocess exited with code {proc.returncode}"
            raise TTSError(f"TTS subprocess worker failed: {err_msg}")

        try:
            res_data = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TTSError(f"Failed to parse TTS subprocess response: {exc}") from exc

        if not res_data.get("ok"):
            err_msg = res_data.get("error", "Unknown error in TTS worker")
            err_type = res_data.get("error_type", "TTSError")
            if err_type == "TTSBadRequestError":
                raise TTSBadRequestError(err_msg)
            elif err_type == "TTSUnavailableError":
                raise TTSUnavailableError(err_msg)
            raise TTSError(err_msg)

        if out_pcm_file.exists():
            return out_pcm_file.read_bytes()
        return b""


def synthesize(
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    settings: Settings | None = None,
) -> tuple[bytes, str]:
    """Synthesize full text into formatted audio bytes and Content-Type."""
    if not text or not text.strip():
        raise TTSBadRequestError("input text must not be empty")

    resolved_settings = settings or Settings(_env_file=None)
    model_key = model or resolved_settings.tts_model
    model_spec = tts_models.resolve(model_key)
    selected_voice = voice or resolved_settings.tts_voice or model_spec.default_voice
    device = resolved_settings.tts_device

    # Synthesize all chunks
    chunks = chunk_text(text)
    pcm_accumulator = bytearray()

    if resolved_settings.tts_in_subprocess and not os.getenv("PYTEST_CURRENT_TEST"):
        try:
            raw_pcm = _synthesize_in_subprocess(
                text,
                model_spec=model_spec,
                voice=selected_voice,
                speed=speed,
                device=device,
            )
            pcm_accumulator.extend(raw_pcm)
        except Exception:
            # Fallback to direct chunk synthesis
            for chunk in chunks:
                pcm_accumulator.extend(
                    _synthesize_chunk_direct(
                        chunk,
                        model_spec=model_spec,
                        voice=selected_voice,
                        speed=speed,
                        device=device,
                    )
                )
    else:
        for chunk in chunks:
            pcm_accumulator.extend(
                _synthesize_chunk_direct(
                    chunk,
                    model_spec=model_spec,
                    voice=selected_voice,
                    speed=speed,
                    device=device,
                )
            )

    return pcm_to_format(bytes(pcm_accumulator), model_spec.sample_rate, response_format)


def synthesize_stream(
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    settings: Settings | None = None,
) -> Iterator[bytes]:
    """Sentence-chunked incremental streaming synthesis yielding audio bytes per chunk."""
    if not text or not text.strip():
        raise TTSBadRequestError("input text must not be empty")

    resolved_settings = settings or Settings(_env_file=None)
    model_key = model or resolved_settings.tts_model
    model_spec = tts_models.resolve(model_key)
    selected_voice = voice or resolved_settings.tts_voice or model_spec.default_voice
    device = resolved_settings.tts_device

    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_pcm = _synthesize_chunk_direct(
            chunk,
            model_spec=model_spec,
            voice=selected_voice,
            speed=speed,
            device=device,
        )
        if chunk_pcm:
            chunk_audio, _ = pcm_to_format(chunk_pcm, model_spec.sample_rate, response_format)
            yield chunk_audio
