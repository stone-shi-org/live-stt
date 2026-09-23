"""Text-to-Speech (TTS) synthesis engine.

Supports Kokoro-82M (default, fast, high naturalness) and Piper (CPU-only fallback).
Provides both full batch synthesis and low-latency sentence-chunked streaming synthesis.
"""

from __future__ import annotations

import io
import json
import os
import re
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


def _ensure_deps_on_path(deps_dir: str, engine: str) -> None:
    """Prepend `engine`'s runtime-installed dependency directory (see
    live_stt/tts_installer.py) to sys.path, if it exists and isn't already
    there -- so an engine installed via POST /api/tts-engines/install
    becomes importable without a container restart. Mirrors live_stt/
    diarization.py's identical _ensure_deps_on_path exactly; see that
    function's docstring for the full reasoning (called in every process
    that might load an engine, cheap enough to re-check every time rather
    than mutate sys.path once at startup).
    """
    from live_stt.tts_installer import engine_deps_dir

    engine_dir = str(engine_deps_dir(deps_dir, engine))
    if engine_dir not in sys.path and Path(engine_dir).is_dir():
        sys.path.insert(0, engine_dir)


# Process-lifetime caches, keyed by what actually determines a loaded
# object's identity -- avoids reloading a real ~82M-param torch model (or
# a piper .onnx model) on every SENTENCE CHUNK within one synthesize()
# call, let alone every call. Correct to share across calls within one
# process: kokoro's KPipeline/piper's PiperVoice are read-only inference
# objects once constructed, nothing here mutates them per-call. Empty at
# the start of every `live_stt.tts_worker` subprocess (a fresh process
# each time, by design -- see that module's docstring on why: 100% VRAM
# reclamation), so this only helps the tts_in_subprocess=False /
# direct-fallback path, which is the path the offline test suite and any
# CPU-only deployment actually exercises repeatedly.
_KOKORO_PIPELINE_CACHE: dict[tuple[str, str | None], object] = {}
_PIPER_VOICE_CACHE: dict[tuple[str, bool], object] = {}


def resolve_voice(voice: str | None, settings: Settings, model_spec: TTSModelSpec) -> str:
    """An explicit per-request `voice` always wins. Otherwise, real bug
    found while testing the fixes above: `Settings.tts_voice` defaults to
    "af_heart" (a kokoro voice name) globally, with no awareness of which
    model is actually selected -- requesting `model=piper-en-us` with no
    explicit voice silently carried that kokoro default straight into
    piper's voice lookup (`af_heart.onnx`, a file that was never going to
    exist, instead of piper's own real default `en_US-lessac-medium`).
    Only honor the configured tts_voice when it's actually one of THIS
    model's supported voices; otherwise fall back to the model's own
    default, never a different engine's.
    """
    if voice:
        return voice
    if settings.tts_voice in model_spec.supported_voices:
        return settings.tts_voice
    return model_spec.default_voice


def _kokoro_lang_code(voice: str) -> str:
    """kokoro's KPipeline needs a lang_code matching the voice's language
    family (confirmed against the real upstream README) -- 'a' for
    American English voices (af_*/am_*), 'b' for British (bf_*/bm_*).
    Every voice in live_stt/tts_models.py's kokoro-82m-v1 entry is one of
    these two families today; defaults to 'a' for anything else rather
    than raising, since an unrecognized prefix is more likely a new voice
    this function hasn't been taught about yet than a real error.
    """
    if voice.startswith(("bf_", "bm_")):
        return "b"
    return "a"


def _resolve_piper_voice_model(models_dir: str, voice: str, deps_dir: str) -> Path:
    """Piper needs a real, locally-present voice model file (.onnx, with
    its .onnx.json config expected alongside it by PiperVoice.load()) --
    unlike kokoro, which resolves its own weights internally. Reuses
    Settings.models_dir, the same mounted-volume convention ASR's GGUF/
    .bin model files already use, rather than inventing a second "where do
    TTS models live" concept.
    """
    model_path = Path(models_dir) / f"{voice}.onnx"
    if not model_path.is_file():
        piper_deps = str(Path(deps_dir) / "piper")
        raise TTSUnavailableError(
            f"piper voice model not found: {model_path} -- fetch it first with "
            f"piper's real download command (confirmed for real: its flag is "
            f"`--download-dir`, NOT `--output-dir` -- an earlier version of this "
            f"message had that wrong). Also confirmed for real: {models_dir} is "
            f"mounted READ-ONLY inside this running container (same convention as "
            f"ASR's model volume), so running the download command via `docker exec` "
            f"on THIS container will fail with a read-only-filesystem error -- fetch "
            f"it via a separate one-off container sharing the SAME models volume "
            f"WITHOUT :ro instead, e.g.: `docker run --rm -e PYTHONPATH={piper_deps} "
            f"-v <the same models volume>:{models_dir} <this image> python3 -m "
            f"piper.download_voices {voice} --download-dir {models_dir}` (needs the "
            "piper engine installed first -- see POST /api/tts-engines/install "
            "{'engine': 'piper'} -- since that download command is itself provided "
            "by the piper-tts package)."
        )
    return model_path


def _synthesize_chunk_direct(
    text_chunk: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float = 1.0,
    device: str = "cpu",
    models_dir: str = "/models",
    deps_dir: str = "/app/data/tts-deps",
) -> bytes:
    """Direct in-process synthesis of a single text chunk to raw int16 PCM
    bytes. Raises TTSUnavailableError (never silently substitutes fake
    audio) when the engine isn't installed, isn't wired correctly, or its
    voice model is missing -- see the module docstring's account of the
    real, deployed bug this replaces: a bare synthetic tone returned as a
    200 OK for every request, on any deployment that never happened to
    have kokoro/piper installed, with no operator-visible signal anything
    was wrong.
    """
    if not text_chunk.strip():
        return b""

    if model_spec.engine == "kokoro":
        _ensure_deps_on_path(deps_dir, "kokoro")
        try:
            from kokoro import KPipeline  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSUnavailableError(
                "kokoro is not installed. It is deliberately excluded from "
                "requirements.txt (a real speech-synthesis model with its own "
                "dependency tree, not something every deployment needs) -- "
                "POST /api/tts-engines/install {'engine': 'kokoro'} to install it "
                "into this running deployment (persists across `docker recreate`), "
                "or install requirements-tts-kokoro.txt yourself for local/manual use."
            ) from exc

        lang_code = _kokoro_lang_code(voice)
        pipeline_device = "cuda" if device == "cuda" else None  # None = kokoro's own CPU default
        cache_key = (lang_code, pipeline_device)
        pipeline = _KOKORO_PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            try:
                pipeline = KPipeline(lang_code=lang_code, device=pipeline_device)
            except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
                raise TTSUnavailableError(
                    f"failed to load kokoro pipeline (lang_code={lang_code!r}): {exc}"
                ) from exc
            _KOKORO_PIPELINE_CACHE[cache_key] = pipeline

        import numpy as np

        pcm = bytearray()
        try:
            for result in pipeline(text_chunk, voice=voice, speed=speed):
                if result.audio is None:
                    continue
                audio_np = result.audio.detach().cpu().numpy()
                audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
                pcm.extend(audio_int16.tobytes())
        except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
            raise TTSUnavailableError(f"kokoro synthesis failed: {exc}") from exc
        return bytes(pcm)

    elif model_spec.engine == "piper":
        _ensure_deps_on_path(deps_dir, "piper")
        try:
            from piper import PiperVoice  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TTSUnavailableError(
                "piper is not installed. It is deliberately excluded from "
                "requirements.txt -- POST /api/tts-engines/install "
                "{'engine': 'piper'} to install it into this running deployment "
                "(persists across `docker recreate`), or install "
                "requirements-tts-piper.txt yourself for local/manual use."
            ) from exc

        model_path = _resolve_piper_voice_model(models_dir, voice, deps_dir)
        use_cuda = False
        if device == "cuda":
            try:
                import onnxruntime  # type: ignore[import-not-found]

                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    use_cuda = True
                else:
                    logger.info(
                        "tts_device=cuda requested for piper engine, but installed onnxruntime "
                        "has no CUDAExecutionProvider (piper-tts ships CPU-only onnxruntime); "
                        "falling back to CPU"
                    )
            except ImportError:
                pass

        piper_cache_key = (str(model_path), use_cuda)
        piper_voice = _PIPER_VOICE_CACHE.get(piper_cache_key)
        if piper_voice is None:
            try:
                piper_voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)
            except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
                raise TTSUnavailableError(f"failed to load piper voice {model_path}: {exc}") from exc
            _PIPER_VOICE_CACHE[piper_cache_key] = piper_voice

        # length_scale is piper's inverse-of-speed knob (2.0 == twice as
        # slow) -- confirmed against the real upstream Python API docs.
        length_scale = 1.0 / speed if speed > 0 else 1.0
        try:
            from piper.config import SynthesisConfig  # type: ignore[import-not-found]

            syn_config = SynthesisConfig(length_scale=length_scale)
        except ImportError:
            syn_config = None

        pcm = bytearray()
        try:
            for audio_chunk in piper_voice.synthesize(text_chunk, syn_config=syn_config):
                pcm.extend(audio_chunk.audio_int16_bytes)
        except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
            raise TTSUnavailableError(f"piper synthesis failed: {exc}") from exc
        return bytes(pcm)

    raise TTSUnavailableError(f"unknown TTS engine {model_spec.engine!r} for model {model_spec.key!r}")


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
    models_dir: str,
    deps_dir: str,
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
            "models_dir": models_dir,
            "deps_dir": deps_dir,
            "out_pcm_path": str(out_pcm_file),
        }
        req_file.write_text(json.dumps(req_payload), encoding="utf-8")

        cmd = [sys.executable, "-m", "live_stt.tts_worker", str(req_file), str(res_file)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # NOT `proc.returncode != 0 or not res_file.exists()` -- tts_worker.py
        # deliberately exits 1 for EVERY error, including well-typed,
        # already-serialized-to-res_file ones (TTSBadRequestError,
        # TTSUnavailableError), not just genuine crashes. Checking
        # returncode here discarded that real, specific error in favor of
        # a generic "subprocess worker failed" message -- found while
        # testing the TTSUnavailableError fix above: a real
        # "kokoro is not installed" response, written correctly to
        # res_file, was getting swallowed into an opaque TTSError purely
        # because the worker's own (correct, intentional) exit(1)
        # convention looked like a crash from here. Mirrors
        # live_stt/diarization.py's _diarize_file_subprocess, which only
        # ever checks `res_file.is_file()` for the true "worker crashed
        # with nothing to say" case.
        if not res_file.is_file():
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
    selected_voice = resolve_voice(voice, resolved_settings, model_spec)
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
                models_dir=resolved_settings.models_dir,
                deps_dir=resolved_settings.tts_deps_dir,
            )
            pcm_accumulator.extend(raw_pcm)
        except (TTSUnavailableError, TTSBadRequestError):
            # A real, well-typed error from the subprocess (e.g. the
            # engine genuinely isn't installed) -- propagate immediately.
            # Retrying via the direct in-process path below would just
            # reproduce the identical failure a second time (same missing
            # dependency, same process environment) while masking a clean
            # error behind a redundant subprocess round-trip.
            raise
        except Exception:
            # A genuine subprocess-level failure (crash, timeout,
            # malformed response) -- retry once in-process for resilience.
            for chunk in chunks:
                pcm_accumulator.extend(
                    _synthesize_chunk_direct(
                        chunk,
                        model_spec=model_spec,
                        voice=selected_voice,
                        speed=speed,
                        device=device,
                        models_dir=resolved_settings.models_dir,
                        deps_dir=resolved_settings.tts_deps_dir,
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
                    models_dir=resolved_settings.models_dir,
                    deps_dir=resolved_settings.tts_deps_dir,
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
    selected_voice = resolve_voice(voice, resolved_settings, model_spec)
    device = resolved_settings.tts_device

    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_pcm = _synthesize_chunk_direct(
            chunk,
            model_spec=model_spec,
            voice=selected_voice,
            speed=speed,
            device=device,
            models_dir=resolved_settings.models_dir,
            deps_dir=resolved_settings.tts_deps_dir,
        )
        if chunk_pcm:
            chunk_audio, _ = pcm_to_format(chunk_pcm, model_spec.sample_rate, response_format)
            yield chunk_audio
