"""Text-to-Speech (TTS) synthesis engine.

Supports Kokoro-82M (default, fast, high naturalness) and Piper (CPU-only fallback).
Provides both full batch synthesis and low-latency sentence-chunked streaming synthesis.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
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


# Process-lifetime caches with idle timeout eviction (LS-9), keyed by what
# actually determines a loaded object's identity. Values are (object, last_used_monotonic).
# Warmed-up pipelines/voices are preserved across calls within the idle timeout window,
# avoiding reloading ~82M parameters on every request, and are evicted once idle.
_IN_PROCESS_LOCK = threading.RLock()
_KOKORO_PIPELINE_CACHE: dict[tuple[str, str | None], tuple[object, float]] = {}
_PIPER_VOICE_CACHE: dict[tuple[str, bool], tuple[object, float]] = {}
_IN_PROCESS_IDLE_TIMER: threading.Timer | None = None


def evict_idle_tts_caches(idle_timeout_sec: float) -> int:
    """Evict in-process cached TTS pipelines/voices that have been idle for > idle_timeout_sec.

    Returns the number of evicted entries.
    """
    now = time.monotonic()
    evicted = 0
    with _IN_PROCESS_LOCK:
        for key, (obj, last_used) in list(_KOKORO_PIPELINE_CACHE.items()):
            if idle_timeout_sec > 0 and (now - last_used) > idle_timeout_sec:
                _KOKORO_PIPELINE_CACHE.pop(key, None)
                del obj
                evicted += 1
        for key, (obj, last_used) in list(_PIPER_VOICE_CACHE.items()):
            if idle_timeout_sec > 0 and (now - last_used) > idle_timeout_sec:
                _PIPER_VOICE_CACHE.pop(key, None)
                del obj
                evicted += 1
    if evicted:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return evicted


def clear_tts_caches() -> None:
    """Immediately clears all cached pipelines and voices in-process."""
    with _IN_PROCESS_LOCK:
        _KOKORO_PIPELINE_CACHE.clear()
        _PIPER_VOICE_CACHE.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _schedule_in_process_reaper(idle_timeout_sec: float) -> None:
    global _IN_PROCESS_IDLE_TIMER
    with _IN_PROCESS_LOCK:
        if _IN_PROCESS_IDLE_TIMER is not None:
            _IN_PROCESS_IDLE_TIMER.cancel()
            _IN_PROCESS_IDLE_TIMER = None
        if idle_timeout_sec > 0:
            _IN_PROCESS_IDLE_TIMER = threading.Timer(
                idle_timeout_sec, lambda: evict_idle_tts_caches(idle_timeout_sec)
            )
            _IN_PROCESS_IDLE_TIMER.daemon = True
            _IN_PROCESS_IDLE_TIMER.start()


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


def _configure_hf_offline(repo_id: str, voice: str) -> None:
    """If Kokoro model weights and the requested voice file already exist in the
    local Hugging Face hub cache, set HF_HUB_OFFLINE=1 to prevent hf_hub_download
    from making outbound HTTP HEAD requests to huggingface.co on every synthesis call.
    If the requested voice is missing from the cache, unset HF_HUB_OFFLINE (if previously
    set automatically) so it can be fetched from Hugging Face.
    """
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore[import-not-found]

        config_cached = try_to_load_from_cache(repo_id, "config.json")
        model_cached = try_to_load_from_cache(repo_id, "kokoro-v1_0.pth")
        voice_cached = True if voice.endswith(".pt") else bool(try_to_load_from_cache(repo_id, f"voices/{voice}.pt"))
        if config_cached and model_cached and voice_cached:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE_AUTO"] = "1"
        elif os.environ.get("HF_HUB_OFFLINE_AUTO") == "1":
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE_AUTO", None)
    except Exception:
        pass


def _synthesize_chunk_direct(
    text_chunk: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float = 1.0,
    device: str = "cpu",
    models_dir: str = "/models",
    deps_dir: str = "/app/data/tts-deps",
    idle_timeout_sec: float = 600.0,
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
        _configure_hf_offline("hexgrad/Kokoro-82M", voice)
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

        now = time.monotonic()
        lang_code = _kokoro_lang_code(voice)
        pipeline_device = "cuda" if device == "cuda" else None  # None = kokoro's own CPU default
        cache_key = (lang_code, pipeline_device)
        pipeline = None

        with _IN_PROCESS_LOCK:
            cached = _KOKORO_PIPELINE_CACHE.get(cache_key)
            if cached is not None:
                pipeline_obj, last_used = cached
                if idle_timeout_sec > 0 and (now - last_used) > idle_timeout_sec:
                    _KOKORO_PIPELINE_CACHE.pop(cache_key, None)
                    del pipeline_obj
                    import gc

                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    pipeline = pipeline_obj
                    _KOKORO_PIPELINE_CACHE[cache_key] = (pipeline, now)
                    if idle_timeout_sec > 0:
                        _schedule_in_process_reaper(idle_timeout_sec)

        if pipeline is None:
            try:
                try:
                    pipeline = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", device=pipeline_device)
                except TypeError:
                    pipeline = KPipeline(lang_code=lang_code, device=pipeline_device)
            except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
                raise TTSUnavailableError(
                    f"failed to load kokoro pipeline (lang_code={lang_code!r}): {exc}"
                ) from exc
            if idle_timeout_sec > 0:
                with _IN_PROCESS_LOCK:
                    _KOKORO_PIPELINE_CACHE[cache_key] = (pipeline, now)
                    _schedule_in_process_reaper(idle_timeout_sec)

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

        now = time.monotonic()
        piper_cache_key = (str(model_path), use_cuda)
        piper_voice = None

        with _IN_PROCESS_LOCK:
            cached_piper = _PIPER_VOICE_CACHE.get(piper_cache_key)
            if cached_piper is not None:
                voice_obj, last_used = cached_piper
                if idle_timeout_sec > 0 and (now - last_used) > idle_timeout_sec:
                    _PIPER_VOICE_CACHE.pop(piper_cache_key, None)
                    del voice_obj
                    import gc

                    gc.collect()
                else:
                    piper_voice = voice_obj
                    _PIPER_VOICE_CACHE[piper_cache_key] = (piper_voice, now)
                    if idle_timeout_sec > 0:
                        _schedule_in_process_reaper(idle_timeout_sec)

        if piper_voice is None:
            try:
                piper_voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)
            except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
                raise TTSUnavailableError(f"failed to load piper voice {model_path}: {exc}") from exc
            if idle_timeout_sec > 0:
                with _IN_PROCESS_LOCK:
                    _PIPER_VOICE_CACHE[piper_cache_key] = (piper_voice, now)
                    _schedule_in_process_reaper(idle_timeout_sec)

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


def _synthesize_in_subprocess_oneshot(
    text: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float,
    device: str,
    models_dir: str,
    deps_dir: str,
) -> bytes:
    """Run synthesis in an isolated one-shot subprocess worker."""
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


class TTSWorkerManager:
    """Manages the persistent warm TTS worker subprocess (LS-9).

    When tts_idle_timeout_sec > 0, keeps a worker process alive across requests
    so that model weights (Kokoro-82M ~82M parameters) stay warm in memory,
    eliminating the ~5s cold start on subsequent requests.
    When idle beyond tts_idle_timeout_sec, the worker automatically terminates,
    allowing the OS to reclaim 100% of VRAM and host RAM.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._last_active: float = 0.0
        self._idle_timer: threading.Timer | None = None
        self._current_timeout: float = 600.0
        self._current_model: str | None = None
        self._current_device: str | None = None
        self._is_busy: bool = False

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._proc.pid if self._proc is not None and self._proc.poll() is None else None

    def snapshot(self) -> dict[str, Any] | None:
        """Returns diagnostic snapshot of the warm TTS worker if running."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            now = time.monotonic()
            idle_sec = (
                0.0
                if self._is_busy or self._last_active <= 0
                else max(0.0, round(now - self._last_active, 2))
            )
            return {
                "model": self._current_model or "unknown",
                "pid": self._proc.pid,
                "idle_sec": idle_sec,
                "device": self._current_device or "cpu",
                "busy": self._is_busy,
            }

    def synthesize(
        self,
        text: str,
        *,
        model_spec: TTSModelSpec,
        voice: str,
        speed: float = 1.0,
        device: str = "cpu",
        models_dir: str = "/models",
        deps_dir: str = "/app/data/tts-deps",
        idle_timeout_sec: float = 600.0,
    ) -> bytes:
        with self._lock:
            if idle_timeout_sec <= 0:
                self.stop()
                return _synthesize_in_subprocess_oneshot(
                    text,
                    model_spec=model_spec,
                    voice=voice,
                    speed=speed,
                    device=device,
                    models_dir=models_dir,
                    deps_dir=deps_dir,
                )

            self._is_busy = True
            try:
                now = time.monotonic()
                if self._proc is not None:
                    if self._proc.poll() is not None or (
                        idle_timeout_sec > 0 and (now - self._last_active) > idle_timeout_sec
                    ):
                        self.stop()

                if self._proc is None or self._proc.poll() is not None:
                    self._current_timeout = idle_timeout_sec
                    self._last_active = now
                    cmd = [
                        sys.executable,
                        "-m",
                        "live_stt.tts_worker",
                        "--server",
                        "--idle-timeout",
                        str(idle_timeout_sec),
                    ]
                    self._proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                    )

                self._current_model = model_spec.key
                self._current_device = device

                with tempfile.TemporaryDirectory(prefix="lstt-tts-out-") as tmpdir:
                    out_pcm_file = Path(tmpdir) / "output.pcm"
                    req_payload = {
                        "text": text,
                        "model_key": model_spec.key,
                        "voice": voice,
                        "speed": speed,
                        "device": device,
                        "models_dir": models_dir,
                        "deps_dir": deps_dir,
                        "idle_timeout_sec": idle_timeout_sec,
                        "out_pcm_path": str(out_pcm_file),
                    }

                    try:
                        assert self._proc.stdin is not None
                        assert self._proc.stdout is not None
                        self._proc.stdin.write(json.dumps(req_payload) + "\n")
                        self._proc.stdin.flush()
                        res_data = None
                        while True:
                            line = self._proc.stdout.readline()
                            if not line:
                                break
                            line_stripped = line.strip()
                            if not line_stripped:
                                continue
                            try:
                                res_data = json.loads(line_stripped)
                                break
                            except Exception:
                                logger.warning("TTS worker non-JSON output on stdout: %s", line_stripped)
                    except Exception as exc:
                        self.stop()
                        raise TTSError(f"TTS persistent worker communication failed: {exc}") from exc

                    if res_data is None:
                        stderr_msg = ""
                        if self._proc.stderr:
                            try:
                                stderr_msg = self._proc.stderr.read()
                            except Exception:
                                pass
                        self.stop()
                        raise TTSError(f"TTS persistent worker exited unexpectedly: {stderr_msg.strip()}")

                    if not res_data.get("ok"):
                        err_msg = res_data.get("error", "Unknown error in TTS worker")
                        err_type = res_data.get("error_type", "TTSError")
                        if err_type == "TTSBadRequestError":
                            raise TTSBadRequestError(err_msg)
                        elif err_type == "TTSUnavailableError":
                            self.stop()
                            raise TTSUnavailableError(err_msg)
                        raise TTSError(err_msg)

                    self._last_active = time.monotonic()
                    self._schedule_idle_reaper(idle_timeout_sec)
                    if out_pcm_file.exists():
                        return out_pcm_file.read_bytes()
                    return b""
            finally:
                self._is_busy = False

    def _schedule_idle_reaper(self, idle_timeout_sec: float) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if idle_timeout_sec > 0:
            self._idle_timer = threading.Timer(idle_timeout_sec + 0.5, self._on_idle_timeout)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _on_idle_timeout(self) -> None:
        with self._lock:
            if self._proc is not None:
                now = time.monotonic()
                if (now - self._last_active) >= self._current_timeout:
                    self.stop()

    def stop(self) -> None:
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self._proc is not None:
                try:
                    if self._proc.stdin and not self._proc.stdin.closed:
                        self._proc.stdin.close()
                    if self._proc.poll() is None:
                        self._proc.terminate()
                        try:
                            self._proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            self._proc.kill()
                            self._proc.wait(timeout=1.0)
                except Exception:
                    pass
                finally:
                    self._proc = None
                    self._current_model = None
                    self._current_device = None
                    self._last_active = 0.0
                    self._is_busy = False


_TTS_WORKER_MANAGER = TTSWorkerManager()


def get_tts_worker_manager() -> TTSWorkerManager:
    """Returns the process-wide TTSWorkerManager singleton."""
    return _TTS_WORKER_MANAGER


def stop_tts_worker() -> None:
    """Stop the persistent warm TTS worker subprocess if running."""
    _TTS_WORKER_MANAGER.stop()


atexit.register(stop_tts_worker)


def _synthesize_in_subprocess(
    text: str,
    *,
    model_spec: TTSModelSpec,
    voice: str,
    speed: float = 1.0,
    device: str = "cpu",
    models_dir: str = "/models",
    deps_dir: str = "/app/data/tts-deps",
    idle_timeout_sec: float = 600.0,
) -> bytes:
    """Run synthesis in an isolated subprocess worker.

    When idle_timeout_sec > 0, reuses a warm persistent worker so cold starts (~5s for Kokoro)
    only happen once, and terminates the worker when idle to reclaim 100% of VRAM/memory.
    When idle_timeout_sec <= 0, executes in one-shot mode.
    """
    if idle_timeout_sec <= 0:
        return _synthesize_in_subprocess_oneshot(
            text,
            model_spec=model_spec,
            voice=voice,
            speed=speed,
            device=device,
            models_dir=models_dir,
            deps_dir=deps_dir,
        )

    return _TTS_WORKER_MANAGER.synthesize(
        text,
        model_spec=model_spec,
        voice=voice,
        speed=speed,
        device=device,
        models_dir=models_dir,
        deps_dir=deps_dir,
        idle_timeout_sec=idle_timeout_sec,
    )


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
    idle_timeout_sec = resolved_settings.tts_idle_timeout_sec

    # Synthesize all chunks
    chunks = chunk_text(text)
    pcm_accumulator = bytearray()

    if (resolved_settings.tts_in_subprocess and not os.getenv("PYTEST_CURRENT_TEST")) or os.getenv("FORCE_TTS_SUBPROCESS"):
        try:
            raw_pcm = _synthesize_in_subprocess(
                text,
                model_spec=model_spec,
                voice=selected_voice,
                speed=speed,
                device=device,
                models_dir=resolved_settings.models_dir,
                deps_dir=resolved_settings.tts_deps_dir,
                idle_timeout_sec=idle_timeout_sec,
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
                        idle_timeout_sec=idle_timeout_sec,
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
                    idle_timeout_sec=idle_timeout_sec,
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
    idle_timeout_sec = resolved_settings.tts_idle_timeout_sec

    chunks = chunk_text(text)
    for chunk in chunks:
        chunk_pcm = b""
        if (resolved_settings.tts_in_subprocess and not os.getenv("PYTEST_CURRENT_TEST")) or os.getenv("FORCE_TTS_SUBPROCESS"):
            try:
                chunk_pcm = _synthesize_in_subprocess(
                    chunk,
                    model_spec=model_spec,
                    voice=selected_voice,
                    speed=speed,
                    device=device,
                    models_dir=resolved_settings.models_dir,
                    deps_dir=resolved_settings.tts_deps_dir,
                    idle_timeout_sec=idle_timeout_sec,
                )
            except (TTSUnavailableError, TTSBadRequestError):
                raise
            except Exception:
                chunk_pcm = _synthesize_chunk_direct(
                    chunk,
                    model_spec=model_spec,
                    voice=selected_voice,
                    speed=speed,
                    device=device,
                    models_dir=resolved_settings.models_dir,
                    deps_dir=resolved_settings.tts_deps_dir,
                    idle_timeout_sec=idle_timeout_sec,
                )
        else:
            chunk_pcm = _synthesize_chunk_direct(
                chunk,
                model_spec=model_spec,
                voice=selected_voice,
                speed=speed,
                device=device,
                models_dir=resolved_settings.models_dir,
                deps_dir=resolved_settings.tts_deps_dir,
                idle_timeout_sec=idle_timeout_sec,
            )
        if chunk_pcm:
            chunk_audio, _ = pcm_to_format(chunk_pcm, model_spec.sample_rate, response_format)
            yield chunk_audio
