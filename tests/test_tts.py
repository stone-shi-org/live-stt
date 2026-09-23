from __future__ import annotations

import io
import json
import sys
import types
import wave
import pytest

from live_stt.config import Settings
from live_stt.tts import (
    TTSBadRequestError,
    TTSUnavailableError,
    chunk_text,
    pcm_to_format,
    pcm_to_wav,
    synthesize,
    synthesize_stream,
)


class _FakeAudioTensor:
    """Duck-types the tiny slice of torch.Tensor's API
    live_stt/tts.py's kokoro branch actually calls
    (.detach().cpu().numpy()) -- so this fixture doesn't need a real torch
    install, which is a multi-hundred-MB dependency this offline test
    suite deliberately avoids (see requirements-tts-kokoro.txt's own
    docstring: kokoro is opt-in, runtime-installed, never baked into the
    always-on test-unit image)."""

    def __init__(self, array) -> None:
        self._array = array

    def detach(self) -> "_FakeAudioTensor":
        return self

    def cpu(self) -> "_FakeAudioTensor":
        return self

    def numpy(self):
        return self._array


def _install_fake_kokoro(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for the real kokoro package -- offline, no torch/real
    model weights needed -- so tests can exercise live_stt/tts.py's REAL
    synthesize()/synthesize_stream() code path end to end (chunking,
    subprocess-vs-direct dispatch, pcm_to_format conversion) without
    depending on the real, heavy dependency being installed. Mirrors the
    fake-pyannote/fake-SortformerEncLabelModel pattern
    tests/test_diarization.py already uses for the same reason. Matches
    the REAL kokoro>=0.9.4 public API confirmed against the actual
    upstream project (KPipeline, not a bare module-level generate()
    function -- see tts.py's own module-level comment on the earlier,
    incorrect assumption this replaced).
    """
    import numpy as np

    fake_kokoro = types.ModuleType("kokoro")

    class _FakeResult:
        def __init__(self, audio) -> None:
            self.audio = audio

    class _FakeKPipeline:
        def __init__(self, lang_code: str, device: str | None = None, **kwargs) -> None:
            self.lang_code = lang_code
            self.device = device

        def __call__(self, text: str, voice: str = "af_heart", speed: float = 1.0, **kwargs):
            n = max(100, len(text) * 10)
            t = np.linspace(0, 1, n, dtype=np.float32)
            audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            yield _FakeResult(_FakeAudioTensor(audio))

    fake_kokoro.KPipeline = _FakeKPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)


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


def test_synthesize_full(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kokoro(monkeypatch)
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


def test_synthesize_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kokoro(monkeypatch)
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


class TestMissingEngineRaisesClearly:
    """Real bug, found against the actual 10.100.0.50 production
    deployment: live_stt/tts.py used to catch ImportError for a missing
    kokoro/piper and silently return a synthetic sine-wave tone instead of
    real speech -- a real HTTP 200 with plausible-looking audio bytes that
    were NOT what the caller asked for, no visible error anywhere. Fixed
    to raise TTSUnavailableError instead, matching
    live_stt/diarization.py's DiarizationError pattern for the same class
    of "heavy engine not installed" condition.
    """

    def test_missing_kokoro_raises_with_install_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "kokoro", None)
        settings = Settings(_env_file=None, tts_in_subprocess=False)
        with pytest.raises(TTSUnavailableError, match="kokoro is not installed"):
            synthesize("Hello", model="kokoro-82m-v1", settings=settings)

    def test_missing_piper_raises_with_install_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "piper", None)
        settings = Settings(_env_file=None, tts_in_subprocess=False)
        with pytest.raises(TTSUnavailableError, match="piper is not installed"):
            synthesize("Hello", model="piper-en-us", settings=settings)

    def test_missing_piper_voice_model_names_the_real_download_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Real bug, found for real while running a piper voice-model
        # download for the first time: this message used to say
        # `--output-dir`, but piper's actual download_voices.py CLI flag
        # is `--download-dir` -- confirmed by actually running the wrong
        # one and getting "unrecognized arguments". Separately also
        # confirmed for real: models_dir is mounted READ-ONLY inside the
        # running container, so the message must not suggest running the
        # download command via `docker exec` on that same container.
        fake_piper = types.ModuleType("piper")
        fake_piper.PiperVoice = type("PiperVoice", (), {})  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "piper", fake_piper)
        settings = Settings(_env_file=None, tts_in_subprocess=False, models_dir=str(tmp_path))
        with pytest.raises(TTSUnavailableError) as exc_info:
            synthesize("Hello", model="piper-en-us", settings=settings)
        message = str(exc_info.value)
        # The real, correct flag must appear in the actual example command.
        assert "piper.download_voices" in message and "--download-dir" in message
        assert "read-only" in message.lower()

    def test_real_subprocess_worker_propagates_the_same_error(self, tmp_path) -> None:
        # synthesize()'s own dispatch always takes the direct path under
        # pytest (see its PYTEST_CURRENT_TEST check, a pre-existing guard
        # this test doesn't touch) -- so exercising the REAL
        # `live_stt.tts_worker` subprocess (no kokoro installed there
        # either, same offline environment) needs calling
        # _synthesize_in_subprocess directly, bypassing that guard. This
        # is a genuine subprocess spawn + real IPC via request/response
        # JSON files, not a mock -- confirms tts_worker.py's own
        # error_type mapping (TTSUnavailableError -> "TTSUnavailableError"
        # -> re-raised as TTSUnavailableError) round-trips correctly.
        from live_stt.tts import _synthesize_in_subprocess
        from live_stt.tts_models import resolve as resolve_tts_model

        with pytest.raises(TTSUnavailableError, match="kokoro is not installed"):
            _synthesize_in_subprocess(
                "Hello",
                model_spec=resolve_tts_model("kokoro-82m-v1"),
                voice="af_heart",
                speed=1.0,
                device="cpu",
                models_dir=str(tmp_path),
                deps_dir=str(tmp_path / "tts-deps"),
            )


class TestPiperDeviceFallback:
    """LS-8: piper-tts packages only CPU onnxruntime by default. When
    tts_device="cuda" is configured (e.g. for Kokoro), Piper synthesis
    must gracefully fall back to CPU rather than raising TTSUnavailableError,
    or use CUDA if CUDAExecutionProvider is actually available.
    """

    def test_piper_cuda_device_falls_back_to_cpu_when_no_cuda_provider(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake_piper = types.ModuleType("piper")
        load_calls = []

        class FakePiperVoice:
            @classmethod
            def load(cls, model_path: str, use_cuda: bool = False):
                load_calls.append({"model_path": model_path, "use_cuda": use_cuda})
                return cls()

            def synthesize(self, text, syn_config=None):
                return [types.SimpleNamespace(audio_int16_bytes=b"\x00" * 3200)]

        fake_piper.PiperVoice = FakePiperVoice
        monkeypatch.setitem(sys.modules, "piper", fake_piper)

        fake_onnxruntime = types.ModuleType("onnxruntime")
        fake_onnxruntime.get_available_providers = lambda: ["CPUExecutionProvider"]
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

        model_file = tmp_path / "en_US-lessac-medium.onnx"
        model_file.write_bytes(b"dummy")

        settings = Settings(
            _env_file=None,
            tts_in_subprocess=False,
            tts_device="cuda",
            models_dir=str(tmp_path),
        )
        audio_bytes, ct = synthesize(
            "Hello world", model="piper-en-us", response_format="wav", settings=settings
        )
        assert audio_bytes
        assert ct == "audio/wav"
        assert len(load_calls) == 1
        assert load_calls[0]["use_cuda"] is False

    def test_piper_cuda_device_uses_cuda_when_cuda_provider_is_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake_piper = types.ModuleType("piper")
        load_calls = []

        class FakePiperVoice:
            @classmethod
            def load(cls, model_path: str, use_cuda: bool = False):
                load_calls.append({"model_path": model_path, "use_cuda": use_cuda})
                return cls()

            def synthesize(self, text, syn_config=None):
                return [types.SimpleNamespace(audio_int16_bytes=b"\x00" * 3200)]

        fake_piper.PiperVoice = FakePiperVoice
        monkeypatch.setitem(sys.modules, "piper", fake_piper)

        fake_onnxruntime = types.ModuleType("onnxruntime")
        fake_onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)

        model_file = tmp_path / "en_US-lessac-medium.onnx"
        model_file.write_bytes(b"dummy")

        settings = Settings(
            _env_file=None,
            tts_in_subprocess=False,
            tts_device="cuda",
            models_dir=str(tmp_path),
        )
        audio_bytes, ct = synthesize(
            "Hello world", model="piper-en-us", response_format="wav", settings=settings
        )
        assert audio_bytes
        assert ct == "audio/wav"
        assert len(load_calls) == 1
        assert load_calls[0]["use_cuda"] is True


class TestResolveVoiceCrossEngineBug:
    """Real bug, found for real against a running container while
    verifying the fixes above: Settings.tts_voice defaults to "af_heart"
    (a kokoro voice name) globally, with no awareness of which model is
    actually selected. Before this fix, requesting model=piper-en-us with
    no explicit voice= silently carried that kokoro default straight into
    piper's voice-model lookup ("af_heart.onnx", a file that could never
    exist) instead of piper's own real default en_US-lessac-medium --
    confirmed via a real HTTP request against a real running container
    before this fix, and again after, to confirm the fix.
    """

    def test_explicit_voice_always_wins(self) -> None:
        from live_stt.tts import resolve_voice
        from live_stt.tts_models import resolve as resolve_tts_model

        settings = Settings(_env_file=None, tts_voice="af_heart")
        spec = resolve_tts_model("piper-en-us")
        assert resolve_voice("en_US-amy-medium", settings, spec) == "en_US-amy-medium"

    def test_global_default_used_only_when_valid_for_the_selected_model(self) -> None:
        from live_stt.tts import resolve_voice
        from live_stt.tts_models import resolve as resolve_tts_model

        settings = Settings(_env_file=None, tts_voice="af_heart")

        # Valid for kokoro -- honored.
        kokoro_spec = resolve_tts_model("kokoro-82m-v1")
        assert resolve_voice(None, settings, kokoro_spec) == "af_heart"

        # NOT valid for piper (a kokoro voice name) -- must fall back to
        # piper's OWN default, not silently carry "af_heart" over.
        piper_spec = resolve_tts_model("piper-en-us")
        assert resolve_voice(None, settings, piper_spec) == piper_spec.default_voice
        assert resolve_voice(None, settings, piper_spec) != "af_heart"

    def test_http_layer_resolves_voice_after_model_not_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The actual bug lived in tts_http.py, which used to compute
        # `payload.get("voice") or settings.tts_voice` BEFORE model_spec
        # even existed. Exercise the real HTTP handler end to end (not
        # just the resolve_voice() unit above) to pin the fix at the
        # layer that actually broke. A minimal fake `piper` module (just
        # enough to satisfy the `from piper import PiperVoice` import) is
        # needed to get PAST the "piper is not installed" check and reach
        # the voice-model-file lookup this test is actually about.
        fake_piper = types.ModuleType("piper")
        fake_piper.PiperVoice = type("PiperVoice", (), {})  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "piper", fake_piper)

        from live_stt import tts_http
        from live_stt.tts_sessions import TTSSessionTracker

        settings = Settings(_env_file=None, tts_voice="af_heart")
        tracker = TTSSessionTracker()
        status, body, ct, _ = tts_http.handle_speech_request(
            content_type="application/json",
            body=json.dumps({"input": "hello", "model": "piper-en-us"}).encode(),
            settings=settings,
            tracker=tracker,
        )
        # The voice model file won't exist in this test environment either
        # -- but the error should name piper's OWN default voice, not
        # kokoro's "af_heart".
        assert status == 503
        assert b"en_US-lessac-medium" in body
        assert b"af_heart" not in body
