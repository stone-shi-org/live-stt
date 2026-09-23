from __future__ import annotations

import json
import sys
import types
import urllib.error
import urllib.request
import pytest

from live_stt import gpu, tts_http
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.state import ServerState
from live_stt.tts_sessions import TTSSessionTracker


def _install_fake_kokoro(monkeypatch: pytest.MonkeyPatch) -> None:
    """See tests/test_tts.py's identical helper for the full reasoning --
    stands in for the real kokoro package (its real KPipeline-based API,
    not a bare generate() function) so the "happy path" tests below
    exercise the real synthesis code path without depending on the real,
    heavy dependency being installed.
    """
    import numpy as np

    fake_kokoro = types.ModuleType("kokoro")

    class _FakeAudioTensor:
        def __init__(self, array) -> None:
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    class _FakeResult:
        def __init__(self, audio) -> None:
            self.audio = audio

    class _FakeKPipeline:
        def __init__(self, lang_code: str, device: str | None = None, **kwargs) -> None:
            pass

        def __call__(self, text: str, voice: str = "af_heart", speed: float = 1.0, **kwargs):
            n = max(100, len(text) * 10)
            t = np.linspace(0, 1, n, dtype=np.float32)
            audio = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            yield _FakeResult(_FakeAudioTensor(audio))

    fake_kokoro.KPipeline = _FakeKPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)


def test_missing_input_returns_400() -> None:
    settings = Settings(_env_file=None)
    tracker = TTSSessionTracker()
    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"model": "kokoro-82m-v1"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 400
    assert b"missing or empty required field 'input'" in body


def test_unknown_model_returns_400() -> None:
    settings = Settings(_env_file=None)
    tracker = TTSSessionTracker()
    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Hello", "model": "invalid-model"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 400
    assert b"unknown TTS model" in body


def test_draining_returns_503() -> None:
    settings = Settings(_env_file=None)
    tracker = TTSSessionTracker()
    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Hello"}).encode(),
        settings=settings,
        tracker=tracker,
        draining=True,
    )
    assert status == 503
    assert b"draining" in body


def test_cuda_vram_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, tts_device="cuda", tts_vram_mb=5000)
    tracker = TTSSessionTracker()

    # Mock gpu.free_vram_mb to return 1000 MiB (less than 5000 MiB required)
    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 1000)

    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Hello"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 503
    assert b"insufficient free VRAM" in body
    assert tracker.rejected_vram_total == 1


def test_cuda_vram_rejection_skipped_for_zero_vram_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # LS-8: piper-en-us has measured_peak_vram_mb=0 and runs on CPU (or falls back to CPU).
    # Even if tts_device="cuda" and free VRAM is below tts_vram_mb, piper requests
    # must not be rejected by the VRAM admission gate.
    fake_piper = types.ModuleType("piper")
    fake_piper.PiperVoice = type("PiperVoice", (), {})
    monkeypatch.setitem(sys.modules, "piper", fake_piper)

    settings = Settings(
        _env_file=None,
        tts_device="cuda",
        tts_vram_mb=5000,
        models_dir=str(tmp_path),
    )
    tracker = TTSSessionTracker()

    monkeypatch.setattr(gpu, "free_vram_mb", lambda: 1000)

    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Hello", "model": "piper-en-us"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    # Gating should NOT have rejected with 503 "insufficient free VRAM".
    # It reaches voice model loading (which raises 503 voice model not found since tmp_path has no voice).
    assert b"insufficient free VRAM" not in body
    assert tracker.rejected_vram_total == 0


def test_missing_engine_returns_503_not_a_silent_fake_tone(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real bug, found against actual production (see live_stt/tts.py's
    # module docstring): before this fix, a missing kokoro/piper made this
    # endpoint return 200 with a fake sine-wave tone -- this path was
    # UNREACHABLE from an HTTP test's perspective (no way to observe the
    # difference between a 200-with-real-speech and a 200-with-a-tone
    # without decoding audio content). Now it's a real, distinguishable
    # 503, and this test can actually see it.
    monkeypatch.setitem(sys.modules, "kokoro", None)
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    tracker = TTSSessionTracker()
    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Hello", "model": "kokoro-82m-v1"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 503
    assert b"kokoro is not installed" in body
    assert tracker.completed_total == 0


def test_happy_path_non_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kokoro(monkeypatch)
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    tracker = TTSSessionTracker()
    status, body, ct, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Testing speech synthesis.", "response_format": "wav"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 200
    assert ct == "audio/wav"
    assert isinstance(body, bytes)
    assert body.startswith(b"RIFF")
    assert tracker.completed_total == 1
    assert tracker.active == 0


def test_happy_path_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kokoro(monkeypatch)
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    tracker = TTSSessionTracker()
    status, generator, ct, headers = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "First. Second.", "stream": True, "response_format": "pcm"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status == 200
    assert headers.get("Transfer-Encoding") == "chunked"
    chunks = list(generator)
    assert len(chunks) == 2
    assert tracker.completed_total == 1
    assert tracker.active == 0


@pytest.mark.asyncio
async def test_live_http_server_speech_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_kokoro(monkeypatch)
    settings = Settings(_env_file=None, tts_in_subprocess=False)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        req_data = json.dumps({"input": "Testing speech synthesis over HTTP.", "response_format": "wav"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/audio/speech",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "audio/wav"
            data = resp.read()
            assert data.startswith(b"RIFF")
    finally:
        server.shutdown()
