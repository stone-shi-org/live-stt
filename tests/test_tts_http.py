from __future__ import annotations

import json
import urllib.error
import urllib.request
import pytest

from live_stt import gpu, tts_http
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.state import ServerState
from live_stt.tts_sessions import TTSSessionTracker


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


def test_happy_path_non_streaming() -> None:
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


def test_happy_path_streaming() -> None:
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
async def test_live_http_server_speech_endpoint() -> None:
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
