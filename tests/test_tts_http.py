from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
import pytest

from live_stt import gpu, tts, tts_http
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


def _install_fake_piper(monkeypatch: pytest.MonkeyPatch, models_dir: Path) -> None:
    """Install a deterministic fake piper module and voice file so HTTP tests
    can exercise the real piper-en-us code path without the runtime-installed
    piper-tts dependency or model weights."""
    fake_piper = types.ModuleType("piper")

    class _FakePiperVoice:
        @classmethod
        def load(cls, model_path: str, use_cuda: bool = False):
            return cls()

        def synthesize(self, text: str, syn_config=None):
            # Deterministic 16-bit PCM bytes whose length depends on the sentence
            n_samples = max(64, len(text) * 16)
            pcm = (b"\x12\x34" * n_samples)
            return [types.SimpleNamespace(audio_int16_bytes=pcm)]

    fake_piper.PiperVoice = _FakePiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake_piper)
    (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake-onnx")


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
    assert "Transfer-Encoding" not in headers
    chunks = list(generator)
    assert len(chunks) == 2
    assert tracker.completed_total == 1
    assert tracker.active == 0


def _raw_http_post_speech(port: int, payload: dict) -> tuple[bytes, bytes]:
    """Send a raw HTTP POST to /v1/audio/speech over a TCP socket and read all
    bytes until the server closes the connection. Returns (raw_headers, raw_body)."""
    body = json.dumps(payload).encode("utf-8")
    req = (
        f"POST /v1/audio/speech HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode("ascii") + body

    chunks: list[bytes] = []
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(req)
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)

    raw = b"".join(chunks)
    sep = b"\r\n\r\n"
    idx = raw.find(sep)
    assert idx != -1, f"missing HTTP header terminator in raw response: {raw[:200]!r}"
    return raw[:idx], raw[idx + len(sep) :]


@pytest.mark.parametrize("model_key", ["kokoro-82m-v1", "piper-en-us"])
def test_live_http_server_streaming_raw_socket_matches_batch_without_chunk_framing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_key: str
) -> None:
    """LS-16: On HTTP/1.0, streaming TTS must not send Transfer-Encoding: chunked
    or write hex chunk markers ({len:X}\\r\\n ... \\r\\n0\\r\\n\\r\\n) into the wire body.
    For both kokoro and piper across a multi-chunk (multi-sentence) stream, the raw
    wire body must be byte-for-byte identical to stream=false and have 16-bit PCM
    frame alignment (even byte length)."""
    _install_fake_kokoro(monkeypatch)
    _install_fake_piper(monkeypatch, tmp_path)

    settings = Settings(_env_file=None, tts_in_subprocess=False, models_dir=str(tmp_path))
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        multi_sentence_input = "First sentence here. Second sentence follows! Third sentence ends."

        batch_headers, batch_body = _raw_http_post_speech(
            port,
            {
                "input": multi_sentence_input,
                "model": model_key,
                "response_format": "pcm",
                "stream": False,
            },
        )
        stream_headers, stream_body = _raw_http_post_speech(
            port,
            {
                "input": multi_sentence_input,
                "model": model_key,
                "response_format": "pcm",
                "stream": True,
            },
        )

        headers_text = stream_headers.decode("iso-8859-1")
        assert headers_text.startswith("HTTP/1.0 200 OK")
        assert "transfer-encoding" not in headers_text.lower()
        assert "content-length" not in headers_text.lower()
        assert "content-type: audio/pcm" in headers_text.lower()

        assert len(stream_body) > 0
        assert len(stream_body) % 2 == 0, f"unaligned 16-bit PCM byte length: {len(stream_body)}"
        assert stream_body == batch_body
        assert not stream_body.endswith(b"0\r\n\r\n")
    finally:
        server.shutdown()


def test_live_http_server_streaming_delivers_audio_progressively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LS-16 regression: raw HTTP/1.0 streaming must still flush each synthesized
    sentence chunk immediately so the client receives the first audio bytes before
    synthesis of subsequent chunks finishes."""
    chunk1_bytes = b"\x01\x02" * 500
    chunk2_bytes = b"\x03\x04" * 600
    chunk1_sent = threading.Event()
    allow_chunk2 = threading.Event()

    def slow_synthesize_stream(*args, **kwargs):
        yield chunk1_bytes
        chunk1_sent.set()
        assert allow_chunk2.wait(timeout=5.0), "timed out waiting for test to release chunk 2"
        yield chunk2_bytes

    monkeypatch.setattr(tts, "synthesize_stream", slow_synthesize_stream)

    settings = Settings(_env_file=None, tts_in_subprocess=False)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        body = json.dumps(
            {"input": "First sentence. Second sentence.", "stream": True, "response_format": "pcm"}
        ).encode("utf-8")
        req = (
            f"POST /v1/audio/speech HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"\r\n"
        ).encode("ascii") + body

        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(req)
            assert chunk1_sent.wait(timeout=5.0), "first chunk was never yielded"

            # Read while chunk 2 synthesis is still blocked on allow_chunk2
            received = b""
            sep = b"\r\n\r\n"
            while True:
                part = sock.recv(65536)
                assert part, "socket closed prematurely before chunk 1 arrived"
                received += part
                idx = received.find(sep)
                if idx != -1 and len(received) - (idx + len(sep)) >= len(chunk1_bytes):
                    break

            idx = received.find(sep)
            early_body = received[idx + len(sep) :]
            assert early_body == chunk1_bytes
            assert not allow_chunk2.is_set()

            # Now release chunk 2 and read until EOF
            allow_chunk2.set()
            rest: list[bytes] = []
            while True:
                part = sock.recv(65536)
                if not part:
                    break
                rest.append(part)

            full_body = early_body + b"".join(rest)
            assert full_body == chunk1_bytes + chunk2_bytes
    finally:
        allow_chunk2.set()
        server.shutdown()


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


def test_consecutive_speech_requests_reuse_worker_pid(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from live_stt.tts import _TTS_WORKER_MANAGER

    fake_out = tmp_path / "output.pcm"
    fake_out.write_bytes(b"\x00\x00" * 100)

    class FakeProc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(json.dumps({"ok": True}) + "\n" + json.dumps({"ok": True}) + "\n")
            self.stderr = io.StringIO()
            self.pid = 54321

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    fake_proc = FakeProc()
    _TTS_WORKER_MANAGER.stop()
    _TTS_WORKER_MANAGER._proc = fake_proc
    _TTS_WORKER_MANAGER._last_active = time.monotonic()

    class DummyTmpDir:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("tempfile.TemporaryDirectory", lambda prefix=None: DummyTmpDir())
    monkeypatch.setenv("FORCE_TTS_SUBPROCESS", "1")

    settings = Settings(_env_file=None, tts_in_subprocess=True, tts_idle_timeout_sec=600.0)
    tracker = TTSSessionTracker()

    status1, body1, ct1, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Sentence one.", "response_format": "wav"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status1 == 200
    pid1 = _TTS_WORKER_MANAGER.pid
    assert pid1 == 54321

    status2, body2, ct2, _ = tts_http.handle_speech_request(
        content_type="application/json",
        body=json.dumps({"input": "Sentence two.", "response_format": "wav"}).encode(),
        settings=settings,
        tracker=tracker,
    )
    assert status2 == 200
    pid2 = _TTS_WORKER_MANAGER.pid
    assert pid2 == pid1
    assert _TTS_WORKER_MANAGER.is_running() is True
    _TTS_WORKER_MANAGER.stop()
