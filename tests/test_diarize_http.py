"""live_stt/diarize_http.py: the multipart parser and the request handler
are both pure functions (no socket, no real pyannote) -- see that module's
docstring for why this can be tested directly rather than through a real
HTTP server. diarize_file itself is monkeypatched; its own behavior is
covered by tests/test_diarization.py.
"""

from __future__ import annotations

import json

import pytest

from live_stt import diarize_http
from live_stt.config import Settings
from live_stt.diarization import DiarizationError
from live_stt.diarize_sessions import DiarizationSessionTracker

BOUNDARY = "----test-boundary"


def _multipart_body(fields: dict[str, bytes | str], file_field: str | None = None, file_bytes: bytes = b"") -> bytes:
    parts = []
    for name, value in fields.items():
        value_bytes = value.encode() if isinstance(value, str) else value
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value_bytes
            + b"\r\n"
        )
    if file_field:
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{file_field}"; '
            f'filename="call.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
            + file_bytes
            + b"\r\n"
        )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _call(*, content_type: str, body: bytes, settings: Settings, tracker: DiarizationSessionTracker | None = None):
    return diarize_http.handle_diarize_request(
        content_type=content_type, body=body, settings=settings, tracker=tracker or DiarizationSessionTracker()
    )


class TestParseMultipartForm:
    def test_round_trips_text_and_binary_fields(self):
        wav_bytes = bytes(range(256)) * 3  # exercise every byte value
        body = _multipart_body({"model": "some-model"}, file_field="file", file_bytes=wav_bytes)
        fields = diarize_http.parse_multipart_form(CONTENT_TYPE, body)
        assert fields["model"] == b"some-model"
        assert fields["file"] == wav_bytes

    def test_rejects_non_multipart_content_type(self):
        with pytest.raises(diarize_http.MultipartError, match="expected multipart"):
            diarize_http.parse_multipart_form("application/json", b"{}")

    def test_rejects_body_with_no_real_boundary_match(self):
        with pytest.raises(diarize_http.MultipartError):
            diarize_http.parse_multipart_form(CONTENT_TYPE, b"not actually multipart at all")


class TestHandleDiarizeRequest:
    def test_missing_file_field_is_400(self):
        body = _multipart_body({"model": "x"})
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 400
        assert "file" in doc["error"]["message"]

    def test_unsupported_response_format_is_400(self):
        body = _multipart_body({"response_format": "text"}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 400
        assert "response_format" in doc["error"]["message"]

    def test_malformed_words_json_is_400(self):
        body = _multipart_body({"words": "not json"}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 400
        assert "words" in doc["error"]["message"]

    def test_non_integer_num_speakers_is_400(self):
        body = _multipart_body({"num_speakers": "two"}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 400
        assert "num_speakers" in doc["error"]["message"]

    def test_missing_dependency_is_503_not_400(self, monkeypatch: pytest.MonkeyPatch):
        def fake_diarize_file(*args, **kwargs):
            raise DiarizationError("pyannote.audio is not installed. ...")

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 503
        assert "not installed" in doc["error"]["message"]

    def test_success_returns_200_and_the_house_json(self, monkeypatch: pytest.MonkeyPatch):
        expected = {"task": "diarize", "num_speakers": 2, "segments": [], "speakers": []}
        captured = {}

        def fake_diarize_file(path, *, settings, words):
            captured["settings"] = settings
            captured["words"] = words
            return expected

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        body = _multipart_body({"model": "override-model", "num_speakers": "3"}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())

        assert status == 200
        assert doc == expected
        assert captured["settings"].diarization_model == "override-model"
        assert captured["settings"].diarization_num_speakers == 3
        assert captured["words"] == []

    def test_words_field_is_parsed_and_forwarded(self, monkeypatch: pytest.MonkeyPatch):
        captured = {}

        def fake_diarize_file(path, *, settings, words):
            captured["words"] = words
            return {"task": "diarize", "num_speakers": 1, "segments": [], "speakers": []}

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        words_json = json.dumps([{"text": "hi", "start_sec": 0.1, "end_sec": 0.3}])
        body = _multipart_body({"words": words_json}, file_field="file", file_bytes=b"wav")
        status, _ = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())

        assert status == 200
        assert len(captured["words"]) == 1
        assert captured["words"][0].text == "hi"
        assert captured["words"][0].start_sec == pytest.approx(0.1)

    def test_unexpected_exception_is_500_not_a_crash(self, monkeypatch: pytest.MonkeyPatch):
        def fake_diarize_file(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings())
        assert status == 500
        assert "boom" in doc["error"]["message"]


class TestVramAdmission:
    """diarization_device="cuda" gates admission on live_stt.gpu.free_vram_mb()
    -- see live_stt/config.py's diarization_vram_mb docstring for the real
    12.3GB measurement behind the default. CPU (the test default) never
    checks VRAM at all, same as the "cpu default never touches torch.cuda"
    invariant tests/test_diarization.py already pins for load_pipeline.
    """

    def test_cpu_device_never_calls_free_vram_mb(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(diarize_http.gpu, "free_vram_mb", lambda: (_ for _ in ()).throw(AssertionError("called")))
        monkeypatch.setattr(diarize_http, "diarize_file", lambda *a, **k: {"task": "diarize", "num_speakers": 0, "segments": [{"id": 0, "speaker": "x", "label": "x", "start": 0, "end": 1, "text": ""}], "speakers": []})
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings(diarization_device="cpu"))
        assert status == 200

    def test_cuda_with_insufficient_vram_is_503_before_touching_the_pipeline(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(diarize_http.gpu, "free_vram_mb", lambda: 5000)

        def fake_diarize_file(*a, **k):
            raise AssertionError("pipeline must not run when VRAM is insufficient")

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        tracker = DiarizationSessionTracker()
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(
            content_type=CONTENT_TYPE,
            body=body,
            settings=_settings(diarization_device="cuda", diarization_vram_mb=13000),
            tracker=tracker,
        )
        assert status == 503
        assert "5000" in doc["error"]["message"] and "13000" in doc["error"]["message"]
        assert tracker.rejected_vram_total == 1
        assert tracker.active == 0  # never started -- rejected before start()

    def test_cuda_with_sufficient_vram_runs_normally(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(diarize_http.gpu, "free_vram_mb", lambda: 20000)
        expected = {"task": "diarize", "num_speakers": 1, "segments": [], "speakers": []}
        monkeypatch.setattr(diarize_http, "diarize_file", lambda *a, **k: expected)
        tracker = DiarizationSessionTracker()
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(
            content_type=CONTENT_TYPE,
            body=body,
            settings=_settings(diarization_device="cuda"),
            tracker=tracker,
        )
        assert status == 200
        assert doc == expected
        assert tracker.completed_total == 1
        assert tracker.active == 0

    def test_free_vram_unavailable_fails_open_not_closed(self, monkeypatch: pytest.MonkeyPatch):
        # nvidia-smi missing/unreachable returns None -- "cannot check", not
        # "zero VRAM free" (see live_stt/gpu.py). Must not block admission.
        monkeypatch.setattr(diarize_http.gpu, "free_vram_mb", lambda: None)
        expected = {"task": "diarize", "num_speakers": 1, "segments": [], "speakers": []}
        monkeypatch.setattr(diarize_http, "diarize_file", lambda *a, **k: expected)
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        status, doc = _call(content_type=CONTENT_TYPE, body=body, settings=_settings(diarization_device="cuda"))
        assert status == 200


class TestSessionTracking:
    def test_active_returns_to_zero_after_success(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(diarize_http, "diarize_file", lambda *a, **k: {"task": "diarize", "num_speakers": 0, "segments": [{"id": 0, "speaker": "x", "label": "x", "start": 0, "end": 1, "text": ""}], "speakers": []})
        tracker = DiarizationSessionTracker()
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        _call(content_type=CONTENT_TYPE, body=body, settings=_settings(), tracker=tracker)
        assert tracker.active == 0
        assert tracker.completed_total == 1
        assert tracker.failed_total == 0

    def test_active_returns_to_zero_after_failure(self, monkeypatch: pytest.MonkeyPatch):
        def fake_diarize_file(*a, **k):
            raise DiarizationError("no segments")

        monkeypatch.setattr(diarize_http, "diarize_file", fake_diarize_file)
        tracker = DiarizationSessionTracker()
        body = _multipart_body({}, file_field="file", file_bytes=b"wav")
        _call(content_type=CONTENT_TYPE, body=body, settings=_settings(), tracker=tracker)
        assert tracker.active == 0
        assert tracker.failed_total == 1
        assert tracker.completed_total == 0
