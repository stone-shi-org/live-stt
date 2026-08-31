"""live_stt/transcribe_http.py.

Two layers, mirroring the split already used for diarize_http/session:
- Pure functions (_pcm16_mono_16k_from_wav, _accumulate) and
  handle_transcribe_request's validation/response-shaping, with
  _run_transcription monkeypatched -- fast, fully offline.
- A handful of tests that go through the REAL CallSession -> WorkerHandle ->
  subprocess path against tests/fakes/fake_worker_main.py (a real Python
  subprocess speaking the real IPC protocol, NOT the real C++ binary/model --
  see that file and tests/test_session.py for the same pattern), to prove
  the wiring itself is correct, not just the request validation around it.
"""

from __future__ import annotations

import io
import json
import os
import wave
from pathlib import Path

import pytest

from live_stt import transcribe_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.pb.livestt.v1 import asr_pb2

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"

BOUNDARY = "----test-boundary"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"


def _wav_bytes(*, rate: int = 16000, channels: int = 1, width: int = 2, n_frames: int = 320) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(b"\x00" * (n_frames * width * channels))
    return buf.getvalue()


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
            f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
            + file_bytes
            + b"\r\n"
        )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **overrides)


def _budget(max_concurrent_calls: int = 3, reserve_slots: int = 1) -> WorkerBudget:
    return WorkerBudget(max_concurrent_calls, reserve_slots)


class TestPcm16MonoFromWav:
    def test_accepts_16k_mono_16bit(self):
        wav = _wav_bytes(rate=16000, channels=1, width=2, n_frames=100)
        pcm = transcribe_http._pcm16_mono_16k_from_wav(wav)
        assert len(pcm) == 100 * 2

    def test_rejects_wrong_sample_rate(self):
        wav = _wav_bytes(rate=8000)
        with pytest.raises(transcribe_http.TranscribeBadRequestError, match="8000Hz"):
            transcribe_http._pcm16_mono_16k_from_wav(wav)

    def test_rejects_stereo(self):
        wav = _wav_bytes(channels=2)
        with pytest.raises(transcribe_http.TranscribeBadRequestError, match="2ch"):
            transcribe_http._pcm16_mono_16k_from_wav(wav)

    def test_rejects_wrong_bit_depth(self):
        wav = _wav_bytes(width=1)
        with pytest.raises(transcribe_http.TranscribeBadRequestError, match="8-bit"):
            transcribe_http._pcm16_mono_16k_from_wav(wav)

    def test_rejects_garbage_bytes(self):
        with pytest.raises(transcribe_http.TranscribeBadRequestError, match="could not parse"):
            transcribe_http._pcm16_mono_16k_from_wav(b"not a wav file at all")


class TestAccumulate:
    def test_delta_event_appends_and_returns_none(self):
        text_parts, words = [], []
        event = asr_pb2.TranscriptionEvent(
            delta=asr_pb2.TranscriptDelta(
                text="hi", words=[asr_pb2.Word(text="hi", start_sec=0.0, end_sec=0.3)]
            )
        )
        result = transcribe_http._accumulate(event, text_parts, words)
        assert result is None
        assert text_parts == ["hi"]
        assert len(words) == 1

    def test_final_event_appends_and_returns_totals(self):
        text_parts, words = [], []
        event = asr_pb2.TranscriptionEvent(
            final=asr_pb2.Final(text=" there", total_audio_sec=1.5, worker_generations=2)
        )
        result = transcribe_http._accumulate(event, text_parts, words)
        assert result == (1.5, 2)
        assert text_parts == [" there"]

    def test_ready_event_is_ignored(self):
        text_parts, words = [], []
        event = asr_pb2.TranscriptionEvent(ready=asr_pb2.Ready(model="m"))
        assert transcribe_http._accumulate(event, text_parts, words) is None
        assert text_parts == []


class TestHandleTranscribeRequestValidation:
    @pytest.mark.asyncio
    async def test_draining_is_503(self):
        status, body, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=b"", settings=_settings(), budget=_budget(), draining=True
        )
        assert status == 503
        assert "draining" in json.loads(body)["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_file_is_400(self):
        body = _multipart_body({"model": "realtime_eou_120m-v1"})
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 400
        assert "file" in json.loads(resp)["error"]["message"]

    @pytest.mark.asyncio
    async def test_bad_wav_is_400(self):
        body = _multipart_body({}, file_field="file", file_bytes=b"not a wav")
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 400

    @pytest.mark.asyncio
    async def test_unknown_model_is_400(self):
        body = _multipart_body({"model": "not-a-real-model"}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 400
        assert "not-a-real-model" in json.loads(resp)["error"]["message"]

    @pytest.mark.asyncio
    async def test_bad_response_format_is_400(self):
        body = _multipart_body(
            {"response_format": "srt"}, file_field="file", file_bytes=_wav_bytes()
        )
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 400
        assert "response_format" in json.loads(resp)["error"]["message"]

    @pytest.mark.asyncio
    async def test_at_capacity_is_503(self):
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        exhausted = _budget(max_concurrent_calls=0, reserve_slots=0)
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=exhausted, draining=False
        )
        assert status == 503
        assert "capacity" in json.loads(resp)["error"]["message"]

    @pytest.mark.asyncio
    async def test_insufficient_vram_on_cuda_backend_is_503(self, monkeypatch: pytest.MonkeyPatch):
        # Mirrors live_stt/servicer.py's gRPC-side gate, now closed here too
        # (see the comment in transcribe_http.py) -- this endpoint spawns
        # the exact same kind of worker process through the exact same
        # CallSession/WorkerHandle path, and is whisper's ONLY entry point.
        monkeypatch.setattr(transcribe_http.gpu, "free_vram_mb", lambda: 100)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE,
            body=body,
            settings=_settings(backend="cuda", vram_per_worker_mb=3000, vram_reserve_mb=2000),
            budget=_budget(),
            draining=False,
        )
        assert status == 503
        assert "VRAM" in json.loads(resp)["error"]["message"]

    @pytest.mark.asyncio
    async def test_vram_check_is_a_noop_on_cpu_backend(self, monkeypatch: pytest.MonkeyPatch):
        # free_vram_mb() would return None anyway without a real nvidia-smi
        # (see tests/test_gpu.py), but this pins the actual guard: the check
        # is skipped entirely off the cuda backend, not merely tolerant of
        # a None reading.
        monkeypatch.setattr(
            transcribe_http.gpu, "free_vram_mb", lambda: (_ for _ in ()).throw(AssertionError("should not be called"))
        )
        self_patch_module = transcribe_http

        async def fake_run(pcm, *, settings, spec, language, budget):
            return "hello", [], 1.0, 1

        monkeypatch.setattr(self_patch_module, "_run_transcription", fake_run)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(backend="cpu"), budget=_budget(), draining=False
        )
        assert status == 200


class TestHandleTranscribeRequestResponses:
    """_run_transcription monkeypatched -- these test response shaping only."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch, *, text="hello world", words=None, total_audio_sec=2.0, generations=1):
        words = words if words is not None else [asr_pb2.Word(text="hello", start_sec=0.0, end_sec=0.5)]

        async def fake_run(pcm, *, settings, spec, language, budget):
            return text, words, total_audio_sec, generations

        monkeypatch.setattr(transcribe_http, "_run_transcription", fake_run)

    @pytest.mark.asyncio
    async def test_default_json_response(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 200
        assert ct == "application/json"
        assert json.loads(resp) == {"text": "hello world"}

    @pytest.mark.asyncio
    async def test_verbose_json_response(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch, total_audio_sec=3.2, generations=2)
        body = _multipart_body(
            {"response_format": "verbose_json", "language": "en-US"}, file_field="file", file_bytes=_wav_bytes()
        )
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        doc = json.loads(resp)
        assert status == 200
        assert doc["task"] == "transcribe"
        assert doc["language"] == "en-US"
        assert doc["duration"] == 3.2
        assert doc["words"] == [{"word": "hello", "start": 0.0, "end": 0.5}]

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse_matching_my_meeting_notes_contract(self, monkeypatch: pytest.MonkeyPatch):
        # The real consumer (my-meeting-notes' _transcribe_window) only reads
        # {"type": "transcript.text.done", "text": ...} and stops at a
        # literal "[DONE]" data line -- this is the exact shape it parses.
        self._patch(monkeypatch, text="a full transcript")
        body = _multipart_body({"stream": "true"}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 200
        assert ct == "text/event-stream"
        text_body = resp.decode("utf-8")
        lines = [l for l in text_body.split("\n\n") if l]
        assert json.loads(lines[0].removeprefix("data: ")) == {
            "type": "transcript.text.done",
            "text": "a full transcript",
        }
        assert lines[1] == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_engine_unavailable_is_503(self, monkeypatch: pytest.MonkeyPatch):
        async def fake_run(*args, **kwargs):
            raise transcribe_http.TranscribeUnavailableError("engine failed to start: boom")

        monkeypatch.setattr(transcribe_http, "_run_transcription", fake_run)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 503

    @pytest.mark.asyncio
    async def test_engine_error_mid_transcription_is_500(self, monkeypatch: pytest.MonkeyPatch):
        async def fake_run(*args, **kwargs):
            raise transcribe_http.TranscribeError("engine error during transcription: boom")

        monkeypatch.setattr(transcribe_http, "_run_transcription", fake_run)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 500

    @pytest.mark.asyncio
    async def test_budget_is_released_on_success_and_on_failure(self, monkeypatch: pytest.MonkeyPatch):
        self._patch(monkeypatch)
        budget = _budget(max_concurrent_calls=1, reserve_slots=0)
        body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes())
        await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=budget, draining=False
        )
        assert budget.active_calls == 0

        async def fake_run_raises(*args, **kwargs):
            raise transcribe_http.TranscribeError("boom")

        monkeypatch.setattr(transcribe_http, "_run_transcription", fake_run_raises)
        await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=budget, draining=False
        )
        assert budget.active_calls == 0


class TestRealCallSessionIntegration:
    """Goes through the REAL CallSession -> WorkerHandle -> subprocess path
    against tests/fakes/fake_worker_main.py -- a real subprocess speaking the
    real IPC protocol, not the real C++ binary/model (see that file and
    tests/test_session.py's own use of FAKE_WORDS_PER_SEC). Proves the
    _run_transcription wiring itself, not just the validation around it.
    """

    @pytest.mark.asyncio
    async def test_real_call_session_produces_a_transcript(self):
        os.environ["FAKE_WORDS_PER_SEC"] = "50"
        try:
            body = _multipart_body({}, file_field="file", file_bytes=_wav_bytes(n_frames=2560))
            status, resp, ct = await transcribe_http.handle_transcribe_request(
                content_type=CONTENT_TYPE,
                body=body,
                settings=_settings(),
                budget=_budget(),
                draining=False,
            )
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)

        assert status == 200
        doc = json.loads(resp)
        assert doc["text"]  # the fake worker emitted synthetic words

    @pytest.mark.asyncio
    async def test_unknown_model_never_spawns_a_worker(self):
        # A bad model must be rejected before _run_transcription (and thus
        # before any subprocess spawn) is ever reached -- this just confirms
        # the 400 path doesn't touch the real worker machinery at all.
        body = _multipart_body({"model": "nope"}, file_field="file", file_bytes=_wav_bytes())
        status, resp, ct = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=body, settings=_settings(), budget=_budget(), draining=False
        )
        assert status == 400

    @pytest.mark.asyncio
    async def test_batch_only_whisper_model_succeeds_over_http_unlike_grpc(self):
        # The whole point of the streaming_capable gate (see
        # tests/test_servicer.py's test_transcribe_rejects_batch_only_whisper_model)
        # is that it's specific to the gRPC Transcribe RPC -- this endpoint
        # is already engine-agnostic and must NOT reject a batch-only model.
        # worker_bin_whisper here points at the same fake as worker_bin --
        # the fake doesn't distinguish engines, only the dispatch logic in
        # live_stt/session.py does (see tests/test_session.py's
        # test_whisper_engine_spawns_the_whisper_binary_not_the_parakeet_one
        # for the test that actually proves the dispatch, not just this
        # end-to-end happy path).
        os.environ["FAKE_WORDS_PER_SEC"] = "50"
        try:
            body = _multipart_body(
                {"model": "whisper-base.en-q8_0"},
                file_field="file",
                file_bytes=_wav_bytes(n_frames=2560),
            )
            status, resp, ct = await transcribe_http.handle_transcribe_request(
                content_type=CONTENT_TYPE,
                body=body,
                settings=_settings(worker_bin_whisper=str(FAKE_WORKER)),
                budget=_budget(),
                draining=False,
            )
        finally:
            os.environ.pop("FAKE_WORDS_PER_SEC", None)

        assert status == 200
        doc = json.loads(resp)
        assert doc["text"]
