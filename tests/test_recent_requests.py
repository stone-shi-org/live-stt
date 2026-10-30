"""GET /api/requests/recent (live_stt/recent_requests.py): the ring itself,
the route's exact JSON contract, what each of the four request kinds
records, and -- the non-negotiable part -- that no entry ever carries
content or caller identity.

Every test uses its OWN RecentRequests instance (passed via ``recent=`` or a
ServerState field), never the process-wide one, so tests stay independent.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import types
import urllib.error
import urllib.request
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import grpc
import pytest

from live_stt import diarize_http, gpu, recent_requests, transcribe_http, tts_http, vram
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.diarize_sessions import DiarizationSessionTracker
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.recent_requests import ALLOWED_FIELDS, RecentRequests
from live_stt.servicer import StreamingASRServicer
from live_stt.state import ServerState
from live_stt.transcribe_sessions import TranscribeSessionTracker
from live_stt.tts_sessions import TTSSessionTracker
from live_stt.worker_pool import STTWorkerPool

FAKE_WORKER = Path(__file__).resolve().parent / "fakes" / "fake_worker_main.py"
BOUNDARY = "----recent-boundary"
CONTENT_TYPE = f"multipart/form-data; boundary={BOUNDARY}"

CONTRACT_KEYS = [
    "id", "seq", "kind", "outcome", "model", "voice", "device", "started_at", "finished_at",
    "duration_s", "audio_seconds", "input_chars", "words", "transcript_chars", "processing_s",
]

# Distinctive strings planted in every request below; none may appear anywhere in the log.
SECRET_TEXT = "zebra-secret-phrase-xyzzy"
SECRET_CALL_ID = "caller-ident-4155550123"


def _wav(n_frames: int = 16000, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def _multipart(fields: dict[str, str], file_bytes: bytes | None = None) -> bytes:
    parts = [
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    if file_bytes is not None:
        parts.append(
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + file_bytes
            + b"\r\n"
        )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def _assert_contract_entry(entry: dict) -> None:
    assert list(entry) == CONTRACT_KEYS
    assert isinstance(entry["id"], str) and len(entry["id"]) >= 16
    int(entry["id"], 16)  # hex
    assert entry["kind"] in ("stream", "transcribe", "tts", "diarize")
    assert entry["device"] in ("cuda", "cpu", None)
    assert entry["finished_at"] >= entry["started_at"]
    assert entry["duration_s"] == pytest.approx(entry["finished_at"] - entry["started_at"], abs=0.002)


FORBIDDEN_KEYS = {"text", "transcript", "input", "call_id", "audio", "file", "words_list", "segments", "metadata"}


def _assert_no_content(ring: RecentRequests) -> None:
    """No key carries content, and no planted text/caller id appears anywhere."""
    snap = ring.snapshot()
    blob = json.dumps(snap)
    assert SECRET_TEXT not in blob
    assert SECRET_CALL_ID not in blob
    for entry in snap["requests"]:
        assert set(entry) == set(ALLOWED_FIELDS)
        assert not (set(entry) & FORBIDDEN_KEYS)
        for key, value in entry.items():
            # Only counts/durations/closed-enum strings: words/transcript_chars/
            # input_chars must be numbers, never the text itself.
            if key in ("words", "transcript_chars", "input_chars"):
                assert value is None or isinstance(value, int)
            if key in ("audio_seconds", "processing_s"):
                assert value is None or isinstance(value, (int, float))


# --------------------------------------------------------------------------- ring


class TestRing:
    def test_seq_monotonic_from_one_and_ids_unique(self):
        ring = RecentRequests()
        for _ in range(5):
            ring.record(kind="tts", outcome="ok", started_at=1.0, finished_at=2.0)
        snap = ring.snapshot()
        assert [e["seq"] for e in snap["requests"]] == [1, 2, 3, 4, 5]
        assert snap["next_seq"] == 5
        assert len({e["id"] for e in snap["requests"]}) == 5

    def test_empty_ring(self):
        snap = RecentRequests().snapshot()
        assert snap["next_seq"] == 0
        assert snap["requests"] == []
        assert snap["retention"] == 500

    def test_bounded(self):
        ring = RecentRequests(maxlen=3)
        for _ in range(10):
            ring.record(kind="stream", outcome="ok", started_at=1.0)
        snap = ring.snapshot()
        assert [e["seq"] for e in snap["requests"]] == [8, 9, 10]
        assert snap["next_seq"] == 10
        assert snap["retention"] == 3

    def test_since_filters_oldest_first(self):
        ring = RecentRequests()
        for _ in range(6):
            ring.record(kind="diarize", outcome="ok", started_at=1.0)
        assert [e["seq"] for e in ring.snapshot(since=4)["requests"]] == [5, 6]
        assert ring.snapshot(since=6)["requests"] == []
        assert ring.snapshot(since=100)["requests"] == []

    def test_boot_id_stable_within_process_and_distinct_per_instance(self):
        ring = RecentRequests()
        assert ring.snapshot()["boot_id"] == ring.snapshot()["boot_id"]
        assert len(ring.boot_id) >= 16
        assert recent_requests.get_recent_requests() is recent_requests.get_recent_requests()
        assert RecentRequests().boot_id != ring.boot_id

    def test_record_never_raises(self):
        ring = RecentRequests()
        ring.record(kind="bogus", outcome="ok", started_at=1.0)  # unknown kind: swallowed
        ring.record(kind="tts", outcome="ok", started_at="not-a-number")  # type: ignore[arg-type]
        assert ring.snapshot()["requests"] == []

    def test_voice_only_for_tts_and_unknown_device_dropped(self):
        ring = RecentRequests()
        ring.record(kind="stream", outcome="ok", started_at=1.0, voice="af_heart", device="tpu")
        e = ring.snapshot()["requests"][0]
        assert e["voice"] is None
        assert e["device"] is None

    def test_bounded_voice(self):
        assert recent_requests.bounded_voice("af_heart", ("af_heart",)) == "af_heart"
        assert recent_requests.bounded_voice(SECRET_TEXT, ("af_heart",)) == "other"
        assert recent_requests.bounded_voice(None, ("af_heart",)) is None

    def test_thread_safe_seq(self):
        ring = RecentRequests(maxlen=10_000)

        def worker():
            for _ in range(200):
                ring.record(kind="tts", outcome="ok", started_at=1.0)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        seqs = [e["seq"] for e in ring.snapshot()["requests"]]
        assert seqs == list(range(1, 1601))


# --------------------------------------------------------------------------- route


class TestRoute:
    def _get(self, port: int, query: str = "") -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/requests/recent{query}", timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_json_shape_and_since(self):
        settings = Settings(_env_file=None)
        ring = RecentRequests()
        state = ServerState(
            settings=settings,
            budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots),
            recent_requests=ring,
        )
        ring.record(
            kind="tts", outcome="ok", started_at=1793383600.12, finished_at=1793383601.05,
            model="kokoro", voice="af_heart", device="cuda", audio_seconds=9.4, input_chars=128,
            processing_s=0.61,
        )
        ring.record(kind="stream", outcome="no_capacity", started_at=1793383602.0, finished_at=1793383602.0)
        server = serve_admin_http("127.0.0.1", 0, state)
        try:
            port = server.server_address[1]
            status, doc = self._get(port)
            assert status == 200
            assert list(doc) == ["boot_id", "next_seq", "retention", "requests"]
            assert doc["boot_id"] == ring.boot_id
            assert doc["next_seq"] == 2
            assert doc["retention"] == 500
            first = doc["requests"][0]
            _assert_contract_entry(first)
            assert {k: v for k, v in first.items() if k != "id"} == {
                "seq": 1, "kind": "tts", "outcome": "ok", "model": "kokoro", "voice": "af_heart",
                "device": "cuda", "started_at": 1793383600.12, "finished_at": 1793383601.05,
                "duration_s": 0.93, "audio_seconds": 9.4, "input_chars": 128, "words": None,
                "transcript_chars": None, "processing_s": 0.61,
            }

            status, doc = self._get(port, "?since=1")
            assert status == 200
            assert [e["seq"] for e in doc["requests"]] == [2]
            assert doc["next_seq"] == 2

            status, doc = self._get(port, "?since=abc")
            assert status == 400
        finally:
            server.shutdown()

    def test_admin_page_lists_endpoint(self):
        from live_stt.admin_http import HTML_STATS_PAGE

        assert "/api/requests/recent" in HTML_STATS_PAGE


# --------------------------------------------------------------------------- stream (gRPC)


class _GrpcServer:
    def __init__(self, settings: Settings, ring: RecentRequests, budget: WorkerBudget | None = None) -> None:
        self.state = ServerState(
            settings=settings,
            budget=budget or WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots),
            worker_pool=STTWorkerPool(),
            recent_requests=ring,
        )
        self.servicer = StreamingASRServicer(self.state)

    async def __aenter__(self) -> "_GrpcServer":
        self._server = grpc.aio.server()
        asr_pb2_grpc.add_StreamingASRServicer_to_server(self.servicer, self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        await self._server.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._server.stop(grace=1.0)
        await self._server.wait_for_termination()
        self.state.worker_pool.close()


def _grpc_settings(**kw) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **kw)


async def _requests(n_chunks: int) -> AsyncIterator[asr_pb2.TranscriptionRequest]:
    yield asr_pb2.TranscriptionRequest(
        config=asr_pb2.StreamConfig(
            call_id=SECRET_CALL_ID, encoding=asr_pb2.AUDIO_ENCODING_LINEAR16, sample_rate_hz=16000
        )
    )
    for _ in range(n_chunks):
        yield asr_pb2.TranscriptionRequest(audio=b"\x00\x00" * 2560)  # 160ms


class TestStream:
    @pytest.mark.asyncio
    async def test_ok_call_records_usage(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FAKE_WORDS_PER_SEC", "20")
        ring = RecentRequests()
        async with _GrpcServer(_grpc_settings(), ring) as srv:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{srv.port}") as ch:
                events = [e async for e in asr_pb2_grpc.StreamingASRStub(ch).Transcribe(_requests(10))]
        words = sum(len(e.delta.words) for e in events if e.WhichOneof("event") == "delta") + sum(
            len(e.final.words) for e in events if e.WhichOneof("event") == "final"
        )
        chars = sum(len(e.delta.text) for e in events if e.WhichOneof("event") == "delta") + sum(
            len(e.final.text) for e in events if e.WhichOneof("event") == "final"
        )
        [entry] = ring.snapshot()["requests"]
        _assert_contract_entry(entry)
        assert entry["kind"] == "stream"
        assert entry["outcome"] == "ok"
        assert entry["model"] == "realtime_eou_120m-v1"
        assert entry["device"] == "cpu"
        assert entry["audio_seconds"] == pytest.approx(1.6)
        assert entry["words"] == words and words > 0
        assert entry["transcript_chars"] == chars
        assert entry["processing_s"] is not None and entry["processing_s"] >= 0
        assert entry["voice"] is None and entry["input_chars"] is None
        _assert_no_content(ring)

    @pytest.mark.asyncio
    async def test_no_capacity_rejection_recorded(self):
        ring = RecentRequests()
        budget = WorkerBudget(1, 0)
        assert budget.try_admit_call()  # fill the only slot
        async with _GrpcServer(_grpc_settings(max_concurrent_calls=1, reserve_slots=0), ring, budget) as srv:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{srv.port}") as ch:
                with pytest.raises(grpc.aio.AioRpcError):
                    [e async for e in asr_pb2_grpc.StreamingASRStub(ch).Transcribe(_requests(0))]
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "no_capacity"
        assert entry["model"] is None
        assert all(entry[k] is None for k in ("audio_seconds", "words", "transcript_chars", "processing_s"))

    @pytest.mark.asyncio
    async def test_draining_rejection_recorded(self):
        ring = RecentRequests()
        async with _GrpcServer(_grpc_settings(), ring) as srv:
            srv.state.draining = True
            async with grpc.aio.insecure_channel(f"127.0.0.1:{srv.port}") as ch:
                with pytest.raises(grpc.aio.AioRpcError):
                    [e async for e in asr_pb2_grpc.StreamingASRStub(ch).Transcribe(_requests(0))]
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "draining"


# --------------------------------------------------------------------------- transcribe (HTTP)


def _xsettings(**kw) -> Settings:
    return Settings(_env_file=None, worker_bin=str(FAKE_WORKER), models_dir="/fake", **kw)


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_ok_real_call_session_records_usage(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FAKE_WORDS_PER_SEC", "50")
        ring = RecentRequests()
        status, resp, _ = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE,
            body=_multipart({"model": "realtime_eou_120m-v1"}, _wav(n_frames=32000)),
            settings=_xsettings(),
            budget=WorkerBudget(3, 1),
            draining=False,
            tracker=TranscribeSessionTracker(),
            recent=ring,
        )
        assert status == 200
        text = json.loads(resp)["text"]
        [entry] = ring.snapshot()["requests"]
        _assert_contract_entry(entry)
        assert entry["kind"] == "transcribe"
        assert entry["outcome"] == "ok"
        assert entry["model"] == "realtime_eou_120m-v1"
        assert entry["device"] == "cpu"
        assert entry["audio_seconds"] == pytest.approx(2.0)
        assert entry["words"] > 0
        assert entry["transcript_chars"] == len(text)
        assert entry["processing_s"] is not None and entry["processing_s"] >= 0
        _assert_no_content(ring)

    @pytest.mark.asyncio
    async def test_failed_records_audio_but_no_counts(self, monkeypatch: pytest.MonkeyPatch):
        async def boom(*a, **kw):
            raise transcribe_http.TranscribeError("boom")

        monkeypatch.setattr(transcribe_http, "_run_transcription", boom)
        ring = RecentRequests()
        status, _, _ = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=_multipart({}, _wav(n_frames=8000)), settings=_xsettings(),
            budget=WorkerBudget(3, 1), draining=False, tracker=TranscribeSessionTracker(), recent=ring,
        )
        assert status == 500
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "failed"
        assert entry["audio_seconds"] == pytest.approx(0.5)
        assert entry["words"] is None and entry["transcript_chars"] is None

    @pytest.mark.asyncio
    async def test_rejected_capacity_recorded(self):
        ring = RecentRequests()
        budget = WorkerBudget(1, 0)
        assert budget.try_admit_call()
        status, _, _ = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=_multipart({}, _wav()), settings=_xsettings(),
            budget=budget, draining=False, tracker=TranscribeSessionTracker(), recent=ring,
        )
        assert status == 503
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "rejected_capacity"
        assert all(entry[k] is None for k in ("audio_seconds", "words", "transcript_chars", "processing_s"))

    @pytest.mark.asyncio
    async def test_rejected_vram_recorded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(vram, "_total_mb_cache", None)
        monkeypatch.setattr(gpu, "free_vram_mb", lambda: 10)
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
        ring = RecentRequests()
        status, _, _ = await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=_multipart({}, _wav()),
            settings=_xsettings(parakeet_tdt_backend="cuda", whisper_backend="cuda", parakeet_backend="cuda"),
            budget=WorkerBudget(3, 1), draining=False, tracker=TranscribeSessionTracker(), recent=ring,
        )
        assert status == 503
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "rejected_vram"
        assert entry["device"] == "cuda"
        assert entry["audio_seconds"] is None

    @pytest.mark.asyncio
    async def test_draining_recorded_and_validation_errors_not(self):
        ring = RecentRequests()
        await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=_multipart({}, _wav()), settings=_xsettings(),
            budget=WorkerBudget(3, 1), draining=True, tracker=TranscribeSessionTracker(), recent=ring,
        )
        await transcribe_http.handle_transcribe_request(
            content_type=CONTENT_TYPE, body=_multipart({}), settings=_xsettings(),
            budget=WorkerBudget(3, 1), draining=False, tracker=TranscribeSessionTracker(), recent=ring,
        )
        assert [e["outcome"] for e in ring.snapshot()["requests"]] == ["draining"]


# --------------------------------------------------------------------------- tts


def _install_fake_kokoro(monkeypatch: pytest.MonkeyPatch, n_samples: int = 24000) -> None:
    import numpy as np

    mod = types.ModuleType("kokoro")

    class _T:
        def __init__(self, a):
            self.a = a

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class _KPipeline:
        def __init__(self, lang_code, device=None, **kw):
            pass

        def __call__(self, text, voice="af_heart", speed=1.0, **kw):
            yield types.SimpleNamespace(audio=_T(np.zeros(n_samples, dtype=np.float32)))

    mod.KPipeline = _KPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro", mod)


class TestTTS:
    def test_non_streaming_ok(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_kokoro(monkeypatch)
        ring = RecentRequests()
        status, _, _, _ = tts_http.handle_speech_request(
            content_type="application/json",
            body=json.dumps({"input": SECRET_TEXT, "voice": "af_heart", "response_format": "wav"}).encode(),
            settings=Settings(_env_file=None, tts_in_subprocess=False),
            tracker=TTSSessionTracker(),
            recent=ring,
        )
        assert status == 200
        [entry] = ring.snapshot()["requests"]
        _assert_contract_entry(entry)
        assert entry["kind"] == "tts"
        assert entry["outcome"] == "ok"
        assert entry["model"] == "kokoro-82m-v1"
        assert entry["voice"] == "af_heart"
        assert entry["device"] == "cpu"
        assert entry["input_chars"] == len(SECRET_TEXT)
        assert entry["audio_seconds"] == pytest.approx(1.0)  # 24000 samples @ 24kHz, one chunk
        assert entry["processing_s"] is not None
        assert entry["words"] is None and entry["transcript_chars"] is None
        _assert_no_content(ring)

    def test_streaming_ok(self, monkeypatch: pytest.MonkeyPatch):
        _install_fake_kokoro(monkeypatch)
        ring = RecentRequests()
        status, gen, _, _ = tts_http.handle_speech_request(
            content_type="application/json",
            body=json.dumps({"input": "First one. Second one.", "stream": True, "response_format": "pcm"}).encode(),
            settings=Settings(_env_file=None, tts_in_subprocess=False),
            tracker=TTSSessionTracker(),
            recent=ring,
        )
        assert status == 200
        assert ring.snapshot()["requests"] == []  # recorded when the stream finishes, not before
        assert len(list(gen)) == 2
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "ok"
        assert entry["audio_seconds"] == pytest.approx(2.0)
        assert entry["input_chars"] == len("First one. Second one.")

    def test_failed_and_unknown_voice_bounded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "kokoro", None)  # engine missing -> 503
        ring = RecentRequests()
        status, _, _, _ = tts_http.handle_speech_request(
            content_type="application/json",
            body=json.dumps({"input": "hello", "voice": SECRET_CALL_ID}).encode(),
            settings=Settings(_env_file=None, tts_in_subprocess=False),
            tracker=TTSSessionTracker(),
            recent=ring,
        )
        assert status in (500, 503)
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "failed"
        assert entry["voice"] == "other"
        assert entry["audio_seconds"] is None
        _assert_no_content(ring)

    def test_rejected_vram_recorded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(vram, "_total_mb_cache", None)
        monkeypatch.setattr(gpu, "free_vram_mb", lambda: 10)
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
        ring = RecentRequests()
        status, _, _, _ = tts_http.handle_speech_request(
            content_type="application/json",
            body=json.dumps({"input": "hello"}).encode(),
            settings=Settings(_env_file=None, tts_device="cuda"),
            tracker=TTSSessionTracker(),
            recent=ring,
        )
        assert status == 503
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "rejected_vram"
        assert entry["input_chars"] is None and entry["audio_seconds"] is None


# --------------------------------------------------------------------------- diarize


class TestDiarize:
    def test_ok_records_audio_seconds(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            diarize_http, "diarize_file",
            lambda path, *, settings, words: {"task": "diarize", "num_speakers": 1, "segments": [], "speakers": []},
        )
        ring = RecentRequests()
        status, _ = diarize_http.handle_diarize_request(
            content_type=CONTENT_TYPE,
            body=_multipart({}, _wav(n_frames=48000)),
            settings=Settings(_env_file=None),
            tracker=DiarizationSessionTracker(),
            recent=ring,
        )
        assert status == 200
        [entry] = ring.snapshot()["requests"]
        _assert_contract_entry(entry)
        assert entry["kind"] == "diarize"
        assert entry["outcome"] == "ok"
        assert entry["model"] == "pyannote/speaker-diarization-community-1"
        assert entry["device"] == "cpu"
        assert entry["audio_seconds"] == pytest.approx(3.0)
        assert entry["processing_s"] is not None

    def test_failed_and_unknown_model_never_recorded_raw(self, monkeypatch: pytest.MonkeyPatch):
        from live_stt.diarization import DiarizationError

        def boom(path, *, settings, words):
            raise DiarizationError("unknown diarization model")

        monkeypatch.setattr(diarize_http, "diarize_file", boom)
        ring = RecentRequests()
        status, _ = diarize_http.handle_diarize_request(
            content_type=CONTENT_TYPE,
            body=_multipart({"model": SECRET_TEXT}, b"not a wav"),
            settings=Settings(_env_file=None),
            tracker=DiarizationSessionTracker(),
            recent=ring,
        )
        assert status == 400
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "failed"
        assert entry["model"] is None
        assert entry["audio_seconds"] is None  # unparseable input
        _assert_no_content(ring)

    def test_rejected_vram_recorded(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(vram, "_total_mb_cache", None)
        monkeypatch.setattr(gpu, "free_vram_mb", lambda: 10)
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
        ring = RecentRequests()
        status, _ = diarize_http.handle_diarize_request(
            content_type=CONTENT_TYPE,
            body=_multipart({}, _wav()),
            settings=Settings(_env_file=None, diarization_device="cuda"),
            tracker=DiarizationSessionTracker(),
            recent=ring,
        )
        assert status == 503
        [entry] = ring.snapshot()["requests"]
        assert entry["outcome"] == "rejected_vram"
        assert entry["device"] == "cuda"
        assert entry["audio_seconds"] is None and entry["processing_s"] is None
