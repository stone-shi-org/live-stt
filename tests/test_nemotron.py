"""NemotronService (live_stt/nemotron.py, LS-19) against a fake ``nemo-speech
serve`` (tests/fakes/fake_nemo_speech.py): a real subprocess speaking real
HTTP on loopback -- no binary, no model, no GPU. nvidia-smi is replaced by a
fake card whose free VRAM drops by ~22 GB while a fake sidecar is alive, so
admission, eviction and preemption run through the real live_stt/vram.py.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import stat
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from live_stt import diarize_http, gpu, metrics, nemotron, vram
from live_stt.backends import installer
from live_stt.config import Settings
from live_stt.diarize_sessions import DiarizationSessionTracker
from live_stt.recent_requests import RecentRequests

FAKE = Path(__file__).resolve().parent / "fakes" / "fake_nemo_speech.py"
CARD_FREE_IDLE = 24125  # nvidia-smi memory.free on homesrv3's idle RTX 3090
SIDECAR_MB = 22000


def make_wav(seconds: int, rate: int = 100) -> bytes:
    """Each second's samples hold (absolute second + 1) -- the fake's convention."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(int(k + 1).to_bytes(2, "little") * rate for k in range(seconds)))
    return buf.getvalue()


class Card:
    def __init__(self, svc: nemotron.NemotronService) -> None:
        self.svc = svc
        self.base_free = CARD_FREE_IDLE
        self.calls = 0
        self.seen: set = set()

    def free(self) -> int:
        # Like the real driver: a sidecar's VRAM is in use until its process
        # has actually exited, whether or not the service still tracks it.
        self.calls += 1
        p = self.svc._proc
        if p is not None:
            self.seen.add(p.popen)
        used = SIDECAR_MB if any(pp.poll() is None for pp in self.seen) else 0
        return max(0, self.base_free - used)


class Env:
    def __init__(self, tmp_path: Path, svc: nemotron.NemotronService, card: Card) -> None:
        self.tmp = tmp_path
        self.svc = svc
        self.card = card
        self.log = tmp_path / "fake.log"

    def settings(self, **kw) -> Settings:
        base = dict(
            nemotron_bin=str(self.tmp / "nemo-speech"),
            nemotron_models_dir=str(self.tmp / "models"),
            db_path=str(self.tmp / "lstt.db"),
            backends_dir=str(self.tmp / "backends"),
            nemotron_ready_timeout_sec=15,
            nemotron_job_timeout_sec=60,
        )
        base.update(kw)
        return Settings(_env_file=None, **base)

    def events(self, kind: str | None = None) -> list[dict]:
        if not self.log.exists():
            return []
        out = [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]
        return [e for e in out if kind is None or e["event"] == kind]

    def wait_for(self, pred, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(0.05)
        return False


@pytest.fixture()
def env(tmp_path, monkeypatch):
    wrapper = tmp_path / "nemo-speech"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    models = tmp_path / "models"
    models.mkdir()
    (models / nemotron.TIERS[2].decoder_filename).write_bytes(b"gguf")
    (models / nemotron.ENCODER_FILENAME).write_bytes(b"gguf")

    svc = nemotron.NemotronService()
    card = Card(svc)
    monkeypatch.setattr(nemotron, "_SERVICE", svc)
    monkeypatch.setattr(vram, "_total_mb_cache", None)
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 24576)
    monkeypatch.setattr(gpu, "compute_cap", lambda: 8.6)
    monkeypatch.setattr(gpu, "free_vram_mb", card.free)
    e = Env(tmp_path, svc, card)
    monkeypatch.setenv("FAKE_NEMO_LOG", str(e.log))
    for k in list(os.environ):
        if k.startswith("FAKE_NEMO_") and k != "FAKE_NEMO_LOG":
            monkeypatch.delenv(k)
    yield e
    svc.stop()


def _run(env: Env, seconds: int, **settings_kw):
    return env.svc.transcribe(env.settings(**settings_kw), nemotron.parse_wav(make_wav(seconds)))


def _true_speaker(word: str, n_spk: int = 3, turn: int = 60) -> int:
    return (int(word[1:]) // turn) % n_spk


def _assert_consistent_speakers(doc: dict) -> None:
    """Every word's global label must map one-to-one onto its true speaker."""
    true_to_label: dict[int, str] = {}
    label_to_true: dict[str, int] = {}
    for w in doc["words"]:
        t = _true_speaker(w["text"])
        assert true_to_label.setdefault(t, w["speaker"]) == w["speaker"], w
        assert label_to_true.setdefault(w["speaker"], t) == t, w


# --------------------------------------------------------------------------- #


def test_request_is_chat_completion_with_explicit_instruction(env):
    doc = _run(env, 30)
    chats = env.events("chat")
    assert len(chats) == 1
    assert chats[0]["path"] == "/v1/chat/completions"
    assert chats[0]["texts"] == [nemotron.SPEAKER_INSTRUCTION]
    assert chats[0]["timestamp_granularities"] == ["word"]
    assert not env.events("other_post")  # never /v1/audio/transcriptions or the default prompt
    assert doc["model"] == nemotron.MODEL_KEY
    assert doc["num_speakers"] == 1
    assert all(seg["text"] for seg in doc["segments"])
    start = env.events("start")[0]["args"]
    assert start["host"] == "127.0.0.1"
    assert start["asr-model"].endswith("AD-IQ4_NL.gguf")
    assert start["salm-encoder"].endswith(nemotron.ENCODER_FILENAME)


def test_25_min_uses_short_and_40_min_uses_long(env):
    _run(env, 25 * 60, nemotron_idle_timeout_sec=300)
    assert "max_sequences: 4" in env.events("start")[-1]["config"]
    assert env.svc.snapshot()["profile"] == "short"
    first_pid = env.svc.snapshot()["pid"]
    # Another short job reuses the warm process.
    _run(env, 60, nemotron_idle_timeout_sec=300)
    assert env.svc.snapshot()["pid"] == first_pid
    _run(env, 40 * 60, nemotron_idle_timeout_sec=300)
    starts = env.events("start")
    assert len(starts) == 2
    assert "max_sequences: 1" in starts[-1]["config"] and "max_context: 73728" in starts[-1]["config"]
    # A long-profile request recycles its process afterwards.
    assert env.wait_for(lambda: env.svc.state == "unloaded", 5)


def test_65_min_completes_through_windowing_with_seams(env):
    doc = _run(env, 65 * 60)
    chats = env.events("chat")
    assert [round(c["audio_seconds"]) for c in chats] == [2400, 1800]
    assert max(c["audio_seconds"] for c in chats) <= 45 * 60  # never one pass over 45 min
    # Each window ran on a fresh long-profile process.
    starts = env.events("start")
    assert len(starts) == 2 and all("max_sequences: 1" in s["config"] for s in starts)
    assert len({c["pid"] for c in chats}) == 2
    assert doc["seams"] == [{"at_sec": 2250.0, "matched_words": 300, "new_speakers": 0}]
    assert "warnings" not in doc
    assert doc["num_speakers"] == 3
    _assert_consistent_speakers(doc)
    # Midpoint join: every second appears exactly once.
    texts = [w["text"] for w in doc["words"]]
    assert texts == [f"w{k}" for k in range(65 * 60)]
    # Segments are the stitched turns: contiguous, one per 60 s speaker turn
    # (the seam cut at 2250 s splits one), each carrying its own words.
    segs = doc["segments"]
    assert len(segs) == 66
    assert segs[0]["start"] == 0.0 and segs[-1]["end"] == 3900.0
    assert all(a["end"] == b["start"] for a, b in zip(segs, segs[1:]))
    assert all(seg["text"] for seg in segs)
    for seg in segs:
        for tok in seg["text"].split():
            assert seg["start"] <= int(tok[1:]) < seg["end"]


def test_without_turns_segments_fall_back_to_word_runs(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_NO_TURNS", "1")
    doc = _run(env, 130)
    assert [round(s["start"], 1) for s in doc["segments"]] == [0.1, 60.1, 120.1]
    assert all(s["text"] for s in doc["segments"])


def test_collapsed_word_speakers_are_relabelled_and_warned(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_COLLAPSE_WORD_SPEAKERS", "1")
    before = metrics.nemotron_word_speaker_repairs_total._value.get()
    doc = _run(env, 130)
    # Words follow the turns again: three speakers, each turn carrying its own words.
    assert doc["num_speakers"] == 3
    assert len({w["speaker"] for w in doc["words"]}) == 3
    assert all(s["text"] for s in doc["segments"])
    for seg in doc["segments"]:
        for tok in seg["text"].split():
            assert seg["start"] <= int(tok[1:]) < seg["end"]
    assert len(doc["warnings"]) == 1 and "relabelled" in doc["warnings"][0]
    assert metrics.nemotron_word_speaker_repairs_total._value.get() == before + 1


def test_low_vote_seam_is_reported_in_warnings(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_WORD_EVERY", "10")  # 30 words in the 5-min overlap
    before = metrics.nemotron_low_vote_seams_total._value.get()
    doc = _run(env, 65 * 60)
    assert doc["seams"][0]["matched_words"] == 30
    assert len(doc["warnings"]) == 1 and "30 matched words" in doc["warnings"][0]
    assert metrics.nemotron_low_vote_seams_total._value.get() == before + 1


def test_cold_load_ready_and_idle_unload(env):
    t0 = time.monotonic()
    _run(env, 20, nemotron_idle_timeout_sec=0.5)
    assert time.monotonic() - t0 < 20
    assert env.svc.loads_total == 1
    assert env.wait_for(lambda: env.svc.state == "unloaded", 5)
    assert env.wait_for(lambda: env.events("term"), 5)
    assert env.wait_for(lambda: vram.reserved_mb() == 0, 5)
    assert env.card.free() == CARD_FREE_IDLE  # card VRAM back to baseline


def test_another_gpu_request_evicts_an_idle_sidecar(env):
    _run(env, 20, nemotron_idle_timeout_sec=300)
    assert env.svc.state == "ready"
    s = env.settings()
    adm = vram.admit(s, model_key="parakeet-tdt-0.6b-v3", model_mb=2300, fallback_mb=3000)
    try:
        assert adm.ok, adm.message()
        assert adm.reclaimed >= 1
        assert env.svc.state == "unloaded"
    finally:
        adm.release()


def test_killed_sidecar_fails_with_503_then_recovers(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_CRASH_ON_CHAT", "1")
    with pytest.raises(nemotron.NemotronRetryable) as exc_info:
        _run(env, 20)
    assert exc_info.value.status == 503 and exc_info.value.retry_after_sec
    assert env.svc.crashes_total == 1
    assert env.svc.state == "unloaded"
    monkeypatch.delenv("FAKE_NEMO_CRASH_ON_CHAT")
    doc = _run(env, 20)
    assert doc["segments"]
    assert env.svc.loads_total == 2


def test_load_failure_is_retryable_503(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_EXIT_BEFORE_READY", "1")
    with pytest.raises(nemotron.NemotronRetryable, match="failed to load"):
        _run(env, 20)
    assert vram.reserved_mb() == 0


def test_alignment_failure_is_not_silently_returned(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_ALIGN_FAIL", "1")
    with pytest.raises(nemotron.NemotronBadResponse, match="word timestamps"):
        _run(env, 20)


def test_admission_refuses_instead_of_loading_when_card_is_not_free(env):
    env.card.base_free = 12000  # another tenant holds half the card
    with pytest.raises(nemotron.NemotronTimeout) as exc_info:
        _run(env, 20, nemotron_job_timeout_sec=1.5)
    assert exc_info.value.retry_after_sec
    assert env.events("start") == []  # never attempted
    assert "insufficient VRAM" in (env.svc.last_error or "")


def test_job_waits_while_a_live_call_holds_vram_and_times_out_with_retry_after(env):
    s = env.settings(nemotron_job_timeout_sec=1.5)
    call = vram.admit(s, model_key="realtime_eou_120m-v1", model_mb=260, fallback_mb=3000)
    assert call.ok
    try:
        body, ctype = _multipart(make_wav(20))
        status, doc = diarize_http.handle_diarize_request(
            content_type=ctype, body=body, settings=s,
            tracker=DiarizationSessionTracker(), recent=RecentRequests(),
        )
    finally:
        call.release()
    assert status == 503
    assert doc["error"]["retry_after_sec"] == nemotron.RETRY_AFTER_TIMEOUT_SEC
    assert env.events("start") == []
    assert env.svc.timeouts_total == 1


def test_live_call_preempts_running_job_which_requeues_and_completes(env, monkeypatch):
    baseline = _run(env, 30, nemotron_idle_timeout_sec=0)
    env.wait_for(lambda: env.svc.state == "unloaded", 5)
    monkeypatch.setenv("FAKE_NEMO_CHAT_DELAY", "4.0")  # SIGTERM alone would take 4 s
    s = env.settings(nemotron_idle_timeout_sec=0)
    result: dict = {}

    def job():
        try:
            result["doc"] = env.svc.transcribe(s, nemotron.parse_wav(make_wav(30)))
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=job)
    t.start()
    assert env.wait_for(lambda: env.svc.state == "busy" and env.events("chat")[1:], 10)

    before = metrics.nemotron_preemptions_total._value.get()

    def admit_call():
        return vram.admit(s, model_key="realtime_eou_120m-v1", model_mb=260, fallback_mb=3000)

    first = admit_call()
    assert not first.ok  # the sidecar holds the card
    t0 = time.monotonic()
    call = nemotron.preempt_and_admit(s, admit_call)
    elapsed = time.monotonic() - t0
    assert call is not None and call.ok, call and call.message()
    assert elapsed < 3.0
    assert metrics.nemotron_preemptions_total._value.get() == before + 1
    try:
        # While the live call holds VRAM the job stays queued, not restarted.
        time.sleep(1.0)
        assert t.is_alive()
        snap = env.svc.snapshot()
        assert snap["requeues_total"] == 1 and snap["queue_depth"] == 1 and snap["state"] == "unloaded"
    finally:
        call.release()
    t.join(20)
    assert not t.is_alive()
    assert "exc" not in result, result.get("exc")
    assert result["doc"]["segments"] == baseline["segments"]
    assert env.svc.preemptions_total == 1


def test_preempt_with_nothing_running_is_a_noop(env):
    assert nemotron.preempt_and_admit(env.settings(), lambda: pytest.fail("must not re-admit")) is None


def test_concurrent_short_jobs_share_one_process(env, monkeypatch):
    monkeypatch.setenv("FAKE_NEMO_CHAT_DELAY", "0.5")
    s = env.settings(nemotron_idle_timeout_sec=300)
    errors = []

    def job():
        try:
            env.svc.transcribe(s, nemotron.parse_wav(make_wav(20)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=job) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(env.events("start")) == 1
    assert len({c["pid"] for c in env.events("chat")}) == 1


def test_snapshot_shape(env):
    snap = env.svc.snapshot()
    for key in ("state", "profile", "pid", "queue_depth", "running_jobs", "preemptions_total", "requeues_total"):
        assert key in snap
    assert snap["state"] == "unloaded"


# --------------------------------------------------------------------------- #
# /v1/audio/diarization routing

BOUNDARY = "----nemotron-test"


def _multipart(wav: bytes, model: str = nemotron.MODEL_KEY) -> tuple[bytes, str]:
    body = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode()
        + f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="include_text"\r\n\r\ntrue\r\n'.encode()
        + f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + wav
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={BOUNDARY}"


def test_diarization_endpoint_routes_to_nemotron_with_text_and_records_usage(env):
    tracker, recent = DiarizationSessionTracker(), RecentRequests()
    body, ctype = _multipart(make_wav(130))
    status, doc = diarize_http.handle_diarize_request(
        content_type=ctype, body=body, settings=env.settings(), tracker=tracker, recent=recent
    )
    assert status == 200, doc
    assert doc["model"] == nemotron.MODEL_KEY
    assert doc["num_speakers"] == 3
    assert all(seg["text"] for seg in doc["segments"])
    assert tracker.completed_total == 1 and tracker.active == 0
    entry = recent.snapshot(0)["requests"][-1]
    assert entry["kind"] == "diarize" and entry["model"] == nemotron.MODEL_KEY and entry["outcome"] == "ok"
    assert entry["audio_seconds"] == 130
    assert "text" not in json.dumps(entry)  # no content


def test_diarization_endpoint_rejects_non_wav_for_nemotron(env):
    body, ctype = _multipart(b"not a wav")
    status, doc = diarize_http.handle_diarize_request(
        content_type=ctype, body=body, settings=env.settings(), tracker=DiarizationSessionTracker(),
        recent=RecentRequests(),
    )
    assert status == 400 and "PCM WAV" in doc["error"]["message"]


def test_missing_binary_or_models_is_503_without_retry_after(env):
    body, ctype = _multipart(make_wav(10))
    status, doc = diarize_http.handle_diarize_request(
        content_type=ctype, body=body, settings=env.settings(nemotron_bin=str(env.tmp / "nope")),
        tracker=DiarizationSessionTracker(), recent=RecentRequests(),
    )
    assert status == 503 and "retry_after_sec" not in doc["error"]
    assert "binary" in doc["error"]["message"]


# --------------------------------------------------------------------------- #
# LS-20: nemo_speech:cpu backend, flavor switching, CPU admission & preemption


def _install_db_backends(env: Env, active_flavor: str | None = "cpu") -> tuple[Path, Path]:
    cpu_dir = env.tmp / "backends" / "nemo_speech" / "cpu" / "v0.3.0" / "bin"
    cuda_dir = env.tmp / "backends" / "nemo_speech" / "cuda" / "v0.3.0" / "bin"
    cpu_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)
    for d in (cpu_dir, cuda_dir):
        b = d / "nemo-speech"
        b.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        b.chmod(b.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with sqlite3.connect(env.tmp / "lstt.db") as conn:
        installer.ensure_schema(conn)
        conn.execute(
            "INSERT OR REPLACE INTO installed_backends VALUES (?,?,?,?,?,?,?)",
            (
                "nemo_speech",
                "cpu",
                "v0.3.0",
                "sha-cpu",
                str(cpu_dir.parent),
                "now",
                1 if active_flavor == "cpu" else 0,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO installed_backends VALUES (?,?,?,?,?,?,?)",
            (
                "nemo_speech",
                "cuda",
                "v0.3.0",
                "sha-cuda",
                str(cuda_dir.parent),
                "now",
                1 if active_flavor == "cuda" else 0,
            ),
        )
        conn.commit()
    return cpu_dir / "nemo-speech", cuda_dir / "nemo-speech"


def test_cpu_flavor_builds_cpu_command_and_works_on_pascal_host(env, monkeypatch):
    _install_db_backends(env, active_flavor="cpu")
    # Simulate homesrv: Quadro P1000 (cc 6.1, 4 GB VRAM) + 48 GB free RAM + AVX2.
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "compute_cap", lambda: 6.1)
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
    monkeypatch.setattr(nemotron, "effective_cpu_threads", lambda s: 16)

    s = env.settings(nemotron_bin=None, nemotron_idle_timeout_sec=300)
    tracker, recent = DiarizationSessionTracker(), RecentRequests()
    body, ctype = _multipart(make_wav(30))
    status, doc = diarize_http.handle_diarize_request(
        content_type=ctype, body=body, settings=s, tracker=tracker, recent=recent
    )
    assert status == 200, doc
    starts = env.events("start")
    assert len(starts) == 1
    args = starts[0]["args"]
    cfg = starts[0]["config"]
    assert args["device"] == "cpu"
    assert args["read-timeout"] == "3600"
    assert args["write-timeout"] == "3600"
    assert "max_sequences: 1" in cfg
    assert "max_context: 32768" in cfg
    assert "threads: 16" in cfg
    snap = env.svc.snapshot(s)
    assert snap["device"] == "cpu"
    assert snap["profile"] == "cpu"
    assert snap["threads"] == 16
    assert vram.reserved_mb() == 0  # CPU flavor never reserves GPU VRAM
    entry = recent.snapshot(0)["requests"][-1]
    assert entry["device"] == "cpu"


def test_cuda_flavor_in_db_builds_cuda_command(env):
    _install_db_backends(env, active_flavor="cuda")
    s = env.settings(nemotron_bin=None, nemotron_idle_timeout_sec=300)
    _run(env, 30, nemotron_bin=None, nemotron_idle_timeout_sec=300)
    starts = env.events("start")
    assert len(starts) == 1
    args = starts[0]["args"]
    cfg = starts[0]["config"]
    assert args["device"] == "cuda"
    assert "max_sequences: 4" in cfg
    assert "threads:" not in cfg
    assert env.svc.snapshot(s)["device"] == "cuda"


def test_switching_active_flavor_stops_idle_sidecar_and_switches_next_load(env, monkeypatch):
    _install_db_backends(env, active_flavor="cuda")
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
    s = env.settings(nemotron_bin=None, nemotron_idle_timeout_sec=300)

    _run(env, 20, nemotron_bin=None, nemotron_idle_timeout_sec=300)
    assert env.svc.state == "ready"
    assert env.svc.snapshot(s)["device"] == "cuda"

    # Switch active flavor to cpu -> idle sidecar stops immediately.
    with sqlite3.connect(env.tmp / "lstt.db") as conn:
        installer.activate(conn, "nemo_speech", "cpu", "v0.3.0")
    assert env.svc.state == "unloaded"

    # Next job launches on CPU.
    _run(env, 20, nemotron_bin=None, nemotron_idle_timeout_sec=300)
    starts = env.events("start")
    assert [st["args"]["device"] for st in starts] == ["cuda", "cpu"]
    assert env.svc.snapshot(s)["device"] == "cpu"


def test_switching_active_flavor_while_busy_finishes_running_job_then_switches(env, monkeypatch):
    _install_db_backends(env, active_flavor="cuda")
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
    monkeypatch.setenv("FAKE_NEMO_CHAT_DELAY", "0.6")
    s = env.settings(nemotron_bin=None, nemotron_idle_timeout_sec=300)
    out: dict = {}

    def first_job():
        out["doc"] = env.svc.transcribe(s, nemotron.parse_wav(make_wav(20)))

    t = threading.Thread(target=first_job)
    t.start()
    assert env.wait_for(lambda: env.svc.state == "busy" and len(env.events("chat")) == 1, 5)

    # Activate CPU while the CUDA job is still running.
    with sqlite3.connect(env.tmp / "lstt.db") as conn:
        installer.activate(conn, "nemo_speech", "cpu", "v0.3.0")

    t.join(10)
    assert not t.is_alive()
    assert out["doc"]["segments"]
    # Sidecar was retired after the busy job finished.
    assert env.wait_for(lambda: env.svc.state == "unloaded", 5)

    monkeypatch.delenv("FAKE_NEMO_CHAT_DELAY")
    _run(env, 20, nemotron_bin=None, nemotron_idle_timeout_sec=300)
    assert [st["args"]["device"] for st in env.events("start")] == ["cuda", "cpu"]


def test_cpu_admission_waits_when_ram_low_and_scales_timeout_and_windows_over_limit(env, monkeypatch):
    _install_db_backends(env, active_flavor="cpu")
    monkeypatch.setattr(nemotron, "effective_cpu_threads", lambda s: 12)

    # 1. Timeout scaling on CPU: base + nemotron_cpu_timeout_per_audio_sec * duration_sec.
    s = env.settings(
        nemotron_bin=None,
        nemotron_job_timeout_sec=60,
        nemotron_cpu_timeout_per_audio_sec=3.0,
    )
    assert env.svc.job_timeout_budget_sec(s, 400.0, flavor="cpu") == 60.0 + 1200.0
    assert env.svc.job_timeout_budget_sec(s, 400.0, flavor="cuda") == 60.0

    # 2. Low available RAM defers load and times out with retry_after_sec.
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 12000)
    with pytest.raises(nemotron.NemotronTimeout) as exc_info:
        _run(
            env,
            10,
            nemotron_bin=None,
            nemotron_job_timeout_sec=0.8,
            nemotron_cpu_timeout_per_audio_sec=0.0,
            nemotron_cpu_ram_mb=24000,
        )
    assert exc_info.value.retry_after_sec == nemotron.RETRY_AFTER_TIMEOUT_SEC
    assert "insufficient available RAM" in (env.svc.last_error or "")
    assert env.events("start") == []

    # 3. Over-limit recording on CPU (> nemotron_cpu_max_sec) windows and stitches with overlap vote.
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
    doc = _run(
        env,
        35 * 60,
        nemotron_bin=None,
        nemotron_cpu_max_sec=1620.0,
        nemotron_window_sec=1620.0,
        nemotron_window_overlap_sec=300.0,
    )
    chats = env.events("chat")
    assert [round(c["audio_seconds"]) for c in chats] == [1620, 780]
    assert doc["seams"] == [{"at_sec": 1470.0, "matched_words": 300, "new_speakers": 0}]
    _assert_consistent_speakers(doc)


def test_live_call_preempts_and_holds_cpu_nemotron_job(env, monkeypatch):
    _install_db_backends(env, active_flavor="cpu")
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
    monkeypatch.setenv("FAKE_NEMO_CHAT_DELAY", "4.0")
    s = env.settings(nemotron_bin=None, nemotron_idle_timeout_sec=0)
    result: dict = {}

    def job():
        try:
            result["doc"] = env.svc.transcribe(s, nemotron.parse_wav(make_wav(30)))
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=job)
    t.start()
    assert env.wait_for(lambda: env.svc.state == "busy" and len(env.events("chat")) == 1, 10)
    assert env.svc.is_active_on("cpu")

    before = metrics.nemotron_preemptions_total._value.get()
    env.svc.enter_live_call()
    try:
        assert nemotron.preempt_cpu_for_live_call(s) is True
        assert metrics.nemotron_preemptions_total._value.get() == before + 1
        # While the live call is active, the CPU job stays queued and does not reload.
        time.sleep(0.8)
        assert t.is_alive()
        snap = env.svc.snapshot(s)
        assert snap["requeues_total"] == 1 and snap["queue_depth"] == 1 and snap["state"] == "unloaded"
        assert snap["live_calls_active"] == 1
        monkeypatch.setenv("FAKE_NEMO_CHAT_DELAY", "0.0")
    finally:
        env.svc.leave_live_call()

    t.join(15)
    assert not t.is_alive()
    assert "exc" not in result, result.get("exc")
    assert result["doc"]["segments"]
    assert env.svc.preemptions_total == 1
