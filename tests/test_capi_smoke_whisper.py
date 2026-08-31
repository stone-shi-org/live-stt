"""The whisper engine's analogue of test_capi_smoke.py: the real
live_stt_worker_whisper binary against a real whisper.cpp model.

Needs the binary built (scripts/build_worker.sh) and a whisper ggml *.bin
mounted at LSTT_WHISPER_MODEL_PATH (default: models/ggml-base.en-q8_0.bin,
fetched by `scripts/fetch_model.sh whisper-base.en-q8_0` -- deliberately the
smallest registered whisper model, 82MB, not the large-v3-turbo one the
service defaults whisper users to elsewhere, to keep this test's fixture
requirement cheap). Skips cleanly if either is missing, same contract as
test_capi_smoke.py, so the unit suite and the test-unit Docker target (no
worker binaries, no models at all) never touch this.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from live_stt.framing import FrameType
from worker_harness import WORKER_BIN_WHISPER, WorkerHandle

pytestmark = [pytest.mark.integration, pytest.mark.model]

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get(
    "LSTT_WHISPER_MODEL_PATH", str(_REPO_ROOT / "models" / "ggml-base.en-q8_0.bin")
)
LONG_AUDIO_FIXTURE = "/data/homes/stoneshi/src/transcript/output.wav"


def _require_binary() -> None:
    if not WORKER_BIN_WHISPER.exists():
        pytest.skip(f"whisper worker binary not built at {WORKER_BIN_WHISPER} -- run scripts/build_worker.sh")


def _require_fixtures() -> None:
    _require_binary()
    if not os.path.exists(MODEL_PATH):
        pytest.skip(
            f"whisper model not found at {MODEL_PATH} -- run "
            "'scripts/fetch_model.sh whisper-base.en-q8_0'"
        )


def _handle() -> WorkerHandle:
    # ggml_lib_dir=None -- this binary is fully statically linked (see
    # worker/CMakeLists.txt), unlike the parakeet WorkerHandle default.
    return WorkerHandle(worker_bin=WORKER_BIN_WHISPER, ggml_lib_dir=None)


def _configure(handle: WorkerHandle, model_path: str, *, language: str = "en") -> dict:
    handle.send_json(
        FrameType.CONFIG, {"gguf_path": model_path, "language": language, "n_threads": 4}
    )
    frame_type, doc = handle.recv_json()
    if frame_type == FrameType.ERROR:
        raise RuntimeError(f"whisper worker CONFIG failed: {doc}")
    assert frame_type == FrameType.READY
    return doc


def test_worker_loads_model_and_reports_ready() -> None:
    _require_fixtures()
    handle = _handle()
    try:
        ready = _configure(handle, MODEL_PATH)
        assert "whisper.cpp" in ready["ggml_features"]
    finally:
        handle.close()


def test_worker_feed_only_buffers_and_returns_an_empty_result() -> None:
    # See worker/session_whisper.hpp -- feed() is deliberately cheap (no
    # inference) for this batch-only engine; only finalize() transcribes.
    _require_fixtures()
    handle = _handle()
    try:
        _configure(handle, MODEL_PATH)
        one_second = b"\x00\x00" * 16000  # 1s of silence @ 16kHz mono int16
        handle.send(FrameType.AUDIO, one_second)
        frame_type, doc = handle.recv_json()
        assert frame_type == FrameType.RESULT
        assert doc["fed_samples"] == 16000
        assert doc.get("text", "") == ""
        assert doc.get("words", []) == []
    finally:
        handle.close()


def test_worker_feed_with_zero_samples_is_legal() -> None:
    _require_fixtures()
    handle = _handle()
    try:
        _configure(handle, MODEL_PATH)
        handle.send(FrameType.PING)
        frame_type, doc = handle.recv_json()
        assert frame_type == FrameType.RESULT
        assert doc["fed_samples"] == 0
    finally:
        handle.close()


def test_worker_reports_error_for_missing_model() -> None:
    _require_binary()
    handle = _handle()
    handle.send_json(FrameType.CONFIG, {"gguf_path": "/nonexistent/model.bin", "n_threads": 4})
    frame_type, doc = handle.recv_json()
    assert frame_type == FrameType.ERROR
    assert "error" in doc
    assert handle.proc.wait(timeout=5) == 1


def test_worker_exits_cleanly_after_eof() -> None:
    _require_fixtures()
    handle = _handle()
    _configure(handle, MODEL_PATH)
    exit_code = handle.close()
    assert exit_code == 0


def test_worker_produces_a_correct_transcript_from_real_speech_with_word_timestamps() -> None:
    # Same real fixture CLAUDE.md's "Serious open risk" entry and
    # test_capi_smoke.py both use. Feeds the ~20s window around 18.0s that
    # CLAUDE.md documents as containing "yes yes absolutely ok perfect well
    # yeah ... twenty thirty minutes or so and i just wanted to informally
    # connect to chat a little bit more" -- real, known content, not just
    # "some words came back". Confirmed manually while building this that
    # whisper (unlike the parakeet repro in that CLAUDE.md entry) transcribes
    # this window correctly and with real punctuation/capitalization.
    _require_fixtures()
    if not os.path.exists(LONG_AUDIO_FIXTURE):
        pytest.skip(f"long audio fixture not found at {LONG_AUDIO_FIXTURE}")

    handle = _handle()
    try:
        _configure(handle, MODEL_PATH)
        wf = wave.open(LONG_AUDIO_FIXTURE, "rb")
        sr = wf.getframerate()
        wf.setpos(int(18.0 * sr))
        pcm = wf.readframes(int(20.0 * sr))

        chunk_bytes = sr * 2  # 1s chunks
        for i in range(0, len(pcm), chunk_bytes):
            chunk = pcm[i : i + chunk_bytes]
            handle.send(FrameType.AUDIO, chunk)
            frame_type, doc = handle.recv_json()
            assert frame_type == FrameType.RESULT
            assert doc.get("text", "") == ""  # feed() never transcribes, see above

        handle.send(FrameType.FINALIZE)
        frame_type, final_doc = handle.recv_json()
        assert frame_type == FrameType.FINAL
    finally:
        handle.close()

    text = final_doc["text"].lower()
    for expected in ("yes", "absolutely", "perfect", "minutes"):
        assert expected in text, f"expected {expected!r} in transcript: {text!r}"

    words = final_doc["words"]
    assert len(words) >= 10
    # Word timestamps must be real, monotonic-ish values within the fed
    # window, not zeros or garbage -- each word's own end must not precede
    # its own start, and start times should span a meaningful fraction of
    # the ~20s window fed.
    for w in words:
        assert w["end"] >= w["start"] >= 0.0
        assert "w" in w and w["w"]
    assert words[-1]["end"] > 5.0
