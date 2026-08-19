"""Phase 0 gate: the real worker binary against a real model.

Needs the binary built (scripts/build_worker.sh) and a GGUF mounted at
LSTT_MODEL_PATH (default: models/realtime_eou_120m-v1-q8_0.gguf, fetched by
scripts/fetch_model.sh). Skips cleanly if either is missing, so the unit
suite (./test.sh, which runs -m "not integration") never touches this and the
test-unit Docker target -- which contains no worker binary and no model at
all -- never even imports it in a context where it could fail.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import pytest

from live_stt.framing import FrameType
from worker_harness import WORKER_BIN, WorkerHandle, configure

pytestmark = [pytest.mark.integration, pytest.mark.model]

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get(
    "LSTT_MODEL_PATH", str(_REPO_ROOT / "models" / "realtime_eou_120m-v1-q8_0.gguf")
)
LONG_AUDIO_FIXTURE = "/data/homes/stoneshi/src/transcript/output.wav"


def _require_binary() -> None:
    if not WORKER_BIN.exists():
        pytest.skip(f"worker binary not built at {WORKER_BIN} -- run scripts/build_worker.sh")


def _require_fixtures() -> None:
    _require_binary()
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model not found at {MODEL_PATH} -- run scripts/fetch_model.sh")


def test_worker_loads_model_and_reports_ready() -> None:
    _require_fixtures()
    handle = WorkerHandle()
    try:
        ready = configure(handle, MODEL_PATH)
        # ABI v6 at the pinned submodule SHA (see .gitmodules) -- a mismatch
        # here means the pin moved without anyone updating this expectation.
        assert ready["abi_version"] == 6
        # Catches the GGML_NATIVE=OFF trap: disabling it without explicitly
        # re-enabling AVX2/FMA/F16C silently ships a scalar build and loses
        # 2-4x throughput with no visible symptom other than a mediocre RTF.
        assert "AVX2" in ready["ggml_features"]
    finally:
        handle.close()


def test_worker_feeds_one_chunk_without_error() -> None:
    _require_fixtures()
    handle = WorkerHandle()
    try:
        configure(handle, MODEL_PATH)
        silence = b"\x00\x00" * 2560  # 160ms @ 16kHz mono int16 -- the 120m model's chunk size
        handle.send(FrameType.AUDIO, silence)
        frame_type, doc = handle.recv_json()
        assert frame_type == FrameType.RESULT
        assert "rss_kb" in doc
        assert doc["fed_samples"] == 2560
    finally:
        handle.close()


def test_worker_feed_with_zero_samples_is_legal() -> None:
    # parakeet_capi.h documents n_samples == 0 as legal -- used for
    # liveness/RSS pings with no audio to feed.
    _require_fixtures()
    handle = WorkerHandle()
    try:
        configure(handle, MODEL_PATH)
        handle.send(FrameType.PING)
        frame_type, doc = handle.recv_json()
        assert frame_type == FrameType.RESULT
        assert doc["fed_samples"] == 0
    finally:
        handle.close()


def test_worker_reports_error_for_missing_model() -> None:
    _require_binary()
    handle = WorkerHandle()
    handle.send_json(FrameType.CONFIG, {"gguf_path": "/nonexistent/model.gguf", "n_threads": 4})
    frame_type, doc = handle.recv_json()
    assert frame_type == FrameType.ERROR
    assert "error" in doc
    assert handle.proc.wait(timeout=5) == 1


def test_worker_exits_cleanly_after_eof() -> None:
    # The production lifecycle always SIGKILLs the worker (see CLAUDE.md);
    # this exercises the OTHER path -- a manual/test caller that simply
    # closes the socket -- and confirms it doesn't hang or crash.
    _require_fixtures()
    handle = WorkerHandle()
    configure(handle, MODEL_PATH)
    exit_code = handle.close()
    assert exit_code == 0


def test_worker_produces_a_sensible_transcript_from_real_speech() -> None:
    _require_fixtures()
    if not os.path.exists(LONG_AUDIO_FIXTURE):
        pytest.skip(f"long audio fixture not found at {LONG_AUDIO_FIXTURE}")

    handle = WorkerHandle()
    try:
        configure(handle, MODEL_PATH)
        wf = wave.open(LONG_AUDIO_FIXTURE, "rb")
        chunk_samples = int(wf.getframerate() * 0.16)
        text = ""
        for _ in range(60):  # ~9.6s of real speech
            pcm = wf.readframes(chunk_samples)
            if not pcm:
                break
            handle.send(FrameType.AUDIO, pcm)
            _, doc = handle.recv_json()
            text += doc.get("text", "")
        handle.send(FrameType.FINALIZE)
        _, final_doc = handle.recv_json()
        text += final_doc.get("text", "")
    finally:
        handle.close()

    # A real transcript, not silence-driven degenerate output: several
    # distinct real words -- catches a session that is silently producing
    # garbage (e.g. a single repeated token) despite returning HTTP-200-shaped
    # JSON on every call.
    words = text.split()
    assert len(words) >= 5
    assert len(set(words)) >= 3
