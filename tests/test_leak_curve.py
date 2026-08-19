"""Phase 1 Gate A, as a repeatable tripwire rather than a one-off measurement.

Rigorous 600s-per-condition runs (see CLAUDE.md) found the CPU backend leaks
at ~0.08 MB per audio-second fed -- roughly 200-500x BELOW the 19-41 MB/s the
upstream issue (github.com/mudler/parakeet.cpp#63) reports, which was only
ever measured on a CUDA Jetson build. This test is a tripwire on THAT
constant, not a re-run of the full 600s measurement (kept short for CI
reasonability) -- deliberately asserting the ASSUMPTION, not "no leak":

- If a future parakeet.cpp pin fixes #63 upstream, the slope collapses
  further and this test's assertion fails, telling you to loosen the bound
  (and consider whether the rotation machinery is still needed at all).
- If a future pin regresses and the CPU backend starts leaking like the CUDA
  report, this test fails LOUD before it reaches a real phone call.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.leak_curve import DEFAULT_GGML_LIB_DIR, DEFAULT_WORKER_BIN, run

pytestmark = [pytest.mark.integration, pytest.mark.model, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get(
    "LSTT_MODEL_PATH", str(_REPO_ROOT / "models" / "realtime_eou_120m-v1-q8_0.gguf")
)
LONG_AUDIO_FIXTURE = "/data/homes/stoneshi/src/transcript/output.wav"

# Measured on this repo's CPU build, 600s runs, both conditions in the same
# order of magnitude (silence 0.076 MB/s, speech 0.085 MB/s -- see CLAUDE.md).
# Wide tolerance band: this is a tripwire on the ORDER OF MAGNITUDE, not a
# tight regression bound -- a 90s run (kept short for CI) is noisier than the
# 600s runs this baseline came from.
CPU_ASSUMED_MB_PER_AUDIO_SEC = 0.1
CPU_TOLERANCE_LOW = 0.2  # 0.02 MB/s
CPU_TOLERANCE_HIGH = 6.0  # 0.6 MB/s -- still ~30-70x below the CUDA report


def _require_fixtures() -> None:
    if not DEFAULT_WORKER_BIN.exists():
        pytest.skip(f"worker binary not built at {DEFAULT_WORKER_BIN} -- run scripts/build_worker.sh")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model not found at {MODEL_PATH} -- run scripts/fetch_model.sh")


def test_cpu_leak_rate_silence_within_measured_order_of_magnitude() -> None:
    _require_fixtures()
    summary = run(
        model_path=MODEL_PATH,
        condition="silence",
        audio_sec_target=90.0,
        chunk_ms=160,
        sample_every_sec=5.0,
        n_threads=4,
        audio_fixture=None,
        worker_bin=DEFAULT_WORKER_BIN,
        ggml_lib_dir=DEFAULT_GGML_LIB_DIR,
    )
    slope = summary["slope_mb_per_audio_sec"]
    assert CPU_TOLERANCE_LOW * CPU_ASSUMED_MB_PER_AUDIO_SEC <= slope <= CPU_TOLERANCE_HIGH * CPU_ASSUMED_MB_PER_AUDIO_SEC, (
        f"slope={slope:.4f} MB/s is outside the tripwire band "
        f"[{CPU_TOLERANCE_LOW * CPU_ASSUMED_MB_PER_AUDIO_SEC}, "
        f"{CPU_TOLERANCE_HIGH * CPU_ASSUMED_MB_PER_AUDIO_SEC}] MB/s -- re-measure with the full "
        f"600s protocol and update CLAUDE.md / this constant before trusting the rotation config"
    )


def test_cpu_leak_rate_speech_within_measured_order_of_magnitude() -> None:
    _require_fixtures()
    if not os.path.exists(LONG_AUDIO_FIXTURE):
        pytest.skip(f"long audio fixture not found at {LONG_AUDIO_FIXTURE}")
    summary = run(
        model_path=MODEL_PATH,
        condition="speech",
        audio_sec_target=90.0,
        chunk_ms=160,
        sample_every_sec=5.0,
        n_threads=4,
        audio_fixture=LONG_AUDIO_FIXTURE,
        worker_bin=DEFAULT_WORKER_BIN,
        ggml_lib_dir=DEFAULT_GGML_LIB_DIR,
    )
    slope = summary["slope_mb_per_audio_sec"]
    assert CPU_TOLERANCE_LOW * CPU_ASSUMED_MB_PER_AUDIO_SEC <= slope <= CPU_TOLERANCE_HIGH * CPU_ASSUMED_MB_PER_AUDIO_SEC, (
        f"slope={slope:.4f} MB/s is outside the tripwire band -- re-measure and update baselines"
    )
