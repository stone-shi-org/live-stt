"""Phase 1 Gate C, as a repeatable regression test.

Measured once on NOTSOFAR-1 meeting MTG_32089 (far-field single-channel
audio, ~6 minutes, 1130 reference words -- see CLAUDE.md): native 16kHz WER
51.8%, mulaw-roundtripped 8kHz WER 70.4%, linear-only 8kHz WER 67.7%. Most of
the damage is bandwidth loss (native->linear_8k: +15.9 points), not mu-law
quantization (linear_8k->mulaw_8k: +2.7 points).

This is ONE meeting on a hard far-field, multi-speaker, single-channel
condition -- harder than a real two-party phone call is likely to be, so the
absolute WER here should not be read as "what telephony calls will score."
The point of this test is not the absolute number, it's catching a
REGRESSION in the gap between conditions (e.g. a future parakeet.cpp pin or
a resampler change that makes narrowband handling worse or better) -- so it
asserts against the recorded deltas with slack, not a fixed target.

No corpus, no model, no worker binary -> skip (never fail): this needs all
three, and none of them are available in every environment this runs in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.leak_curve import DEFAULT_WORKER_BIN
from tools.telephony_band_wer import (
    build_reference_text,
    load_native_pcm,
    to_linear_8k_roundtrip,
    to_mulaw_roundtrip,
    transcribe,
)
from tools.wer import tokenize, word_error_rate

pytestmark = [pytest.mark.integration, pytest.mark.model, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get(
    "LSTT_MODEL_PATH", str(_REPO_ROOT / "models" / "realtime_eou_120m-v1-q8_0.gguf")
)
MEETING_DIR = Path(
    "/data/vmfs/main01a_shared/Download/NOTSOFAR-1/eval_set/240629.1_eval_small_with_GT/MTG/MTG_32089"
)

# Recorded baseline (see module docstring). Wide slack: this test's job is to
# catch a gross regression in the telephony-band penalty, not to pin WER to
# the decimal against normal model/decoder nondeterminism.
RECORDED_NATIVE_WER = 0.518
RECORDED_MULAW_8K_WER = 0.704
RECORDED_LINEAR_8K_WER = 0.677
SLACK = 0.15  # absolute WER points


def _require_fixtures() -> None:
    if not DEFAULT_WORKER_BIN.exists():
        pytest.skip(f"worker binary not built at {DEFAULT_WORKER_BIN} -- run scripts/build_worker.sh")
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"model not found at {MODEL_PATH} -- run scripts/fetch_model.sh")
    if not MEETING_DIR.exists():
        pytest.skip(f"NOTSOFAR-1 corpus not found at {MEETING_DIR}")


def test_telephony_band_penalty_matches_recorded_baseline() -> None:
    _require_fixtures()
    reference_words = tokenize(build_reference_text(MEETING_DIR / "gt_transcription.json"))
    native_pcm = load_native_pcm(MEETING_DIR / "sc_meetup_0" / "ch0.wav")

    arms = {
        "native": native_pcm,
        "mulaw_8k": to_mulaw_roundtrip(native_pcm),
        "linear_8k": to_linear_8k_roundtrip(native_pcm),
    }
    wer = {}
    for name, pcm in arms.items():
        hyp_words = tokenize(transcribe(pcm, model_path=MODEL_PATH, n_threads=4))
        wer[name] = word_error_rate(reference_words, hyp_words)

    assert abs(wer["native"] - RECORDED_NATIVE_WER) < SLACK
    assert abs(wer["mulaw_8k"] - RECORDED_MULAW_8K_WER) < SLACK
    assert abs(wer["linear_8k"] - RECORDED_LINEAR_8K_WER) < SLACK

    # The structural claim that matters regardless of the absolute numbers:
    # narrowbanding hurts, and most of the hurt is bandwidth, not the mu-law
    # codec itself. If a future change makes mu-law's OWN contribution
    # dominate (e.g. a resampler regression), this is the line that catches it.
    assert wer["mulaw_8k"] > wer["native"]
    assert wer["linear_8k"] > wer["native"]
    codec_only = wer["mulaw_8k"] - wer["linear_8k"]
    bandwidth_only = wer["linear_8k"] - wer["native"]
    assert bandwidth_only > codec_only
