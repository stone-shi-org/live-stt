"""Real nemo-speech binary + real Nemotron-3.5-Transcribe GGUFs on a real GPU
(LS-19). Skipped unless LSTT_NEMOTRON_BIN, LSTT_NEMOTRON_MODELS_DIR and
LSTT_NEMOTRON_TEST_WAV point at real files -- see CLAUDE.md's Nemotron
section for the homesrv3 paths used to verify it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.model, pytest.mark.slow]

BIN = os.environ.get("LSTT_NEMOTRON_TEST_BIN") or os.environ.get("LSTT_NEMOTRON_BIN")
MODELS = os.environ.get("LSTT_NEMOTRON_TEST_MODELS_DIR") or os.environ.get("LSTT_NEMOTRON_MODELS_DIR")
WAV = os.environ.get("LSTT_NEMOTRON_TEST_WAV")


@pytest.mark.skipif(not (BIN and MODELS and WAV), reason="needs a real nemo-speech binary, models and a WAV")
def test_real_sidecar_returns_speaker_segments_with_text(tmp_path):
    from live_stt import nemotron
    from live_stt.config import Settings

    settings = Settings(
        _env_file=None,
        nemotron_bin=BIN,
        nemotron_models_dir=MODELS,
        db_path=str(tmp_path / "x.db"),
        backends_dir=str(tmp_path / "b"),
        nemotron_idle_timeout_sec=0,
    )
    svc = nemotron.NemotronService()
    try:
        doc = svc.transcribe(settings, nemotron.parse_wav(Path(WAV).read_bytes()))
    finally:
        svc.stop()
    assert doc["num_speakers"] >= 1
    assert doc["segments"] and all(seg["text"] for seg in doc["segments"])
    assert svc.loads_total == 1
