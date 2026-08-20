from __future__ import annotations

from live_stt.gpu import free_vram_mb


def test_free_vram_mb_returns_none_without_nvidia_smi() -> None:
    # This dev host has no nvidia-smi at all -- confirms the "cannot check"
    # path never raises and never claims zero VRAM (which would be read as
    # "definitely can't fit a call" rather than "unknown").
    assert free_vram_mb() is None
