from __future__ import annotations

import subprocess

import pytest

from live_stt import gpu
from live_stt.gpu import free_vram_mb, total_vram_mb, used_vram_mb, utilization_pct


def test_free_vram_mb_returns_none_without_nvidia_smi() -> None:
    # This dev host has no nvidia-smi at all -- confirms the "cannot check"
    # path never raises and never claims zero VRAM (which would be read as
    # "definitely can't fit a call" rather than "unknown").
    assert free_vram_mb() is None


def test_total_used_utilization_all_return_none_without_nvidia_smi() -> None:
    assert total_vram_mb() is None
    assert used_vram_mb() is None
    assert utilization_pct() is None


def test_snapshot_is_all_none_together_when_nvidia_smi_is_unavailable() -> None:
    # Never a mix of some real, some None -- see gpu.snapshot()'s docstring.
    snap = gpu.snapshot()
    assert snap == {
        "free_vram_mb": None,
        "total_vram_mb": None,
        "used_vram_mb": None,
        "utilization_pct": None,
    }


def _fake_run(field_to_value: dict[str, str]):
    def fake_run(cmd, **kwargs):
        field = cmd[1].removeprefix("--query-gpu=")
        return subprocess.CompletedProcess(cmd, 0, stdout=field_to_value[field] + "\n", stderr="")

    return fake_run


def test_snapshot_parses_real_nvidia_smi_shaped_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu.subprocess,
        "run",
        _fake_run(
            {
                "memory.free": "11813",
                "memory.total": "24576",
                "memory.used": "12312",
                "utilization.gpu": "45",
            }
        ),
    )
    assert gpu.snapshot() == {
        "free_vram_mb": 11813,
        "total_vram_mb": 24576,
        "used_vram_mb": 12312,
        "utilization_pct": 45,
    }
