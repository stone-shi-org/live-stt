"""Runtime TTS-engine dependency installer (live_stt/tts_installer.py).
Unit tests must not touch the network or run a real pip install -- every
subprocess.run call is monkeypatched, matching the offline philosophy
tests/test_diarization_installer.py already uses for the diarization
analogue of this module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from live_stt import tts_installer as ti


@pytest.fixture(autouse=True)
def _fake_repo_root(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The real requirements-tts-*.txt files are deliberately NOT shipped
    in the test-unit Docker image -- point install()'s repo-root lookup at
    a throwaway directory with fake stand-ins instead."""
    fake_root = tmp_path_factory.mktemp("fake-repo-root")
    (fake_root / "requirements-tts-kokoro.txt").write_text("kokoro>=0.8.4\n")
    (fake_root / "requirements-tts-piper.txt").write_text("piper-tts>=1.8.0\n")
    monkeypatch.setattr(ti, "_repo_root", lambda: fake_root)
    return fake_root


def test_unknown_engine_raises_without_touching_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(ti.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    with pytest.raises(ti.TTSInstallError, match="unknown TTS engine"):
        ti.install(str(tmp_path), "not_a_real_engine")

    assert calls == []


def test_install_runs_pip_with_target_and_requirements_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_cmd: list[str] = []

    def fake_run(cmd, capture_output, text, timeout, check):
        captured_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ti.subprocess, "run", fake_run)

    result = ti.install(str(tmp_path), "kokoro")

    assert result.engine == "kokoro"
    assert result.already_installed is False
    assert "--target" in captured_cmd
    target_idx = captured_cmd.index("--target") + 1
    assert captured_cmd[target_idx] == str(tmp_path / "kokoro")
    assert "-r" in captured_cmd
    req_idx = captured_cmd.index("-r") + 1
    assert captured_cmd[req_idx].endswith("requirements-tts-kokoro.txt")
    assert (tmp_path / "kokoro" / ".installed").is_file()


def test_install_is_idempotent_and_does_not_rerun_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(cmd, capture_output, text, timeout, check):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ti.subprocess, "run", fake_run)

    first = ti.install(str(tmp_path), "piper")
    second = ti.install(str(tmp_path), "piper")

    assert calls["n"] == 1
    assert first.already_installed is False
    assert second.already_installed is True


def test_install_rolls_back_directory_on_pip_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="pip resolver exploded")

    monkeypatch.setattr(ti.subprocess, "run", fake_run)

    with pytest.raises(ti.TTSInstallError, match="pip resolver exploded"):
        ti.install(str(tmp_path), "kokoro")

    assert not (tmp_path / "kokoro").exists()


def test_install_rolls_back_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(ti.subprocess, "run", fake_run)

    with pytest.raises(ti.TTSInstallError, match="timed out"):
        ti.install(str(tmp_path), "piper")

    assert not (tmp_path / "piper").exists()


def test_is_installed_reflects_marker_file_only(tmp_path: Path) -> None:
    assert ti.is_installed(str(tmp_path), "kokoro") is False

    engine_dir = tmp_path / "kokoro"
    engine_dir.mkdir()
    (engine_dir / "some_package").mkdir()
    assert ti.is_installed(str(tmp_path), "kokoro") is False

    (engine_dir / ".installed").write_text("1")
    assert ti.is_installed(str(tmp_path), "kokoro") is True


def test_status_reports_every_known_engine_regardless_of_install_state(tmp_path: Path) -> None:
    (tmp_path / "kokoro").mkdir()
    (tmp_path / "kokoro" / ".installed").write_text("1")

    doc = ti.status(str(tmp_path))

    assert set(doc) == {"kokoro", "piper"}
    assert doc["kokoro"]["installed"] is True
    assert doc["piper"]["installed"] is False


def test_install_missing_requirements_file_raises_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ti, "REQUIREMENTS_FILES", {"kokoro": "requirements-does-not-exist.txt"})
    calls = []
    monkeypatch.setattr(ti.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    with pytest.raises(ti.TTSInstallError, match="requirements file not found"):
        ti.install(str(tmp_path), "kokoro")

    assert calls == []
