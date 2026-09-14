"""Runtime diarization-engine dependency installer (live_stt/
diarization_installer.py). Unit tests must not touch the network or run a
real pip install -- every subprocess.run call is monkeypatched, matching
the offline philosophy tests/test_backend_installer.py already uses for
the ASR analogue of this module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from live_stt import diarization_installer as di


@pytest.fixture(autouse=True)
def _fake_repo_root(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The real requirements-diarization*.txt files are deliberately NOT
    shipped in the test-unit Docker image (see _repo_root's own docstring)
    -- point install()'s repo-root lookup at a throwaway directory with
    fake stand-ins instead, so these tests exercise the real
    file-exists-then-pip-install control flow without depending on the
    actual repo layout at test-run time.
    """
    fake_root = tmp_path_factory.mktemp("fake-repo-root")
    (fake_root / "requirements-diarization.txt").write_text("pyannote.audio>=4.0.0\n")
    (fake_root / "requirements-diarization-sortformer.txt").write_text("nemo_toolkit[asr]==3.0.0\n")
    monkeypatch.setattr(di, "_repo_root", lambda: fake_root)
    return fake_root


def test_unknown_engine_raises_without_touching_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(di.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    with pytest.raises(di.DiarizationInstallError, match="unknown diarization engine"):
        di.install(str(tmp_path), "not_a_real_engine")

    assert calls == []


def test_install_runs_pip_with_target_and_requirements_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_cmd: list[str] = []

    def fake_run(cmd, capture_output, text, timeout, check):
        captured_cmd.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(di.subprocess, "run", fake_run)

    result = di.install(str(tmp_path), "pyannote")

    assert result.engine == "pyannote"
    assert result.already_installed is False
    assert "--target" in captured_cmd
    target_idx = captured_cmd.index("--target") + 1
    assert captured_cmd[target_idx] == str(tmp_path / "pyannote")
    assert "-r" in captured_cmd
    req_idx = captured_cmd.index("-r") + 1
    assert captured_cmd[req_idx].endswith("requirements-diarization.txt")
    # A real marker file must exist afterward -- this is what is_installed()
    # and a later idempotent install() call both key off of.
    assert (tmp_path / "pyannote" / ".installed").is_file()


def test_install_is_idempotent_and_does_not_rerun_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_run(cmd, capture_output, text, timeout, check):
        calls["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(di.subprocess, "run", fake_run)

    first = di.install(str(tmp_path), "pyannote")
    second = di.install(str(tmp_path), "pyannote")

    assert calls["n"] == 1
    assert first.already_installed is False
    assert second.already_installed is True


def test_install_rolls_back_directory_on_pip_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="pip resolver exploded")

    monkeypatch.setattr(di.subprocess, "run", fake_run)

    with pytest.raises(di.DiarizationInstallError, match="pip resolver exploded"):
        di.install(str(tmp_path), "nemo_sortformer")

    # A half-installed directory (created before pip ran, per install()'s
    # own mkdir) must not survive a failed install -- otherwise a later
    # is_installed() check (marker-file-based, not an import probe) could
    # never distinguish "never attempted" from "failed but left debris".
    assert not (tmp_path / "nemo_sortformer").exists()


def test_install_rolls_back_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(di.subprocess, "run", fake_run)

    with pytest.raises(di.DiarizationInstallError, match="timed out"):
        di.install(str(tmp_path), "pyannote")

    assert not (tmp_path / "pyannote").exists()


def test_is_installed_reflects_marker_file_only(tmp_path: Path) -> None:
    assert di.is_installed(str(tmp_path), "pyannote") is False

    engine_dir = tmp_path / "pyannote"
    engine_dir.mkdir()
    # A populated-but-marker-less directory (e.g. an interrupted install)
    # must NOT read as installed.
    (engine_dir / "some_package").mkdir()
    assert di.is_installed(str(tmp_path), "pyannote") is False

    (engine_dir / ".installed").write_text("1")
    assert di.is_installed(str(tmp_path), "pyannote") is True


def test_status_reports_every_known_engine_regardless_of_install_state(tmp_path: Path) -> None:
    (tmp_path / "pyannote").mkdir()
    (tmp_path / "pyannote" / ".installed").write_text("1")

    doc = di.status(str(tmp_path))

    assert set(doc) == {"pyannote", "nemo_sortformer"}
    assert doc["pyannote"]["installed"] is True
    assert doc["nemo_sortformer"]["installed"] is False


def test_install_missing_requirements_file_raises_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(di, "REQUIREMENTS_FILES", {"pyannote": "requirements-does-not-exist.txt"})
    calls = []
    monkeypatch.setattr(di.subprocess, "run", lambda *a, **k: calls.append(a) or None)

    with pytest.raises(di.DiarizationInstallError, match="requirements file not found"):
        di.install(str(tmp_path), "pyannote")

    assert calls == []
