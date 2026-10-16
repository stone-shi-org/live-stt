"""An engine installed against an outdated torch (marker "1", written before the
cu126 index existed -- PyPI's default torch is CUDA 13, which has no Pascal
support) must be replaceable through the same install API, not skipped forever.
No network: pip is monkeypatched."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from live_stt import diarization_installer as di
from live_stt import tts_installer as ti
from live_stt import torch_index

CASES = [
    pytest.param(di, "pyannote", "DiarizationInstallError", id="pyannote"),
    pytest.param(di, "nemo_sortformer", "DiarizationInstallError", id="sortformer"),
    pytest.param(ti, "kokoro", "TTSInstallError", id="kokoro"),
]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch.delenv("LSTT_TORCH_INDEX_URL", raising=False)
    root = tmp_path_factory.mktemp("fake-root")
    for name in (
        "requirements-diarization.txt",
        "requirements-diarization-sortformer.txt",
        "requirements-tts-kokoro.txt",
        "requirements-tts-piper.txt",
    ):
        (root / name).write_text("x\n")
    monkeypatch.setattr(di, "_repo_root", lambda: root)
    monkeypatch.setattr(ti, "_repo_root", lambda: root)


def _fake_pip(calls: list[list[str]], *, rc: int = 0, payload: str = "NEW"):
    def run(cmd, capture_output, text, timeout, check):
        calls.append(list(cmd))
        target = Path(cmd[cmd.index("--target") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "payload.txt").write_text(payload)
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="boom")

    return run


def _make_old_install(root: Path, engine: str, marker: str = "1") -> Path:
    d = root / engine
    d.mkdir(parents=True)
    (d / "payload.txt").write_text("OLD")
    (d / ".installed").write_text(marker)
    return d


@pytest.mark.parametrize("mod,engine,_err", CASES)
def test_stale_marker_is_rebuilt_in_staging_and_swapped_in(mod, engine, _err, tmp_path, monkeypatch):
    live = _make_old_install(tmp_path, engine)
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_pip(calls))

    result = mod.install(str(tmp_path), engine)

    assert result.already_installed is False and result.replaced_stale is True
    # pip never wrote into the live tree
    assert calls[0][calls[0].index("--target") + 1] == str(tmp_path / f"{engine}.new")
    assert (live / "payload.txt").read_text() == "NEW"
    assert (live / ".installed").read_text() == torch_index.deps_marker(engine)
    assert not (tmp_path / f"{engine}.new").exists() and not (tmp_path / f"{engine}.old").exists()


@pytest.mark.parametrize("mod,engine,err", CASES)
def test_failed_rebuild_leaves_old_install_untouched(mod, engine, err, tmp_path, monkeypatch):
    live = _make_old_install(tmp_path, engine)
    monkeypatch.setattr(mod.subprocess, "run", _fake_pip([], rc=1))

    with pytest.raises(getattr(mod, err)):
        mod.install(str(tmp_path), engine)

    assert (live / "payload.txt").read_text() == "OLD"
    assert (live / ".installed").read_text() == "1"
    assert not (tmp_path / f"{engine}.new").exists()
    assert mod.is_installed(str(tmp_path), engine)  # still usable


@pytest.mark.parametrize("mod,engine,_err", CASES)
def test_current_marker_is_a_noop(mod, engine, _err, tmp_path, monkeypatch):
    _make_old_install(tmp_path, engine, marker=torch_index.deps_marker(engine))
    calls: list[list[str]] = []
    monkeypatch.setattr(mod.subprocess, "run", _fake_pip(calls))

    result = mod.install(str(tmp_path), engine)

    assert result.already_installed is True and result.replaced_stale is False
    assert calls == []


def test_changing_the_torch_index_makes_an_install_stale(tmp_path, monkeypatch):
    _make_old_install(tmp_path, "pyannote", marker=torch_index.deps_marker("pyannote"))
    monkeypatch.setenv("LSTT_TORCH_INDEX_URL", "https://example.test/whl/cu121")
    calls: list[list[str]] = []
    monkeypatch.setattr(di.subprocess, "run", _fake_pip(calls))
    assert di.install(str(tmp_path), "pyannote").replaced_stale is True


def test_piper_marker_is_never_stale(tmp_path, monkeypatch):
    # piper has no torch: an old "1" marker is current, and must not be reinstalled
    _make_old_install(tmp_path, "piper", marker="1")
    calls: list[list[str]] = []
    monkeypatch.setattr(ti.subprocess, "run", _fake_pip(calls))
    assert ti.install(str(tmp_path), "piper").already_installed is True
    assert calls == []


def test_status_reports_stale_without_hiding_that_it_is_installed(tmp_path):
    _make_old_install(tmp_path, "pyannote")  # legacy "1"
    st = di.status(str(tmp_path))
    assert st["pyannote"]["installed"] is True and st["pyannote"]["stale"] is True
    assert st["nemo_sortformer"]["installed"] is False and st["nemo_sortformer"]["stale"] is False


def test_parakeet_cuda_has_its_own_install_label_but_downloads_from_the_same_tag():
    from live_stt.backends.registry import resolve

    spec = resolve("parakeet", "cuda")
    assert spec.tag == "v0.5.0"  # download URL unchanged
    assert spec.pinned_version == "v0.5.0-cuda12"  # but never collides with the CUDA 13 install
    assert resolve("parakeet", "cpu").pinned_version == "v0.5.0"
    assert resolve("whisper", "cuda").pinned_version == "backend-whisper-v1.9.4-cuda12"
