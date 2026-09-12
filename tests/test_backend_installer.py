"""Backend installer (LS-5 Phase 7b). Unit tests must not touch the
network -- installer._download_and_verify is monkeypatched to copy a
locally-built fake release tarball instead of hitting GitHub, mirroring
the offline-safety philosophy CLAUDE.md documents elsewhere (a fake
process/fixture standing in for the real external artifact, not a mock of
this module's own logic).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from live_stt import db
from live_stt.backends import installer
from live_stt.backends.registry import BackendSpec


def _fake_tarball(tmp_path: Path, top_dir: str, files: dict[str, bytes]) -> tuple[Path, str]:
    tar_path = tmp_path / "release.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=f"{top_dir}/{name}")
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    return tar_path, sha256


def _spec(sha256: str, engine: str = "parakeet") -> BackendSpec:
    return BackendSpec(
        engine=engine,
        flavor="cpu",
        repo="fake/repo",
        tag="v9.9.9",
        asset_filename="release.tar.gz",
        sha256=sha256,
        lib_filename="libparakeet.so" if engine == "parakeet" else "libwhisper.so",
        ships_headers=True,
        wired=True,
    )


def test_install_extracts_stripping_the_top_level_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path, sha256 = _fake_tarball(
        tmp_path, "release-v9.9.9", {"libparakeet.so": b"fake-so-bytes", "parakeet_capi.h": b"//fake"}
    )
    spec = _spec(sha256)
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)
    monkeypatch.setattr(
        installer, "_download_and_verify", lambda spec_, dest: dest.write_bytes(tar_path.read_bytes())
    )
    monkeypatch.setattr(installer, "probe_parakeet_private_symbols", lambda so_path: True)

    conn = db.connect(":memory:")
    result = installer.install(conn, str(tmp_path / "install-root"), "parakeet", "cpu")

    assert result.version == "v9.9.9"
    assert result.active is False
    assert (Path(result.path) / "libparakeet.so").read_bytes() == b"fake-so-bytes"
    assert (Path(result.path) / "parakeet_capi.h").exists()


def test_install_is_idempotent_and_does_not_redownload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path, sha256 = _fake_tarball(tmp_path, "release-v9.9.9", {"libparakeet.so": b"x"})
    spec = _spec(sha256)
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)
    calls = {"n": 0}

    def fake_download(spec_, dest):
        calls["n"] += 1
        dest.write_bytes(tar_path.read_bytes())

    monkeypatch.setattr(installer, "_download_and_verify", fake_download)
    monkeypatch.setattr(installer, "probe_parakeet_private_symbols", lambda so_path: True)

    conn = db.connect(":memory:")
    root = str(tmp_path / "install-root")
    installer.install(conn, root, "parakeet", "cpu")
    installer.install(conn, root, "parakeet", "cpu")
    assert calls["n"] == 1


def test_install_rejects_a_version_that_fails_the_private_symbol_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_path, sha256 = _fake_tarball(tmp_path, "release-v9.9.9", {"libparakeet.so": b"x"})
    spec = _spec(sha256)
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)
    monkeypatch.setattr(
        installer, "_download_and_verify", lambda spec_, dest: dest.write_bytes(tar_path.read_bytes())
    )
    monkeypatch.setattr(installer, "probe_parakeet_private_symbols", lambda so_path: False)

    conn = db.connect(":memory:")
    with pytest.raises(installer.BackendInstallError, match="private-symbol safety probe"):
        installer.install(conn, str(tmp_path / "install-root"), "parakeet", "cpu")


def test_install_does_not_probe_symbols_for_non_parakeet_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_path, sha256 = _fake_tarball(tmp_path, "release-v9.9.9", {"libwhisper.so": b"x"})
    spec = _spec(sha256, engine="whisper")
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)
    monkeypatch.setattr(
        installer, "_download_and_verify", lambda spec_, dest: dest.write_bytes(tar_path.read_bytes())
    )
    probe_called = []
    monkeypatch.setattr(
        installer, "probe_parakeet_private_symbols", lambda so_path: probe_called.append(so_path) or True
    )

    conn = db.connect(":memory:")
    installer.install(conn, str(tmp_path / "install-root"), "whisper", "cpu")
    assert probe_called == []


def test_download_and_verify_rejects_sha256_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(sha256="0" * 64)

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"not matching the pinned sha256")

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(installer.BackendInstallError, match="sha256 mismatch"):
        installer._download_and_verify(spec, tmp_path / "out.tar.gz")


def test_download_and_verify_wraps_network_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(sha256="0" * 64)

    def fake_urlretrieve(url, filename):
        raise OSError("connection refused")

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", fake_urlretrieve)
    with pytest.raises(installer.BackendInstallError, match="download failed"):
        installer._download_and_verify(spec, tmp_path / "out.tar.gz")


def test_probe_parakeet_private_symbols_true_when_both_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLib:
        def __getattr__(self, name):
            if name in ("_ZN2pk15set_num_threadsEi", "_ZN2pk16shutdown_backendEv"):
                return object()
            raise AttributeError(name)

    monkeypatch.setattr(installer.ctypes, "CDLL", lambda path: _FakeLib())
    assert installer.probe_parakeet_private_symbols(Path("/fake/libparakeet.so")) is True


def test_probe_parakeet_private_symbols_false_when_a_symbol_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLib:
        def __getattr__(self, name):
            if name == "_ZN2pk15set_num_threadsEi":
                return object()
            raise AttributeError(name)

    monkeypatch.setattr(installer.ctypes, "CDLL", lambda path: _FakeLib())
    assert installer.probe_parakeet_private_symbols(Path("/fake/libparakeet.so")) is False


def test_probe_parakeet_private_symbols_false_when_dlopen_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_oserror(path):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(installer.ctypes, "CDLL", raise_oserror)
    assert installer.probe_parakeet_private_symbols(Path("/nonexistent.so")) is False


def test_activate_deactivates_other_versions_of_the_same_engine(tmp_path: Path) -> None:
    conn = db.connect(":memory:")
    installer.ensure_schema(conn)
    now = "2024-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO installed_backends VALUES (?,?,?,?,?,?,1)",
        ("parakeet", "cpu", "v1", "sha1", "/p/v1", now),
    )
    conn.execute(
        "INSERT INTO installed_backends VALUES (?,?,?,?,?,?,0)",
        ("parakeet", "cpu", "v2", "sha2", "/p/v2", now),
    )
    conn.commit()

    installer.activate(conn, "parakeet", "cpu", "v2")

    active = installer.get_active(conn, "parakeet")
    assert active is not None
    assert active.version == "v2"
    v1 = next(b for b in installer.list_installed(conn) if b.version == "v1")
    assert v1.active is False


def test_activate_unknown_version_raises() -> None:
    conn = db.connect(":memory:")
    with pytest.raises(installer.BackendInstallError, match="is not installed"):
        installer.activate(conn, "parakeet", "cpu", "v999")


def test_get_active_returns_none_when_nothing_active() -> None:
    conn = db.connect(":memory:")
    assert installer.get_active(conn, "parakeet") is None


def test_get_active_lib_dir_returns_none_when_db_file_does_not_exist(tmp_path: Path) -> None:
    missing_db = tmp_path / "nonexistent" / "live_stt.db"
    assert installer.get_active_lib_dir(str(missing_db), "parakeet") is None


def test_get_active_lib_dir_returns_the_active_path(tmp_path: Path) -> None:
    db_path = str(tmp_path / "live_stt.db")
    conn = db.connect(db_path)
    installer.ensure_schema(conn)
    now = "2024-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO installed_backends VALUES (?,?,?,?,?,?,1)",
        ("parakeet", "cpu", "v1", "sha1", "/some/path", now),
    )
    conn.commit()
    conn.close()

    assert installer.get_active_lib_dir(db_path, "parakeet") == "/some/path"
