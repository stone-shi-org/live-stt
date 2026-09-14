"""Backend installer (LS-5 Phase 7b/7d). Unit tests must not touch the
network or a real docker daemon -- installer._download_and_verify and
installer._run_docker are monkeypatched, mirroring the offline-safety
philosophy CLAUDE.md documents elsewhere (a fake process/fixture standing
in for the real external artifact, not a mock of this module's own logic).
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


def _spec(sha256: str, engine: str = "parakeet", header_urls: dict[str, str] | None = None) -> BackendSpec:
    return BackendSpec(
        engine=engine,
        flavor="cpu",
        source="github_release",
        repo="fake/repo",
        tag="v9.9.9",
        asset_filename="release.tar.gz",
        sha256=sha256,
        lib_filename="libparakeet.so" if engine == "parakeet" else "libwhisper.so",
        header_urls=header_urls or {},
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


def test_install_fetches_registered_headers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tar_path, sha256 = _fake_tarball(tmp_path, "release-v9.9.9", {"libwhisper.so": b"x"})
    spec = _spec(sha256, engine="whisper", header_urls={"whisper.h": "https://example.invalid/whisper.h"})
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)
    monkeypatch.setattr(
        installer, "_download_and_verify", lambda spec_, dest: dest.write_bytes(tar_path.read_bytes())
    )

    fetched = []

    def fake_urlretrieve(url, filename):
        fetched.append(url)
        Path(filename).write_text("// fake header")

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", fake_urlretrieve)

    conn = db.connect(":memory:")
    result = installer.install(conn, str(tmp_path / "install-root"), "whisper", "cpu")
    assert fetched == ["https://example.invalid/whisper.h"]
    assert (Path(result.path) / "whisper.h").exists()


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
    version_root = tmp_path / "install-root"
    with pytest.raises(installer.BackendInstallError, match="private-symbol safety probe"):
        installer.install(conn, str(version_root), "parakeet", "cpu")
    # A failed probe must not leave a half-installed directory behind for a
    # later get_active_lib_dir() to ever point a real worker spawn at.
    assert not (version_root / "parakeet" / "cpu" / "v9.9.9").exists()


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


def test_install_via_parakeet_tdt_alias_uses_the_parakeet_spec_and_probes_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_path, sha256 = _fake_tarball(tmp_path, "release-v9.9.9", {"libparakeet.so": b"x"})
    spec = _spec(sha256, engine="parakeet")
    resolved_with = []

    def fake_resolve(engine, flavor):
        resolved_with.append(engine)
        return spec

    monkeypatch.setattr(installer, "resolve", fake_resolve)
    monkeypatch.setattr(
        installer, "_download_and_verify", lambda spec_, dest: dest.write_bytes(tar_path.read_bytes())
    )
    monkeypatch.setattr(installer, "probe_parakeet_private_symbols", lambda so_path: True)

    conn = db.connect(":memory:")
    result = installer.install(conn, str(tmp_path / "install-root"), "parakeet_tdt", "cpu")
    # canonical_engine() maps parakeet_tdt -> parakeet before resolve() and
    # before the row is written -- installed_backends is keyed by the
    # canonical name, shared with the streaming engine.
    assert resolved_with == ["parakeet"]
    assert result.engine == "parakeet"


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


# --- oci_image source (whisper:cuda) -----------------------------------


def _oci_spec(image_files: dict[str, tuple[str, str]], symlinks: dict[str, str]) -> BackendSpec:
    return BackendSpec(
        engine="whisper",
        flavor="cuda",
        source="oci_image",
        image_repo="fake/whisper.cpp",
        image_registry="ghcr.io",
        image_digest="sha256:" + "a" * 64,
        image_files=image_files,
        symlinks=symlinks,
        lib_filename="libwhisper.so",
        wired=True,
    )


def test_install_oci_image_copies_verifies_and_symlinks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"fake-cuda-so-bytes"
    sha256 = hashlib.sha256(content).hexdigest()
    spec = _oci_spec(
        image_files={"libwhisper.so.1.9.4": ("/app/build/bin/libwhisper.so.1.9.4", sha256)},
        symlinks={"libwhisper.so": "libwhisper.so.1.9.4", "libwhisper.so.1": "libwhisper.so.1.9.4"},
    )
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)

    docker_calls = []

    def fake_run_docker(args):
        docker_calls.append(args)
        if args[0] == "cp":
            # args = ["cp", "<container>:<path>", "<dest>"]
            dest = Path(args[2])
            dest.write_bytes(content)

    monkeypatch.setattr(installer, "_run_docker", fake_run_docker)
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: None)  # the `docker rm -f` cleanup call

    conn = db.connect(":memory:")
    result = installer.install(conn, str(tmp_path / "install-root"), "whisper", "cuda")

    assert result.version == spec.image_digest
    real_file = Path(result.path) / "libwhisper.so.1.9.4"
    assert real_file.read_bytes() == content
    assert (Path(result.path) / "libwhisper.so").resolve() == real_file.resolve()
    assert (Path(result.path) / "libwhisper.so.1").resolve() == real_file.resolve()
    assert any(call[0] == "create" for call in docker_calls)
    assert any(call[0] == "cp" for call in docker_calls)

    # Real regression: version_dir.name IS spec.image_digest for an
    # oci_image source (e.g. "sha256:aaaa..." here, via _oci_spec above),
    # and a bare, unsanitized use of that in --name would embed the colon.
    # The fake _run_docker above never validates this (it's not the real
    # docker daemon), which is exactly why this shipped once and only broke
    # against a REAL `docker create` on a real deployment -- see
    # installer.py's own comment on _install_oci_image's container_name
    # line. Assert directly on the actual argument passed to _run_docker so
    # a future regression fails here, not in production.
    create_call = next(call for call in docker_calls if call[0] == "create")
    container_name = create_call[create_call.index("--name") + 1]
    assert ":" not in container_name

    # Real regression, found separately and later than the container-name
    # one above: version_dir.name (the ON-DISK directory, not just the
    # docker container name) also embeds spec.image_digest -- and this
    # directory can become session.py's ggml_lib_dir, which
    # WorkerHandle.spawn() puts straight into LD_LIBRARY_PATH, a
    # COLON-DELIMITED path list. A colon in the directory name silently
    # splits it into two bogus paths there, so every transitive dependency
    # of that directory's OWN .so files (libcudart.so.13 etc, physically
    # present) fails to resolve -- confirmed for real via `ldd` against the
    # actual production deployment. `version` itself (the API-facing/DB
    # value) must stay exactly the real digest; only the directory name is
    # sanitized.
    assert result.version == spec.image_digest  # unsanitized -- API contract unchanged
    assert ":" not in Path(result.path).name  # but the actual directory must be


def test_install_oci_image_rejects_a_file_with_the_wrong_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _oci_spec(
        image_files={"libwhisper.so.1.9.4": ("/app/build/bin/libwhisper.so.1.9.4", "0" * 64)},
        symlinks={},
    )
    monkeypatch.setattr(installer, "resolve", lambda engine, flavor: spec)

    def fake_run_docker(args):
        if args[0] == "cp":
            Path(args[2]).write_bytes(b"not matching the pinned sha256")

    monkeypatch.setattr(installer, "_run_docker", fake_run_docker)
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: None)

    conn = db.connect(":memory:")
    with pytest.raises(installer.BackendInstallError, match="sha256 mismatch"):
        installer.install(conn, str(tmp_path / "install-root"), "whisper", "cuda")


def test_run_docker_raises_a_clear_error_when_docker_binary_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_subprocess_run(cmd, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(installer.subprocess, "run", fake_subprocess_run)
    with pytest.raises(installer.BackendInstallError, match="docker is not available"):
        installer._run_docker(["create", "some-image"])


def test_run_docker_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        returncode = 1
        stderr = "no such image"

    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: _FakeResult())
    with pytest.raises(installer.BackendInstallError, match="no such image"):
        installer._run_docker(["create", "some-image"])


# --- activate / get_active ------------------------------------------------


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


def test_activate_via_parakeet_tdt_alias_affects_the_shared_parakeet_row(tmp_path: Path) -> None:
    conn = db.connect(":memory:")
    installer.ensure_schema(conn)
    now = "2024-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO installed_backends VALUES (?,?,?,?,?,?,0)",
        ("parakeet", "cuda", "v1", "sha1", "/p/v1", now),
    )
    conn.commit()

    installer.activate(conn, "parakeet_tdt", "cuda", "v1")

    # Reading back via either name sees the same row -- there is only ever
    # one "parakeet" row, shared by both engine names.
    assert installer.get_active(conn, "parakeet").version == "v1"
    assert installer.get_active(conn, "parakeet_tdt").version == "v1"


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
    # Same underlying row, reached via the alias.
    assert installer.get_active_lib_dir(db_path, "parakeet_tdt") == "/some/path"
