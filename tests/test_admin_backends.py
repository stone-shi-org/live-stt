"""LS-5 Phase 7b: GET /api/backends + POST /api/backends/install,activate,
against a real socket and a real serve_admin_http() server -- same pattern
tests/test_admin_settings.py already uses for /api/settings. Network calls
are monkeypatched out (installer.install is patched directly) -- these
tests exercise the HTTP wiring/status-code contract, not a real download.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from live_stt import gpu, nemotron
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.backends import installer
from live_stt.config import Settings
from live_stt.settings_store import SettingsStore
from live_stt.state import ServerState


def _state(settings: Settings | None = None, settings_store: SettingsStore | None = None) -> ServerState:
    settings = settings or Settings(_env_file=None)
    return ServerState(
        settings=settings,
        budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots),
        settings_store=settings_store,
    )


def _post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.mark.asyncio
async def test_get_api_backends_reports_store_unavailable_when_none_configured() -> None:
    server = serve_admin_http("127.0.0.1", 0, _state(settings_store=None))
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/backends", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
            assert doc["store_available"] is False
            assert doc["registered"] == []
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_get_api_backends_lists_the_real_registry_when_store_available() -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None)
    store.migrate_if_empty(settings)
    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/backends", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
            assert doc["store_available"] is True
            keys = {(r["engine"], r["flavor"]) for r in doc["registered"]}
            assert ("parakeet", "cpu") in keys
            assert ("whisper", "cpu") in keys
            assert doc["installed"] == []
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_without_a_store_is_503() -> None:
    server = serve_admin_http("127.0.0.1", 0, _state(settings_store=None))
    try:
        port = server.server_address[1]
        status, _ = _post(f"http://127.0.0.1:{port}/api/backends/install", {"engine": "parakeet", "flavor": "cpu"})
        assert status == 503
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_missing_fields_is_400() -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None)
    store.migrate_if_empty(settings)
    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]
        status, doc = _post(f"http://127.0.0.1:{port}/api/backends/install", {"engine": "parakeet"})
        assert status == 400
        assert "engine" in doc["error"] or "flavor" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_success_then_activate(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_install(conn, install_root, engine, flavor):
        # Must actually write the row -- activate() (called next, for
        # real, against the real DB) looks it up there, not in whatever
        # this fake returns.
        installer.ensure_schema(conn)
        conn.execute(
            "INSERT OR IGNORE INTO installed_backends VALUES (?,?,?,?,?,?,0)",
            (engine, flavor, "v9.9.9", "sha", str(tmp_path / "p"), "now"),
        )
        conn.commit()
        return installer.InstalledBackend(engine, flavor, "v9.9.9", "sha", str(tmp_path / "p"), "now", False)

    monkeypatch.setattr(installer, "install", fake_install)

    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"))
    store.migrate_if_empty(settings)
    state = _state(settings=settings, settings_store=store)
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/install", {"engine": "parakeet", "flavor": "cpu"}
        )
        assert status == 200
        assert doc["version"] == "v9.9.9"
        assert doc["active"] is False

        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "parakeet", "flavor": "cpu", "version": "v9.9.9"},
        )
        assert status == 200
        assert doc["activated"]["version"] == "v9.9.9"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_download_failure_is_502(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def fake_install(conn, install_root, engine, flavor):
        raise installer.BackendInstallError("sha256 mismatch")

    monkeypatch.setattr(installer, "install", fake_install)

    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"))
    store.migrate_if_empty(settings)
    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/install", {"engine": "parakeet", "flavor": "cpu"}
        )
        assert status == 502
        assert "sha256 mismatch" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_activate_unknown_version_is_400(tmp_path) -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"))
    store.migrate_if_empty(settings)
    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "parakeet", "flavor": "cpu", "version": "v999"},
        )
        assert status == 400
        assert "is not installed" in doc["error"]
    finally:
        server.shutdown()


def _seed_installed(conn, tmp_path, engine: str, flavor: str, version: str = "v0.3.0", active: int = 0) -> None:
    installer.ensure_schema(conn)
    conn.execute(
        "INSERT OR REPLACE INTO installed_backends VALUES (?,?,?,?,?,?,?)",
        (engine, flavor, version, "sha", str(tmp_path / f"{engine}-{flavor}"), "now", active),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_activate_nemo_speech_cuda_refuses_pascal_4gb_and_preserves_active_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(gpu, "device_name", lambda: "Quadro P1000")
    monkeypatch.setattr(gpu, "total_vram_mb", lambda: 4096)
    monkeypatch.setattr(gpu, "compute_cap", lambda: 6.1)
    monkeypatch.setattr(nemotron, "cpu_has_avx2", lambda: True)
    monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)

    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"))
    store.migrate_if_empty(settings)
    _seed_installed(store._conn, tmp_path, "nemo_speech", "cpu", active=1)
    _seed_installed(store._conn, tmp_path, "nemo_speech", "cuda", active=0)

    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "nemo_speech", "flavor": "cuda", "version": "v0.3.0"},
        )
        assert status == 422
        assert doc["reason"] == "gpu_unsupported"
        assert "Quadro P1000" in doc["error"]
        assert "6.1" in doc["error"]
        assert "4" in doc["error"] or "4096" in doc["error"]
        assert doc["suggestion"] == "Activate nemo_speech:cpu instead"
        assert doc["details"]["compute_cap"] == 6.1
        assert doc["details"]["total_vram_mb"] == 4096
        assert doc["suggested_flavor"] == "cpu"
        assert doc["suggested_engine"] == "nemo_speech"

        # Active flavor must remain unchanged (nemo_speech:cpu).
        active = installer.get_active(store._conn, "nemo_speech")
        assert active is not None and active.flavor == "cpu"

        # GET /api/backends includes activatable and unsupported_reason.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/backends", timeout=5) as resp:
            bdoc = json.loads(resp.read().decode())
        by_key = {(r["engine"], r["flavor"]): r for r in bdoc["registered"]}
        assert by_key[("nemo_speech", "cuda")]["activatable"] is False
        assert "Quadro P1000" in (by_key[("nemo_speech", "cuda")]["unsupported_reason"] or "")
        assert by_key[("nemo_speech", "cpu")]["activatable"] is True
        assert by_key[("nemo_speech", "cpu")]["unsupported_reason"] is None
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_activate_nemo_speech_cuda_refuses_no_gpu_and_8gb_ampere(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"))
    store.migrate_if_empty(settings)
    _seed_installed(store._conn, tmp_path, "nemo_speech", "cuda", active=0)
    _seed_installed(store._conn, tmp_path, "parakeet", "cuda", version="v0.5.0", active=0)
    _seed_installed(store._conn, tmp_path, "whisper", "cuda", version="v1.8.3-cuda12.1", active=0)

    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]

        # Case 1: no GPU visible at all -> nemo_speech:cuda, parakeet:cuda, whisper:cuda all refused.
        monkeypatch.setattr(gpu, "device_name", lambda: None)
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: None)
        monkeypatch.setattr(gpu, "compute_cap", lambda: None)

        for eng, ver in [
            ("nemo_speech", "v0.3.0"),
            ("parakeet", "v0.5.0"),
            ("whisper", "v1.8.3-cuda12.1"),
        ]:
            status, doc = _post(
                f"http://127.0.0.1:{port}/api/backends/activate",
                {"engine": eng, "flavor": "cuda", "version": ver},
            )
            assert status == 422, (eng, doc)
            assert doc["reason"] == "no_gpu"
            assert f"{eng}:cpu" in doc["suggestion"]
            assert installer.get_active(store._conn, eng) is None

        # Case 2: Ampere GPU with 8 GB VRAM (cc 8.6, 8192 MB) -> refused for nemo_speech:cuda.
        monkeypatch.setattr(gpu, "device_name", lambda: "NVIDIA GeForce RTX 3070")
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: 8192)
        monkeypatch.setattr(gpu, "compute_cap", lambda: 8.6)

        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "nemo_speech", "flavor": "cuda", "version": "v0.3.0"},
        )
        assert status == 422
        assert doc["reason"] == "gpu_unsupported"
        assert "RTX 3070" in doc["error"]
        assert "24 GB" in doc["error"]
        assert doc["suggestion"] == "Activate nemo_speech:cpu instead"
        assert installer.get_active(store._conn, "nemo_speech") is None

        # Case 3: Ampere GPU with 24 GB VRAM (cc 8.6, 24576 MB) -> succeeds.
        monkeypatch.setattr(gpu, "device_name", lambda: "NVIDIA GeForce RTX 3090")
        monkeypatch.setattr(gpu, "total_vram_mb", lambda: 24576)
        monkeypatch.setattr(gpu, "compute_cap", lambda: 8.6)

        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "nemo_speech", "flavor": "cuda", "version": "v0.3.0"},
        )
        assert status == 200
        assert doc["activated"]["flavor"] == "cuda"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_activate_nemo_speech_cpu_checks_avx2_and_warns_on_low_ram(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None, db_path=str(tmp_path / "live_stt.db"), nemotron_cpu_ram_mb=24000)
    store.migrate_if_empty(settings)
    _seed_installed(store._conn, tmp_path, "nemo_speech", "cpu", active=0)

    server = serve_admin_http("127.0.0.1", 0, _state(settings=settings, settings_store=store))
    try:
        port = server.server_address[1]

        # Without AVX2: refused with 422.
        monkeypatch.setattr(nemotron, "cpu_has_avx2", lambda: False)
        monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 48000)
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "nemo_speech", "flavor": "cpu", "version": "v0.3.0"},
        )
        assert status == 422
        assert doc["reason"] == "cpu_unsupported"
        assert "AVX2" in doc["error"]
        assert installer.get_active(store._conn, "nemo_speech") is None

        # With AVX2 and low RAM: allowed with warning.
        monkeypatch.setattr(nemotron, "cpu_has_avx2", lambda: True)
        monkeypatch.setattr(nemotron, "available_ram_mb", lambda: 16000)
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/backends/activate",
            {"engine": "nemo_speech", "flavor": "cpu", "version": "v0.3.0"},
        )
        assert status == 200
        assert doc["activated"]["flavor"] == "cpu"
        assert "warning" in doc and "16000 MiB" in doc["warning"]
        active = installer.get_active(store._conn, "nemo_speech")
        assert active is not None and active.flavor == "cpu"
    finally:
        server.shutdown()
