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
