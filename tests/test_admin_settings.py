"""LS-5: /settings dashboard page + GET/PUT /api/settings, against a real
socket and a real serve_admin_http() server -- same pattern
tests/test_metrics.py already uses for /api/config, not a mock.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
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


def _put(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


@pytest.mark.asyncio
async def test_settings_page_serves_html() -> None:
    server = serve_admin_http("127.0.0.1", 0, _state())
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/settings", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read().decode()
            assert "settingsBody" in body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_settings_page_includes_diarization_engines_install_ui() -> None:
    # Mirrors the pre-existing "Engine Backends" install/activate UI on
    # this same page -- pyannote/nemo_sortformer are installable at
    # runtime (live_stt/diarization_installer.py) the same way ASR cuda
    # flavors are, and an operator needs a UI to trigger it, not just curl.
    server = serve_admin_http("127.0.0.1", 0, _state())
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/settings", timeout=5) as resp:
            body = resp.read().decode()
            assert "diarizationBody" in body
            assert "installDiarizationEngine" in body
            assert "/api/diarization-engines/install" in body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_settings_page_includes_tts_engines_install_ui() -> None:
    # Same reasoning as the diarization-engines UI test above, for
    # live_stt/tts_installer.py's kokoro/piper install endpoints.
    server = serve_admin_http("127.0.0.1", 0, _state())
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/settings", timeout=5) as resp:
            body = resp.read().decode()
            assert "ttsBody" in body
            assert "installTtsEngine" in body
            assert "/api/tts-engines/install" in body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_dashboard_ui_multi_tab_navigation_and_sectioned_settings() -> None:
    # LS-6: Multi-page tabbed navigation (Running Status, Models & Endpoints,
    # and Configuration & Settings) with search and sectioned categories.
    server = serve_admin_http("127.0.0.1", 0, _state())
    try:
        port = server.server_address[1]
        for path in ("/", "/dashboard", "/stats", "/settings"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
                assert resp.status == 200
                body = resp.read().decode()
                # Menu tabs
                assert 'id="nav-status"' in body
                assert 'id="nav-models"' in body
                assert 'id="nav-settings"' in body
                assert "switchTab(" in body
                # Tab content containers
                assert 'id="tab-status"' in body
                assert 'id="tab-models"' in body
                assert 'id="tab-settings"' in body
                # Settings toolbar & sectioned grouping
                assert 'id="settingsSearchInput"' in body
                assert 'id="categoryPills"' in body
                assert "SETTINGS_SECTIONS" in body
                assert "settingsBody" in body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_get_api_settings_reports_store_unavailable_when_none_configured() -> None:
    server = serve_admin_http("127.0.0.1", 0, _state(settings_store=None))
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/settings", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
            assert doc["store_available"] is False
            assert "parakeet_backend" in doc["settings"]
            assert "tts_voice" in doc["hot_reload_fields"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_put_api_settings_without_a_store_is_503() -> None:
    server = serve_admin_http("127.0.0.1", 0, _state(settings_store=None))
    try:
        port = server.server_address[1]
        status, doc = _put(f"http://127.0.0.1:{port}/api/settings", {"tts_voice": "af_sky"})
        assert status == 503
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_put_api_settings_applies_a_hot_field_immediately() -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None)
    store.migrate_if_empty(settings)
    state = _state(settings=settings, settings_store=store)
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        status, doc = _put(f"http://127.0.0.1:{port}/api/settings", {"tts_voice": "af_sky"})
        assert status == 200
        assert doc["applied_hot"] == ["tts_voice"]
        assert doc["pending_restart"] == []
        # Mutated on the LIVE settings object, not just persisted.
        assert state.settings.tts_voice == "af_sky"
        assert store.load_overrides()["tts_voice"] == "af_sky"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_put_api_settings_persists_but_does_not_apply_a_restart_required_field() -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None)
    store.migrate_if_empty(settings)
    state = _state(settings=settings, settings_store=store)
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        status, doc = _put(f"http://127.0.0.1:{port}/api/settings", {"max_concurrent_calls": 9})
        assert status == 200
        assert doc["applied_hot"] == []
        assert doc["pending_restart"] == ["max_concurrent_calls"]
        # NOT mutated live -- the running WorkerBudget was already sized off
        # the old value and can't safely be resized out from under it.
        assert state.settings.max_concurrent_calls != 9
        assert store.load_overrides()["max_concurrent_calls"] == 9
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_put_api_settings_unknown_key_is_400() -> None:
    store = SettingsStore(":memory:")
    settings = Settings(_env_file=None)
    store.migrate_if_empty(settings)
    state = _state(settings=settings, settings_store=store)
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        status, doc = _put(f"http://127.0.0.1:{port}/api/settings", {"not_a_real_field": 1})
        assert status == 400
        assert "unknown setting" in doc["error"]
    finally:
        server.shutdown()
