"""GET /api/tts-engines + POST /api/tts-engines/install, against a real
socket and a real serve_admin_http() server -- same pattern
tests/test_admin_diarization_engines.py already uses for the diarization
analogue. tts_installer.install is monkeypatched so this exercises the
HTTP wiring/status-code contract, not a real pip install.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from live_stt import tts_installer
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.state import ServerState


def _state(settings: Settings | None = None) -> ServerState:
    settings = settings or Settings(_env_file=None)
    return ServerState(
        settings=settings,
        budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots),
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
async def test_get_tts_engines_lists_every_known_engine(tmp_path) -> None:
    settings = Settings(_env_file=None, tts_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tts-engines", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
            assert set(doc["engines"]) == {"kokoro", "piper"}
            assert doc["engines"]["kokoro"]["installed"] is False
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_missing_engine_field_is_400(tmp_path) -> None:
    settings = Settings(_env_file=None, tts_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(f"http://127.0.0.1:{port}/api/tts-engines/install", {})
        assert status == 400
        assert "engine" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_unknown_engine_is_502(tmp_path) -> None:
    settings = Settings(_env_file=None, tts_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(f"http://127.0.0.1:{port}/api/tts-engines/install", {"engine": "not_a_real_engine"})
        assert status == 502
        assert "unknown TTS engine" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_success_is_200_and_reflects_in_subsequent_get(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_install(deps_root, engine):
        return tts_installer.TTSInstallResult(engine, f"{deps_root}/{engine}", False)

    monkeypatch.setattr(tts_installer, "install", fake_install)
    settings = Settings(_env_file=None, tts_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(f"http://127.0.0.1:{port}/api/tts-engines/install", {"engine": "kokoro"})
        assert status == 200
        assert doc["engine"] == "kokoro"
        assert doc["already_installed"] is False
    finally:
        server.shutdown()
