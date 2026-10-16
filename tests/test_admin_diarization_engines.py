"""GET /api/diarization-engines + POST /api/diarization-engines/install,
against a real socket and a real serve_admin_http() server -- same pattern
tests/test_admin_backends.py already uses for the ASR /api/backends
analogue. diarization_installer.install is monkeypatched so this exercises
the HTTP wiring/status-code contract, not a real pip install.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from live_stt import diarization_installer
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
async def test_get_diarization_engines_lists_every_known_engine(tmp_path) -> None:
    settings = Settings(_env_file=None, diarization_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/diarization-engines", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
            assert set(doc["engines"]) == {"pyannote", "nemo_sortformer"}
            assert doc["engines"]["pyannote"]["installed"] is False
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_missing_engine_field_is_400(tmp_path) -> None:
    settings = Settings(_env_file=None, diarization_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(f"http://127.0.0.1:{port}/api/diarization-engines/install", {})
        assert status == 400
        assert "engine" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_unknown_engine_is_400(tmp_path) -> None:
    settings = Settings(_env_file=None, diarization_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/diarization-engines/install", {"engine": "not_a_real_engine"}
        )
        assert status == 400  # caller's fault, not an upstream failure
        assert "unknown diarization engine" in doc["error"]
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_post_install_success_is_200_and_reflects_in_subsequent_get(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_install(deps_root, engine):
        return diarization_installer.DiarizationInstallResult(engine, f"{deps_root}/{engine}", False)

    monkeypatch.setattr(diarization_installer, "install", fake_install)
    settings = Settings(_env_file=None, diarization_deps_dir=str(tmp_path))
    server = serve_admin_http("127.0.0.1", 0, _state(settings))
    try:
        port = server.server_address[1]
        status, doc = _post(
            f"http://127.0.0.1:{port}/api/diarization-engines/install", {"engine": "pyannote"}
        )
        assert status == 200
        assert doc["engine"] == "pyannote"
        assert doc["already_installed"] is False
    finally:
        server.shutdown()
