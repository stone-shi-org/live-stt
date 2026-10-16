"""Auto-upgrade check (LS-5 Phase 7c). Unit tests must not touch the
network -- every urllib call and installer.install() are monkeypatched,
matching the offline philosophy tests/test_backend_installer.py already
uses.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from live_stt import db
from live_stt.backends import installer, upgrade_checker
from live_stt.backends.registry import BACKENDS


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_check_for_upgrades_reports_already_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = db.connect(":memory:")
    installer.ensure_schema(conn)
    spec = BACKENDS[("parakeet", "cpu")]
    conn.execute(
        "INSERT INTO installed_backends VALUES (?,?,?,?,?,?,0)",
        ("parakeet", "cpu", spec.tag, spec.sha256, str(tmp_path / "p"), "2024-01-01T00:00:00+00:00"),
    )
    conn.commit()

    # Only parakeet:cpu is seeded as installed above -- every OTHER
    # registered (engine, flavor) pair is "not yet installed", and with
    # auto_upgrade_enabled=True below, check_for_upgrades() would call the
    # REAL installer.install() for each of them (parakeet:cuda, whisper:cpu,
    # whisper:cuda at time of writing) if this weren't mocked. That is a
    # real, large network download plus a real `docker create`/`docker cp`
    # against a real ghcr.io image -- exactly the unmocked call that made
    # this test hang for 20+ minutes on Bamboo's bare CI agent (LIV-LIV-33/
    # 34) until its build-hang detector force-killed the process tree. It
    # didn't reproduce locally because this repo's own sandboxed Docker
    # build blocks/rejects egress fast, so installer.install() failed
    # quickly with BackendInstallError -- which check_for_upgrades already
    # catches and reports as action="failed" -- masking the bug rather than
    # tripping it. Mocking here (matching the correct pattern already used
    # by test_check_for_upgrades_installs_when_missing_and_enabled below,
    # and matching this file's own docstring claim) is the actual fix, not
    # environment-dependent luck.
    def fake_install(conn_, install_root, engine, flavor):
        other_spec = BACKENDS[(engine, flavor)]
        return installer.InstalledBackend(
            engine,
            flavor,
            other_spec.pinned_version,
            other_spec.sha256 or other_spec.image_digest,
            "/fake/path",
            "now",
            False,
        )

    monkeypatch.setattr(installer, "install", fake_install)

    results = upgrade_checker.check_for_upgrades(conn, str(tmp_path), auto_upgrade_enabled=True)
    parakeet_result = next(r for r in results if r.engine == "parakeet" and r.flavor == "cpu")
    assert parakeet_result.action == "already_installed"


def test_check_for_upgrades_skips_when_auto_upgrade_disabled(tmp_path: Path) -> None:
    conn = db.connect(":memory:")
    results = upgrade_checker.check_for_upgrades(conn, str(tmp_path), auto_upgrade_enabled=False)
    assert results  # at least one registered backend
    assert all(r.action == "skipped_auto_upgrade_disabled" for r in results)


def test_check_for_upgrades_installs_when_missing_and_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = db.connect(":memory:")
    calls: list[tuple[str, str]] = []

    def fake_install(conn_, install_root, engine, flavor):
        calls.append((engine, flavor))
        spec = BACKENDS[(engine, flavor)]
        return installer.InstalledBackend(
            engine, flavor, spec.pinned_version, spec.sha256 or spec.image_digest, "/fake/path", "now", False
        )

    monkeypatch.setattr(installer, "install", fake_install)
    results = upgrade_checker.check_for_upgrades(conn, str(tmp_path), auto_upgrade_enabled=True)
    assert all(r.action == "installed" for r in results)
    assert len(calls) == len(BACKENDS)


def test_check_for_upgrades_reports_failed_installs_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conn = db.connect(":memory:")

    def fake_install(conn_, install_root, engine, flavor):
        raise installer.BackendInstallError("network unreachable")

    monkeypatch.setattr(installer, "install", fake_install)
    results = upgrade_checker.check_for_upgrades(conn, str(tmp_path), auto_upgrade_enabled=True)
    assert all(r.action == "failed" for r in results)
    assert all("network unreachable" in r.detail for r in results)


def test_fetch_upstream_latest_tag_returns_none_on_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_urlerror(url, timeout=5.0):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(upgrade_checker.urllib.request, "urlopen", raise_urlerror)
    assert upgrade_checker.fetch_upstream_latest_tag("mudler/parakeet.cpp") is None


def test_fetch_upstream_latest_tag_returns_none_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        upgrade_checker.urllib.request, "urlopen", lambda url, timeout=5.0: _FakeResponse(b"not json")
    )
    assert upgrade_checker.fetch_upstream_latest_tag("mudler/parakeet.cpp") is None


def test_fetch_upstream_latest_tag_parses_a_real_shaped_response(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps({"tag_name": "v0.6.0"}).encode()
    monkeypatch.setattr(upgrade_checker.urllib.request, "urlopen", lambda url, timeout=5.0: _FakeResponse(body))
    assert upgrade_checker.fetch_upstream_latest_tag("mudler/parakeet.cpp") == "v0.6.0"


def test_upstream_visibility_report_covers_every_registered_backend_and_caches_per_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_fetch(repo, timeout=5.0):
        calls.append(repo)
        return "vX"

    monkeypatch.setattr(upgrade_checker, "fetch_upstream_latest_tag", fake_fetch)
    report = upgrade_checker.upstream_visibility_report()

    assert set(report) == {f"{e}:{f}" for (e, f) in BACKENDS}
    github_release_repos = {
        spec.repo for spec in BACKENDS.values() if spec.source == "github_release"
    }
    assert len(calls) == len(github_release_repos)
    for (engine, flavor), spec in BACKENDS.items():
        key = f"{engine}:{flavor}"
        # github_release compares the upstream TAG (the install label can carry a
        # build-variant suffix like "-cuda12" that no release tag ever has)
        assert report[key]["pinned"] == (spec.tag if spec.source == "github_release" else spec.pinned_version)
        if spec.source == "github_release":
            assert report[key]["upstream_latest"] == "vX"
        else:
            # oci_image sources have no "latest release" equivalent to
            # check (see upgrade_checker.py's docstring) -- always None,
            # and fetch_upstream_latest_tag is never called for them.
            assert report[key]["upstream_latest"] is None
