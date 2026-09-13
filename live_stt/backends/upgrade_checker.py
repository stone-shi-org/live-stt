"""Auto-upgrade check (LS-5 Phase 7c): keeps the persistent backend install
directory in sync with live_stt/backends/registry.py's currently PINNED
versions, and separately surfaces (read-only) whether upstream has
published something newer than what's pinned.

Two deliberately different things, per the explicit design constraint from
the LS-5 Jira thread ("pin exact tags the wrapper was verified against,
never latest"):

1. check_for_upgrades(): compares the registry's pinned tag for each
   (engine, flavor) against what's already INSTALLED locally. If they
   differ (an operator shipped a new pinned tag in a code/image update)
   and `auto_upgrade_enabled` is True, downloads (installs) the newly
   pinned version -- but never activates it (see
   live_stt/backends/installer.py's activate(): a separate, explicit
   step, so a bad download never auto-takes-effect). This is
   "auto-upgrade" in the safe sense LS-5 asked for: staying in sync with
   what THIS codebase has already verified, automatically, without a
   manual admin click on every deploy.

2. fetch_upstream_latest_tag() / upstream_visibility_report(): a live,
   best-effort GitHub API read of what upstream currently calls its
   latest release -- informational only. Never installs anything based on
   this by itself; surfacing "upstream has vX.Y.Z, this deployment is
   pinned to v0.5.0" is a signal for a HUMAN to go re-verify and update
   the registry (a code change -- see registry.py's docstring) before
   it's ever installable. Blindly trusting "latest" is exactly the
   failure mode pinning exists to avoid.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass

from live_stt import db
from live_stt.backends import installer
from live_stt.backends.registry import BACKENDS
from live_stt.config import Settings
from live_stt.logging_config import get_logger

logger = get_logger("backends.upgrade_checker")


@dataclass(frozen=True, slots=True)
class UpgradeResult:
    engine: str
    flavor: str
    pinned_tag: str
    action: str  # "installed" | "already_installed" | "skipped_auto_upgrade_disabled" | "failed"
    detail: str = ""


def check_for_upgrades(
    conn: sqlite3.Connection, install_root: str, *, auto_upgrade_enabled: bool
) -> list[UpgradeResult]:
    results: list[UpgradeResult] = []
    for (engine, flavor), spec in sorted(BACKENDS.items()):
        installed = [
            b for b in installer.list_installed(conn) if b.engine == engine and b.flavor == flavor
        ]
        pinned = spec.pinned_version
        if any(b.version == pinned for b in installed):
            results.append(UpgradeResult(engine, flavor, pinned, "already_installed"))
            continue
        if not auto_upgrade_enabled:
            results.append(
                UpgradeResult(
                    engine,
                    flavor,
                    pinned,
                    "skipped_auto_upgrade_disabled",
                    detail="not yet installed locally; enable auto_upgrade_enabled or install manually",
                )
            )
            continue
        try:
            installer.install(conn, install_root, engine, flavor)
            results.append(UpgradeResult(engine, flavor, pinned, "installed"))
        except installer.BackendInstallError as exc:
            results.append(UpgradeResult(engine, flavor, pinned, "failed", detail=str(exc)))
    return results


def fetch_upstream_latest_tag(repo: str, timeout: float = 5.0) -> str | None:
    """GET https://api.github.com/repos/{repo}/releases/latest -- returns
    the tag_name, or None on ANY failure (network, non-200, bad JSON).
    Never raises: this is a periodic background/informational check, not a
    request path, and a GitHub outage or rate limit must not be able to
    break anything that depends on it.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 -- fixed api.github.com URL
            doc = json.loads(resp.read().decode())
        tag = doc.get("tag_name")
        return tag if isinstance(tag, str) else None
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def upstream_visibility_report() -> dict[str, dict[str, str | None]]:
    """{"parakeet:cpu": {"pinned": "v0.5.0", "upstream_latest": ... or None}, ...}
    -- read-only, informational (see this module's docstring). A
    difference here means "a human should go re-verify and re-pin," never
    "auto-install this."

    oci_image-sourced entries (whisper:cuda -- see registry.py's docstring)
    have no GitHub "latest release" equivalent to compare against (GHCR's
    rolling "main-cuda" tag has no meaningful "latest version" the way a
    real release does) -- upstream_latest is always None for those,
    distinguishable from "checked, no answer" by their source field, not
    silently conflated with a real check that happened to fail.
    """
    report: dict[str, dict[str, str | None]] = {}
    seen_repos: dict[str, str | None] = {}
    for (engine, flavor), spec in BACKENDS.items():
        if spec.source != "github_release":
            report[f"{engine}:{flavor}"] = {"pinned": spec.pinned_version, "upstream_latest": None}
            continue
        if spec.repo not in seen_repos:
            seen_repos[spec.repo] = fetch_upstream_latest_tag(spec.repo)
        report[f"{engine}:{flavor}"] = {"pinned": spec.pinned_version, "upstream_latest": seen_repos[spec.repo]}
    return report


async def run_periodic_upgrade_check(settings: Settings) -> None:
    """Background task (live_stt/server.py spawns this once, only on the
    real entrypoint path -- see that module). Checks immediately on
    startup (so a fresh deploy with a new pin doesn't wait a full interval
    before catching up), then sleeps upgrade_check_interval_sec between
    checks. Never lets a single failed check (a locked DB, a transient
    network blip during install) kill the loop or the process -- this is
    maintenance, not a request path.
    """
    while True:
        try:
            conn = db.connect(settings.db_path)
            try:
                results = check_for_upgrades(
                    conn, settings.backends_dir, auto_upgrade_enabled=settings.auto_upgrade_enabled
                )
                for result in results:
                    if result.action == "installed":
                        logger.info(
                            "auto-upgrade installed %s:%s -> %s", result.engine, result.flavor, result.pinned_tag
                        )
                    elif result.action == "failed":
                        logger.warning(
                            "auto-upgrade failed for %s:%s -> %s: %s",
                            result.engine, result.flavor, result.pinned_tag, result.detail,
                        )
            finally:
                conn.close()
        except Exception:  # noqa: BLE001 -- a background maintenance loop must never take the process down
            logger.exception("periodic upgrade check failed")
        await asyncio.sleep(settings.upgrade_check_interval_sec)
