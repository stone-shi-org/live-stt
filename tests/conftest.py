"""Offline-safety autouse fixtures.

Unit tests must not be able to reach a real model, a real worker binary, or
the network -- see CLAUDE.md's testing philosophy. Integration/model/slow/gpu
-marked tests opt out by requesting the real environment explicitly through
their own fixtures/marks, not by fighting this one.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Must happen here, not in a per-test fixture: grpc-core picks its
    # polling engine ONCE per process, at the first channel/server it
    # creates, and caches that choice for the process's whole lifetime. A
    # fixture that sets this later would be too late if any earlier test
    # already touched grpc. pytest_configure runs before test collection
    # even imports the test modules, so it is early enough. See
    # CLAUDE.md/the Dockerfile for why this is needed at all: the default
    # epoll1 poller has a fork-safety bug that can crash the whole process
    # when a worker subprocess is spawned from within a live grpc.aio
    # server -- reproduced under load in tests/test_servicer.py.
    os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")


@pytest.fixture(autouse=True)
def _offline_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LSTT_MODELS_DIR", "/nonexistent-lstt-model-dir")
    monkeypatch.setenv("LSTT_ALLOW_PII", "false")
    monkeypatch.delenv("LSTT_MODEL_PATH", raising=False)
    # Same reasoning as LSTT_MODEL_PATH above, for the whisper engine's own
    # binary path (live_stt/config.py's worker_bin_whisper) -- a dev host
    # that happens to have a real one built at the default path must not
    # let a "unit" test accidentally exercise it.
    monkeypatch.delenv("LSTT_WORKER_BIN_WHISPER", raising=False)
    # A real HF token lives in this dev host's shell env (~/.secrets/huggingface.env,
    # sourced from .zshenv -- see CLAUDE.md) so ad-hoc diarization runs don't need
    # it typed in. Settings(_env_file=None) only skips the .env FILE, not real
    # process env vars, so without this a Settings()-defaults test here would
    # silently pass or fail depending on which machine/shell it runs in --
    # confirmed the hard way when installing requirements-diarization.txt for a
    # real end-to-end test made test_diarization_defaults fail on this exact host.
    monkeypatch.delenv("LSTT_DIARIZATION_HF_TOKEN", raising=False)
    # LS-5: Settings.db_path/backends_dir default to /app/... (the real
    # container layout) -- a real bug this fixture didn't catch: a test
    # that forgets to override db_path only fails in SOME environments
    # (passed locally and in the Docker test-unit image, where the process
    # runs as root and can create /app/data; failed for real on the bare
    # Bamboo CI agent, a non-root native pytest run where it can't -- see
    # LS-5's Bamboo investigation). A plain "/nonexistent-..." path
    # wouldn't reliably catch this either: root can still mkdir a brand
    # new top-level directory. Pointing at a path THROUGH an existing
    # regular file (/etc/passwd, present on every Linux box including
    # minimal container images) makes os.makedirs() fail with
    # NotADirectoryError deterministically regardless of privilege level --
    # so a test that forgets to override these now fails the SAME way
    # everywhere, not just on whichever environment happens to be stricter.
    monkeypatch.setenv("LSTT_DB_PATH", "/etc/passwd/lstt-tests-must-override-db-path/live_stt.db")
    monkeypatch.setenv("LSTT_BACKENDS_DIR", "/etc/passwd/lstt-tests-must-override-backends-dir")


@pytest.fixture()
def fixtures_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(autouse=True)
def _clear_worker_pool_fixture():
    from live_stt.worker_pool import clear_worker_pool

    clear_worker_pool()
    yield
    clear_worker_pool()
