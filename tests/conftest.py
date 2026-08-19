"""Offline-safety autouse fixtures.

Unit tests must not be able to reach a real model, a real worker binary, or
the network -- see CLAUDE.md's testing philosophy. Integration/model/slow/gpu
-marked tests opt out by requesting the real environment explicitly through
their own fixtures/marks, not by fighting this one.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _offline_safety(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LSTT_MODELS_DIR", "/nonexistent-lstt-model-dir")
    monkeypatch.setenv("LSTT_ALLOW_PII", "false")
    monkeypatch.delenv("LSTT_MODEL_PATH", raising=False)


@pytest.fixture()
def fixtures_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures")
