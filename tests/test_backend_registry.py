from __future__ import annotations

import pytest

from live_stt.backends.registry import BACKENDS, resolve, available_flavors


def test_resolve_known_pair_returns_the_real_pinned_spec() -> None:
    spec = resolve("parakeet", "cpu")
    assert spec.repo == "mudler/parakeet.cpp"
    assert spec.tag == "v0.5.0"
    assert len(spec.sha256) == 64


def test_resolve_unknown_pair_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no registered backend"):
        resolve("parakeet", "cuda")


def test_available_flavors_for_parakeet() -> None:
    assert available_flavors("parakeet") == ["cpu"]


def test_available_flavors_for_unknown_engine_is_empty() -> None:
    assert available_flavors("does-not-exist") == []


def test_parakeet_cpu_is_wired_whisper_cpu_is_not() -> None:
    # Real, documented gap (see registry.py's docstring and
    # installer.py's): only parakeet's worker binary is built against a
    # dynamically-linked .so today, so only its entry actually changes
    # behavior when installed+activated.
    assert resolve("parakeet", "cpu").wired is True
    assert resolve("whisper", "cpu").wired is False


def test_no_cuda_flavor_is_registered_yet() -> None:
    # Deliberate -- see registry.py's docstring on why a real cuda entry
    # wasn't added this pass (full download + real GPU needed to verify).
    assert all(flavor != "cuda" for (_, flavor) in BACKENDS)
