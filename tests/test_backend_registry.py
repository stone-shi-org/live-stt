from __future__ import annotations

import pytest

from live_stt.backends.registry import BACKENDS, ENGINE_ALIASES, available_flavors, canonical_engine, resolve


def test_resolve_known_pair_returns_the_real_pinned_spec() -> None:
    spec = resolve("parakeet", "cpu")
    assert spec.source == "github_release"
    assert spec.repo == "mudler/parakeet.cpp"
    assert spec.tag == "v0.5.0"
    assert len(spec.sha256) == 64


def test_resolve_unknown_pair_raises_key_error() -> None:
    with pytest.raises(KeyError, match="no registered backend"):
        resolve("parakeet", "vulkan")


def test_parakeet_cuda_is_registered_and_self_contained_source() -> None:
    spec = resolve("parakeet", "cuda")
    assert spec.source == "github_release"
    assert spec.tag == "v0.5.0"
    assert len(spec.sha256) == 64
    assert spec.sha256 != resolve("parakeet", "cpu").sha256


def test_whisper_cpu_is_a_github_release_with_headers_fetched_separately() -> None:
    spec = resolve("whisper", "cpu")
    assert spec.source == "github_release"
    assert spec.repo == "ggerganov/whisper.cpp"
    assert "whisper.h" in spec.header_urls
    assert "ggml.h" in spec.header_urls


def test_whisper_cuda_is_an_oci_image_source_pinned_by_digest() -> None:
    spec = resolve("whisper", "cuda")
    assert spec.source == "oci_image"
    assert spec.image_digest.startswith("sha256:")
    assert spec.tag == ""  # no floating tag pin for oci_image -- digest only
    assert spec.pinned_version == spec.image_digest
    # Every image_files entry must have a real-looking sha256 and a real
    # absolute in-image path.
    assert len(spec.image_files) > 0
    for path_in_image, sha256 in spec.image_files.values():
        assert path_in_image.startswith("/")
        assert len(sha256) == 64
    # The SONAME symlink chain installer.py must create -- every symlink
    # target must correspond to a file this spec actually fetches.
    fetched_filenames = set(spec.image_files)
    for target in spec.symlinks.values():
        assert target in fetched_filenames


def test_available_flavors_for_parakeet() -> None:
    assert available_flavors("parakeet") == ["cpu", "cuda"]


def test_available_flavors_for_unknown_engine_is_empty() -> None:
    assert available_flavors("does-not-exist") == []


def test_all_registered_backends_are_wired() -> None:
    # As of LS-5 Phase 7d, every registered entry is actually consumed by
    # worker/CMakeLists.txt + live_stt/session.py's runtime dispatch -- the
    # `wired` field is kept as an explicit place to land a future
    # registered-but-not-yet-consumed entry, not because one exists today.
    assert all(spec.wired for spec in BACKENDS.values())


def test_parakeet_tdt_aliases_to_parakeet_not_a_separate_registry_entry() -> None:
    # parakeet_tdt (live_stt_worker_parakeet_tdt) links the exact same
    # `parakeet` CMake target as the streaming engine -- see
    # worker/CMakeLists.txt -- so it deliberately reuses "parakeet"'s
    # registry entry rather than duplicating it.
    assert ENGINE_ALIASES["parakeet_tdt"] == "parakeet"
    assert canonical_engine("parakeet_tdt") == "parakeet"
    assert resolve("parakeet_tdt", "cpu") == resolve("parakeet", "cpu")
    assert ("parakeet_tdt", "cpu") not in BACKENDS


def test_canonical_engine_is_a_noop_for_non_aliased_engines() -> None:
    assert canonical_engine("parakeet") == "parakeet"
    assert canonical_engine("whisper") == "whisper"
