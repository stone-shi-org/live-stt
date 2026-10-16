"""Extra pip index args so runtime-installed torch matches the GPU.

PyPI's default torch wheel is now built against CUDA 13, which dropped
Pascal (sm_6x): on a Quadro P1000 it fails with "no kernel image is
available" (diarization/sortformer) and "cuDNN ... not compatible with
devices with SM < 7.5" (pyannote/kokoro). PyTorch's cu126 wheels still ship
sm_60 SASS (runs on sm_61) and cuDNN 9.10, and also cover Turing/Ampere, so
one index serves every GPU this service targets. Verified on the P1000:
torch 2.14.1+cu126, cuDNN conv+LSTM on the device.

pip prefers `X+cu126` over plain `X` when both indexes offer the same version.
Override with LSTT_TORCH_INDEX_URL; set it to the empty string to use PyPI only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"


def pip_index_args() -> list[str]:
    url = os.environ.get("LSTT_TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URL).strip()
    return ["--extra-index-url", url] if url else []


# Engines whose dependency tree contains torch -- only these are sensitive to the
# index above. (piper is onnxruntime, unaffected, so it must never be reinstalled
# just because the torch index changed.)
TORCH_ENGINES = frozenset({"pyannote", "nemo_sortformer", "kokoro"})


def deps_marker(engine: str) -> str:
    """Content written into an engine's `.installed` marker. For torch engines it
    records which torch index the tree was built from, so an install made before
    this module existed (marker "1", i.e. PyPI's CUDA 13 torch that cannot run on
    Pascal) is recognisable as stale instead of being skipped forever."""
    if engine not in TORCH_ENGINES:
        return "1"
    return "torch-index=" + (os.environ.get("LSTT_TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URL).strip() or "pypi")


def marker_is_current(marker: Path, engine: str) -> bool:
    try:
        return marker.read_text(encoding="utf-8").strip() == deps_marker(engine)
    except OSError:
        return False


def swap_in(staged: Path, final: Path) -> None:
    """Replace `final` with the fully-built `staged` directory. The old tree is
    moved aside first and only deleted once the new one is in place, so the
    window with no usable install is two renames, not a whole pip run -- and a
    failed build never reaches this function, leaving the old install intact."""
    old = final.with_name(final.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    staged.rename(final)
    shutil.rmtree(old, ignore_errors=True)
