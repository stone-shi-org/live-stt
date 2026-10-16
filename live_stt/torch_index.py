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

DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"


def pip_index_args() -> list[str]:
    url = os.environ.get("LSTT_TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URL).strip()
    return ["--extra-index-url", url] if url else []
