"""version.txt reader. Written by build.sh at image build time (hash,
timestamp, parakeet_ref -- see that script). Missing outside a built image
(e.g. `python run.py` from a checkout), in which case everything reads "dev".
"""

from __future__ import annotations

import os

_VERSION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")


def info() -> dict[str, str]:
    fields = {"hash": "dev", "timestamp": "dev", "parakeet_ref": "dev"}
    try:
        with open(_VERSION_FILE) as f:
            for line in f:
                if "=" in line:
                    key, _, value = line.strip().partition("=")
                    fields[key] = value
    except FileNotFoundError:
        pass
    return fields
