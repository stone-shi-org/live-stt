"""version.txt reader. Written by build.sh at image build time (hash,
timestamp, parakeet_ref, whisper_ref -- see that script). Missing outside a
built image (e.g. `python run.py` from a checkout), in which case everything
reads "dev".

whisper_ref is recorded here for the same traceability reason parakeet_ref
is, but -- unlike parakeet_ref -- is NOT currently threaded through
GetServerInfo/the admin dashboard/Prometheus labels (all three are baked
into the ServerInfoResponse proto message, which would need regenerating);
it is only readable via version.txt / this function today. Extend those
three the same way parakeet_ref is wired in (see live_stt/servicer.py,
live_stt/session.py, live_stt/metrics.py, live_stt/admin_http.py) if
whisper's vendored SHA ever needs to be visible at runtime, not just in the
image.
"""

from __future__ import annotations

import os

_VERSION_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "version.txt")


def info() -> dict[str, str]:
    fields = {"hash": "dev", "timestamp": "dev", "parakeet_ref": "dev", "whisper_ref": "dev"}
    try:
        with open(_VERSION_FILE) as f:
            for line in f:
                if "=" in line:
                    key, _, value = line.strip().partition("=")
                    fields[key] = value
    except FileNotFoundError:
        pass
    return fields
