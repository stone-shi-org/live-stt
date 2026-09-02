"""Standalone subprocess worker for speaker diarization.

Executed via ``python -m live_stt.diarize_worker <request_path> <response_path>``.
Running diarization in a one-shot subprocess ensures that PyTorch/CUDA driver context
and all allocated VRAM (~398 MiB baseline + transient tensors) are 100% reclaimed
by the operating system upon process exit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from live_stt.config import Settings
from live_stt.diarization import DiarizationError, _diarize_file_direct


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m live_stt.diarize_worker <request_json_path> <response_json_path>", file=sys.stderr)
        sys.exit(2)

    req_path = Path(sys.argv[1])
    res_path = Path(sys.argv[2])

    try:
        req_data = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        res_path.write_text(json.dumps({"ok": False, "error": f"Failed to read request file: {exc}"}), encoding="utf-8")
        sys.exit(1)

    wav_path = req_data["wav_path"]
    settings_dict = req_data.get("settings", {})
    settings = Settings(_env_file=None, **settings_dict)

    try:
        result = _diarize_file_direct(wav_path, settings=settings)
        res_path.write_text(json.dumps({"ok": True, "result": result}), encoding="utf-8")
    except DiarizationError as exc:
        res_path.write_text(json.dumps({"ok": False, "error_type": "DiarizationError", "error": str(exc)}), encoding="utf-8")
        sys.exit(1)
    except Exception as exc:
        res_path.write_text(json.dumps({"ok": False, "error_type": "Exception", "error": str(exc)}), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
