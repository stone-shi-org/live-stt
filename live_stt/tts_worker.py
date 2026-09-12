"""Standalone subprocess worker for TTS synthesis.

Executed via ``python -m live_stt.tts_worker <request_path> <response_path>``.
Running TTS synthesis in an isolated subprocess ensures that PyTorch/CUDA driver context
and transient allocated VRAM are 100% reclaimed by the operating system upon exit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from live_stt import tts_models
from live_stt.tts import (
    TTSBadRequestError,
    TTSError,
    TTSUnavailableError,
    _synthesize_chunk_direct,
    chunk_text,
)


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python -m live_stt.tts_worker <request_json_path> <response_json_path>",
            file=sys.stderr,
        )
        sys.exit(2)

    req_path = Path(sys.argv[1])
    res_path = Path(sys.argv[2])

    try:
        req_data = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error": f"Failed to read request file: {exc}"}),
            encoding="utf-8",
        )
        sys.exit(1)

    text = req_data.get("text", "")
    model_key = req_data.get("model_key")
    voice = req_data.get("voice", "af_heart")
    speed = float(req_data.get("speed", 1.0))
    device = req_data.get("device", "cpu")
    out_pcm_path = Path(req_data["out_pcm_path"])

    try:
        model_spec = tts_models.resolve(model_key)
        chunks = chunk_text(text)
        pcm_acc = bytearray()
        for chunk in chunks:
            pcm_acc.extend(
                _synthesize_chunk_direct(
                    chunk,
                    model_spec=model_spec,
                    voice=voice,
                    speed=speed,
                    device=device,
                )
            )
        out_pcm_path.write_bytes(bytes(pcm_acc))
        res_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    except TTSBadRequestError as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error_type": "TTSBadRequestError", "error": str(exc)}),
            encoding="utf-8",
        )
        sys.exit(1)
    except TTSUnavailableError as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error_type": "TTSUnavailableError", "error": str(exc)}),
            encoding="utf-8",
        )
        sys.exit(1)
    except TTSError as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error_type": "TTSError", "error": str(exc)}),
            encoding="utf-8",
        )
        sys.exit(1)
    except Exception as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error_type": "Exception", "error": str(exc)}),
            encoding="utf-8",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
