"""Standalone subprocess worker for TTS synthesis.

Executed via ``python -m live_stt.tts_worker <request_path> <response_path>`` (one-shot mode)
or ``python -m live_stt.tts_worker --server [--idle-timeout <seconds>]`` (persistent warm mode).

Running TTS synthesis in an isolated subprocess ensures that PyTorch/CUDA driver context
and transient allocated VRAM are 100% reclaimed by the operating system upon exit.
Persistent server mode keeps the loaded model warm across requests to eliminate the ~5s
cold start for Kokoro, exiting automatically when idle beyond the configured timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
from pathlib import Path

# Isolate IPC protocol stdout from library prints and C-level stdout before any
# heavy module imports (PyTorch, Kokoro, Hugging Face, etc.).
# Duplicating stdout gives us a dedicated file descriptor for IPC communication.
# Redirecting fd 1 and sys.stdout to sys.stderr ensures that any third-party
# print statements, warnings (e.g. Kokoro repo_id warning), or C extensions
# write to stderr without corrupting the NDJSON protocol stream.
_IPC_STDOUT = None
if "--server" in sys.argv:
    try:
        _ipc_stdout_fd = os.dup(sys.stdout.fileno())
        _IPC_STDOUT = os.fdopen(_ipc_stdout_fd, "w", encoding="utf-8", buffering=1)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
    except Exception:
        _IPC_STDOUT = sys.stdout

from live_stt import tts_models
from live_stt.logging_config import get_logger
from live_stt.tts import (
    TTSBadRequestError,
    TTSError,
    TTSUnavailableError,
    _synthesize_chunk_direct,
    chunk_text,
)

logger = get_logger("tts_worker")


def process_request(req_data: dict) -> tuple[bool, dict]:
    """Execute TTS synthesis for a single request dictionary.

    Returns (success, response_dict).
    """
    text = req_data.get("text", "")
    model_key = req_data.get("model_key")
    voice = req_data.get("voice", "af_heart")
    speed = float(req_data.get("speed", 1.0))
    device = req_data.get("device", "cpu")
    models_dir = req_data.get("models_dir", "/models")
    deps_dir = req_data.get("deps_dir", "/app/data/tts-deps")
    idle_timeout_sec = float(req_data.get("idle_timeout_sec", 600.0))
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
                    models_dir=models_dir,
                    deps_dir=deps_dir,
                    idle_timeout_sec=idle_timeout_sec,
                )
            )
        out_pcm_path.write_bytes(bytes(pcm_acc))
        return True, {"ok": True}
    except TTSBadRequestError as exc:
        return False, {"ok": False, "error_type": "TTSBadRequestError", "error": str(exc)}
    except TTSUnavailableError as exc:
        return False, {"ok": False, "error_type": "TTSUnavailableError", "error": str(exc)}
    except TTSError as exc:
        return False, {"ok": False, "error_type": "TTSError", "error": str(exc)}
    except Exception as exc:
        return False, {"ok": False, "error_type": "Exception", "error": str(exc)}


def run_oneshot(req_path: Path, res_path: Path) -> None:
    """Execute a single synthesis request and exit, preserving legacy CLI behavior."""
    try:
        req_data = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as exc:
        res_path.write_text(
            json.dumps({"ok": False, "error": f"Failed to read request file: {exc}"}),
            encoding="utf-8",
        )
        sys.exit(1)

    ok, res_obj = process_request(req_data)
    res_path.write_text(json.dumps(res_obj), encoding="utf-8")
    if not ok:
        sys.exit(1)


def run_server(idle_timeout_sec: float) -> None:
    """Run persistent server loop reading NDJSON from stdin and writing NDJSON to stdout.

    Exits automatically when idle for longer than idle_timeout_sec.
    """
    global _IPC_STDOUT
    if _IPC_STDOUT is None:
        try:
            ipc_stdout_fd = os.dup(sys.stdout.fileno())
            _IPC_STDOUT = os.fdopen(ipc_stdout_fd, "w", encoding="utf-8", buffering=1)
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            sys.stdout = sys.stderr
        except Exception:
            _IPC_STDOUT = sys.stdout
    ipc_stdout = _IPC_STDOUT

    logger.info("TTS persistent worker started (idle_timeout_sec=%.1f)", idle_timeout_sec)
    current_timeout = idle_timeout_sec
    try:
        while True:
            if current_timeout > 0:
                ready, _, _ = select.select([sys.stdin], [], [], current_timeout)
                if not ready:
                    logger.info(
                        "TTS persistent worker idle timeout (%.1fs) expired. Exiting cleanly.",
                        current_timeout,
                    )
                    break

            line = sys.stdin.readline()
            if not line:
                # EOF on stdin (parent closed pipe)
                break
            line = line.strip()
            if not line:
                continue

            try:
                req_data = json.loads(line)
            except Exception as exc:
                res_obj = {
                    "ok": False,
                    "error_type": "TTSBadRequestError",
                    "error": f"Invalid JSON request: {exc}",
                }
                ipc_stdout.write(json.dumps(res_obj) + "\n")
                ipc_stdout.flush()
                continue

            if "idle_timeout_sec" in req_data:
                try:
                    current_timeout = float(req_data["idle_timeout_sec"])
                except (ValueError, TypeError):
                    pass

            ok, res_obj = process_request(req_data)
            ipc_stdout.write(json.dumps(res_obj) + "\n")
            ipc_stdout.flush()
    finally:
        try:
            ipc_stdout.close()
        except Exception:
            pass


def main() -> None:
    if len(sys.argv) == 3 and not sys.argv[1].startswith("--"):
        run_oneshot(Path(sys.argv[1]), Path(sys.argv[2]))
        return

    parser = argparse.ArgumentParser(description="Standalone subprocess worker for TTS synthesis")
    parser.add_argument("--server", action="store_true", help="Run in persistent server mode")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=600.0,
        help="Idle timeout in seconds before clean exit (default: 600.0)",
    )
    args = parser.parse_args()

    if args.server:
        run_server(args.idle_timeout)
    else:
        parser.print_help(sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
