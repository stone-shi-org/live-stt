#!/usr/bin/env python3
"""Post-call speaker diarization: run pyannote.audio over a recorded WAV and
print the house JSON shape (see live_stt/diarization.py's module docstring
for why this is batch-only and why the shape mirrors
my-meeting-notes/app/services/diarize.py rather than pyannote's native one).

Requires requirements-diarization.txt installed (torch/torchaudio/
pyannote.audio -- NOT part of the runtime image, see that file) and a
HuggingFace token with access accepted for the gated
pyannote/speaker-diarization-community-1 model.

The token is read from LSTT_DIARIZATION_HF_TOKEN (Settings is plain
pydantic-settings, env_prefix="LSTT_", same as every other config value in
this repo) -- NOT passed on the command line. A CLI flag would land in `ps`
output and shell history for the lifetime of the process and the history
file respectively; an env var set in the calling shell does not. --hf-token
still exists below as an escape hatch for one-off/CI use where that's
already been weighed against the alternative, not as the documented path.

Usage:
    export LSTT_DIARIZATION_HF_TOKEN="$HF_TOKEN"
    python tools/diarize_call.py --wav call.wav

    # With per-word ASR timestamps merged in as segment text (same shape
    # session.py already produces, dumped here as a flat JSON list):
    python tools/diarize_call.py --wav call.wav \\
        --words words.json --out diarization.json

``words.json`` shape: ``[{"text": "hello", "start_sec": 0.48, "end_sec": 0.64}, ...]``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from live_stt.config import Settings  # noqa: E402
from live_stt.diarization import DiarizationError, diarize_file  # noqa: E402
from live_stt.pb.livestt.v1 import asr_pb2  # noqa: E402


def _load_words(path: str | None) -> list[asr_pb2.Word]:
    if not path:
        return []
    raw = json.loads(Path(path).read_text())
    return [
        asr_pb2.Word(text=w["text"], start_sec=w["start_sec"], end_sec=w["end_sec"])
        for w in raw
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav", required=True, help="recorded call audio, 16kHz mono")
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "escape hatch only -- prefer LSTT_DIARIZATION_HF_TOKEN (env var). "
            "A token passed here is visible in `ps` output and shell history "
            "for as long as either persists."
        ),
    )
    parser.add_argument("--model", default=None, help="overrides LSTT_DIARIZATION_MODEL")
    parser.add_argument("--num-speakers", type=int, default=None, help="overrides LSTT_DIARIZATION_NUM_SPEAKERS")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="overrides LSTT_DIARIZATION_DEVICE (default cpu; cuda needs a visible CUDA GPU/driver)",
    )
    parser.add_argument("--words", default=None, help="path to a words.json to merge in as segment text")
    parser.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = parser.parse_args()

    settings = Settings(_env_file=None)
    if args.hf_token:
        settings = settings.model_copy(update={"diarization_hf_token": args.hf_token})
    if args.model:
        settings = settings.model_copy(update={"diarization_model": args.model})
    if args.num_speakers is not None:
        settings = settings.model_copy(update={"diarization_num_speakers": args.num_speakers})
    if args.device:
        settings = settings.model_copy(update={"diarization_device": args.device})

    try:
        words = _load_words(args.words)
        result = diarize_file(args.wav, settings=settings, words=words)
    except DiarizationError as exc:
        print(f"diarization failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(output)
        print(f"wrote {args.out}: {result['num_speakers']} speakers, {len(result['segments'])} segments")
    else:
        print(output)


if __name__ == "__main__":
    main()
