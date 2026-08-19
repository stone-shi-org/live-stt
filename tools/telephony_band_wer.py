#!/usr/bin/env python3
"""Phase 1 Gate C: does narrowband telephony audio wreck WER?

Both parakeet.cpp streaming models are trained on 16kHz wideband audio
(the published WER figures are all 16kHz OpenASR/FLEURS numbers). G.711
gives 300-3400 Hz and nothing above it, no matter who upsamples afterward.
This is a real product risk independent of everything else in this repo, and
the point of this script is to produce the number, not assume it.

Three arms on the same NOTSOFAR-1 meeting audio:
  1. native   -- 16kHz as recorded (the baseline)
  2. mulaw_8k -- 16k -> 8k -> mu-law encode -> mu-law decode -> 8k -> 16k
                 (the full simulated telephony leg, codec loss AND bandwidth loss)
  3. linear_8k -- 16k -> 8k -> 16k, no mu-law (isolates bandwidth loss alone)

Uses audioop (via the audioop-lts backport) for the mu-law ENCODE step only,
since that direction is needed only by this test harness to simulate "the
telephony leg" -- the service itself never encodes G.711, only decodes it
(live_stt/client/telephony.py), so audioop is deliberately NOT a runtime
dependency of that module.

Usage:
    python tools/telephony_band_wer.py \
        --meeting-dir /data/vmfs/main01a_shared/Download/NOTSOFAR-1/eval_set/240629.1_eval_small_with_GT/MTG/MTG_32089 \
        --model models/realtime_eou_120m-v1-q8_0.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import soxr  # noqa: E402

from live_stt.client.telephony import decode_ulaw, resample_8k_to_16k  # noqa: E402
from tools.leak_curve import DEFAULT_GGML_LIB_DIR, DEFAULT_WORKER_BIN, WorkerHandle  # noqa: E402
from tools.wer import tokenize, word_error_rate  # noqa: E402
from live_stt.framing import FrameType  # noqa: E402


def build_reference_text(gt_transcription_path: Path) -> str:
    with open(gt_transcription_path) as f:
        utterances = json.load(f)
    utterances.sort(key=lambda u: u["start_time"])
    return " ".join(u["text"] for u in utterances)


def load_native_pcm(wav_path: Path) -> bytes:
    wf = wave.open(str(wav_path), "rb")
    assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
    return wf.readframes(wf.getnframes())


def to_mulaw_roundtrip(pcm16_16k: bytes) -> bytes:
    # Imported here, not at module level: this is the only function in the
    # repo that needs audioop (as audioop-lts -- stdlib audioop was removed
    # in Python 3.13), and it exists ONLY to simulate the encode side of a
    # telephony leg for this diagnostic. Keeping the import lazy means
    # `import tools.telephony_band_wer` (which tests/test_telephony_band.py
    # does at collection time, before any skip check runs) doesn't require
    # audioop-lts in every environment this repo's test suite runs in --
    # including the offline-safe unit-test path, which never calls this.
    import audioop

    import numpy as np

    samples_16k = np.frombuffer(pcm16_16k, dtype="<i2")
    samples_8k = soxr.resample(samples_16k, 16000, 8000, quality="HQ").astype("<i2")
    ulaw_bytes = audioop.lin2ulaw(samples_8k.tobytes(), 2)
    decoded_8k = decode_ulaw(ulaw_bytes)
    decoded_16k = resample_8k_to_16k(decoded_8k)
    return decoded_16k.astype("<i2").tobytes()


def to_linear_8k_roundtrip(pcm16_16k: bytes) -> bytes:
    import numpy as np

    samples_16k = np.frombuffer(pcm16_16k, dtype="<i2")
    samples_8k = soxr.resample(samples_16k, 16000, 8000, quality="HQ").astype("<i2")
    samples_16k_back = soxr.resample(samples_8k, 8000, 16000, quality="HQ").astype("<i2")
    return samples_16k_back.astype("<i2").tobytes()


def transcribe(pcm16le: bytes, *, model_path: str, n_threads: int, chunk_ms: int = 160) -> str:
    handle = WorkerHandle(DEFAULT_WORKER_BIN, DEFAULT_GGML_LIB_DIR)
    config = json.dumps({"gguf_path": model_path, "language": "", "n_threads": n_threads})
    handle.send(FrameType.CONFIG, config.encode())
    frame_type, doc = handle.recv_json()
    if frame_type == FrameType.ERROR:
        raise RuntimeError(f"CONFIG failed: {doc}")

    chunk_bytes = int(16000 * chunk_ms / 1000) * 2
    text_parts = []
    for i in range(0, len(pcm16le), chunk_bytes):
        chunk = pcm16le[i : i + chunk_bytes]
        handle.send(FrameType.AUDIO, chunk)
        _, doc = handle.recv_json()
        if doc.get("text"):
            text_parts.append(doc["text"])

    handle.send(FrameType.FINALIZE)
    _, doc = handle.recv_json()
    if doc.get("text"):
        text_parts.append(doc["text"])
    handle.close()
    return "".join(text_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--channel-wav", default="sc_meetup_0/ch0.wav")
    args = parser.parse_args()

    meeting_dir = Path(args.meeting_dir)
    reference_text = build_reference_text(meeting_dir / "gt_transcription.json")
    reference_words = tokenize(reference_text)
    print(f"# reference: {len(reference_words)} words")

    native_pcm = load_native_pcm(meeting_dir / args.channel_wav)
    print(f"# audio: {len(native_pcm) / 2 / 16000:.1f}s native 16kHz")

    arms = {
        "native": native_pcm,
        "mulaw_8k": to_mulaw_roundtrip(native_pcm),
        "linear_8k": to_linear_8k_roundtrip(native_pcm),
    }

    print("arm,wer,hyp_words,ref_words")
    results = {}
    for name, pcm in arms.items():
        hyp_text = transcribe(pcm, model_path=args.model, n_threads=args.n_threads)
        hyp_words = tokenize(hyp_text)
        wer = word_error_rate(reference_words, hyp_words)
        results[name] = wer
        print(f"{name},{wer:.4f},{len(hyp_words)},{len(reference_words)}")
        sys.stdout.flush()

    print(f"\n# delta (mulaw_8k - native): {results['mulaw_8k'] - results['native']:+.4f}")
    print(f"# delta (linear_8k - native): {results['linear_8k'] - results['native']:+.4f}")
    print(f"# codec-loss-only (mulaw_8k - linear_8k): {results['mulaw_8k'] - results['linear_8k']:+.4f}")


if __name__ == "__main__":
    main()
