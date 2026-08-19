#!/usr/bin/env python3
"""Manual Phase-2 end-to-end smoke test: real gRPC client -> real server ->
real CallSession -> real worker -> real model, streaming a real WAV file in
small chunks. Not a pytest test -- see tests/test_servicer.py for that (to be
written); this is the fastest way to eyeball the whole chain."""

from __future__ import annotations

import asyncio
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live_stt.client.asr_client import ASRClient  # noqa: E402
from live_stt.pb.livestt.v1 import asr_pb2  # noqa: E402

TARGET = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:50161"
WAV_PATH = "/data/homes/stoneshi/src/transcript/output.wav"
CHUNK_MS = 20  # deliberately NOT the model's 160ms chunk -- exercises coalescing


async def audio_chunks():
    wf = wave.open(WAV_PATH, "rb")
    chunk_samples = int(wf.getframerate() * CHUNK_MS / 1000)
    n_chunks = int(20 / (CHUNK_MS / 1000))  # ~20s of audio
    for _ in range(n_chunks):
        pcm = wf.readframes(chunk_samples)
        if not pcm:
            break
        yield pcm


async def main() -> None:
    client = ASRClient(TARGET)
    info = await client.get_server_info()
    print(f"server: {info}")

    config = asr_pb2.StreamConfig(
        call_id="smoke-test",
        encoding=asr_pb2.AUDIO_ENCODING_LINEAR16,
        sample_rate_hz=16000,
    )
    full_text = ""
    async for event in client.transcribe(config, audio_chunks()):
        kind = event.WhichOneof("event")
        if kind == "ready":
            print(f"READY: {event.ready}")
        elif kind == "delta":
            full_text += event.delta.text
            print(f"DELTA: {event.delta.text!r} (offset={event.delta.audio_offset_sec:.2f}s)")
        elif kind == "final":
            full_text += event.final.text
            print(f"FINAL: {event.final}")
        else:
            print(f"{kind.upper()}: {getattr(event, kind)}")

    print("=" * 60)
    print("full transcript:", full_text)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
