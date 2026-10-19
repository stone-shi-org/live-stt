#!/usr/bin/env python3
"""A real Python subprocess speaking the REAL IPC protocol on fd 3, driven by
env vars, standing in for worker/build/live_stt_worker in tests that need a
controllable worker without a C++ toolchain or a real GGUF.

This is what makes live_stt/session.py and live_stt/servicer.py testable at
all without the real binary: every path that depends on process/signal
semantics (crash, hang, SIGKILL cleanup, RSS growth) is exercised against a
REAL process and REAL signals, just not a real model. Matches the house
precedent for hand-rolled fakes over mock-patching (e.g.
my-meeting-notes/tests/test_live_caption.py's FakeRealtimeConnection) --
mock-patching would be worthless here since the bugs this guards against
live in process/fd/signal semantics, not Python call graphs.

Env vars (all optional):
    FAKE_LOAD_DELAY_SEC     sleep this long before sending READY
    FAKE_CONFIG_ERROR       if set (any value), respond ERROR to CONFIG
    FAKE_WORDS_PER_SEC      emit one synthetic word per this much audio fed
    FAKE_RTF                sleep rtf * audio_chunk_sec per feed (simulates
                             decode taking real compute time)
    FAKE_LEAK_MB_PER_SEC    actually allocate (and retain) this many MB per
                             audio-second fed, so a real RSS watchdog test
                             has something real to observe
    FAKE_CRASH_AFTER_SEC    os._exit(1) once this much audio has been fed
    FAKE_ABORT_AFTER_SEC    os.abort() (SIGABRT) once this much audio has
                             been fed
    FAKE_HANG_AFTER_SEC     stop responding (but stay alive) once this much
                             audio has been fed
    FAKE_EOU_EVERY_SEC      fire a synthetic <EOU> (eou=1, plus an "eou"
                             entry in "events") every time this much audio
                             has been fed since the last one -- for testing
                             rotation's early-cutover-at-EOU path
    FAKE_BATCH_WORDS_PER_SEC buffer AUDIO frames per window and emit words
                             only on FINALIZE with window-relative timestamps
                             [0, window_sec], clearing the window buffer on
                             each FINALIZE (mirrors ParakeetTdtSession and
                             WhisperSession). If a word's starting int16
                             sample is non-zero (uint16 tag), emits "w{tag}"
                             so overlapping windows over tagged PCM produce
                             matching words in the overlap region.
    FAKE_CUDA_OOM_MIB       on FINALIZE, print a ggml cudaMalloc OOM line for
                             this many MiB to stderr and exit(1)
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from live_stt.framing import FrameType, HEADER_SIZE, pack, parse_header  # noqa: E402

IPC_FD = 3


def _read_frame() -> tuple[FrameType, bytes]:
    header = _read_exact(HEADER_SIZE)
    payload_len, frame_type = parse_header(header)
    payload = _read_exact(payload_len) if payload_len else b""
    return frame_type, payload


def _read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = os.read(IPC_FD, n - len(buf))
        if not chunk:
            sys.exit(0)  # peer closed -- normal end of a generation
        buf += chunk
    return buf


def _write_frame(frame_type: FrameType, payload: bytes = b"") -> None:
    os.write(IPC_FD, pack(frame_type, payload))


def main() -> None:
    load_delay = float(os.environ.get("FAKE_LOAD_DELAY_SEC", "0"))
    words_per_sec = float(os.environ.get("FAKE_WORDS_PER_SEC", "0"))
    batch_words_per_sec = float(os.environ.get("FAKE_BATCH_WORDS_PER_SEC", "0"))
    cuda_oom_mib = os.environ.get("FAKE_CUDA_OOM_MIB")
    rtf = float(os.environ.get("FAKE_RTF", "0"))
    leak_mb_per_sec = float(os.environ.get("FAKE_LEAK_MB_PER_SEC", "0"))
    crash_after_sec = os.environ.get("FAKE_CRASH_AFTER_SEC")
    abort_after_sec = os.environ.get("FAKE_ABORT_AFTER_SEC")
    hang_after_sec = os.environ.get("FAKE_HANG_AFTER_SEC")
    eou_every_sec = os.environ.get("FAKE_EOU_EVERY_SEC")

    frame_type, payload = _read_frame()
    assert frame_type == FrameType.CONFIG
    config = json.loads(payload.decode())

    if load_delay:
        time.sleep(load_delay)

    if os.environ.get("FAKE_CONFIG_ERROR"):
        _write_frame(FrameType.ERROR, json.dumps({"error": "fake CONFIG error"}).encode())
        return

    _write_frame(
        FrameType.READY,
        json.dumps(
            {
                "abi_version": 6,
                "n_threads": config.get("n_threads", 1),
                "ggml_features": "fake",
                # Echoed back purely so tests can assert on what
                # WorkerHandle.spawn() actually put in the CONFIG frame
                # (see tests/test_session.py's use_gpu dispatch test) --
                # not read by any real worker's own CONFIG contract.
                "use_gpu_received": config.get("use_gpu", False),
                # LS-5 Phase 7b: echoes the LD_LIBRARY_PATH env var this
                # process actually got, so tests can assert on
                # CallSession._spawn_worker's ggml_lib_dir choice (active
                # installed backend vs. the image-baked default) the same
                # real-subprocess way use_gpu_received above already does,
                # rather than mocking WorkerHandle.spawn directly.
                "ld_library_path_received": os.environ.get("LD_LIBRARY_PATH"),
                # Same reasoning, for GGML_BACKEND_PATH (see
                # WorkerHandle.spawn()'s own comment on why this exists
                # SEPARATELY from LD_LIBRARY_PATH -- ggml's own
                # ggml_backend_load_all() dynamic-backend-discovery
                # mechanism does not consult LD_LIBRARY_PATH at all, so a
                # runtime-installed cuda flavor of the whisper engine needs
                # this instead. Only ever set for whisper+cuda in real
                # session.py code, but this fake worker doesn't care which
                # "engine" it's pretending to be.
                "ggml_backend_path_received": os.environ.get("GGML_BACKEND_PATH"),
                # Same pattern for the TDT engine's attention-window override
                # (live_stt/session.py's _extra_env).
                "att_context_received": os.environ.get("PARAKEET_ATT_CONTEXT"),
            }
        ).encode(),
    )

    fed_samples = 0
    window_samples = 0
    window_buf = bytearray()
    max_window_samples = 0
    finalize_count = 0
    leaked_chunks: list[bytes] = []  # retained on purpose, to simulate a real leak
    word_counter = 0
    words_owed = 0.0
    audio_sec_since_last_eou = 0.0

    def rss_kb() -> int:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except FileNotFoundError:
            pass
        return 0

    while True:
        frame_type, payload = _read_frame()

        if frame_type in (FrameType.AUDIO, FrameType.FINALIZE):
            n_samples = len(payload) // 2 if frame_type == FrameType.AUDIO else 0
            audio_sec_this_feed = n_samples / 16000.0
            fed_samples += n_samples
            window_samples += n_samples
            if frame_type == FrameType.AUDIO and batch_words_per_sec:
                window_buf.extend(payload[: n_samples * 2])
            if window_samples > max_window_samples:
                max_window_samples = window_samples
            fed_audio_sec = fed_samples / 16000.0
            window_audio_sec = window_samples / 16000.0

            if rtf:
                time.sleep(rtf * audio_sec_this_feed)
            if leak_mb_per_sec and audio_sec_this_feed:
                leaked_chunks.append(bytes(int(leak_mb_per_sec * audio_sec_this_feed * 1_000_000)))

            if crash_after_sec is not None and fed_audio_sec >= float(crash_after_sec):
                os._exit(1)
            if abort_after_sec is not None and fed_audio_sec >= float(abort_after_sec):
                os.abort()
            if hang_after_sec is not None and fed_audio_sec >= float(hang_after_sec):
                while True:
                    time.sleep(3600)

            if frame_type == FrameType.FINALIZE and cuda_oom_mib is not None:
                sys.stderr.write(
                    f"ggml_backend_cuda_buffer_type_alloc_buffer: allocating {cuda_oom_mib} MiB "
                    f"on device 0: cudaMalloc failed: out of memory\n"
                    f"ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer\n"
                    f"[parakeet] Backend::compute: ggml_gallocr_alloc_graph failed\n"
                )
                sys.stderr.flush()
                os._exit(1)

            text = ""
            words = []
            if words_per_sec:
                words_owed += audio_sec_this_feed * words_per_sec
                while words_owed >= 1.0:
                    word_counter += 1
                    w = f"word{word_counter}"
                    text += (" " if text else "") + w
                    words.append(
                        {"w": w, "start": window_audio_sec, "end": window_audio_sec, "conf": 0.9}
                    )
                    words_owed -= 1.0

            if frame_type == FrameType.FINALIZE and batch_words_per_sec and window_samples > 0:
                n_words = int(round(window_audio_sec * batch_words_per_sec))
                step_sec = 1.0 / batch_words_per_sec
                for k in range(n_words):
                    w_start = round(k * step_sec, 4)
                    w_end = round(min(window_audio_sec, (k + 0.8) * step_sec), 4)
                    sample_idx = min(int(round(w_start * 16000)), window_samples - 1)
                    tag = 0
                    if sample_idx * 2 + 2 <= len(window_buf):
                        tag = struct.unpack_from("<H", window_buf, sample_idx * 2)[0]
                    if tag > 0:
                        w = f"w{tag}"
                    else:
                        word_counter += 1
                        w = f"word{word_counter}"
                    text += (" " if text else "") + w
                    words.append({"w": w, "start": w_start, "end": w_end, "conf": 0.9})

            eou_fired = 0
            events = []
            if eou_every_sec:
                audio_sec_since_last_eou += audio_sec_this_feed
                threshold = float(eou_every_sec)
                if threshold > 0 and audio_sec_since_last_eou >= threshold:
                    eou_fired = 1
                    events.append({"type": "eou", "frame": 0, "t": window_audio_sec})
                    audio_sec_since_last_eou = 0.0

            if frame_type == FrameType.FINALIZE:
                finalize_count += 1
                window_samples = 0
                window_buf.clear()
                words_owed = 0.0

            doc = {
                "rss_kb": rss_kb(),
                "fed_samples": fed_samples,
                "max_window_samples": max_window_samples,
                "finalize_count": finalize_count,
                "text": text,
                "eou": eou_fired,
                "eob": 0,
                "frame_sec": 0.08,
                "events": events,
                "words": words,
            }
            response_type = FrameType.FINAL if frame_type == FrameType.FINALIZE else FrameType.RESULT
            _write_frame(response_type, json.dumps(doc).encode())

        elif frame_type == FrameType.PING:
            _write_frame(
                FrameType.RESULT,
                json.dumps(
                    {
                        "rss_kb": rss_kb(),
                        "fed_samples": fed_samples,
                        "text": "",
                        "eou": 0,
                        "eob": 0,
                        "frame_sec": 0.08,
                        "events": [],
                        "words": [],
                    }
                ).encode(),
            )
        else:
            _write_frame(FrameType.ERROR, json.dumps({"error": "unexpected frame"}).encode())
            return


if __name__ == "__main__":
    main()
