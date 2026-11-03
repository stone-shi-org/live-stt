"""Stand-in for NeMo-Speech.cpp's ``nemo-speech serve`` (LS-19) -- a real
subprocess speaking real HTTP on 127.0.0.1, no binary, no model, no GPU.

Accepts the same command line ``live_stt/nemotron.py`` builds
(``serve --asr-model ... --port N --config cfg.yaml ...``) and serves
``GET /ready`` and ``POST /v1/chat/completions`` in the response shape seen
from the real server on homesrv3 (``choices[0].message.content`` plus
``transcription.words`` / ``transcription.word_timestamp_alignment``).

**Synthetic audio convention** (so windowed jobs can be checked end to end):
the WAV is read one second at a time and the first sample of each second is
taken as ``k + 1``, where ``k`` is that second's ABSOLUTE index in the
original recording. The fake emits one word ``w<k>`` per second (every
``FAKE_NEMO_WORD_EVERY`` seconds), spoken by true speaker
``(k // FAKE_NEMO_TURN_SEC) % FAKE_NEMO_SPEAKERS``, and labels it with a
window-local label ``spk:<(true + shift) % n>`` where ``shift`` depends on
the window's first absolute second -- so each window's labels are a
different permutation, as real independent windows' are.

Behaviour knobs (environment):
  FAKE_NEMO_LOG            append one JSON line per event (start/ready/chat/term)
  FAKE_NEMO_READY_DELAY    seconds before /ready reports ready (default 0.2)
  FAKE_NEMO_CHAT_DELAY     seconds each chat completion takes (default 0)
  FAKE_NEMO_CRASH_ON_CHAT  "1": the process dies mid-request (compute-buffer OOM)
  FAKE_NEMO_ALIGN_FAIL     "1": text but no aligned word timestamps (the 55-min case)
  FAKE_NEMO_EXIT_BEFORE_READY "1": exit 3 during load
  FAKE_NEMO_NO_TURNS       "1": no diarization_timestamps (segments fall back to word runs)
  FAKE_NEMO_SPEAKERS / FAKE_NEMO_TURN_SEC / FAKE_NEMO_WORD_EVERY
"""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STARTED = time.monotonic()
# Like the real server (measured on homesrv3): SIGTERM while a request is in
# flight does not exit promptly -- here it is deferred until the request ends.
_busy = 0
_term_pending = False
_busy_lock = threading.Lock()
LOG = os.environ.get("FAKE_NEMO_LOG")
_log_lock = threading.Lock()


def log(event: dict) -> None:
    if not LOG:
        return
    event = {"pid": os.getpid(), **event}
    with _log_lock, open(LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def parse_args(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    it = iter(argv)
    for a in it:
        if a.startswith("--"):
            if a in ("--no-ui",):
                out[a[2:]] = "1"
            else:
                out[a[2:]] = next(it, "")
    return out


def transcribe(wav_bytes: bytes) -> dict:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate, n = wf.getframerate(), wf.getnframes()
        frames = wf.readframes(n)
    n_spk = int(env_f("FAKE_NEMO_SPEAKERS", 3))
    turn = env_f("FAKE_NEMO_TURN_SEC", 60)
    every = max(1, int(env_f("FAKE_NEMO_WORD_EVERY", 1)))
    secs = n // rate
    abs_secs = []
    for i in range(secs):
        v = int.from_bytes(frames[i * rate * 2 : i * rate * 2 + 2], "little", signed=True)
        abs_secs.append(v - 1)
    first = abs_secs[0] if abs_secs else 0
    shift = (first // 7) % n_spk
    words = []
    for i, k in enumerate(abs_secs):
        if k < 0 or k % every:
            continue
        true = int(k // turn) % n_spk
        words.append(
            {
                "word": f"w{k}",
                "word_index": len(words),
                "start": round(i + 0.1, 3),
                "end": round(i + 0.5, 3),
                "speaker": f"spk:{(true + shift) % n_spk}",
                "speaker_slot": (true + shift) % n_spk,
            }
        )
    # Speaker turns like the real diarization_timestamps: integer speaker,
    # covering each run of one speaker's seconds (including the gaps the
    # tight word spans leave).
    turns: list[dict] = []
    for i, k in enumerate(abs_secs):
        if k < 0:
            continue
        slot = (int(k // turn) % n_spk + shift) % n_spk
        if turns and turns[-1]["speaker"] == slot and abs(turns[-1]["end"] - i) < 1e-6:
            turns[-1]["end"] = float(i + 1)
        else:
            turns.append({"speaker": slot, "start": float(i), "end": float(i + 1)})
    text = " ".join(f"<{w['speaker']}> {w['word']}" for w in words)
    if os.environ.get("FAKE_NEMO_ALIGN_FAIL") == "1":
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text or "hello"}}],
            "transcription": {
                "text": text or "hello",
                "words": [],
                "word_timestamp_alignment": {"status": "failed", "failure_reason": "cannot allocate 82502 rows"},
            },
        }
    return {
        "object": "chat.completion",
        "model": "salm",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "transcription": {
            "task": "transcribe",
            "duration": n / rate,
            "text": text,
            "words": words,
            "word_timestamp_alignment": {"status": "aligned", "failure_reason": None},
            "diarization_timestamps": [] if os.environ.get("FAKE_NEMO_NO_TURNS") == "1" else turns,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet
        pass

    def _json(self, status: int, doc: dict) -> None:
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/ready":
            ready = time.monotonic() - STARTED >= env_f("FAKE_NEMO_READY_DELAY", 0.2)
            self._json(200, {"ready": ready})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            log({"event": "other_post", "path": self.path})
            self._json(404, {"error": "only chat completions are faked"})
            return
        req = json.loads(raw)
        texts, audio = [], b""
        for m in req.get("messages", []):
            for part in m.get("content", []):
                if part.get("type") == "text":
                    texts.append(part["text"])
                elif part.get("type") == "input_audio":
                    audio = base64.b64decode(part["input_audio"]["data"])
        with wave.open(io.BytesIO(audio), "rb") as wf:
            seconds = wf.getnframes() / wf.getframerate()
        log(
            {
                "event": "chat",
                "path": self.path,
                "texts": texts,
                "timestamp_granularities": req.get("timestamp_granularities"),
                "model": req.get("model"),
                "audio_seconds": seconds,
            }
        )
        if os.environ.get("FAKE_NEMO_CRASH_ON_CHAT") == "1":
            os._exit(1)
        global _busy
        with _busy_lock:
            _busy += 1
        try:
            time.sleep(env_f("FAKE_NEMO_CHAT_DELAY", 0.0))
            self._json(200, transcribe(audio))
        finally:
            with _busy_lock:
                _busy -= 1
                if _term_pending and _busy == 0:
                    log({"event": "term"})
                    os._exit(0)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        print("usage: fake_nemo_speech.py serve [options]", file=sys.stderr)
        sys.exit(2)
    args = parse_args(sys.argv[2:])
    cfg = ""
    if args.get("config"):
        with open(args["config"]) as f:
            cfg = f.read()
    log({"event": "start", "args": args, "config": cfg})
    if os.environ.get("FAKE_NEMO_EXIT_BEFORE_READY") == "1":
        sys.exit(3)
    server = ThreadingHTTPServer((args.get("host", "127.0.0.1"), int(args["port"])), Handler)
    server.daemon_threads = True

    def on_term(signum: int, frame: object) -> None:
        global _term_pending
        with _busy_lock:
            if _busy:
                _term_pending = True
                log({"event": "term_deferred"})
                return
        log({"event": "term"})
        os._exit(0)

    signal.signal(signal.SIGTERM, on_term)
    server.serve_forever()


if __name__ == "__main__":
    main()
