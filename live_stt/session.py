"""CallSession: the RPC IS the session.

one phone call = one gRPC Transcribe stream = one CallSession
  = (Phase 2) one WorkerHandle for the life of the call

No rotation yet -- that's Phase 3's live_stt/pool/supervisor.py. A
CallSession is constructed inside servicer.Transcribe()'s call and lives
only in that coroutine's local scope: there is no registry, no dict, no
lookup-by-id anywhere. That absence is what makes "no session_id" true
rather than aspirational (see the proto's service-level comment).
"""

from __future__ import annotations

from pathlib import Path

from live_stt.config import Settings
from live_stt.events import worker_json_to_events
from live_stt.logging_config import get_logger
from live_stt.models import ModelSpec, strip_language_tag
from live_stt.pb.livestt.v1 import asr_pb2
from live_stt.worker import WorkerError, WorkerHandle

logger = get_logger("session")


class CallSession:
    def __init__(self, settings: Settings, model_spec: ModelSpec, config: asr_pb2.StreamConfig) -> None:
        self._settings = settings
        self._model_spec = model_spec
        self._config = config
        self._worker: WorkerHandle | None = None
        self._fed_samples = 0
        self._buf = bytearray()
        # Coalesce arbitrary client-sized frames to exactly one model chunk
        # before writing to the worker -- the model can't emit anything
        # until a full chunk (plus right context) is buffered anyway, so
        # this costs zero latency and cuts IPC round trips substantially for
        # a client sending small (e.g. 20ms RTP-sized) frames.
        self._chunk_bytes = int(16000 * model_spec.model_chunk_ms / 1000) * 2

    @property
    def audio_offset_sec(self) -> float:
        return self._fed_samples / 16000.0

    @property
    def worker_generations(self) -> int:
        return 1  # always, until Phase 3's rotation exists

    async def start(self) -> asr_pb2.TranscriptionEvent:
        gguf_path = str(Path(self._settings.models_dir) / self._model_spec.gguf_filename)
        ggml_lib_dir = (
            Path(self._settings.worker_ggml_lib_dir) if self._settings.worker_ggml_lib_dir else None
        )
        self._worker = await WorkerHandle.spawn(
            worker_bin=Path(self._settings.worker_bin),
            gguf_path=gguf_path,
            language=self._config.language,
            n_threads=self._settings.n_threads_per_worker,
            ggml_lib_dir=ggml_lib_dir,
        )
        return asr_pb2.TranscriptionEvent(
            ready=asr_pb2.Ready(
                model=self._model_spec.key,
                supports_turn_detection=self._model_spec.has_eou,
                has_punctuation=self._model_spec.has_punctuation,
                model_chunk_ms=self._model_spec.model_chunk_ms,
                accepted_sample_rate_hz=16000,
            )
        )

    async def feed_audio(self, pcm16le: bytes) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "feed_audio called before start()"
        self._buf += pcm16le
        events: list[asr_pb2.TranscriptionEvent] = []
        while len(self._buf) >= self._chunk_bytes:
            chunk = bytes(self._buf[: self._chunk_bytes])
            del self._buf[: self._chunk_bytes]
            doc = await self._worker.feed(chunk)
            self._fed_samples += len(chunk) // 2
            events.extend(self._doc_to_events(doc))
        return events

    async def finalize(self) -> list[asr_pb2.TranscriptionEvent]:
        assert self._worker is not None, "finalize called before start()"
        events: list[asr_pb2.TranscriptionEvent] = []
        if self._buf:
            # n_samples == 0 is legal per parakeet_capi.h, and so is any
            # other length -- feed whatever's left, even a partial chunk.
            doc = await self._worker.feed(bytes(self._buf))
            self._fed_samples += len(self._buf) // 2
            self._buf.clear()
            events.extend(self._doc_to_events(doc))

        doc = await self._worker.finalize()
        text = doc.get("text", "")
        if self._model_spec.strip_language_tag:
            text = strip_language_tag(text)
        words = [
            asr_pb2.Word(
                text=w["w"], start_sec=w["start"], end_sec=w["end"], confidence=w.get("conf", 0.0)
            )
            for w in doc.get("words", [])
        ]
        events.append(
            asr_pb2.TranscriptionEvent(
                final=asr_pb2.Final(
                    text=text,
                    words=words,
                    total_audio_sec=self.audio_offset_sec,
                    worker_generations=self.worker_generations,
                )
            )
        )
        return events

    def _doc_to_events(self, doc: dict) -> list[asr_pb2.TranscriptionEvent]:
        return worker_json_to_events(
            doc,
            audio_offset_sec=self.audio_offset_sec,
            strip_tag=self._model_spec.strip_language_tag,
        )

    async def close(self) -> None:
        """Always SIGKILLs the worker -- see CLAUDE.md: a worker process is
        never reused for a second call, on any exit path, clean or not.
        Safe to call after finalize() (idempotent) and safe to call without
        ever having called finalize() (the client-disconnect path, which
        must NOT finalize -- there is nowhere to send the output).
        """
        if self._worker is not None:
            self._worker.kill()
            try:
                await self._worker.wait_closed()
            except Exception:
                logger.warning("worker did not report an exit code during close()")
