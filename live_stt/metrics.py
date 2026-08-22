"""Prometheus metrics. All metrics live in this one module so a drifted
name/type/label set is a one-file diff to catch (see tests/test_metrics.py).

The estate already runs prom/prometheus and Grafana (home-docker-script's
litellm.yml / ha.yml) -- this adds a tenant rather than standing up new
infrastructure. Exposed at GET /metrics (live_stt/admin_http.py).

No ``stream_id`` (or anything call-identifying) as a label, anywhere:
unbounded cardinality and a PII vector. ``reason``/``outcome``/``kind`` are
small closed enums and are the only labels used.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Defaults top out at 10s, useless for calls that run 30-120 minutes.
STREAM_DURATION_BUCKETS = (60, 300, 900, 1800, 3600, 5400, 7200, 10800)
# .32 is nemotron's realtime budget (320ms/chunk) -- a bucket edge on purpose.
FEED_DURATION_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.32, 0.5, 1, 2)

build_info = Gauge(
    "live_stt_build_info",
    "Build metadata; value is always 1, information is in the labels",
    ["version", "parakeet_ref", "backend", "model", "n_threads", "ggml_features"],
)
streams_active = Gauge("live_stt_streams_active", "Currently active calls")
streams_total = Counter(
    "live_stt_streams_total",
    "Completed calls",
    ["outcome"],  # ok|client_cancel|invalid_argument|resource_exhausted|unavailable|internal
)
stream_duration_seconds = Histogram(
    "live_stt_stream_duration_seconds",
    "Wall-clock call duration",
    buckets=STREAM_DURATION_BUCKETS,
)
audio_seconds_total = Counter("live_stt_audio_seconds_total", "Audio seconds successfully fed")
feed_duration_seconds = Histogram(
    "live_stt_feed_duration_seconds",
    "Per-feed worker round-trip latency",
    buckets=FEED_DURATION_BUCKETS,
)
model_load_duration_seconds = Histogram(
    "live_stt_model_load_duration_seconds", "Worker spawn-to-READY latency"
)
worker_restarts_total = Counter(
    "live_stt_worker_restarts_total",
    "Worker recycle events",
    ["reason"],  # rss_threshold|audio_cap|eou_opportunistic|crash
)
rotations_total = Counter(
    "live_stt_rotations_total", "Completed worker rotations", ["kind"]  # warm|cold
)
stream_init_failures_total = Counter(
    "live_stt_stream_init_failures_total", "Call failed before Ready", ["reason"]
)
asr_errors_total = Counter(
    "live_stt_asr_errors_total", "Worker-reported errors", ["code"]  # bucketed, never the raw message
)
admission_rejections_total = Counter(
    "live_stt_admission_rejections_total", "Rejected calls", ["reason"]  # no_capacity|draining
)
words_total = Counter("live_stt_words_total", "Finalized words -- volume without content")
transcript_chars_total = Counter(
    "live_stt_transcript_chars_total", "Finalized characters -- volume without content"
)
gpu_free_vram_mb = Gauge(
    "live_stt_gpu_free_vram_mb", "Free VRAM as last observed at admission time (CUDA backend only)"
)
diarization_sessions_active = Gauge(
    "live_stt_diarization_sessions_active", "Diarization requests currently running (pyannote inference in flight)"
)
diarization_requests_total = Counter(
    "live_stt_diarization_requests_total",
    "Completed diarization HTTP requests",
    ["outcome"],  # ok|failed|rejected_vram
)


def set_build_info(
    *, version: str, parakeet_ref: str, backend: str, model: str, n_threads: int, ggml_features: str
) -> None:
    build_info.labels(
        version=version,
        parakeet_ref=parakeet_ref,
        backend=backend,
        model=model,
        n_threads=str(n_threads),
        ggml_features=ggml_features,
    ).set(1)
