from __future__ import annotations

import urllib.error
import urllib.request

import pytest
from prometheus_client import REGISTRY
from prometheus_client.parser import text_string_to_metric_families

from live_stt import diarization_models, metrics, models
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings
from live_stt.state import ServerState

EXPECTED_METRICS = {
    "live_stt_build_info",
    "live_stt_streams_active",
    "live_stt_streams_total",
    "live_stt_stream_duration_seconds",
    "live_stt_audio_seconds_total",
    "live_stt_feed_duration_seconds",
    "live_stt_model_load_duration_seconds",
    "live_stt_worker_restarts_total",
    "live_stt_rotations_total",
    "live_stt_stream_init_failures_total",
    "live_stt_asr_errors_total",
    "live_stt_admission_rejections_total",
    "live_stt_words_total",
    "live_stt_transcript_chars_total",
    "live_stt_gpu_free_vram_mb",
    "live_stt_diarization_sessions_active",
    "live_stt_diarization_requests_total",
}


def _dump() -> str:
    from prometheus_client import generate_latest

    return generate_latest(REGISTRY).decode()


def test_all_expected_metrics_are_registered() -> None:
    families = list(text_string_to_metric_families(_dump()))
    names = set()
    for fam in families:
        names.add(fam.name)
        # Counters render as e.g. "live_stt_words_total" with the family name
        # already stripped of the _total suffix by the parser for some
        # client versions -- cover both forms defensively.
        names.add(fam.name + "_total")
    missing = EXPECTED_METRICS - names
    assert not missing, f"missing metrics: {missing}"


def test_metrics_output_parses_as_valid_prometheus_text() -> None:
    # If this doesn't raise, the exposition format is well-formed.
    list(text_string_to_metric_families(_dump()))


def test_no_metric_carries_a_stream_identifying_label() -> None:
    forbidden = {"stream_id", "call_id", "call_ref"}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            assert not (forbidden & sample.labels.keys()), (
                f"metric {sample.name} has a forbidden label: {sample.labels.keys()}"
            )


def test_set_build_info_sets_expected_labels() -> None:
    metrics.set_build_info(
        version="abc1234",
        parakeet_ref="deadbeef",
        backend="cpu",
        model="realtime_eou_120m-v1",
        n_threads=2,
        ggml_features="AVX2 FMA F16C",
    )
    dump = _dump()
    assert 'version="abc1234"' in dump
    assert 'model="realtime_eou_120m-v1"' in dump
    assert 'n_threads="2"' in dump


def test_stream_duration_buckets_cover_a_two_hour_call() -> None:
    # The default prometheus_client buckets top out at 10s -- useless for a
    # 30-120 minute call. Confirm the custom buckets actually go that high.
    assert max(metrics.STREAM_DURATION_BUCKETS) >= 7200


def test_feed_duration_has_a_bucket_edge_at_the_realtime_budget() -> None:
    # 320ms is nemotron's per-chunk realtime budget -- a deliberate bucket
    # edge, not an arbitrary histogram shape.
    assert 0.32 in metrics.FEED_DURATION_BUCKETS


@pytest.mark.asyncio
async def test_metrics_http_endpoint_serves_valid_output() -> None:
    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read().decode()
        list(text_string_to_metric_families(body))  # parses without raising
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_health_endpoint_is_200_at_capacity_and_503_while_draining() -> None:
    settings = Settings(_env_file=None, max_concurrent_calls=1)
    budget = WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots)
    state = ServerState(settings=settings, budget=budget)
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]

        budget.try_admit_call()  # now "at capacity" (1/1) -- still healthy
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
            assert resp.status == 200

        state.draining = True
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5)
            assert False, "expected HTTPError for 503"
        except urllib.error.HTTPError as exc:
            assert exc.code == 503
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_stats_web_page_endpoint() -> None:
    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        for path in ("/", "/stats", "/dashboard"):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
                assert resp.status == 200
                assert "text/html" in resp.headers.get("Content-Type", "")
                html_body = resp.read().decode()
                assert "Live-STT Streaming ASR" in html_body
                assert "Admin & Real-time Stats Dashboard" in html_body
                # GPU/VRAM and diarization-session cards (see admin_http.py).
                assert 'id="valGpuVramFree"' in html_body
                assert 'id="valGpuUtil"' in html_body
                assert 'id="valDiarizeActive"' in html_body
                assert 'id="valDiarizeTotals"' in html_body
                assert 'id="vramProgressBar"' in html_body
                # Diarization model row (in addition to the pre-existing
                # default_model/ASR row) -- CLAUDE.md's requested addition.
                assert 'id="cfgDiarizationModel"' in html_body
                # Live "what's running right now" table for diarization.
                assert 'id="diarizeActiveTbody"' in html_body
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_config_http_endpoint() -> None:
    import json

    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=5) as resp:
            assert resp.status == 200
            doc = json.loads(resp.read().decode())
            assert "grpc_port" in doc
            assert "backend" in doc
            assert "max_concurrent_calls" in doc
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_stats_json_endpoint_includes_gpu_and_diarization() -> None:
    import json

    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats", timeout=5) as resp:
            assert resp.status == 200
            doc = json.loads(resp.read().decode())

        # Pre-existing fields still present.
        assert "active_calls" in doc
        assert "max_concurrent_calls" in doc

        # This dev host has no nvidia-smi -- all four gpu fields None
        # together (never a mix of some real, some None; see gpu.py).
        assert doc["gpu"] == {
            "free_vram_mb": None,
            "total_vram_mb": None,
            "used_vram_mb": None,
            "utilization_pct": None,
        }
        # Fresh ServerState -- diarization_sessions starts at all zeros,
        # with no in-flight requests in the live list.
        assert doc["diarization"] == {
            "active": 0,
            "completed_total": 0,
            "failed_total": 0,
            "rejected_vram_total": 0,
            "active_requests": [],
        }
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_stats_json_endpoint_reflects_an_in_flight_diarization_request() -> None:
    import json

    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]

        request_id = state.diarization_sessions.start(device="cuda")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
        active_requests = doc["diarization"]["active_requests"]
        assert len(active_requests) == 1
        assert active_requests[0]["id"] == request_id
        assert active_requests[0]["device"] == "cuda"
        assert active_requests[0]["elapsed_sec"] >= 0

        state.diarization_sessions.finish(request_id, ok=True)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stats", timeout=5) as resp:
            doc = json.loads(resp.read().decode())
        assert doc["diarization"]["active_requests"] == []
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_v1_models_endpoint_lists_asr_and_diarization_models() -> None:
    import json

    settings = Settings(_env_file=None)
    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))
    server = serve_admin_http("127.0.0.1", 0, state)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
            assert resp.status == 200
            assert "application/json" in resp.headers.get("Content-Type", "")
            doc = json.loads(resp.read().decode())

        assert doc["object"] == "list"
        by_id = {entry["id"]: entry for entry in doc["data"]}

        # Every ASR model in the registry is present, typed correctly, and
        # exactly one is marked default (the configured default_model).
        for key, spec in models.MODELS.items():
            assert key in by_id
            entry = by_id[key]
            assert entry["object"] == "model"
            assert entry["type"] == "asr"
            assert entry["model_chunk_ms"] == spec.model_chunk_ms
            assert entry["has_eou"] == spec.has_eou
            assert entry["default"] == (key == settings.default_model)
        assert sum(e["default"] for e in by_id.values() if e["type"] == "asr") == 1

        # Every diarization model in the registry is present too, typed
        # correctly, with exactly one marked default.
        for key, spec in diarization_models.DIARIZATION_MODELS.items():
            assert key in by_id
            entry = by_id[key]
            assert entry["object"] == "model"
            assert entry["type"] == "diarization"
            assert entry["gated"] == spec.gated
            assert entry["supports_num_speakers_hint"] == spec.supports_num_speakers_hint
            assert entry["default"] == (key == settings.diarization_model)
        assert sum(e["default"] for e in by_id.values() if e["type"] == "diarization") == 1
    finally:
        server.shutdown()

