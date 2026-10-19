"""gRPC server entrypoint. See run.py."""

from __future__ import annotations

import asyncio
import signal

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from live_stt import gpu, models, tts, vram
from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.backends.upgrade_checker import run_periodic_upgrade_check
from live_stt.config import Settings
from live_stt.logging_config import configure_logging, get_logger
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.redaction import validate as validate_redaction_config
from live_stt.servicer import StreamingASRServicer
from live_stt.settings_store import SettingsStore, load_effective_settings
from live_stt.state import ServerState
from live_stt import worker_pool

logger = get_logger("server")


def _warn_if_concurrency_exceeds_gpu(settings: Settings) -> None:
    cap = vram.gpu_capacity(settings)
    if not cap:
        return
    default_key = settings.default_model or models.DEFAULT_MODEL_KEY
    model_cap = cap["models"].get(default_key)
    if model_cap is None:
        return
    if settings.max_concurrent_calls > model_cap["effective_max_calls"]:
        logger.warning(
            "max_concurrent_calls=%d exceeds estimated GPU capacity: "
            "max_concurrent_calls=%d but only %d %s calls fit in %dMB VRAM "
            "(per_call_mb=%d); VRAM admission will reject excess concurrent requests with 503",
            settings.max_concurrent_calls,
            settings.max_concurrent_calls,
            model_cap["max_calls"],
            default_key,
            cap["budget_mb"],
            model_cap["per_call_mb"],
        )


async def serve(settings: Settings | None = None) -> None:
    # settings is only ever passed explicitly by a caller that wants pure,
    # deterministic env-based behavior with no SQLite involved (a one-off
    # script, or a future test) -- the real default path (settings=None)
    # goes through the SQLite-backed store (LS-5), which fully replaces env
    # vars as the ongoing source of truth after a one-time migration; see
    # live_stt/settings_store.py's docstring.
    settings_store: SettingsStore | None = None
    if settings is None:
        settings, settings_store = load_effective_settings()
    validate_redaction_config(settings)  # fail fast, not silently downgrade

    state = ServerState(
        settings=settings,
        budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots),
        settings_store=settings_store,
    )

    server = grpc.aio.server()
    asr_pb2_grpc.add_StreamingASRServicer_to_server(StreamingASRServicer(state), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("livestt.v1.StreamingASR", health_pb2.HealthCheckResponse.SERVING)

    service_names = (
        asr_pb2.DESCRIPTOR.services_by_name["StreamingASR"].full_name,
        health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)

    admin_http = serve_admin_http(settings.admin_host, settings.admin_port, state)
    logger.info(
        "starting grpc=%s admin_http=%s:%s parakeet_backend=%s whisper_backend=%s "
        "parakeet_tdt_backend=%s default_model=%s default_batch_model=%s max_concurrent_calls=%s",
        listen_addr,
        settings.admin_host,
        settings.admin_port,
        settings.parakeet_backend,
        settings.whisper_backend,
        settings.parakeet_tdt_backend,
        settings.default_model,
        settings.default_batch_model,
        settings.max_concurrent_calls,
    )
    _warn_if_concurrency_exceeds_gpu(settings)

    await server.start()

    # LS-5 Phase 7c: only on the real entrypoint path (settings_store is
    # None for a caller that passed settings=... explicitly -- see the
    # comment above) -- a background task, never awaited to completion,
    # kept alive by this reference for serve()'s lifetime and cancelled on
    # shutdown below like any other in-process resource.
    upgrade_check_task: asyncio.Task | None = None
    if settings_store is not None:
        upgrade_check_task = asyncio.create_task(run_periodic_upgrade_check(settings))

    stop_event = asyncio.Event()

    def _handle_sigterm() -> None:
        # Flip everything to "don't send me new work" INSTANTLY, before
        # grpc.aio's own drain grace period even starts: the health service
        # (for gRPC-aware load balancers), state.draining (checked by
        # StreamingASRServicer.Transcribe's admission check, so new calls
        # get UNAVAILABLE immediately), and /api/health (for anything
        # polling HTTP). Active calls are NOT touched here -- they get up to
        # drain_timeout_sec to finish naturally via server.stop()'s grace
        # period below.
        logger.info("SIGTERM received, draining (up to %.0fs)", settings.drain_timeout_sec)
        health_servicer.set("livestt.v1.StreamingASR", health_pb2.HealthCheckResponse.NOT_SERVING)
        state.draining = True
        worker_pool.stop_worker_pool()
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

    await stop_event.wait()
    await server.stop(grace=settings.drain_timeout_sec)
    admin_http.shutdown()
    tts.stop_tts_worker()
    worker_pool.stop_worker_pool()
    if upgrade_check_task is not None:
        upgrade_check_task.cancel()


def main() -> None:
    configure_logging()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
