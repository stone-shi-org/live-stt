"""gRPC server entrypoint. See run.py."""

from __future__ import annotations

import asyncio
import signal

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from live_stt.admin_http import serve_admin_http
from live_stt.admission import WorkerBudget
from live_stt.config import Settings, get_settings
from live_stt.logging_config import configure_logging, get_logger
from live_stt.pb.livestt.v1 import asr_pb2, asr_pb2_grpc
from live_stt.redaction import validate as validate_redaction_config
from live_stt.servicer import StreamingASRServicer
from live_stt.state import ServerState

logger = get_logger("server")


async def serve(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    validate_redaction_config(settings)  # fail fast, not silently downgrade

    state = ServerState(settings=settings, budget=WorkerBudget(settings.max_concurrent_calls, settings.reserve_slots))

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
        "starting grpc=%s admin_http=%s:%s backend=%s default_model=%s max_concurrent_calls=%s",
        listen_addr,
        settings.admin_host,
        settings.admin_port,
        settings.backend,
        settings.default_model,
        settings.max_concurrent_calls,
    )

    await server.start()

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
        stop_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

    await stop_event.wait()
    await server.stop(grace=settings.drain_timeout_sec)
    admin_http.shutdown()


def main() -> None:
    configure_logging()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
