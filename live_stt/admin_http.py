"""Admin HTTP surface: /api/health, /api/version, /api/stats, /metrics. Runs
on a daemon thread, deliberately NOT in the asyncio loop -- nothing on the
admin path should ever be able to stall the stream pump (see CLAUDE.md).

Being at capacity is 200 ok, not 503: 503 is reserved for structural failure
(``state.degraded``) or draining (``state.draining``). Marking a busy box
unhealthy would make a load balancer pull a perfectly healthy instance out of
rotation at exactly the moment it's needed.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import generate_latest

from live_stt import __about__
from live_stt.state import ServerState


def _make_handler(state: ServerState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_json(self, status: int, doc: dict) -> None:
            import json

            self._write(status, json.dumps(doc).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
            if self.path == "/api/health":
                status = state.health_status()
                http_status = 200 if status == "ok" else 503
                self._write_json(
                    http_status,
                    {
                        "status": status,
                        "backend": state.settings.backend,
                        "model": state.settings.default_model,
                        "capacity": {
                            "used": state.budget.active_calls,
                            "total": state.settings.max_concurrent_calls,
                            "admitting": status == "ok",
                        },
                    },
                )
            elif self.path == "/api/version":
                self._write_json(200, __about__.info())
            elif self.path == "/api/stats":
                self._write_json(
                    200,
                    {
                        "active_calls": state.budget.active_calls,
                        "active_workers": state.budget.active_workers,
                        "max_concurrent_calls": state.budget.max_concurrent_calls,
                        "max_workers": state.budget.max_workers,
                        "draining": state.draining,
                    },
                )
            elif self.path == "/metrics":
                self._write(200, generate_latest(), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._write_json(404, {"error": "not found"})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # the house logger, not BaseHTTPRequestHandler's stderr default

    return Handler


def serve_admin_http(host: str, port: int, state: ServerState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="admin-http")
    thread.start()
    return server
