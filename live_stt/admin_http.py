"""Admin HTTP surface: /api/health, /api/version. Runs on a daemon thread,
deliberately NOT in the asyncio loop -- nothing on the admin path should ever
be able to stall the stream pump (see CLAUDE.md).

Being at capacity is 200 ok, not 503: 503 is reserved for structural failure
(model unreadable, repeated spawn failures) or draining. Marking a busy box
unhealthy would make a load balancer pull a perfectly healthy instance out of
rotation at exactly the moment it's needed. Not yet implemented in this
skeleton: capacity/worker-pool fields (Phase 2/3), /api/stats, /metrics.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from live_stt import __about__


class _Handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, doc: dict) -> None:
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
        if self.path == "/api/health":
            self._write_json(200, {"status": "ok"})
        elif self.path == "/api/version":
            self._write_json(200, __about__.info())
        else:
            self._write_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # the house logger, not BaseHTTPRequestHandler's stderr default


def serve_admin_http(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="admin-http")
    thread.start()
    return server
