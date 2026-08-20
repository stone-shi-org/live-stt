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


HTML_STATS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Live-STT Admin & Statistics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --panel-border: #334155;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --hover-bg: #334155;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 2rem;
        }

        header {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--panel-border);
            padding: 1.25rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .logo-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            font-size: 1.75rem;
            background: rgba(99, 102, 241, 0.2);
            padding: 0.5rem;
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.4);
        }

        h1 {
            font-size: 1.35rem;
            font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-badges {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .badge.degraded {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-yellow);
            border-color: rgba(245, 158, 11, 0.3);
        }
        .badge.draining {
            background: rgba(239, 68, 68, 0.15);
            color: var(--accent-red);
            border-color: rgba(239, 68, 68, 0.3);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: currentColor;
            border-radius: 50%;
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .container {
            max-width: 1400px;
            width: 100%;
            margin: 1.5rem auto 0 auto;
            padding: 0 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .grid-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.25rem;
        }

        .card {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .card-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .card-value {
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        .card-subtext {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .section-card {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            color: var(--accent-blue);
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 0.75rem;
        }

        .progress-group {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .progress-item {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .progress-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .progress-bar-bg {
            height: 10px;
            background-color: rgba(15, 23, 42, 0.8);
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--panel-border);
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-indigo), var(--accent-blue));
            width: 0%;
            transition: width 0.4s ease;
            border-radius: 6px;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.25rem;
        }

        .config-group {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
        }

        .config-group-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 0.8rem;
        }

        .kv-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        .kv-table tr {
            border-bottom: 1px solid rgba(51, 65, 85, 0.4);
        }

        .kv-table tr:last-child {
            border-bottom: none;
        }

        .kv-table td {
            padding: 0.4rem 0;
        }

        .kv-key {
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            width: 55%;
        }

        .kv-val {
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 500;
            word-break: break-all;
        }

        .links-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .btn-link {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--hover-bg);
            color: var(--text-main);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            border: 1px solid var(--panel-border);
            transition: all 0.2s;
        }

        .btn-link:hover {
            background: var(--accent-indigo);
            color: white;
            border-color: var(--accent-indigo);
        }

        .refresh-tag {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-title">
            <span class="logo-icon">🎙️</span>
            <div>
                <h1>Live-STT Streaming ASR</h1>
                <div class="refresh-tag">Admin & Real-time Stats Dashboard</div>
            </div>
        </div>
        <div class="header-badges">
            <div id="statusPill" class="badge">
                <span class="pulse-dot"></span>
                <span id="statusText">HEALTH: OK</span>
            </div>
        </div>
    </header>

    <div class="container">
        <!-- Metric Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-label">Active Calls</div>
                <div class="card-value" id="valActiveCalls">0</div>
                <div class="card-subtext" id="subActiveCalls">Max Capacity: -</div>
            </div>
            <div class="card">
                <div class="card-label">Active Workers</div>
                <div class="card-value" id="valActiveWorkers">0</div>
                <div class="card-subtext" id="subActiveWorkers">Max Workers: -</div>
            </div>
            <div class="card">
                <div class="card-label">Reserve Slots</div>
                <div class="card-value" id="valReserveSlots">-</div>
                <div class="card-subtext">Reserved for Overlap Shadow</div>
            </div>
            <div class="card">
                <div class="card-label">Server State</div>
                <div class="card-value" id="valServerState" style="font-size:1.4rem;">OK</div>
                <div class="card-subtext" id="subDraining">Draining: False</div>
            </div>
        </div>

        <!-- Connection & Capacity Progress -->
        <div class="section-card">
            <div class="section-title">📊 Capacity & Resource Usage</div>
            <div class="progress-group">
                <div class="progress-item">
                    <div class="progress-label-row">
                        <span>Active Concurrent Calls</span>
                        <span id="callUsageText">0 / 0 (0%)</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="callProgressBar"></div>
                    </div>
                </div>
                <div class="progress-item">
                    <div class="progress-label-row">
                        <span>Worker Pool Utilization</span>
                        <span id="workerUsageText">0 / 0 (0%)</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="workerProgressBar"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Server Configuration -->
        <div class="section-card">
            <div class="section-title">⚙️ Server Configuration & Settings</div>
            <div class="config-grid">
                <div class="config-group">
                    <div class="config-group-header">Network & Engine</div>
                    <table class="kv-table">
                        <tr><td class="kv-key">grpc_host:port</td><td class="kv-val" id="cfgGrpc">-</td></tr>
                        <tr><td class="kv-key">admin_host:port</td><td class="kv-val" id="cfgAdmin">-</td></tr>
                        <tr><td class="kv-key">backend</td><td class="kv-val" id="cfgBackend">-</td></tr>
                        <tr><td class="kv-key">default_model</td><td class="kv-val" id="cfgModel">-</td></tr>
                        <tr><td class="kv-key">models_dir</td><td class="kv-val" id="cfgModelsDir">-</td></tr>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">Worker & Threading</div>
                    <table class="kv-table">
                        <tr><td class="kv-key">n_threads_per_worker</td><td class="kv-val" id="cfgThreads">-</td></tr>
                        <tr><td class="kv-key">max_concurrent_calls</td><td class="kv-val" id="cfgMaxCalls">-</td></tr>
                        <tr><td class="kv-key">reserve_slots</td><td class="kv-val" id="cfgReserve">-</td></tr>
                        <tr><td class="kv-key">worker_rss_soft_kb</td><td class="kv-val" id="cfgRssSoft">-</td></tr>
                        <tr><td class="kv-key">rotate_after_sec</td><td class="kv-val" id="cfgRotateAfter">-</td></tr>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">Timeouts & Safety</div>
                    <table class="kv-table">
                        <tr><td class="kv-key">queue_max_sec</td><td class="kv-val" id="cfgQueueMax">-</td></tr>
                        <tr><td class="kv-key">ring_history_sec</td><td class="kv-val" id="cfgRingHist">-</td></tr>
                        <tr><td class="kv-key">max_call_sec</td><td class="kv-val" id="cfgMaxCall">-</td></tr>
                        <tr><td class="kv-key">idle_timeout_sec</td><td class="kv-val" id="cfgIdleTimeout">-</td></tr>
                        <tr><td class="kv-key">drain_timeout_sec</td><td class="kv-val" id="cfgDrainTimeout">-</td></tr>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">Redaction & System</div>
                    <table class="kv-table">
                        <tr><td class="kv-key">transcript_log</td><td class="kv-val" id="cfgTranscriptLog">-</td></tr>
                        <tr><td class="kv-key">audio_dump</td><td class="kv-val" id="cfgAudioDump">-</td></tr>
                        <tr><td class="kv-key">allow_pii</td><td class="kv-val" id="cfgAllowPii">-</td></tr>
                        <tr><td class="kv-key">build_version</td><td class="kv-val" id="verHash">-</td></tr>
                        <tr><td class="kv-key">parakeet_ref</td><td class="kv-val" id="verParakeet">-</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- API Links -->
        <div class="section-card">
            <div class="section-title">🔗 API & Telemetry Endpoints</div>
            <div class="links-row">
                <a href="/api/health" target="_blank" class="btn-link">🏥 /api/health</a>
                <a href="/api/stats" target="_blank" class="btn-link">📈 /api/stats</a>
                <a href="/api/config" target="_blank" class="btn-link">🛠️ /api/config</a>
                <a href="/api/version" target="_blank" class="btn-link">ℹ️ /api/version</a>
                <a href="/metrics" target="_blank" class="btn-link">📊 /metrics (Prometheus)</a>
            </div>
        </div>
    </div>

    <script>
        async function fetchJSON(url) {
            try {
                const r = await fetch(url);
                if (!r.ok && r.status !== 503) return null;
                return await r.json();
            } catch (e) {
                return null;
            }
        }

        async function updateDashboard() {
            const [health, stats, config, version] = await Promise.all([
                fetchJSON('/api/health'),
                fetchJSON('/api/stats'),
                fetchJSON('/api/config'),
                fetchJSON('/api/version')
            ]);

            // Status Pill
            const statusPill = document.getElementById('statusPill');
            const statusText = document.getElementById('statusText');
            const valServerState = document.getElementById('valServerState');
            if (health) {
                const st = health.status || 'ok';
                statusText.innerText = 'HEALTH: ' + st.toUpperCase();
                valServerState.innerText = st.toUpperCase();
                statusPill.className = 'badge ' + (st !== 'ok' ? st : '');
            } else {
                statusText.innerText = 'HEALTH: OFFLINE';
                valServerState.innerText = 'OFFLINE';
                statusPill.className = 'badge draining';
            }

            // Stats
            if (stats) {
                const activeCalls = stats.active_calls ?? 0;
                const maxCalls = stats.max_concurrent_calls ?? 1;
                const activeWorkers = stats.active_workers ?? 0;
                const maxWorkers = stats.max_workers ?? 1;

                document.getElementById('valActiveCalls').innerText = activeCalls;
                document.getElementById('subActiveCalls').innerText = 'Max Capacity: ' + maxCalls;
                document.getElementById('valActiveWorkers').innerText = activeWorkers;
                document.getElementById('subActiveWorkers').innerText = 'Max Workers: ' + maxWorkers;
                document.getElementById('subDraining').innerText = 'Draining: ' + (stats.draining ? 'True' : 'False');

                // Call Capacity Bar
                const callPct = Math.min(100, Math.round((activeCalls / maxCalls) * 100));
                document.getElementById('callUsageText').innerText = `${activeCalls} / ${maxCalls} (${callPct}%)`;
                document.getElementById('callProgressBar').style.width = callPct + '%';

                // Worker Capacity Bar
                const workerPct = Math.min(100, Math.round((activeWorkers / maxWorkers) * 100));
                document.getElementById('workerUsageText').innerText = `${activeWorkers} / ${maxWorkers} (${workerPct}%)`;
                document.getElementById('workerProgressBar').style.width = workerPct + '%';
            }

            // Config
            if (config) {
                document.getElementById('cfgGrpc').innerText = `${config.grpc_host}:${config.grpc_port}`;
                document.getElementById('cfgAdmin').innerText = `${config.admin_host}:${config.admin_port}`;
                document.getElementById('cfgBackend').innerText = config.backend || 'cpu';
                document.getElementById('cfgModel').innerText = config.default_model || '-';
                document.getElementById('cfgModelsDir').innerText = config.models_dir || '-';

                document.getElementById('cfgThreads').innerText = config.n_threads_per_worker ?? '-';
                document.getElementById('cfgMaxCalls').innerText = config.max_concurrent_calls ?? '-';
                document.getElementById('cfgReserve').innerText = config.reserve_slots ?? '-';
                document.getElementById('valReserveSlots').innerText = config.reserve_slots ?? '-';
                
                const softKb = config.worker_rss_soft_kb;
                document.getElementById('cfgRssSoft').innerText = softKb ? `${Math.round(softKb / 1024)} MB` : '-';
                document.getElementById('cfgRotateAfter').innerText = config.rotate_after_sec ? `${config.rotate_after_sec}s` : '-';

                document.getElementById('cfgQueueMax').innerText = config.queue_max_sec ? `${config.queue_max_sec}s` : '-';
                document.getElementById('cfgRingHist').innerText = config.ring_history_sec ? `${config.ring_history_sec}s` : '-';
                document.getElementById('cfgMaxCall').innerText = config.max_call_sec ? `${config.max_call_sec}s` : '-';
                document.getElementById('cfgIdleTimeout').innerText = config.idle_timeout_sec ? `${config.idle_timeout_sec}s` : '-';
                document.getElementById('cfgDrainTimeout').innerText = config.drain_timeout_sec ? `${config.drain_timeout_sec}s` : '-';

                document.getElementById('cfgTranscriptLog').innerText = config.transcript_log || 'off';
                document.getElementById('cfgAudioDump').innerText = config.audio_dump || 'off';
                document.getElementById('cfgAllowPii').innerText = config.allow_pii ? 'True' : 'False';
            }

            // Version
            if (version) {
                document.getElementById('verHash').innerText = version.hash || 'dev';
                document.getElementById('verParakeet').innerText = version.parakeet_ref || 'dev';
            }
        }

        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""


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
            if self.path in ("/", "/stats", "/dashboard"):
                self._write(200, HTML_STATS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/health":
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
            elif self.path == "/api/config":
                self._write_json(200, state.settings.model_dump())
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
