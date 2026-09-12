"""Admin HTTP surface: /api/health, /api/version, /api/stats, /metrics, and
(LS-5) /settings + GET/PUT /api/settings. Runs on a daemon thread,
deliberately NOT in the asyncio loop -- nothing on the admin path should
ever be able to stall the stream pump (see CLAUDE.md).

Being at capacity is 200 ok, not 503: 503 is reserved for structural failure
(``state.degraded``) or draining (``state.draining``). Marking a busy box
unhealthy would make a load balancer pull a perfectly healthy instance out of
rotation at exactly the moment it's needed.

Dashboard (today's /, /stats, /dashboard, /api/stats, /metrics -- all
read-only, unchanged) and Settings (/settings, GET/PUT /api/settings) are
kept as a deliberate split per LS-5: the former is "what is this instance
doing right now," the latter is "what is this instance configured to do,"
and only the latter ever mutates anything.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import generate_latest

from live_stt import (
    __about__,
    diarization_models,
    diarize_http,
    gpu,
    models,
    transcribe_http,
    tts_http,
    tts_models,
)
from live_stt.settings_store import HOT_RELOAD_FIELDS, apply_update
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

        .col-id {
            width: 100px;
        }

        .col-elapsed {
            width: 120px;
        }

        .col-info {
            width: auto;
        }

        .model-name-cell {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }

        .copy-btn {
            background: none;
            border: none;
            cursor: pointer;
            padding: 0.15rem 0.25rem;
            border-radius: 4px;
            opacity: 0.6;
            transition: opacity 0.15s, background 0.15s, color 0.15s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            vertical-align: middle;
            color: var(--text-muted);
        }

        .copy-btn:hover {
            opacity: 1;
            background: var(--hover-bg);
            color: var(--text-main);
        }

        .copy-btn svg {
            width: 14px;
            height: 14px;
            fill: none;
            stroke: currentColor;
            stroke-width: 2;
            stroke-linecap: round;
            stroke-linejoin: round;
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

        <!-- GPU & Diarization Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-label">GPU VRAM Free</div>
                <div class="card-value" id="valGpuVramFree">-</div>
                <div class="card-subtext" id="subGpuVramTotal">of - total</div>
            </div>
            <div class="card">
                <div class="card-label">GPU Utilization</div>
                <div class="card-value" id="valGpuUtil">-</div>
                <div class="card-subtext">Compute, whole card</div>
            </div>
            <div class="card">
                <div class="card-label">Diarization Sessions</div>
                <div class="card-value" id="valDiarizeActive">0</div>
                <div class="card-subtext">Currently running</div>
            </div>
            <div class="card">
                <div class="card-label">Diarization Totals</div>
                <div class="card-value" id="valDiarizeTotals" style="font-size:1.1rem;">0 / 0 / 0</div>
                <div class="card-subtext">ok / failed / VRAM-rejected</div>
            </div>
        </div>

        <!-- Transcript (batch HTTP) Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-label">Transcript Requests</div>
                <div class="card-value" id="valTranscribeActive">0</div>
                <div class="card-subtext">Currently running (POST /v1/audio/transcriptions)</div>
            </div>
            <div class="card">
                <div class="card-label">Transcript Totals</div>
                <div class="card-value" id="valTranscribeTotals" style="font-size:1.1rem;">0 / 0 / 0 / 0</div>
                <div class="card-subtext">ok / failed / VRAM-rejected / capacity-rejected</div>
            </div>
        </div>

        <!-- TTS (Speech Synthesis) Cards -->
        <div class="grid-cards">
            <div class="card">
                <div class="card-label">TTS Requests</div>
                <div class="card-value" id="valTtsActive">0</div>
                <div class="card-subtext">Currently running (POST /v1/audio/speech)</div>
            </div>
            <div class="card">
                <div class="card-label">TTS Totals</div>
                <div class="card-value" id="valTtsTotals" style="font-size:1.1rem;">0 / 0 / 0 / 0</div>
                <div class="card-subtext">ok / failed / VRAM-rejected / capacity-rejected</div>
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
                <div class="progress-item" id="vramProgressRow" style="display:none;">
                    <div class="progress-label-row">
                        <span>GPU VRAM Usage (whole card, all tenants)</span>
                        <span id="vramUsageText">- / - (0%)</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" id="vramProgressBar"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Active Diarization Requests -->
        <div class="section-card">
            <div class="section-title">🎤 Active Diarization Requests</div>
            <table class="kv-table" id="diarizeActiveTable">
                <thead>
                    <tr>
                        <td class="kv-key col-id">ID</td>
                        <td class="kv-key col-elapsed">Elapsed</td>
                        <td class="kv-key col-info">Device</td>
                    </tr>
                </thead>
                <tbody id="diarizeActiveTbody">
                    <tr><td colspan="3" class="card-subtext">No active diarization requests</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Active Transcript Requests -->
        <div class="section-card">
            <div class="section-title">📝 Active Transcript Requests</div>
            <table class="kv-table" id="transcribeActiveTable">
                <thead>
                    <tr>
                        <td class="kv-key col-id">ID</td>
                        <td class="kv-key col-elapsed">Elapsed</td>
                        <td class="kv-key col-info">Model</td>
                    </tr>
                </thead>
                <tbody id="transcribeActiveTbody">
                    <tr><td colspan="3" class="card-subtext">No active transcript requests</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Active TTS Requests -->
        <div class="section-card">
            <div class="section-title">🔊 Active Speech Synthesis Requests</div>
            <table class="kv-table" id="ttsActiveTable">
                <thead>
                    <tr>
                        <td class="kv-key col-id">ID</td>
                        <td class="kv-key col-elapsed">Elapsed</td>
                        <td class="kv-key col-info">Model / Voice</td>
                    </tr>
                </thead>
                <tbody id="ttsActiveTbody">
                    <tr><td colspan="3" class="card-subtext">No active TTS requests</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Supported Models by Service -->
        <div class="section-card">
            <div class="section-title">🧩 Supported Models by Service</div>
            <div class="config-grid">
                <div class="config-group">
                    <div class="config-group-header">📞 Live Call &mdash; gRPC <code>Transcribe</code> (streaming)</div>
                    <table class="kv-table" id="svcLiveCallTable">
                        <tbody id="svcLiveCallTbody">
                            <tr><td class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">📝 Batch Transcript &mdash; <code>POST /v1/audio/transcriptions</code></div>
                    <table class="kv-table" id="svcTranscriptTable">
                        <tbody id="svcTranscriptTbody">
                            <tr><td class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">🎤 Diarization &mdash; <code>POST /v1/audio/diarization</code></div>
                    <table class="kv-table" id="svcDiarizationTable">
                        <tbody id="svcDiarizationTbody">
                            <tr><td class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
                <div class="config-group">
                    <div class="config-group-header">🔊 Text-to-Speech &mdash; <code>POST /v1/audio/speech</code></div>
                    <table class="kv-table" id="svcTtsTable">
                        <tbody id="svcTTSTbody">
                            <tr><td class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Server Configuration -->
        <div class="section-card">
            <div class="section-title">⚙️ Server Configuration & Settings (<a href="/settings" style="color:inherit;">edit in Settings →</a>)</div>
            <div class="config-grid">
                <div class="config-group">
                    <div class="config-group-header">Network & Engine</div>
                    <table class="kv-table">
                        <tr><td class="kv-key">grpc_host:port</td><td class="kv-val" id="cfgGrpc">-</td></tr>
                        <tr><td class="kv-key">admin_host:port</td><td class="kv-val" id="cfgAdmin">-</td></tr>
                        <tr><td class="kv-key">parakeet_backend</td><td class="kv-val" id="cfgBackendParakeet">-</td></tr>
                        <tr><td class="kv-key">whisper_backend</td><td class="kv-val" id="cfgBackendWhisper">-</td></tr>
                        <tr><td class="kv-key">parakeet_tdt_backend</td><td class="kv-val" id="cfgBackendParakeetTdt">-</td></tr>
                        <tr><td class="kv-key">default_model</td><td class="kv-val" id="cfgModel">-</td></tr>
                        <tr><td class="kv-key">diarization_model</td><td class="kv-val" id="cfgDiarizationModel">-</td></tr>
                        <tr><td class="kv-key">tts_model</td><td class="kv-val" id="cfgTtsModel">-</td></tr>
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
                <a href="/v1/models" target="_blank" class="btn-link">🧩 /v1/models</a>
                <a href="/metrics" target="_blank" class="btn-link">📊 /metrics (Prometheus)</a>
            </div>
        </div>
    </div>

    <script>
        const COPY_ICON_SVG = `<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
        const CHECK_ICON_SVG = `<svg viewBox="0 0 24 24" style="stroke:var(--accent-green);"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

        // Copy-to-clipboard for model names on the Supported Models by Service page
        // so a full model key can be grabbed with one click.
        function copyToClipboard(text, btn) {
            const done = () => {
                btn.innerHTML = CHECK_ICON_SVG;
                setTimeout(() => { btn.innerHTML = COPY_ICON_SVG; }, 900);
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
            } else {
                fallbackCopy(text, done);
            }
        }

        function fallbackCopy(text, done) {
            // Non-secure-context / older-browser fallback -- this admin
            // dashboard is often reached over plain http:// on a home LAN,
            // where navigator.clipboard is unavailable (Clipboard API is
            // restricted to secure contexts).
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); done(); } catch (e) { /* give up quietly */ }
            document.body.removeChild(ta);
        }

        function copyBtn(name) {
            return `<button class="copy-btn" onclick="copyToClipboard('${name}', this)" title="Copy '${name}' to clipboard">${COPY_ICON_SVG}</button>`;
        }

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
            const [health, stats, config, version, modelsResp] = await Promise.all([
                fetchJSON('/api/health'),
                fetchJSON('/api/stats'),
                fetchJSON('/api/config'),
                fetchJSON('/api/version'),
                fetchJSON('/v1/models')
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

                // GPU / VRAM -- gpu.snapshot() is all-None together when
                // nvidia-smi is unavailable (CPU backend / no GPU), never a
                // mix of some real some None (see live_stt/gpu.py), so one
                // null check covers the whole block.
                const gpuInfo = stats.gpu || {};
                const vramRow = document.getElementById('vramProgressRow');
                if (gpuInfo.free_vram_mb != null && gpuInfo.total_vram_mb != null) {
                    const freeGb = (gpuInfo.free_vram_mb / 1024).toFixed(1);
                    const totalGb = (gpuInfo.total_vram_mb / 1024).toFixed(1);
                    document.getElementById('valGpuVramFree').innerText = freeGb + ' GB';
                    document.getElementById('subGpuVramTotal').innerText = 'of ' + totalGb + ' GB total';

                    const usedMb = gpuInfo.used_vram_mb ?? (gpuInfo.total_vram_mb - gpuInfo.free_vram_mb);
                    const vramPct = Math.min(100, Math.round((usedMb / gpuInfo.total_vram_mb) * 100));
                    document.getElementById('vramUsageText').innerText =
                        `${(usedMb / 1024).toFixed(1)} GB / ${totalGb} GB (${vramPct}%)`;
                    document.getElementById('vramProgressBar').style.width = vramPct + '%';
                    vramRow.style.display = '';
                } else {
                    document.getElementById('valGpuVramFree').innerText = 'N/A';
                    document.getElementById('subGpuVramTotal').innerText = 'no GPU / nvidia-smi';
                    vramRow.style.display = 'none';
                }
                document.getElementById('valGpuUtil').innerText =
                    gpuInfo.utilization_pct != null ? gpuInfo.utilization_pct + '%' : 'N/A';

                // Diarization sessions -- see live_stt/diarize_sessions.py.
                // Aggregate counters, plus a small live list of in-flight
                // requests (opaque id / elapsed / device) -- a deliberate,
                // narrow exception to "not a per-call registry" (see that
                // module's docstring), not the full ASR-style registry.
                const diar = stats.diarization || {};
                document.getElementById('valDiarizeActive').innerText = diar.active ?? 0;
                document.getElementById('valDiarizeTotals').innerText =
                    `${diar.completed_total ?? 0} / ${diar.failed_total ?? 0} / ${diar.rejected_vram_total ?? 0}`;

                const activeRequests = diar.active_requests || [];
                const diarizeTbody = document.getElementById('diarizeActiveTbody');
                if (activeRequests.length === 0) {
                    diarizeTbody.innerHTML = '<tr><td colspan="3" class="card-subtext">No active diarization requests</td></tr>';
                } else {
                    diarizeTbody.innerHTML = activeRequests.map(r =>
                        `<tr><td class="kv-val col-id">#${r.id}</td><td class="kv-val col-elapsed">${r.elapsed_sec.toFixed(1)}s</td><td class="kv-val col-info">${r.device}</td></tr>`
                    ).join('');
                }

                // Batch transcription requests -- see live_stt/transcribe_sessions.py.
                // Same shape as diarization above (aggregate counters +
                // small live list), plus a capacity-rejected count this
                // path has and diarization doesn't (diarization has no
                // WorkerBudget capacity gate, only VRAM).
                const xscribe = stats.transcribe || {};
                document.getElementById('valTranscribeActive').innerText = xscribe.active ?? 0;
                document.getElementById('valTranscribeTotals').innerText =
                    `${xscribe.completed_total ?? 0} / ${xscribe.failed_total ?? 0} / ${xscribe.rejected_vram_total ?? 0} / ${xscribe.rejected_capacity_total ?? 0}`;

                const activeTranscribeRequests = xscribe.active_requests || [];
                const transcribeTbody = document.getElementById('transcribeActiveTbody');
                if (activeTranscribeRequests.length === 0) {
                    transcribeTbody.innerHTML = '<tr><td colspan="3" class="card-subtext">No active transcript requests</td></tr>';
                } else {
                    transcribeTbody.innerHTML = activeTranscribeRequests.map(r =>
                        `<tr><td class="kv-val col-id">#${r.id}</td><td class="kv-val col-elapsed">${r.elapsed_sec.toFixed(1)}s</td><td class="kv-val col-info">${r.model}</td></tr>`
                    ).join('');
                }

                // TTS synthesis requests -- see live_stt/tts_sessions.py.
                const tts = stats.tts || {};
                document.getElementById('valTtsActive').innerText = tts.active ?? 0;
                document.getElementById('valTtsTotals').innerText =
                    `${tts.completed_total ?? 0} / ${tts.failed_total ?? 0} / ${tts.rejected_vram_total ?? 0} / ${tts.rejected_capacity_total ?? 0}`;

                const activeTtsRequests = tts.active_requests || [];
                const ttsTbody = document.getElementById('ttsActiveTbody');
                if (activeTtsRequests.length === 0) {
                    ttsTbody.innerHTML = '<tr><td colspan="3" class="card-subtext">No active TTS requests</td></tr>';
                } else {
                    ttsTbody.innerHTML = activeTtsRequests.map(r =>
                        `<tr><td class="kv-val col-id">#${r.id}</td><td class="kv-val col-elapsed">${r.elapsed_sec.toFixed(1)}s</td><td class="kv-val col-info">${r.model} (${r.voice || '-'})</td></tr>`
                    ).join('');
                }
            }

            // Config
            if (config) {
                document.getElementById('cfgGrpc').innerText = `${config.grpc_host}:${config.grpc_port}`;
                document.getElementById('cfgAdmin').innerText = `${config.admin_host}:${config.admin_port}`;
                document.getElementById('cfgBackendParakeet').innerText = config.parakeet_backend || 'cpu';
                document.getElementById('cfgBackendWhisper').innerText = config.whisper_backend || 'cpu';
                document.getElementById('cfgBackendParakeetTdt').innerText = config.parakeet_tdt_backend || 'cpu';
                document.getElementById('cfgModel').innerText = config.default_model || '-';
                document.getElementById('cfgDiarizationModel').innerText = config.diarization_model || '-';
                document.getElementById('cfgTtsModel').innerText = config.tts_model || '-';
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

            // Supported Models by Service -- derived from /v1/models
            // (live_stt/models.py + live_stt/diarization_models.py, see
            // admin_http.py's /v1/models handler), not a second hardcoded
            // list: "Live Call" is ASR models with streaming_capable=true
            // only (the gRPC Transcribe RPC rejects the rest -- see
            // live_stt/servicer.py), "Batch Transcript" is EVERY ASR model
            // (that endpoint is engine-agnostic, per-model capability
            // doesn't matter there), "Diarization" is the separate
            // diarization_models registry entirely.
            const modelRow = (spec, badges) =>
                `<tr><td class="kv-key"><span class="model-name-cell">${spec.id}${copyBtn(spec.id)}</span></td><td class="kv-val">${badges.filter(Boolean).join(', ') || '&mdash;'}</td></tr>`;
            const noModelsRow = '<tr><td class="card-subtext">No models registered</td></tr>';

            if (modelsResp && modelsResp.data) {
                const asrModels = modelsResp.data.filter(m => m.type === 'asr');
                const diarizationModels = modelsResp.data.filter(m => m.type === 'diarization');
                const ttsModels = modelsResp.data.filter(m => m.type === 'tts');

                const liveCallModels = asrModels.filter(m => m.streaming_capable !== false);
                document.getElementById('svcLiveCallTbody').innerHTML =
                    liveCallModels.length
                        ? liveCallModels.map(m => modelRow(m, [
                              m.default && 'default',
                              m.has_eou && 'EOU',
                              m.multilingual && 'multilingual',
                          ])).join('')
                        : noModelsRow;

                document.getElementById('svcTranscriptTbody').innerHTML =
                    asrModels.length
                        ? asrModels.map(m => modelRow(m, [
                              m.default && 'default',
                              m.streaming_capable === false && 'batch-only',
                              m.multilingual && 'multilingual',
                          ])).join('')
                        : noModelsRow;

                document.getElementById('svcDiarizationTbody').innerHTML =
                    diarizationModels.length
                        ? diarizationModels.map(m => modelRow(m, [
                              m.default && 'default',
                              m.gated && 'gated',
                              m.supports_num_speakers_hint && 'num_speakers hint',
                          ])).join('')
                        : noModelsRow;

                document.getElementById('svcTTSTbody').innerHTML =
                    ttsModels.length
                        ? ttsModels.map(m => modelRow(m, [
                              m.default && 'default',
                              m.engine,
                              m.supports_streaming && 'streaming',
                          ])).join('')
                        : noModelsRow;
            } else {
                const failedRow = '<tr><td class="card-subtext">Failed to load /v1/models</td></tr>';
                document.getElementById('svcLiveCallTbody').innerHTML = failedRow;
                document.getElementById('svcTranscriptTbody').innerHTML = failedRow;
                document.getElementById('svcDiarizationTbody').innerHTML = failedRow;
                document.getElementById('svcTTSTbody').innerHTML = failedRow;
            }
        }

        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""


# LS-5: a deliberately generic (field-name, current-value) form rather than
# a hand-crafted widget per Settings field -- Settings has 60+ fields and
# hand-authoring each one would be a maintenance trap (a new field added to
# live_stt/config.py would silently never appear here). Input type is
# inferred client-side from the JSON value's JS type. Every field is shown
# and editable; the badge communicates which of them actually take effect
# immediately (HOT_RELOAD_FIELDS, live_stt/settings_store.py) vs. only on
# the next process restart -- the store never pretends both are the same.
HTML_SETTINGS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚙️ Live-STT Settings</title>
    <style>
        :root {
            --bg-dark: #0f172a;
            --panel-bg: #1e293b;
            --panel-border: #334155;
            --accent-blue: #38bdf8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        body { background: var(--bg-dark); color: var(--text-main); font-family: -apple-system, sans-serif; margin: 0; padding: 24px; }
        h1 { font-size: 1.4rem; }
        a { color: var(--accent-blue); }
        #banner { padding: 10px 14px; border-radius: 8px; margin-bottom: 16px; display: none; }
        #banner.ok { display: block; background: rgba(16,185,129,0.15); border: 1px solid var(--accent-green); }
        #banner.err { display: block; background: rgba(239,68,68,0.15); border: 1px solid var(--accent-red); }
        table { width: 100%; border-collapse: collapse; background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 8px; overflow: hidden; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--panel-border); font-size: 0.85rem; }
        th { color: var(--text-muted); font-weight: 600; }
        input[type=text], input[type=number] { background: var(--bg-dark); color: var(--text-main); border: 1px solid var(--panel-border); border-radius: 4px; padding: 4px 8px; width: 100%; box-sizing: border-box; }
        .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 999px; }
        .badge.hot { background: rgba(16,185,129,0.2); color: var(--accent-green); }
        .badge.restart { background: rgba(245,158,11,0.2); color: var(--accent-yellow); }
        #saveBtn { margin-top: 16px; padding: 10px 20px; background: var(--accent-blue); color: #0f172a; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; }
        #saveBtn:disabled { opacity: 0.5; cursor: default; }
    </style>
</head>
<body>
    <h1>⚙️ Live-STT Settings <a href="/">&larr; back to dashboard</a></h1>
    <p style="color:var(--text-muted);">
        Fields badged <span class="badge hot">hot</span> take effect immediately on save.
        Fields badged <span class="badge restart">restart required</span> are persisted but
        only take effect the next time this process starts (see live_stt/settings_store.py --
        several fields are unsafe to change under a live worker pool).
    </p>
    <div id="banner"></div>
    <table>
        <thead><tr><th>Setting</th><th>Value</th><th>Applies</th></tr></thead>
        <tbody id="settingsBody"><tr><td colspan="3">Loading...</td></tr></tbody>
    </table>
    <button id="saveBtn" disabled>Save Changes</button>

    <script>
        let original = {};
        let hotFields = new Set();
        const dirty = {};

        function cell(key, value) {
            const isBool = (typeof value === 'boolean');
            const isNum = (typeof value === 'number');
            let input;
            if (isBool) {
                input = `<input type="checkbox" data-key="${key}" ${value ? 'checked' : ''}>`;
            } else if (isNum) {
                input = `<input type="number" step="any" data-key="${key}" value="${value}">`;
            } else {
                input = `<input type="text" data-key="${key}" value="${value === null ? '' : value}">`;
            }
            const badge = hotFields.has(key)
                ? '<span class="badge hot">hot</span>'
                : '<span class="badge restart">restart required</span>';
            return `<tr><td>${key}</td><td>${input}</td><td>${badge}</td></tr>`;
        }

        function markDirty(key, value) {
            if (JSON.stringify(value) === JSON.stringify(original[key])) {
                delete dirty[key];
            } else {
                dirty[key] = value;
            }
            document.getElementById('saveBtn').disabled = Object.keys(dirty).length === 0;
        }

        async function load() {
            const resp = await fetch('/api/settings');
            const doc = await resp.json();
            original = doc.settings;
            hotFields = new Set(doc.hot_reload_fields || []);
            const tbody = document.getElementById('settingsBody');
            const keys = Object.keys(original).sort();
            tbody.innerHTML = keys.map((k) => cell(k, original[k])).join('');
            tbody.querySelectorAll('input').forEach((el) => {
                el.addEventListener('input', () => {
                    const key = el.dataset.key;
                    let value = el.value;
                    if (el.type === 'checkbox') value = el.checked;
                    else if (el.type === 'number') value = value === '' ? null : Number(value);
                    markDirty(key, value);
                });
            });
            if (doc.store_available === false) {
                showBanner('err', 'No settings store configured on this process -- changes cannot be saved (see live_stt/server.py).');
            }
        }

        function showBanner(kind, text) {
            const b = document.getElementById('banner');
            b.className = kind;
            b.innerText = text;
        }

        async function save() {
            const resp = await fetch('/api/settings', {
                method: 'PUT',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(dirty),
            });
            const doc = await resp.json();
            if (resp.ok) {
                const hot = (doc.applied_hot || []).join(', ') || 'none';
                const restart = (doc.pending_restart || []).join(', ') || 'none';
                showBanner('ok', `Saved. Applied immediately: ${hot}. Needs restart: ${restart}.`);
                for (const k of Object.keys(dirty)) delete dirty[k];
                document.getElementById('saveBtn').disabled = true;
                await load();
            } else {
                showBanner('err', doc.error || 'save failed');
            }
        }

        document.getElementById('saveBtn').addEventListener('click', save);
        load();
    </script>
</body>
</html>
"""


def _settings_doc(state: ServerState) -> dict:
    current = state.settings.model_dump()
    return {
        "settings": current,
        "hot_reload_fields": sorted(HOT_RELOAD_FIELDS),
        "restart_required_fields": sorted(set(current) - HOT_RELOAD_FIELDS),
        "store_available": state.settings_store is not None,
    }


def _apply_settings_update(state: ServerState, body: bytes) -> tuple[int, dict]:
    if state.settings_store is None:
        return 503, {"error": "no settings store configured on this process"}
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    if not isinstance(payload, dict) or not payload:
        return 400, {"error": "expected a non-empty JSON object of {field: value}"}
    try:
        validated = apply_update(state.settings_store, state.settings, payload)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    dumped = validated.model_dump()
    applied_hot: list[str] = []
    pending_restart: list[str] = []
    for key in payload:
        if key in HOT_RELOAD_FIELDS:
            # Mutating the LIVE settings object here, not `validated` as a
            # whole -- only the fields actually in HOT_RELOAD_FIELDS should
            # ever take effect without a restart; everything else in
            # `validated` (a full model reconstruction) is discarded except
            # for what apply_update already persisted to the store.
            setattr(state.settings, key, dumped[key])
            applied_hot.append(key)
        else:
            pending_restart.append(key)
    return 200, {"applied_hot": applied_hot, "pending_restart": pending_restart}


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
            elif self.path == "/settings":
                self._write(200, HTML_SETTINGS_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/settings":
                self._write_json(200, _settings_doc(state))
            elif self.path == "/api/health":
                status = state.health_status()
                http_status = 200 if status == "ok" else 503
                self._write_json(
                    http_status,
                    {
                        "status": status,
                        # "backend" (singular) preserved for existing
                        # consumers of this exact shape -- reports the
                        # streaming (parakeet) engine's backend, the one
                        # most relevant to "is this instance healthy for a
                        # live call." "backends" (plural, LS-5) is the full
                        # per-engine breakdown now that each engine's
                        # cpu/cuda choice is independent -- see
                        # live_stt/config.py's parakeet_backend/
                        # whisper_backend/parakeet_tdt_backend fields.
                        "backend": state.settings.parakeet_backend,
                        "backends": {
                            "parakeet": state.settings.parakeet_backend,
                            "whisper": state.settings.whisper_backend,
                            "parakeet_tdt": state.settings.parakeet_tdt_backend,
                        },
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
                diar = state.diarization_sessions
                xscribe = state.transcribe_sessions
                self._write_json(
                    200,
                    {
                        "active_calls": state.budget.active_calls,
                        "active_workers": state.budget.active_workers,
                        "max_concurrent_calls": state.budget.max_concurrent_calls,
                        "max_workers": state.budget.max_workers,
                        "draining": state.draining,
                        # None fields throughout when nvidia-smi is
                        # unavailable (CPU backend / no GPU) -- see
                        # live_stt/gpu.py's snapshot() docstring.
                        "gpu": gpu.snapshot(),
                        "diarization": {
                            "active": diar.active,
                            "completed_total": diar.completed_total,
                            "failed_total": diar.failed_total,
                            "rejected_vram_total": diar.rejected_vram_total,
                            # Small live list of in-flight requests
                            # (opaque per-process id, elapsed seconds,
                            # device) -- see diarize_sessions.py's
                            # docstring for why this is a deliberate,
                            # narrow exception to "not a session registry":
                            # no call-identifying info, bounded to exactly
                            # what's currently active.
                            "active_requests": diar.snapshot_active(),
                        },
                        # Batch transcription (POST /v1/audio/transcriptions)
                        # -- see live_stt/transcribe_sessions.py. Same shape
                        # as "diarization" above, plus rejected_capacity_total
                        # since this path (unlike diarization) also has a
                        # WorkerBudget capacity gate, not just VRAM.
                        # Deliberately separate from active_calls above,
                        # which conflates this with live gRPC streaming
                        # calls (both share the one WorkerBudget) -- this is
                        # the number that answers "how much of that is
                        # batch transcription specifically."
                        "transcribe": {
                            "active": xscribe.active,
                            "completed_total": xscribe.completed_total,
                            "failed_total": xscribe.failed_total,
                            "rejected_vram_total": xscribe.rejected_vram_total,
                            "rejected_capacity_total": xscribe.rejected_capacity_total,
                            "active_requests": xscribe.snapshot_active(),
                        },
                        # Text-to-Speech (POST /v1/audio/speech)
                        # -- see live_stt/tts_sessions.py.
                        "tts": {
                            "active": state.tts_sessions.active,
                            "completed_total": state.tts_sessions.completed_total,
                            "failed_total": state.tts_sessions.failed_total,
                            "rejected_vram_total": state.tts_sessions.rejected_vram_total,
                            "rejected_capacity_total": state.tts_sessions.rejected_capacity_total,
                            "active_requests": state.tts_sessions.snapshot_active(),
                        },
                    },
                )
            elif self.path == "/api/config":
                self._write_json(200, state.settings.model_dump())
            elif self.path == "/v1/models":
                # OpenAI-style listing ({"object": "list", "data": [...]})
                # covering ALL model registries this service has --
                # live_stt/models.py (ASR), live_stt/diarization_models.py (diarization),
                # and live_stt/tts_models.py (TTS).
                asr_data = [
                    {
                        "id": spec.key,
                        "object": "model",
                        "type": "asr",
                        "default": spec.key == state.settings.default_model,
                        "model_chunk_ms": spec.model_chunk_ms,
                        "has_eou": spec.has_eou,
                        "has_punctuation": spec.has_punctuation,
                        "multilingual": spec.multilingual,
                        # engine/streaming_capable (live_stt/models.py, added
                        # for the whisper.cpp addition) is what actually
                        # determines which SERVICE can serve this model --
                        # streaming_capable=False models are batch-only
                        # (POST /v1/audio/transcriptions), rejected with
                        # INVALID_ARGUMENT on the gRPC Transcribe RPC (see
                        # live_stt/servicer.py) -- surfaced here so the admin
                        # dashboard's "supported models by service" section
                        # can derive the split from real data instead of a
                        # second hardcoded list.
                        "engine": spec.engine,
                        "streaming_capable": spec.streaming_capable,
                    }
                    for spec in models.MODELS.values()
                ]
                diarization_data = [
                    {
                        "id": spec.key,
                        "object": "model",
                        "type": "diarization",
                        "default": spec.key == state.settings.diarization_model,
                        "gated": spec.gated,
                        "supports_num_speakers_hint": spec.supports_num_speakers_hint,
                        "measured_peak_vram_mb": spec.measured_peak_vram_mb,
                    }
                    for spec in diarization_models.DIARIZATION_MODELS.values()
                ]
                tts_data = [
                    {
                        "id": spec.key,
                        "object": "model",
                        "type": "tts",
                        "default": spec.key == state.settings.tts_model,
                        "engine": spec.engine,
                        "sample_rate": spec.sample_rate,
                        "default_voice": spec.default_voice,
                        "supported_voices": list(spec.supported_voices),
                        "supports_streaming": spec.supports_streaming,
                        "measured_peak_vram_mb": spec.measured_peak_vram_mb,
                    }
                    for spec in tts_models.TTS_MODELS.values()
                ]
                self._write_json(200, {"object": "list", "data": asr_data + diarization_data + tts_data})
            elif self.path == "/metrics":
                self._write(200, generate_latest(), "text/plain; version=0.0.4; charset=utf-8")
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
            if self.path == diarize_http.DIARIZE_PATH:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "")
                status, doc = diarize_http.handle_diarize_request(
                    content_type=content_type,
                    body=body,
                    settings=state.settings,
                    tracker=state.diarization_sessions,
                )
                self._write_json(status, doc)
            elif self.path == transcribe_http.TRANSCRIBE_PATH:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "")
                # ThreadingHTTPServer gives this request its own OS thread
                # with no pre-existing event loop, so asyncio.run() here is
                # safe -- CallSession/WorkerHandle are async (the same
                # production code servicer.Transcribe drives), and this
                # handler is otherwise plain synchronous BaseHTTPRequestHandler
                # code, same as diarize_http's branch above.
                status, response_body, content_type = asyncio.run(
                    transcribe_http.handle_transcribe_request(
                        content_type=content_type,
                        body=body,
                        settings=state.settings,
                        budget=state.budget,
                        draining=state.draining,
                        tracker=state.transcribe_sessions,
                    )
                )
                self._write(status, response_body, content_type)
            elif self.path == tts_http.SPEECH_PATH:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                content_type = self.headers.get("Content-Type", "")
                status, response_data, resp_content_type, headers = tts_http.handle_speech_request(
                    content_type=content_type,
                    body=body,
                    settings=state.settings,
                    tracker=state.tts_sessions,
                    draining=state.draining,
                )
                if headers.get("Transfer-Encoding") == "chunked" and hasattr(response_data, "__iter__") and not isinstance(response_data, (bytes, bytearray)):
                    self.send_response(status)
                    self.send_header("Content-Type", resp_content_type)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    try:
                        for chunk in response_data:
                            if chunk:
                                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                                self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                else:
                    self._write(status, response_data, resp_content_type)
            else:
                self._write_json(404, {"error": "not found"})

        def do_PUT(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
            if self.path == "/api/settings":
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                status, doc = _apply_settings_update(state, body)
                self._write_json(status, doc)
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
