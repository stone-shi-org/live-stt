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
    diarization_installer,
    diarization_models,
    diarize_http,
    gpu,
    models,
    transcribe_http,
    tts_http,
    tts_installer,
    tts_models,
)
from live_stt.backends import installer
from live_stt.backends import registry as backend_registry
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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --panel-bg: #111827;
            --panel-header-bg: #162032;
            --panel-border: #1f293d;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --hover-bg: #1e293b;
            --input-bg: #0b1120;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding-bottom: 3rem;
        }

        header {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--panel-border);
            padding: 1.15rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .logo-title {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .logo-icon {
            font-size: 1.85rem;
            background: rgba(99, 102, 241, 0.2);
            padding: 0.45rem 0.55rem;
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.4);
            display: inline-flex;
            align-items: center;
            justify-content: center;
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

        /* Top Navigation Bar */
        .nav-bar {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(17, 24, 39, 0.95);
            border-bottom: 1px solid var(--panel-border);
            padding: 0 2rem;
            position: sticky;
            top: 0;
            z-index: 50;
            backdrop-filter: blur(10px);
        }

        .nav-tab {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.95rem 1.35rem;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        }

        .nav-tab:hover {
            color: var(--text-main);
            background: rgba(255, 255, 255, 0.03);
        }

        .nav-tab.active {
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
            font-weight: 600;
        }

        .nav-badge {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-blue);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 9999px;
            padding: 0.15rem 0.5rem;
            font-size: 0.7rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .container {
            max-width: 1400px;
            width: 100%;
            margin: 1.25rem auto 0 auto;
            padding: 0 1.5rem;
        }

        .tab-content {
            display: none;
            flex-direction: column;
            gap: 1.25rem;
        }

        .tab-content.active {
            display: flex;
        }

        /* Compact KPI Strip */
        .kpi-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.85rem;
        }

        .kpi-pill {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: border-color 0.15s;
        }

        .kpi-pill:hover {
            border-color: rgba(99, 102, 241, 0.35);
        }

        .kpi-icon {
            font-size: 1.35rem;
            background: rgba(255,255,255,0.04);
            padding: 0.4rem;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .kpi-data {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            min-width: 0;
        }

        .kpi-label {
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .kpi-val {
            font-size: 1.15rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
        }

        .kpi-sub {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        .kpi-badge {
            margin-left: auto;
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            border-radius: 9999px;
            background: rgba(255,255,255,0.06);
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Charts Grid */
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 1.25rem;
        }

        .section-card {
            background-color: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.25rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }

        .section-title {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--accent-blue);
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 0.65rem;
        }

        .live-tag {
            font-size: 0.7rem;
            color: var(--accent-green);
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Circular Gauges */
        .gauges-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.85rem;
            margin-bottom: 0.85rem;
        }

        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.85rem 0.5rem;
            gap: 0.55rem;
        }

        .gauge-svg-wrap {
            position: relative;
            width: 82px;
            height: 82px;
        }

        .gauge-svg {
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }

        .gauge-track {
            fill: none;
            stroke: rgba(255, 255, 255, 0.06);
            stroke-width: 8;
        }

        .gauge-fill {
            fill: none;
            stroke-width: 8;
            stroke-linecap: round;
            stroke-dasharray: 263.894;
            stroke-dashoffset: 263.894;
            transition: stroke-dashoffset 0.5s ease;
        }

        .calls-gauge { stroke: var(--accent-blue); background: var(--accent-blue); }
        .workers-gauge { stroke: var(--accent-indigo); background: var(--accent-indigo); }
        .vram-gauge { stroke: #a855f7; background: #a855f7; }

        .gauge-center {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .gauge-pct {
            font-size: 1.05rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        .gauge-unit {
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }

        .gauge-info {
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .gauge-name {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .gauge-detail {
            font-size: 0.7rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .mini-bar-bg {
            height: 4px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 0.2rem;
        }

        .mini-bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.4s ease;
        }

        /* Sparkline */
        .sparkline-section {
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .sparkline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .sparkline-title {
            font-weight: 600;
        }

        .sparkline-legend {
            display: flex;
            align-items: center;
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .legend-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 4px;
        }

        .sparkline-svg-wrap {
            width: 100%;
            height: 70px;
        }

        .sparkline-svg {
            width: 100%;
            height: 100%;
            overflow: visible;
        }

        /* Outcome Distribution Bars */
        .outcome-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .outcome-box {
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 0.75rem 0.95rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .outcome-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .outcome-name {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .outcome-name code {
            font-size: 0.75rem;
            background: rgba(255,255,255,0.06);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            color: var(--accent-blue);
        }

        .outcome-active-badge {
            font-size: 0.75rem;
            color: var(--accent-green);
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 9999px;
            padding: 0.15rem 0.6rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        .stacked-bar {
            height: 9px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            border: 1px solid var(--panel-border);
        }

        .bar-seg {
            height: 100%;
            transition: width 0.4s ease;
        }

        .seg-ok { background: var(--accent-green); }
        .seg-fail { background: var(--accent-red); }
        .seg-vram { background: var(--accent-yellow); }
        .seg-cap { background: #c084fc; }

        .outcome-metrics {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            font-size: 0.72rem;
        }

        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--text-muted);
        }

        .metric-pill strong {
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }

        .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
        }
        .dot.ok { background: var(--accent-green); }
        .dot.fail { background: var(--accent-red); }
        .dot.vram { background: var(--accent-yellow); }
        .dot.cap { background: #c084fc; }

        .raw-total-text {
            margin-left: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        /* Requests Grid at Bottom */
        .requests-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 1.25rem;
        }

        .request-table-card {
            margin-bottom: 0;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.25rem;
        }

        .config-group {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 1rem 1.25rem;
        }

        
        /* Models & Endpoints: Row Layout Styles */
        .service-row-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            margin-bottom: 1.25rem;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            transition: border-color var(--transition-fast), box-shadow var(--transition-fast);
        }

        .service-row-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }

        .service-row-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.25rem;
            background: rgba(15, 23, 42, 0.6);
            border-bottom: 1px solid var(--border-color);
            flex-wrap: wrap;
            gap: 0.75rem;
        }

        .service-row-title {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .service-icon {
            font-size: 1.35rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 38px;
            height: 38px;
            background: rgba(99, 102, 241, 0.12);
            border-radius: 8px;
            border: 1px solid rgba(99, 102, 241, 0.25);
        }

        .service-title-text {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .service-name {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .service-endpoint-desc {
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        .service-row-badges {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .endpoint-chip {
            font-family: monospace;
            font-size: 0.8rem;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            border: 1px solid transparent;
        }

        .chip-grpc {
            background: rgba(14, 165, 233, 0.12);
            color: var(--accent-cyan);
            border-color: rgba(14, 165, 233, 0.3);
        }

        .chip-http {
            background: rgba(99, 102, 241, 0.12);
            color: var(--accent-indigo);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .tag-chip {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.25rem 0.55rem;
            border-radius: 6px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .tag-streaming {
            background: rgba(16, 185, 129, 0.12);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .tag-http {
            background: rgba(59, 130, 246, 0.12);
            color: var(--accent-blue);
            border: 1px solid rgba(59, 130, 246, 0.25);
        }

        .tag-vram {
            background: rgba(245, 158, 11, 0.12);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.25);
        }

        .tag-tts {
            background: rgba(168, 85, 247, 0.12);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.25);
        }

        .service-table-wrap {
            padding: 0;
            overflow-x: auto;
        }

        .service-models-table {
            width: 100%;
            margin-bottom: 0;
        }

        .service-models-table thead th {
            background: rgba(15, 23, 42, 0.4);
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            padding: 0.65rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }

        .service-models-table td {
            padding: 0.8rem 1.25rem;
        }

        .service-models-table tr:not(:last-child) td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        /* Attribute Badges */
        .model-badge {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 500;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            margin-right: 0.4rem;
            line-height: 1.3;
        }

        .badge-def {
            background: rgba(99, 102, 241, 0.18);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.35);
            font-weight: 600;
        }

        .badge-feat {
            background: rgba(16, 185, 129, 0.14);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .badge-warn {
            background: rgba(245, 158, 11, 0.14);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.25);
        }

        /* Endpoints Table */
        .endpoints-table thead th {
            background: rgba(15, 23, 42, 0.4);
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            padding: 0.65rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            text-align: left;
        }

        .endpoints-table td {
            padding: 0.8rem 1.25rem;
            vertical-align: middle;
        }

        .endpoints-table tr:not(:last-child) td {
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .method-tag {
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            letter-spacing: 0.04em;
        }

        .method-get {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .btn-endpoint-open {
            display: inline-flex;
            align-items: center;
            font-size: 0.78rem;
            color: var(--accent-blue);
            text-decoration: none;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            background: rgba(59, 130, 246, 0.08);
            border: 1px solid rgba(59, 130, 246, 0.2);
            transition: all var(--transition-fast);
        }

        .btn-endpoint-open:hover {
            background: rgba(59, 130, 246, 0.2);
            border-color: var(--accent-blue);
            color: #fff;
        }

        .config-group-header {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
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
            padding: 0.45rem 0;
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

        .col-id { width: 100px; }
        .col-elapsed { width: 120px; }
        .col-info { width: auto; }

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
            padding: 0.55rem 1.1rem;
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

        /* Settings Tab UI */
        .settings-header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-blue));
            color: white;
            font-weight: 600;
            padding: 0.65rem 1.4rem;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 0.9rem;
            transition: opacity 0.2s, transform 0.1s;
        }
        .btn-primary:hover:not(:disabled) {
            opacity: 0.92;
            transform: translateY(-1px);
        }
        .btn-primary:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .settings-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: center;
            margin: 1.25rem 0;
            justify-content: space-between;
        }

        .search-box {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 0.45rem 0.85rem;
            color: var(--text-muted);
            min-width: 260px;
        }
        .search-box input {
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-main);
            font-family: inherit;
            font-size: 0.85rem;
            width: 100%;
        }

        .category-pills {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }

        .cat-pill {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid var(--panel-border);
            color: var(--text-muted);
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.15s;
            font-family: inherit;
        }
        .cat-pill:hover {
            color: var(--text-main);
            border-color: var(--accent-indigo);
        }
        .cat-pill.active {
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent-blue);
            border-color: var(--accent-blue);
            font-weight: 600;
        }

        .section-divider-row {
            background: rgba(30, 41, 59, 0.7) !important;
        }
        .section-divider-row td {
            padding: 0.65rem 0.75rem !important;
            border-top: 1px solid var(--panel-border);
            border-bottom: 1px solid var(--panel-border);
        }
        .section-divider-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-blue);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .settings-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .settings-table th {
            text-align: left;
            padding: 0.65rem 0.75rem;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--panel-border);
        }
        .settings-table td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.3);
            vertical-align: middle;
        }
        .settings-table tr:hover:not(.section-divider-row) {
            background: rgba(255, 255, 255, 0.02);
        }
        .settings-table input[type=text], .settings-table input[type=number] {
            background: var(--input-bg);
            color: var(--text-main);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.4rem 0.6rem;
            width: 100%;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            box-sizing: border-box;
        }
        .settings-table input[type=text]:focus, .settings-table input[type=number]:focus {
            border-color: var(--accent-blue);
            outline: none;
        }
        .settings-table input[type=checkbox] {
            accent-color: var(--accent-indigo);
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .settings-table button {
            background: var(--hover-bg);
            color: var(--text-main);
            border: 1px solid var(--panel-border);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            font-size: 0.8rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.15s;
        }
        .settings-table button:hover {
            background: var(--accent-indigo);
            border-color: var(--accent-indigo);
            color: white;
        }

        .badge.hot { background: rgba(16,185,129,0.2); color: var(--accent-green); }
        .badge.restart { background: rgba(245,158,11,0.2); color: var(--accent-yellow); }

        #banner, #backendsBanner, #diarizationBanner, #ttsBanner {
            display: none;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 0.85rem;
        }
        #banner.ok, #backendsBanner.ok, #diarizationBanner.ok, #ttsBanner.ok {
            display: block !important;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-green);
            color: #34d399;
        }
        #banner.err, #backendsBanner.err, #diarizationBanner.err, #ttsBanner.err {
            display: block !important;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--accent-red);
            color: #f87171;
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

    <nav class="nav-bar">
        <button class="nav-tab active" id="nav-status" onclick="switchTab('status')">
            <span>📊</span>
            <span>Running Status</span>
            <span class="nav-badge" id="tabBadgeCalls">0 active</span>
        </button>
        <button class="nav-tab" id="nav-models" onclick="switchTab('models')">
            <span>🧩</span>
            <span>Models & Endpoints</span>
        </button>
        <button class="nav-tab" id="nav-settings" onclick="switchTab('settings')">
            <span>⚙️</span>
            <span>Configuration & Settings</span>
        </button>
    </nav>

    <div class="container">
        <!-- TAB 1: RUNNING STATUS -->
        <div id="tab-status" class="tab-content active">
            <!-- Compact KPI Strip -->
            <div class="kpi-strip">
                <div class="kpi-pill">
                    <span class="kpi-icon">📞</span>
                    <div class="kpi-data">
                        <div class="kpi-label">Active Calls</div>
                        <div class="kpi-val"><span id="valActiveCalls">0</span> <span class="kpi-sub" id="subActiveCalls">/ Max -</span></div>
                    </div>
                </div>
                <div class="kpi-pill">
                    <span class="kpi-icon">👷</span>
                    <div class="kpi-data">
                        <div class="kpi-label">Active Workers</div>
                        <div class="kpi-val"><span id="valActiveWorkers">0</span> <span class="kpi-sub" id="subActiveWorkers">/ Max -</span></div>
                    </div>
                </div>
                <div class="kpi-pill">
                    <span class="kpi-icon">🛡️</span>
                    <div class="kpi-data">
                        <div class="kpi-label">Reserve Slots</div>
                        <div class="kpi-val" id="valReserveSlots">-</div>
                    </div>
                </div>
                <div class="kpi-pill">
                    <span class="kpi-icon">🖥️</span>
                    <div class="kpi-data">
                        <div class="kpi-label">Server State</div>
                        <div class="kpi-val" id="valServerState">OK</div>
                    </div>
                    <span class="kpi-badge" id="subDraining">Draining: False</span>
                </div>
                <div class="kpi-pill" id="kpiGpuPill">
                    <span class="kpi-icon">🎮</span>
                    <div class="kpi-data">
                        <div class="kpi-label">GPU Free VRAM</div>
                        <div class="kpi-val"><span id="valGpuVramFree">-</span> <span class="kpi-sub" id="subGpuVramTotal">of -</span></div>
                    </div>
                    <span class="kpi-badge" id="valGpuUtil">-</span>
                </div>
            </div>

            <!-- Charts Section (2 Columns) -->
            <div class="charts-grid">
                <!-- Column 1: Resource Capacity Gauges & Sparkline -->
                <div class="section-card">
                    <div class="section-title">
                        <span>📈 Resource Capacity & Real-Time Telemetry</span>
                        <span class="live-tag"><span class="pulse-dot"></span> LIVE 3s</span>
                    </div>

                    <div class="gauges-row">
                        <!-- Gauge 1: Calls -->
                        <div class="gauge-container">
                            <div class="gauge-svg-wrap">
                                <svg viewBox="0 0 100 100" class="gauge-svg">
                                    <circle cx="50" cy="50" r="42" class="gauge-track" />
                                    <circle cx="50" cy="50" r="42" class="gauge-fill calls-gauge" id="gaugeCircleCalls" />
                                </svg>
                                <div class="gauge-center">
                                    <span class="gauge-pct" id="gaugePctCalls">0%</span>
                                    <span class="gauge-unit">Calls</span>
                                </div>
                            </div>
                            <div class="gauge-info">
                                <div class="gauge-name">Call Capacity</div>
                                <div class="gauge-detail" id="callUsageText">0 / 0 (0%)</div>
                                <div class="mini-bar-bg"><div class="mini-bar-fill calls-gauge" id="callProgressBar" style="width:0%"></div></div>
                            </div>
                        </div>

                        <!-- Gauge 2: Workers -->
                        <div class="gauge-container">
                            <div class="gauge-svg-wrap">
                                <svg viewBox="0 0 100 100" class="gauge-svg">
                                    <circle cx="50" cy="50" r="42" class="gauge-track" />
                                    <circle cx="50" cy="50" r="42" class="gauge-fill workers-gauge" id="gaugeCircleWorkers" />
                                </svg>
                                <div class="gauge-center">
                                    <span class="gauge-pct" id="gaugePctWorkers">0%</span>
                                    <span class="gauge-unit">Workers</span>
                                </div>
                            </div>
                            <div class="gauge-info">
                                <div class="gauge-name">Worker Pool</div>
                                <div class="gauge-detail" id="workerUsageText">0 / 0 (0%)</div>
                                <div class="mini-bar-bg"><div class="mini-bar-fill workers-gauge" id="workerProgressBar" style="width:0%"></div></div>
                            </div>
                        </div>

                        <!-- Gauge 3: GPU VRAM -->
                        <div class="gauge-container" id="vramProgressRow">
                            <div class="gauge-svg-wrap">
                                <svg viewBox="0 0 100 100" class="gauge-svg">
                                    <circle cx="50" cy="50" r="42" class="gauge-track" />
                                    <circle cx="50" cy="50" r="42" class="gauge-fill vram-gauge" id="gaugeCircleVram" />
                                </svg>
                                <div class="gauge-center">
                                    <span class="gauge-pct" id="gaugePctVram">0%</span>
                                    <span class="gauge-unit">VRAM</span>
                                </div>
                            </div>
                            <div class="gauge-info">
                                <div class="gauge-name">GPU VRAM</div>
                                <div class="gauge-detail" id="vramUsageText">- / - (0%)</div>
                                <div class="mini-bar-bg"><div class="mini-bar-fill vram-gauge" id="vramProgressBar" style="width:0%"></div></div>
                            </div>
                        </div>
                    </div>

                    <!-- Sparkline Live Trend -->
                    <div class="sparkline-section">
                        <div class="sparkline-header">
                            <span class="sparkline-title">Real-Time Activity Stream Waveform (Last 90s)</span>
                            <div class="sparkline-legend">
                                <span class="legend-dot" style="background:var(--accent-blue);"></span> Calls
                                <span class="legend-dot" style="background:var(--accent-indigo);margin-left:8px;"></span> Workers
                            </div>
                        </div>
                        <div class="sparkline-svg-wrap">
                            <svg id="activitySparkline" viewBox="0 0 400 70" preserveAspectRatio="none" class="sparkline-svg">
                                <defs>
                                    <linearGradient id="callsGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stop-color="var(--accent-blue)" stop-opacity="0.35" />
                                        <stop offset="100%" stop-color="var(--accent-blue)" stop-opacity="0.0" />
                                    </linearGradient>
                                </defs>
                                <line x1="0" y1="35" x2="400" y2="35" stroke="rgba(255,255,255,0.06)" stroke-dasharray="3,3" />
                                <path id="sparklineAreaCalls" d="" fill="url(#callsGrad)" />
                                <path id="sparklinePathCalls" d="" fill="none" stroke="var(--accent-blue)" stroke-width="2" />
                                <path id="sparklinePathWorkers" d="" fill="none" stroke="var(--accent-indigo)" stroke-width="1.8" stroke-dasharray="4,2" />
                            </svg>
                        </div>
                    </div>
                </div>

                <!-- Column 2: Service Request Outcome Distribution Charts -->
                <div class="section-card">
                    <div class="section-title">
                        <span>📊 Request Outcome Distribution Charts</span>
                    </div>

                    <div class="outcome-list">
                        <!-- Diarization -->
                        <div class="outcome-box">
                            <div class="outcome-top">
                                <div class="outcome-name">
                                    <span>🎤</span>
                                    <span>Post-Call Diarization</span>
                                </div>
                                <span class="outcome-active-badge"><span id="valDiarizeActive">0</span> in-flight</span>
                            </div>
                            <div class="stacked-bar">
                                <div class="bar-seg seg-ok" id="diarSegOk" style="width:0%"></div>
                                <div class="bar-seg seg-fail" id="diarSegFail" style="width:0%"></div>
                                <div class="bar-seg seg-vram" id="diarSegVram" style="width:0%"></div>
                            </div>
                            <div class="outcome-metrics">
                                <div class="metric-pill"><span class="dot ok"></span> OK: <strong id="diarOkCount">0</strong></div>
                                <div class="metric-pill"><span class="dot fail"></span> Failed: <strong id="diarFailCount">0</strong></div>
                                <div class="metric-pill"><span class="dot vram"></span> VRAM Rej: <strong id="diarVramCount">0</strong></div>
                                <span id="valDiarizeTotals" class="raw-total-text">0 / 0 / 0</span>
                            </div>
                        </div>

                        <!-- Batch Transcribe -->
                        <div class="outcome-box">
                            <div class="outcome-top">
                                <div class="outcome-name">
                                    <span>📝</span>
                                    <span>Batch Transcriptions (<code>/v1/audio/transcriptions</code>)</span>
                                </div>
                                <span class="outcome-active-badge"><span id="valTranscribeActive">0</span> in-flight</span>
                            </div>
                            <div class="stacked-bar">
                                <div class="bar-seg seg-ok" id="xscribeSegOk" style="width:0%"></div>
                                <div class="bar-seg seg-fail" id="xscribeSegFail" style="width:0%"></div>
                                <div class="bar-seg seg-vram" id="xscribeSegVram" style="width:0%"></div>
                                <div class="bar-seg seg-cap" id="xscribeSegCap" style="width:0%"></div>
                            </div>
                            <div class="outcome-metrics">
                                <div class="metric-pill"><span class="dot ok"></span> OK: <strong id="xscribeOkCount">0</strong></div>
                                <div class="metric-pill"><span class="dot fail"></span> Failed: <strong id="xscribeFailCount">0</strong></div>
                                <div class="metric-pill"><span class="dot vram"></span> VRAM: <strong id="xscribeVramCount">0</strong></div>
                                <div class="metric-pill"><span class="dot cap"></span> Capacity: <strong id="xscribeCapCount">0</strong></div>
                                <span id="valTranscribeTotals" class="raw-total-text">0 / 0 / 0 / 0</span>
                            </div>
                        </div>

                        <!-- Speech Synthesis (TTS) -->
                        <div class="outcome-box">
                            <div class="outcome-top">
                                <div class="outcome-name">
                                    <span>🔊</span>
                                    <span>Speech Synthesis (<code>/v1/audio/speech</code>)</span>
                                </div>
                                <span class="outcome-active-badge"><span id="valTtsActive">0</span> in-flight</span>
                            </div>
                            <div class="stacked-bar">
                                <div class="bar-seg seg-ok" id="ttsSegOk" style="width:0%"></div>
                                <div class="bar-seg seg-fail" id="ttsSegFail" style="width:0%"></div>
                                <div class="bar-seg seg-vram" id="ttsSegVram" style="width:0%"></div>
                                <div class="bar-seg seg-cap" id="ttsSegCap" style="width:0%"></div>
                            </div>
                            <div class="outcome-metrics">
                                <div class="metric-pill"><span class="dot ok"></span> OK: <strong id="ttsOkCount">0</strong></div>
                                <div class="metric-pill"><span class="dot fail"></span> Failed: <strong id="ttsFailCount">0</strong></div>
                                <div class="metric-pill"><span class="dot vram"></span> VRAM: <strong id="ttsVramCount">0</strong></div>
                                <div class="metric-pill"><span class="dot cap"></span> Capacity: <strong id="ttsCapCount">0</strong></div>
                                <span id="valTtsTotals" class="raw-total-text">0 / 0 / 0 / 0</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom: Detailed Requests Info Tables -->
            <div class="requests-grid">
                <div class="section-card request-table-card">
                    <div class="section-title">
                        <span>🎤 Active Diarization Requests</span>
                    </div>
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

                <div class="section-card request-table-card">
                    <div class="section-title">
                        <span>📝 Active Transcript Requests</span>
                    </div>
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

                <div class="section-card request-table-card">
                    <div class="section-title">
                        <span>🔊 Active Speech Synthesis Requests</span>
                    </div>
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
            </div>
        </div>

        <!-- TAB 2: MODELS & ENDPOINTS -->
        <div id="tab-models" class="tab-content">
            <!-- Row 1: Live Call (gRPC Transcribe) -->
            <div class="service-row-card">
                <div class="service-row-header">
                    <div class="service-row-title">
                        <span class="service-icon">📞</span>
                        <div class="service-title-text">
                            <span class="service-name">Live Call (Streaming ASR)</span>
                            <span class="service-endpoint-desc">Low-latency bidirectional streaming speech-to-text</span>
                        </div>
                    </div>
                    <div class="service-row-badges">
                        <code class="endpoint-chip chip-grpc">gRPC Transcribe</code>
                        <span class="tag-chip tag-streaming">Streaming Capable</span>
                    </div>
                </div>
                <div class="service-table-wrap">
                    <table class="kv-table service-models-table" id="svcLiveCallTable">
                        <thead>
                            <tr>
                                <th style="width: 55%;">Model Identifier</th>
                                <th style="width: 45%;">Attributes & Capabilities</th>
                            </tr>
                        </thead>
                        <tbody id="svcLiveCallTbody">
                            <tr><td colspan="2" class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Row 2: Batch Transcript (HTTP POST /v1/audio/transcriptions) -->
            <div class="service-row-card">
                <div class="service-row-header">
                    <div class="service-row-title">
                        <span class="service-icon">📝</span>
                        <div class="service-title-text">
                            <span class="service-name">Batch Audio Transcription</span>
                            <span class="service-endpoint-desc">OpenAI-compatible file upload & transcription pipeline</span>
                        </div>
                    </div>
                    <div class="service-row-badges">
                        <code class="endpoint-chip chip-http">POST /v1/audio/transcriptions</code>
                        <span class="tag-chip tag-http">HTTP REST</span>
                    </div>
                </div>
                <div class="service-table-wrap">
                    <table class="kv-table service-models-table" id="svcTranscriptTable">
                        <thead>
                            <tr>
                                <th style="width: 55%;">Model Identifier</th>
                                <th style="width: 45%;">Attributes & Capabilities</th>
                            </tr>
                        </thead>
                        <tbody id="svcTranscriptTbody">
                            <tr><td colspan="2" class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Row 3: Diarization (HTTP POST /v1/audio/diarization) -->
            <div class="service-row-card">
                <div class="service-row-header">
                    <div class="service-row-title">
                        <span class="service-icon">🎤</span>
                        <div class="service-title-text">
                            <span class="service-name">Speaker Diarization</span>
                            <span class="service-endpoint-desc">Multi-speaker segmentation & speaker clustering</span>
                        </div>
                    </div>
                    <div class="service-row-badges">
                        <code class="endpoint-chip chip-http">POST /v1/audio/diarization</code>
                        <span class="tag-chip tag-vram">VRAM Gated</span>
                    </div>
                </div>
                <div class="service-table-wrap">
                    <table class="kv-table service-models-table" id="svcDiarizationTable">
                        <thead>
                            <tr>
                                <th style="width: 55%;">Model Identifier</th>
                                <th style="width: 45%;">Attributes & Capabilities</th>
                            </tr>
                        </thead>
                        <tbody id="svcDiarizationTbody">
                            <tr><td colspan="2" class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Row 4: Text-to-Speech (HTTP POST /v1/audio/speech) -->
            <div class="service-row-card">
                <div class="service-row-header">
                    <div class="service-row-title">
                        <span class="service-icon">🔊</span>
                        <div class="service-title-text">
                            <span class="service-name">Text-to-Speech (TTS)</span>
                            <span class="service-endpoint-desc">OpenAI-compatible speech synthesis & audio streaming</span>
                        </div>
                    </div>
                    <div class="service-row-badges">
                        <code class="endpoint-chip chip-http">POST /v1/audio/speech</code>
                        <span class="tag-chip tag-tts">TTS Engine</span>
                    </div>
                </div>
                <div class="service-table-wrap">
                    <table class="kv-table service-models-table" id="svcTtsTable">
                        <thead>
                            <tr>
                                <th style="width: 55%;">Model Identifier</th>
                                <th style="width: 45%;">Attributes & Capabilities</th>
                            </tr>
                        </thead>
                        <tbody id="svcTTSTbody">
                            <tr><td colspan="2" class="card-subtext">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Row 5: API & Telemetry Endpoints Table -->
            <div class="service-row-card">
                <div class="service-row-header">
                    <div class="service-row-title">
                        <span class="service-icon">🔗</span>
                        <div class="service-title-text">
                            <span class="service-name">API & Telemetry Endpoints</span>
                            <span class="service-endpoint-desc">System health probes, configuration, and monitoring feeds</span>
                        </div>
                    </div>
                    <div class="links-row" style="margin: 0;">
                        <a href="/api/health" target="_blank" class="btn-link">🏥 /api/health</a>
                        <a href="/api/stats" target="_blank" class="btn-link">📈 /api/stats</a>
                        <a href="/api/config" target="_blank" class="btn-link">🛠️ /api/config</a>
                        <a href="/api/version" target="_blank" class="btn-link">ℹ️ /api/version</a>
                        <a href="/v1/models" target="_blank" class="btn-link">🧩 /v1/models</a>
                        <a href="/metrics" target="_blank" class="btn-link">📊 /metrics</a>
                    </div>
                </div>
                <div class="service-table-wrap">
                    <table class="kv-table endpoints-table">
                        <thead>
                            <tr>
                                <th style="width: 10%;">Method</th>
                                <th style="width: 25%;">Endpoint Path</th>
                                <th style="width: 53%;">Description</th>
                                <th style="width: 12%; text-align: right;">Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/api/health</code></td>
                                <td class="card-subtext">Liveness probe and worker admission capacity check (HTTP 200 / 503)</td>
                                <td style="text-align: right;"><a href="/api/health" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/api/stats</code></td>
                                <td class="card-subtext">Real-time JSON stats for calls, workers, diarization, batch transcription & TTS</td>
                                <td style="text-align: right;"><a href="/api/stats" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/api/config</code></td>
                                <td class="card-subtext">Active configuration, worker concurrency limits, timeouts, and engine settings</td>
                                <td style="text-align: right;"><a href="/api/config" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/api/version</code></td>
                                <td class="card-subtext">Git commit hash and Parakeet runtime engine build reference</td>
                                <td style="text-align: right;"><a href="/api/version" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/v1/models</code></td>
                                <td class="card-subtext">OpenAI-compatible registry of all available ASR, diarization & TTS models</td>
                                <td style="text-align: right;"><a href="/v1/models" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                            <tr>
                                <td><span class="method-tag method-get">GET</span></td>
                                <td><code>/metrics</code></td>
                                <td class="card-subtext">Prometheus format metrics exporter for Prometheus/Grafana scrapers</td>
                                <td style="text-align: right;"><a href="/metrics" target="_blank" class="btn-endpoint-open">Open ↗</a></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- TAB 3: CONFIGURATION & SETTINGS -->
        <div id="tab-settings" class="tab-content">
            <!-- Active Runtime Configuration Summary -->
            <div class="section-card">
                <div class="section-title">
                    <span>⚙️ Active Runtime Overview</span>
                </div>
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

            <!-- Editable Settings Section -->
            <div class="section-card">
                <div class="settings-header-row">
                    <div>
                        <div class="section-title" style="border-bottom:none;margin-bottom:0.25rem;padding-bottom:0;">🛠️ Settings Configuration</div>
                        <p style="color:var(--text-muted);font-size:0.85rem;">
                            Fields badged <span class="badge hot" style="display:inline-flex;">hot</span> apply immediately on save.
                            Fields badged <span class="badge restart" style="display:inline-flex;">restart required</span> persist to store and apply on next process start.
                        </p>
                    </div>
                    <button id="saveBtn" disabled class="btn-primary">Save Changes</button>
                </div>

                <div id="banner"></div>

                <div class="settings-toolbar">
                    <div class="search-box">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                        <input type="text" id="settingsSearchInput" placeholder="Filter settings by name..." oninput="filterSettings()">
                    </div>
                    <div class="category-pills" id="categoryPills"></div>
                </div>

                <table class="settings-table">
                    <thead>
                        <tr>
                            <th style="width:38%;">Setting</th>
                            <th style="width:44%;">Value</th>
                            <th style="width:18%;">Applies</th>
                        </tr>
                    </thead>
                    <tbody id="settingsBody">
                        <tr><td colspan="3" class="card-subtext">Loading settings...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Engine Backends -->
            <div class="section-card">
                <div class="section-title">
                    <span>🚀 Engine Backends</span>
                </div>
                <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1rem;">
                    Registered upstream artifacts (<code>live_stt/backends/registry.py</code>) and what is installed/activated on this process (<code>live_stt/backends/installer.py</code>).
                </p>
                <div id="backendsBanner"></div>
                <table class="settings-table">
                    <thead>
                        <tr><th>Engine</th><th>Flavor</th><th>Registered tag</th><th>Wired</th><th>Installed versions</th><th>Actions</th></tr>
                    </thead>
                    <tbody id="backendsBody">
                        <tr><td colspan="6" class="card-subtext">Loading backends...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Diarization Engines -->
            <div class="section-card">
                <div class="section-title">
                    <span>🎤 Diarization Engines</span>
                </div>
                <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1rem;">
                    Post-call speaker diarization dependency trees (pyannote.audio+torch, NVIDIA Sortformer). Install once per deployment under mounted data volume.
                </p>
                <div id="diarizationBanner"></div>
                <table class="settings-table">
                    <thead>
                        <tr><th>Engine</th><th>Requirements file</th><th>Installed</th><th>Actions</th></tr>
                    </thead>
                    <tbody id="diarizationBody">
                        <tr><td colspan="4" class="card-subtext">Loading diarization engines...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- TTS Engines -->
            <div class="section-card">
                <div class="section-title">
                    <span>🔊 Text-to-Speech Engines</span>
                </div>
                <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:1rem;">
                    Text-to-speech engine dependency trees (kokoro, piper). Install once per deployment under mounted data volume.
                </p>
                <div id="ttsBanner"></div>
                <table class="settings-table">
                    <thead>
                        <tr><th>Engine</th><th>Requirements file</th><th>Installed</th><th>Actions</th></tr>
                    </thead>
                    <tbody id="ttsBody">
                        <tr><td colspan="4" class="card-subtext">Loading TTS engines...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const COPY_ICON_SVG = `<svg viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
        const CHECK_ICON_SVG = `<svg viewBox="0 0 24 24" style="stroke:var(--accent-green);"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

        // Tab Navigation
        function switchTab(tabName) {
            const tabs = ['status', 'models', 'settings'];
            if (!tabs.includes(tabName)) tabName = 'status';

            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

            const targetContent = document.getElementById('tab-' + tabName);
            const targetNav = document.getElementById('nav-' + tabName);
            if (targetContent && targetNav) {
                targetContent.classList.add('active');
                targetNav.classList.add('active');
                if (window.location.hash !== '#' + tabName) {
                    history.replaceState(null, '', '#' + tabName);
                }
            }
        }

        function initTabs() {
            let initialTab = 'status';
            if (window.location.pathname === '/settings' || window.location.pathname.endsWith('/settings')) {
                initialTab = 'settings';
            }
            const hash = (window.location.hash || '').replace('#', '');
            if (['status', 'models', 'settings', 'config'].includes(hash)) {
                initialTab = (hash === 'config') ? 'settings' : hash;
            }
            switchTab(initialTab);

            window.addEventListener('hashchange', () => {
                const h = (window.location.hash || '').replace('#', '');
                if (['status', 'models', 'settings', 'config'].includes(h)) {
                    switchTab((h === 'config') ? 'settings' : h);
                }
            });
        }

        // Copy-to-clipboard for model names
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

        // Real-Time Activity Waveform (Sparkline) Buffer
        const sparkHistory = [];
        const MAX_HISTORY = 30;

        function updateSparkline(calls, workers, maxCalls) {
            sparkHistory.push({ calls, workers });
            if (sparkHistory.length > MAX_HISTORY) sparkHistory.shift();

            const svgW = 400;
            const svgH = 70;
            const maxVal = Math.max(maxCalls || 1, 1, ...sparkHistory.map(d => Math.max(d.calls, d.workers)));
            const n = sparkHistory.length;
            if (n < 2) return;

            const step = svgW / (MAX_HISTORY - 1);
            const startX = svgW - (n - 1) * step;

            const callsPts = sparkHistory.map((d, i) => {
                const x = startX + i * step;
                const y = svgH - 8 - ((d.calls / maxVal) * (svgH - 16));
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            });

            const workersPts = sparkHistory.map((d, i) => {
                const x = startX + i * step;
                const y = svgH - 8 - ((d.workers / maxVal) * (svgH - 16));
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            });

            const pathCalls = document.getElementById('sparklinePathCalls');
            const areaCalls = document.getElementById('sparklineAreaCalls');
            const pathWorkers = document.getElementById('sparklinePathWorkers');

            if (pathCalls) pathCalls.setAttribute('d', 'M ' + callsPts.join(' L '));
            if (pathWorkers) pathWorkers.setAttribute('d', 'M ' + workersPts.join(' L '));
            if (areaCalls) {
                const firstX = startX.toFixed(1);
                const lastX = svgW.toFixed(1);
                areaCalls.setAttribute('d', `M ${firstX},${svgH} L ${callsPts.join(' L ')} L ${lastX},${svgH} Z`);
            }
        }

        function setGauge(circleId, pctId, pct) {
            const circle = document.getElementById(circleId);
            const text = document.getElementById(pctId);
            const C = 263.894;
            const p = Math.max(0, Math.min(100, Math.round(pct)));
            if (circle) circle.style.strokeDashoffset = C * (1 - p / 100);
            if (text) text.innerText = p + '%';
        }

        function updateStackedBar(segPrefix, ok, fail, vram, cap) {
            const total = (ok + fail + vram + (cap || 0)) || 0;
            const okPct = total ? (ok / total) * 100 : 0;
            const failPct = total ? (fail / total) * 100 : 0;
            const vramPct = total ? (vram / total) * 100 : 0;
            const capPct = total && cap ? (cap / total) * 100 : 0;

            const segOk = document.getElementById(segPrefix + 'SegOk');
            const segFail = document.getElementById(segPrefix + 'SegFail');
            const segVram = document.getElementById(segPrefix + 'SegVram');
            const segCap = document.getElementById(segPrefix + 'SegCap');

            if (segOk) segOk.style.width = okPct + '%';
            if (segFail) segFail.style.width = failPct + '%';
            if (segVram) segVram.style.width = vramPct + '%';
            if (segCap) segCap.style.width = capPct + '%';
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
                document.getElementById('subActiveCalls').innerText = '/ Max ' + maxCalls;
                document.getElementById('valActiveWorkers').innerText = activeWorkers;
                document.getElementById('subActiveWorkers').innerText = '/ Max ' + maxWorkers;
                document.getElementById('subDraining').innerText = 'Draining: ' + (stats.draining ? 'True' : 'False');

                const tabBadge = document.getElementById('tabBadgeCalls');
                if (tabBadge) tabBadge.innerText = `${activeCalls} active`;

                // Call Capacity Gauge & Bar
                const callPct = Math.min(100, Math.round((activeCalls / maxCalls) * 100));
                document.getElementById('callUsageText').innerText = `${activeCalls} / ${maxCalls} (${callPct}%)`;
                document.getElementById('callProgressBar').style.width = callPct + '%';
                setGauge('gaugeCircleCalls', 'gaugePctCalls', callPct);

                // Worker Capacity Gauge & Bar
                const workerPct = Math.min(100, Math.round((activeWorkers / maxWorkers) * 100));
                document.getElementById('workerUsageText').innerText = `${activeWorkers} / ${maxWorkers} (${workerPct}%)`;
                document.getElementById('workerProgressBar').style.width = workerPct + '%';
                setGauge('gaugeCircleWorkers', 'gaugePctWorkers', workerPct);

                // Update Sparkline Trend
                updateSparkline(activeCalls, activeWorkers, maxCalls);

                // GPU / VRAM
                const gpuInfo = stats.gpu || {};
                const vramRow = document.getElementById('vramProgressRow');
                const kpiGpuPill = document.getElementById('kpiGpuPill');
                if (gpuInfo.free_vram_mb != null && gpuInfo.total_vram_mb != null) {
                    const freeGb = (gpuInfo.free_vram_mb / 1024).toFixed(1);
                    const totalGb = (gpuInfo.total_vram_mb / 1024).toFixed(1);
                    document.getElementById('valGpuVramFree').innerText = freeGb + ' GB';
                    document.getElementById('subGpuVramTotal').innerText = 'of ' + totalGb + ' GB';

                    const usedMb = gpuInfo.used_vram_mb ?? (gpuInfo.total_vram_mb - gpuInfo.free_vram_mb);
                    const vramPct = Math.min(100, Math.round((usedMb / gpuInfo.total_vram_mb) * 100));
                    document.getElementById('vramUsageText').innerText =
                        `${(usedMb / 1024).toFixed(1)} GB / ${totalGb} GB (${vramPct}%)`;
                    document.getElementById('vramProgressBar').style.width = vramPct + '%';
                    setGauge('gaugeCircleVram', 'gaugePctVram', vramPct);
                    vramRow.style.display = '';
                    if (kpiGpuPill) kpiGpuPill.style.display = '';
                } else {
                    document.getElementById('valGpuVramFree').innerText = 'N/A';
                    document.getElementById('subGpuVramTotal').innerText = 'no GPU / nvidia-smi';
                    vramRow.style.display = 'none';
                    setGauge('gaugeCircleVram', 'gaugePctVram', 0);
                }
                const utilText = gpuInfo.utilization_pct != null ? gpuInfo.utilization_pct + '%' : 'N/A';
                document.getElementById('valGpuUtil').innerText = utilText;

                // Diarization Outcomes & Table
                const diar = stats.diarization || {};
                const diarOk = diar.completed_total ?? 0;
                const diarFail = diar.failed_total ?? 0;
                const diarVram = diar.rejected_vram_total ?? 0;
                document.getElementById('valDiarizeActive').innerText = diar.active ?? 0;
                document.getElementById('valDiarizeTotals').innerText = `${diarOk} / ${diarFail} / ${diarVram}`;
                document.getElementById('diarOkCount').innerText = diarOk;
                document.getElementById('diarFailCount').innerText = diarFail;
                document.getElementById('diarVramCount').innerText = diarVram;
                updateStackedBar('diar', diarOk, diarFail, diarVram);

                const activeRequests = diar.active_requests || [];
                const diarizeTbody = document.getElementById('diarizeActiveTbody');
                if (activeRequests.length === 0) {
                    diarizeTbody.innerHTML = '<tr><td colspan="3" class="card-subtext">No active diarization requests</td></tr>';
                } else {
                    diarizeTbody.innerHTML = activeRequests.map(r =>
                        `<tr><td class="kv-val col-id">#${r.id}</td><td class="kv-val col-elapsed">${r.elapsed_sec.toFixed(1)}s</td><td class="kv-val col-info">${r.device}</td></tr>`
                    ).join('');
                }

                // Batch Transcription Outcomes & Table
                const xscribe = stats.transcribe || {};
                const xOk = xscribe.completed_total ?? 0;
                const xFail = xscribe.failed_total ?? 0;
                const xVram = xscribe.rejected_vram_total ?? 0;
                const xCap = xscribe.rejected_capacity_total ?? 0;
                document.getElementById('valTranscribeActive').innerText = xscribe.active ?? 0;
                document.getElementById('valTranscribeTotals').innerText = `${xOk} / ${xFail} / ${xVram} / ${xCap}`;
                document.getElementById('xscribeOkCount').innerText = xOk;
                document.getElementById('xscribeFailCount').innerText = xFail;
                document.getElementById('xscribeVramCount').innerText = xVram;
                document.getElementById('xscribeCapCount').innerText = xCap;
                updateStackedBar('xscribe', xOk, xFail, xVram, xCap);

                const activeTranscribeRequests = xscribe.active_requests || [];
                const transcribeTbody = document.getElementById('transcribeActiveTbody');
                if (activeTranscribeRequests.length === 0) {
                    transcribeTbody.innerHTML = '<tr><td colspan="3" class="card-subtext">No active transcript requests</td></tr>';
                } else {
                    transcribeTbody.innerHTML = activeTranscribeRequests.map(r =>
                        `<tr><td class="kv-val col-id">#${r.id}</td><td class="kv-val col-elapsed">${r.elapsed_sec.toFixed(1)}s</td><td class="kv-val col-info">${r.model}</td></tr>`
                    ).join('');
                }

                // TTS Synthesis Outcomes & Table
                const tts = stats.tts || {};
                const tOk = tts.completed_total ?? 0;
                const tFail = tts.failed_total ?? 0;
                const tVram = tts.rejected_vram_total ?? 0;
                const tCap = tts.rejected_capacity_total ?? 0;
                document.getElementById('valTtsActive').innerText = tts.active ?? 0;
                document.getElementById('valTtsTotals').innerText = `${tOk} / ${tFail} / ${tVram} / ${tCap}`;
                document.getElementById('ttsOkCount').innerText = tOk;
                document.getElementById('ttsFailCount').innerText = tFail;
                document.getElementById('ttsVramCount').innerText = tVram;
                document.getElementById('ttsCapCount').innerText = tCap;
                updateStackedBar('tts', tOk, tFail, tVram, tCap);

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

            // Config runtime overview
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

            // Supported Models
            const badgePill = (b) => {
                if (!b) return '';
                let cls = 'badge-feat';
                if (b === 'default') cls = 'badge-def';
                else if (b === 'gated') cls = 'badge-warn';
                return `<span class="model-badge ${cls}">${b}</span>`;
            };
            const modelRow = (spec, badges) => {
                const validBadges = badges.filter(Boolean);
                const badgeHtml = validBadges.length
                    ? validBadges.map(badgePill).join('')
                    : '<span class="card-subtext">&mdash;</span>';
                return `<tr><td class="kv-key"><span class="model-name-cell">${spec.id}${copyBtn(spec.id)}</span></td><td class="kv-val">${badgeHtml}</td></tr>`;
            };
            const noModelsRow = '<tr><td colspan="2" class="card-subtext">No models registered</td></tr>';

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

        // Settings Management
        const SETTINGS_SECTIONS = [
            {
                id: "network",
                name: "Network & Service Ports",
                icon: "🌐",
                keys: ["grpc_host", "grpc_port", "admin_host", "admin_port", "db_path"]
            },
            {
                id: "asr",
                name: "ASR Engines & Models",
                icon: "🎙️",
                keys: [
                    "parakeet_backend", "whisper_backend", "parakeet_tdt_backend",
                    "default_model", "models_dir",
                    "worker_bin", "worker_ggml_lib_dir",
                    "worker_bin_whisper", "worker_ggml_lib_dir_whisper",
                    "worker_bin_parakeet_tdt", "worker_ggml_lib_dir_parakeet_tdt"
                ]
            },
            {
                id: "workers",
                name: "Worker Pool & Concurrency",
                icon: "⚡",
                keys: [
                    "n_threads_per_worker", "n_threads_whisper", "n_threads_parakeet_tdt",
                    "max_concurrent_calls", "reserve_slots",
                    "worker_rss_soft_kb", "rotate_after_sec", "rotation_overlap_sec"
                ]
            },
            {
                id: "gpu",
                name: "GPU & VRAM Budgets",
                icon: "🎮",
                keys: ["vram_per_worker_mb", "vram_reserve_mb", "diarization_vram_mb", "tts_vram_mb"]
            },
            {
                id: "timeouts",
                name: "Timeouts & Lifecycle Safety",
                icon: "⏱️",
                keys: [
                    "queue_max_sec", "ring_history_sec", "warn_behind_sec", "abort_behind_sec",
                    "max_call_sec", "idle_timeout_sec", "finalize_timeout_sec", "drain_timeout_sec"
                ]
            },
            {
                id: "diarization",
                name: "Speaker Diarization",
                icon: "🎤",
                keys: [
                    "diarization_model", "diarization_device", "diarization_hf_token",
                    "diarization_num_speakers", "diarization_min_speakers", "diarization_max_speakers",
                    "diarize_in_subprocess", "diarization_deps_dir"
                ]
            },
            {
                id: "tts",
                name: "Text-to-Speech (TTS)",
                icon: "🔊",
                keys: ["tts_model", "tts_voice", "tts_device", "tts_idle_timeout_sec", "tts_in_subprocess", "tts_deps_dir"]
            },
            {
                id: "system",
                name: "Redaction & Auto-Upgrade",
                icon: "🔒",
                keys: [
                    "transcript_log", "audio_dump", "allow_pii",
                    "auto_upgrade_enabled", "upgrade_check_interval_sec", "backends_dir"
                ]
            }
        ];

        let original = {};
        let hotFields = new Set();
        const dirty = {};
        let activeCategory = 'all';

        function cell(key, value, secId) {
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
            return `<tr class="setting-row" data-key="${key}" data-section="${secId}"><td><code>${key}</code></td><td>${input}</td><td>${badge}</td></tr>`;
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
            original = doc.settings || {};
            hotFields = new Set(doc.hot_reload_fields || []);
            const tbody = document.getElementById('settingsBody');

            const allKeys = new Set(Object.keys(original));
            const assignedKeys = new Set();

            let html = '';
            for (const sec of SETTINGS_SECTIONS) {
                const secKeys = sec.keys.filter(k => k in original);
                secKeys.forEach(k => assignedKeys.add(k));
                if (secKeys.length > 0) {
                    html += `<tr class="section-divider-row" data-section="${sec.id}"><td colspan="3"><div class="section-divider-title">${sec.icon} ${sec.name}</div></td></tr>`;
                    html += secKeys.map(k => cell(k, original[k], sec.id)).join('');
                }
            }

            const remainingKeys = Array.from(allKeys).filter(k => !assignedKeys.has(k)).sort();
            if (remainingKeys.length > 0) {
                html += `<tr class="section-divider-row" data-section="other"><td colspan="3"><div class="section-divider-title">📦 Other Settings</div></td></tr>`;
                html += remainingKeys.map(k => cell(k, original[k], 'other')).join('');
            }

            tbody.innerHTML = html || '<tr><td colspan="3" class="card-subtext">No settings found</td></tr>';

            tbody.querySelectorAll('input').forEach(el => {
                el.addEventListener('input', () => {
                    const key = el.dataset.key;
                    let val = el.value;
                    if (el.type === 'checkbox') val = el.checked;
                    else if (el.type === 'number') val = val === '' ? null : Number(val);
                    markDirty(key, val);
                });
            });

            renderCategoryPills(SETTINGS_SECTIONS, remainingKeys.length > 0);

            if (doc.store_available === false) {
                showBanner('err', 'No settings store configured on this process -- changes cannot be saved (see live_stt/server.py).');
            }
        }

        function renderCategoryPills(sections, hasOther) {
            const container = document.getElementById('categoryPills');
            let pills = `<button class="cat-pill ${activeCategory === 'all' ? 'active' : ''}" onclick="setCategoryFilter('all', this)">All</button>`;
            sections.forEach(s => {
                pills += `<button class="cat-pill ${activeCategory === s.id ? 'active' : ''}" onclick="setCategoryFilter('${s.id}', this)">${s.icon} ${s.name.split(' ')[0]}</button>`;
            });
            if (hasOther) {
                pills += `<button class="cat-pill ${activeCategory === 'other' ? 'active' : ''}" onclick="setCategoryFilter('other', this)">📦 Other</button>`;
            }
            container.innerHTML = pills;
        }

        function setCategoryFilter(cat, btn) {
            activeCategory = cat;
            document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
            if (btn) btn.classList.add('active');
            filterSettings();
        }

        function filterSettings() {
            const query = (document.getElementById('settingsSearchInput').value || '').toLowerCase().trim();
            const rows = document.querySelectorAll('#settingsBody tr.setting-row');
            const secDividers = document.querySelectorAll('#settingsBody tr.section-divider-row');

            const visibleSections = new Set();
            rows.forEach(r => {
                const key = r.dataset.key || '';
                const sec = r.dataset.section || '';
                const matchesCat = (activeCategory === 'all' || sec === activeCategory);
                const matchesQuery = !query || key.toLowerCase().includes(query);
                if (matchesCat && matchesQuery) {
                    r.style.display = '';
                    visibleSections.add(sec);
                } else {
                    r.style.display = 'none';
                }
            });

            secDividers.forEach(d => {
                const sec = d.dataset.section || '';
                d.style.display = visibleSections.has(sec) ? '' : 'none';
            });
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

        function showBackendsBanner(kind, text) {
            const b = document.getElementById('backendsBanner');
            b.className = kind;
            b.innerText = text;
        }

        async function installBackend(engine, flavor) {
            showBackendsBanner('ok', `Installing ${engine}:${flavor}... (may take a moment)`);
            const resp = await fetch('/api/backends/install', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({engine, flavor}),
            });
            const doc = await resp.json();
            if (resp.ok) {
                showBackendsBanner('ok', `Installed ${engine}:${flavor} -> ${doc.version}. Not activated yet.`);
            } else {
                showBackendsBanner('err', doc.error || 'install failed');
            }
            await loadBackends();
        }

        async function activateBackend(engine, flavor, version) {
            const resp = await fetch('/api/backends/activate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({engine, flavor, version}),
            });
            const doc = await resp.json();
            if (resp.ok) {
                showBackendsBanner('ok', `Activated ${engine}:${flavor}:${version}. New worker spawns will use it.`);
            } else {
                showBackendsBanner('err', doc.error || 'activate failed');
            }
            await loadBackends();
        }

        async function loadBackends() {
            const resp = await fetch('/api/backends');
            const doc = await resp.json();
            const tbody = document.getElementById('backendsBody');
            if (doc.store_available === false) {
                tbody.innerHTML = '<tr><td colspan="6" class="card-subtext">No settings store configured on this process.</td></tr>';
                return;
            }
            const rows = (doc.registered || []).map((spec) => {
                const installedVersions = (doc.installed || []).filter(
                    (b) => b.engine === spec.engine && b.flavor === spec.flavor
                );
                const versionsText = installedVersions.length
                    ? installedVersions.map((b) => `${b.version}${b.active ? ' (active)' : ''}`).join(', ')
                    : '(none installed)';
                const installBtn = `<button onclick="installBackend('${spec.engine}','${spec.flavor}')">Install ${spec.tag}</button>`;
                const activateBtns = installedVersions
                    .filter((b) => !b.active)
                    .map((b) => `<button onclick="activateBackend('${spec.engine}','${spec.flavor}','${b.version}')">Activate ${b.version}</button>`)
                    .join(' ');
                return `<tr><td><code>${spec.engine}</code></td><td>${spec.flavor}</td><td>${spec.tag}</td>` +
                    `<td>${spec.wired ? 'yes' : 'no (registered only)'}</td>` +
                    `<td>${versionsText}</td><td>${installBtn} ${activateBtns}</td></tr>`;
            });
            tbody.innerHTML = rows.join('') || '<tr><td colspan="6" class="card-subtext">No backends registered.</td></tr>';
        }

        function showDiarizationBanner(kind, text) {
            const b = document.getElementById('diarizationBanner');
            b.className = kind;
            b.innerText = text;
        }

        async function installDiarizationEngine(engine) {
            showDiarizationBanner('ok', `Installing ${engine}... (a real pip install, can take several minutes -- do not navigate away)`);
            const resp = await fetch('/api/diarization-engines/install', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({engine}),
            });
            const doc = await resp.json();
            if (resp.ok) {
                showDiarizationBanner('ok', `${doc.already_installed ? 'Already installed' : 'Installed'}: ${engine} -> ${doc.path}`);
            } else {
                showDiarizationBanner('err', doc.error || 'install failed');
            }
            await loadDiarizationEngines();
        }

        async function loadDiarizationEngines() {
            const resp = await fetch('/api/diarization-engines');
            const doc = await resp.json();
            const tbody = document.getElementById('diarizationBody');
            const rows = Object.entries(doc.engines || {}).map(([engine, info]) => {
                const installBtn = info.installed
                    ? ''
                    : `<button onclick="installDiarizationEngine('${engine}')">Install</button>`;
                return `<tr><td><code>${engine}</code></td><td>${info.requirements_file}</td>` +
                    `<td>${info.installed ? 'yes' : 'no'}</td><td>${installBtn}</td></tr>`;
            });
            tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="card-subtext">No diarization engines registered.</td></tr>';
        }

        function showTtsBanner(kind, text) {
            const b = document.getElementById('ttsBanner');
            b.className = kind;
            b.innerText = text;
        }

        async function installTtsEngine(engine) {
            showTtsBanner('ok', `Installing ${engine}... (a real pip install, can take several minutes -- do not navigate away)`);
            const resp = await fetch('/api/tts-engines/install', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({engine}),
            });
            const doc = await resp.json();
            if (resp.ok) {
                showTtsBanner('ok', `${doc.already_installed ? 'Already installed' : 'Installed'}: ${engine} -> ${doc.path}`);
            } else {
                showTtsBanner('err', doc.error || 'install failed');
            }
            await loadTtsEngines();
        }

        async function loadTtsEngines() {
            const resp = await fetch('/api/tts-engines');
            const doc = await resp.json();
            const tbody = document.getElementById('ttsBody');
            const rows = Object.entries(doc.engines || {}).map(([engine, info]) => {
                const installBtn = info.installed
                    ? ''
                    : `<button onclick="installTtsEngine('${engine}')">Install</button>`;
                return `<tr><td><code>${engine}</code></td><td>${info.requirements_file}</td>` +
                    `<td>${info.installed ? 'yes' : 'no'}</td><td>${installBtn}</td></tr>`;
            });
            tbody.innerHTML = rows.join('') || '<tr><td colspan="4" class="card-subtext">No TTS engines registered.</td></tr>';
        }

        // Initialize UI
        initTabs();
        document.getElementById('saveBtn').addEventListener('click', save);
        updateDashboard();
        load();
        loadBackends();
        loadDiarizationEngines();
        loadTtsEngines();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>
"""

# Settings page uses the unified multi-page dashboard UI, pre-routed to #settings
HTML_SETTINGS_PAGE = HTML_STATS_PAGE



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


def _backends_doc(state: ServerState) -> dict:
    """GET /api/backends: the registry (live_stt/backends/registry.py --
    what CAN be fetched) alongside what's actually been installed/activated
    on THIS process (live_stt/backends/installer.py). Deliberately excludes
    a live upstream-GitHub-latest-tag check (upgrade_checker.py's
    upstream_visibility_report()) -- that's a real network call per repo,
    which this read-only GET should not incur on every dashboard poll.
    """
    if state.settings_store is None:
        return {"store_available": False, "registered": [], "installed": []}
    # Reuse the settings store's OWN already-open connection (see
    # SettingsStore.connection's docstring) rather than independently
    # calling db.connect(state.settings.db_path) again -- besides being a
    # redundant connection, that second call re-derives the path from
    # Settings.db_path instead of whatever this store was ACTUALLY
    # constructed with, which is exactly what let a real bug slip through
    # unit tests locally (passed against a writable /app in the Docker
    # test image, then hit a real PermissionError on a bare CI agent where
    # /app doesn't exist) before being caught for real on Bamboo.
    installed = installer.list_installed(state.settings_store.connection)
    return {
        "store_available": True,
        "registered": [
            {
                "engine": spec.engine,
                "flavor": spec.flavor,
                "repo": spec.repo,
                "tag": spec.tag,
                "wired": spec.wired,
            }
            for spec in backend_registry.BACKENDS.values()
        ],
        "installed": [
            {
                "engine": b.engine,
                "flavor": b.flavor,
                "version": b.version,
                "path": b.path,
                "installed_at": b.installed_at,
                "active": b.active,
            }
            for b in installed
        ],
    }


def _handle_backend_install(state: ServerState, body: bytes) -> tuple[int, dict]:
    if state.settings_store is None:
        return 503, {"error": "no settings store configured on this process"}
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    engine, flavor = payload.get("engine"), payload.get("flavor")
    if not engine or not flavor:
        return 400, {"error": "expected a JSON object: {'engine': ..., 'flavor': ...}"}

    try:
        installed = installer.install(
            state.settings_store.connection, state.settings.backends_dir, engine, flavor
        )
    except KeyError as exc:
        return 400, {"error": str(exc)}
    except installer.BackendInstallError as exc:
        # Download/checksum/extraction/safety-probe failure -- an upstream
        # or network problem, not a bad request; 502 (not 500) says so.
        return 502, {"error": str(exc)}
    return 200, {
        "engine": installed.engine,
        "flavor": installed.flavor,
        "version": installed.version,
        "path": installed.path,
        "active": installed.active,
    }


def _handle_backend_activate(state: ServerState, body: bytes) -> tuple[int, dict]:
    if state.settings_store is None:
        return 503, {"error": "no settings store configured on this process"}
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    engine, flavor, version = payload.get("engine"), payload.get("flavor"), payload.get("version")
    if not engine or not flavor or not version:
        return 400, {"error": "expected a JSON object: {'engine': ..., 'flavor': ..., 'version': ...}"}

    try:
        installer.activate(state.settings_store.connection, engine, flavor, version)
    except installer.BackendInstallError as exc:
        return 400, {"error": str(exc)}
    return 200, {"activated": {"engine": engine, "flavor": flavor, "version": version}}


def _diarization_engines_doc(state: ServerState) -> dict:
    """GET /api/diarization-engines: every KNOWN diarization engine
    (live_stt/diarization_installer.py's REQUIREMENTS_FILES -- "pyannote",
    "nemo_sortformer") alongside whether its heavy dependency tree has
    actually been installed into THIS deployment's persistent
    diarization_deps_dir yet. The ASR analogue of GET /api/backends, for
    the same reason: a dashboard/operator needs to see what's installable
    vs. what's already usable before calling
    POST /api/diarization-engines/install.
    """
    return {"engines": diarization_installer.status(state.settings.diarization_deps_dir)}


def _handle_diarization_engine_install(state: ServerState, body: bytes) -> tuple[int, dict]:
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    engine = payload.get("engine")
    if not engine:
        return 400, {"error": "expected a JSON object: {'engine': 'pyannote' | 'nemo_sortformer'}"}

    try:
        result = diarization_installer.install(state.settings.diarization_deps_dir, engine)
    except diarization_installer.DiarizationInstallError as exc:
        # Same reasoning as _handle_backend_install's 502: this is a real
        # pip/network/disk failure, an unknown engine name is the one case
        # that's actually the caller's fault -- but distinguishing the two
        # here would mean parsing the exception message, which is exactly
        # the fragile string-matching this codebase avoids elsewhere (see
        # diarize_http.py's own handler for the same tradeoff already made
        # and accepted). 502 for all of it, consistent with the ASR path.
        return 502, {"error": str(exc)}
    return 200, {
        "engine": result.engine,
        "path": result.path,
        "already_installed": result.already_installed,
    }


def _tts_engines_doc(state: ServerState) -> dict:
    """GET /api/tts-engines: every KNOWN TTS engine (live_stt/
    tts_installer.py's REQUIREMENTS_FILES -- "kokoro", "piper") alongside
    whether its dependency tree has actually been installed into THIS
    deployment's persistent tts_deps_dir yet. Direct analogue of GET
    /api/diarization-engines, added for the same reason: before this,
    neither engine had an install path at all, and live_stt/tts.py used to
    silently substitute a synthetic tone for real speech when they weren't
    importable -- see that module's docstring for the real bug this closes.
    """
    return {"engines": tts_installer.status(state.settings.tts_deps_dir)}


def _handle_tts_engine_install(state: ServerState, body: bytes) -> tuple[int, dict]:
    try:
        payload = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    engine = payload.get("engine")
    if not engine:
        return 400, {"error": "expected a JSON object: {'engine': 'kokoro' | 'piper'}"}

    try:
        result = tts_installer.install(state.settings.tts_deps_dir, engine)
    except tts_installer.TTSInstallError as exc:
        # Same 502-for-everything tradeoff as _handle_diarization_engine_install.
        return 502, {"error": str(exc)}
    return 200, {
        "engine": result.engine,
        "path": result.path,
        "already_installed": result.already_installed,
    }


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
            elif self.path == "/api/backends":
                self._write_json(200, _backends_doc(state))
            elif self.path == "/api/diarization-engines":
                self._write_json(200, _diarization_engines_doc(state))
            elif self.path == "/api/tts-engines":
                self._write_json(200, _tts_engines_doc(state))
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
            elif self.path == "/api/backends/install":
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                status, doc = _handle_backend_install(state, body)
                self._write_json(status, doc)
            elif self.path == "/api/backends/activate":
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                status, doc = _handle_backend_activate(state, body)
                self._write_json(status, doc)
            elif self.path == "/api/diarization-engines/install":
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                status, doc = _handle_diarization_engine_install(state, body)
                self._write_json(status, doc)
            elif self.path == "/api/tts-engines/install":
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                status, doc = _handle_tts_engine_install(state, body)
                self._write_json(status, doc)
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
