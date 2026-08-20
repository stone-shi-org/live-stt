"""Redaction: two independent switches, because one env var should never be
a single typo away from logging every phone call in the building.

- ``LSTT_TRANSCRIPT_LOG`` (``off|hash|sample:1inN|full``) controls whether
  transcript TEXT ever reaches the logs. ``off`` (default) logs only counts.
  ``hash`` adds a stable, non-recoverable fingerprint -- useful for
  confirming two runs produced byte-identical output, without storing
  anything recoverable. ``sample:1inN`` and ``full`` require
  ``LSTT_ALLOW_PII=true`` as well, checked at startup by :func:`validate`,
  which refuses to start rather than silently downgrading to a safer mode.
- ``LSTT_AUDIO_DUMP`` (``off|on_error|always``) controls whether raw audio
  is ever persisted to disk. Also gated on ``LSTT_ALLOW_PII=true``. Not yet
  wired to an actual dump path -- Phase 3's backpressure ring buffer (which
  would hold the audio to dump) was never built, so this setting is
  validated but currently has nothing to act on. See CLAUDE.md.

Raw audio is NEVER logged, at any level, regardless of these switches --
there is no configuration that turns that on.
"""

from __future__ import annotations

import hashlib
import re

from live_stt.config import Settings
from live_stt.logging_config import get_logger

logger = get_logger("redaction")

_FIXED_MODES = {"off", "hash", "full"}


def transcript_hash(text: str) -> str:
    """Stable, non-recoverable fingerprint. Proves two runs produced
    byte-identical output without storing anything an attacker (or a
    curious operator) could recover the transcript from."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def hash_call_ref(call_ref: str, salt: str = "") -> str:
    """A phone number or SIP call-ID must never become a log key, a metric
    label, or a filename -- only this hash may."""
    return hashlib.sha256((salt + call_ref).encode()).hexdigest()[:16]


def _sample_n(mode: str) -> int | None:
    if mode.startswith("sample:1in"):
        try:
            n = int(mode[len("sample:1in") :])
            return n if n > 0 else None
        except ValueError:
            return None
    return None


def _is_valid_mode(mode: str) -> bool:
    return mode in _FIXED_MODES or _sample_n(mode) is not None


def _is_pii_mode(mode: str) -> bool:
    return mode == "full" or _sample_n(mode) is not None


def validate(settings: Settings) -> None:
    """Called once at startup (see live_stt/server.py). Raises ValueError on
    a nonsensical configuration; logs a loud, hard-to-miss banner if any
    PII-visible mode is genuinely active. Two switches means a single typo
    in one of them can't turn PII logging on by accident -- but a
    deliberate combination of both should still be impossible to miss in
    the startup log, not just quietly effective.
    """
    mode = settings.transcript_log
    if not _is_valid_mode(mode):
        raise ValueError(f"unknown LSTT_TRANSCRIPT_LOG mode: {mode!r}")

    transcript_pii = _is_pii_mode(mode)
    if transcript_pii and not settings.allow_pii:
        raise ValueError(
            f"LSTT_TRANSCRIPT_LOG={mode!r} requires LSTT_ALLOW_PII=true -- "
            "refusing to start rather than silently falling back to a safer mode"
        )

    if settings.audio_dump not in ("off", "on_error", "always"):
        raise ValueError(f"unknown LSTT_AUDIO_DUMP mode: {settings.audio_dump!r}")

    audio_pii = settings.audio_dump != "off"
    if audio_pii and not settings.allow_pii:
        raise ValueError(f"LSTT_AUDIO_DUMP={settings.audio_dump!r} requires LSTT_ALLOW_PII=true")

    if transcript_pii or audio_pii:
        logger.warning(
            "=" * 70
            + "\nPII-VISIBLE LOGGING IS ACTIVE: transcript_log=%s audio_dump=%s\n"
            "Call transcripts and/or raw audio may be written to logs or disk.\n"
            + "=" * 70,
            mode,
            settings.audio_dump,
        )


def transcript_log_fields(text: str, call_ref_hash: str, settings: Settings) -> dict[str, str]:
    """What to fold into a lifecycle log line for this call's transcript.
    Always includes char/word counts (never sensitive). Adds a hash in
    ``hash`` mode. Adds the actual text only in ``full`` mode, or in
    ``sample:1inN`` mode for a deterministic 1-in-N subset of calls keyed on
    the call's own hash -- deterministic (not random) so a specific
    problem call, once sampled, reproduces the same way on a retry.
    """
    words = text.split()
    fields = {"chars": str(len(text)), "words": str(len(words))}
    mode = settings.transcript_log

    if mode == "hash":
        fields["sha"] = transcript_hash(text)
    elif mode == "full":
        fields["text"] = text
    else:
        n = _sample_n(mode)
        if n and int(call_ref_hash, 16) % n == 0:
            fields["text"] = text
            fields["sampled"] = "1"
    return fields
