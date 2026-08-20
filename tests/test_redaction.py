from __future__ import annotations

import logging

import pytest

from live_stt import redaction
from live_stt.config import Settings

CANARY = "the secret canary phrase nobody should see in logs"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_default_mode_never_emits_transcript_text() -> None:
    fields = redaction.transcript_log_fields(CANARY, "abc123", _settings())
    assert "text" not in fields
    assert CANARY not in " ".join(fields.values())
    assert fields["chars"] == str(len(CANARY))
    assert fields["words"] == str(len(CANARY.split()))


def test_hash_mode_emits_a_stable_fingerprint_not_the_text() -> None:
    settings = _settings(transcript_log="hash", allow_pii=False)  # hash needs no PII flag
    fields = redaction.transcript_log_fields(CANARY, "abc123", settings)
    assert CANARY not in fields.get("sha", "")
    assert "text" not in fields
    assert len(fields["sha"]) == 12


def test_hash_is_deterministic_and_case_insensitive_to_whitespace() -> None:
    a = redaction.transcript_hash("Hello   World")
    b = redaction.transcript_hash("hello world")
    assert a == b


def test_full_mode_without_allow_pii_refuses_at_startup() -> None:
    with pytest.raises(ValueError, match="LSTT_ALLOW_PII"):
        redaction.validate(_settings(transcript_log="full", allow_pii=False))


def test_sample_mode_without_allow_pii_refuses_at_startup() -> None:
    with pytest.raises(ValueError, match="LSTT_ALLOW_PII"):
        redaction.validate(_settings(transcript_log="sample:1in10", allow_pii=False))


def test_full_mode_with_allow_pii_passes_validation_and_logs_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="stt.redaction"):
        redaction.validate(_settings(transcript_log="full", allow_pii=True))
    assert any("PII-VISIBLE" in r.message for r in caplog.records)


def test_full_mode_emits_the_actual_text() -> None:
    settings = _settings(transcript_log="full", allow_pii=True)
    fields = redaction.transcript_log_fields(CANARY, "abc123", settings)
    assert fields["text"] == CANARY


def test_sample_mode_is_deterministic_on_call_ref_not_random() -> None:
    settings = _settings(transcript_log="sample:1in1", allow_pii=True)  # 1-in-1 => always sampled
    fields1 = redaction.transcript_log_fields(CANARY, "abc123", settings)
    fields2 = redaction.transcript_log_fields(CANARY, "abc123", settings)
    assert fields1 == fields2
    assert fields1.get("text") == CANARY


def test_audio_dump_without_allow_pii_refuses_at_startup() -> None:
    with pytest.raises(ValueError, match="LSTT_ALLOW_PII"):
        redaction.validate(_settings(audio_dump="on_error", allow_pii=False))


def test_audio_dump_off_is_always_allowed() -> None:
    redaction.validate(_settings(audio_dump="off", allow_pii=False))  # must not raise


def test_unknown_transcript_mode_rejected() -> None:
    with pytest.raises(ValueError, match="unknown LSTT_TRANSCRIPT_LOG"):
        redaction.validate(_settings(transcript_log="verbose"))


def test_hash_call_ref_never_reversible_and_never_the_raw_value() -> None:
    call_id = "+15551234567"
    hashed = redaction.hash_call_ref(call_id)
    assert call_id not in hashed
    assert len(hashed) == 16


def test_hash_call_ref_is_deterministic() -> None:
    assert redaction.hash_call_ref("call-1") == redaction.hash_call_ref("call-1")
    assert redaction.hash_call_ref("call-1") != redaction.hash_call_ref("call-2")
