"""Resilience behaviour of llmops.client: retry -> fallback -> escalation."""
import pytest

import config
from llmops import client, telemetry


@pytest.fixture(autouse=True)
def _real_path(monkeypatch):
    # exercise the real (non dry-run) code path but with a scripted _invoke
    monkeypatch.setattr(config, "DRY_RUN", False)
    monkeypatch.setattr(client, "BACKOFF_BASE_S", 0.0)
    monkeypatch.setattr(client, "MAX_RETRIES", 2)


class Boom(Exception):
    pass


def script(monkeypatch, behaviours):
    """behaviours: list of str results or Exceptions, consumed per _invoke call; also records models used."""
    calls = []

    def fake(model, system_prompt, user_message, temperature, json_mode):
        calls.append(model)
        b = behaviours.pop(0)
        if isinstance(b, Exception):
            raise b
        return b, 100, 20

    monkeypatch.setattr(client, "_invoke", fake)
    return calls


def test_happy_path_uses_routed_tier_and_logs(monkeypatch):
    calls = script(monkeypatch, ["hello"])
    mark = telemetry.mark()
    out = client.complete("coach", "sys", "user", prompt_version="coach@x")
    assert out.text == "hello" and out.tier == "strong" and not out.escalated
    rec = telemetry.records_since(mark)[0]
    assert rec["status"] == "ok" and rec["task"] == "coach" and rec["prompt_version"] == "coach@x"
    assert rec["input_tokens"] == 100 and rec["est_cost_usd"] > 0
    assert len(calls) == 1


def test_non_retryable_error_falls_back_to_other_tier(monkeypatch):
    calls = script(monkeypatch, [Boom("bad request"), "from fallback"])
    mark = telemetry.mark()
    out = client.complete("coach", "sys", "user")  # coach -> strong, fallback -> fast
    assert out.text == "from fallback" and out.fell_back and out.tier == "fast"
    recs = telemetry.records_since(mark)
    assert [r["status"] for r in recs] == ["error", "ok"]
    assert recs[1]["fallback_from"] is not None
    assert len(calls) == 2


def test_fast_tier_invalid_output_escalates_to_strong(monkeypatch):
    calls = script(monkeypatch, ["not json", '{"ok": true}'])
    mark = telemetry.mark()
    validator = lambda raw: (raw.startswith("{"), "bad format")
    out = client.complete("interviewer", "sys", "user", ctx={"turn": 1}, validator=validator)  # opener -> fast
    assert out.escalated and out.tier == "strong" and out.text == '{"ok": true}'
    recs = telemetry.records_since(mark)
    assert [r["status"] for r in recs] == ["invalid", "ok"]
    assert recs[1]["escalated_from"] is not None


def test_forced_mode_disables_fallback_and_escalation(monkeypatch):
    monkeypatch.setenv("ROUTING_MODE", "fast")
    script(monkeypatch, [Boom("down")])
    with pytest.raises(Boom):
        client.complete("coach", "sys", "user")

    calls = script(monkeypatch, ["still bad"])
    out = client.complete("coach", "sys", "user", validator=lambda r: (False, "nope"))
    assert not out.escalated and len(calls) == 1


class Transient(Exception):
    pass


def test_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(client, "_is_retryable", lambda e: isinstance(e, Transient))
    calls = script(monkeypatch, [Transient("429"), Transient("429"), "finally"])
    mark = telemetry.mark()
    out = client.complete("coach", "sys", "user")
    assert out.text == "finally" and len(calls) == 3 and not out.fell_back
    assert [r["attempt"] for r in telemetry.records_since(mark)] == [1, 2, 3]


def test_retries_exhausted_then_falls_back(monkeypatch):
    monkeypatch.setattr(client, "_is_retryable", lambda e: isinstance(e, Transient))
    calls = script(monkeypatch, [Transient("1"), Transient("2"), Transient("3"), "fallback ok"])
    out = client.complete("coach", "sys", "user")
    assert out.fell_back and out.text == "fallback ok" and len(calls) == 4
