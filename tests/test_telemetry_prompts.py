import json

from llmops import telemetry
from llmops.models import estimate_cost
from llmops.prompts import all_versions, load_prompt


def test_cost_estimate_fast_is_cheaper_than_strong():
    fast = estimate_cost("llama-3.1-8b-instant", 1000, 200)
    strong = estimate_cost("llama-3.3-70b-versatile", 1000, 200)
    assert 0 < fast < strong
    assert abs(estimate_cost("llama-3.3-70b-versatile", 1_000_000, 0) - 0.59) < 1e-9


def test_percentile_nearest_rank():
    assert telemetry.percentile([], 95) == 0.0
    assert telemetry.percentile([10, 20, 30, 40], 50) == 20
    assert telemetry.percentile([10, 20, 30, 40], 95) == 40


def test_summarize_counts_and_shares():
    recs = [
        {"model": "a", "tier": "fast", "task": "evaluator", "status": "invalid", "latency_ms": 100, "input_tokens": 10, "output_tokens": 5, "est_cost_usd": 0.001},
        {"model": "b", "tier": "strong", "task": "evaluator", "status": "ok", "latency_ms": 300, "input_tokens": 10, "output_tokens": 5, "est_cost_usd": 0.01, "escalated_from": "a"},
        {"model": "a", "tier": "fast", "task": "interviewer", "status": "error", "latency_ms": 50, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0},
    ]
    s = telemetry.summarize(recs)
    assert s["calls"] == 3 and s["errors"] == 1 and s["invalid"] == 1 and s["escalations"] == 1
    assert s["strong_share"] == 0.5  # 1 strong of 2 completed calls
    assert abs(s["est_cost_usd"] - 0.011) < 1e-9
    assert set(s["by_model"]) == {"a", "b"} and set(s["by_task"]) == {"evaluator", "interviewer"}
    assert "est. cost" in telemetry.format_summary(s)


def test_log_file_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "calls.jsonl"
    monkeypatch.setenv("LLM_LOG", "1")
    monkeypatch.setenv("LLM_LOG_PATH", str(path))
    telemetry.record(model="m", tier="fast", task="t", status="ok")
    rows = telemetry.load_log(path)
    assert rows and rows[-1]["model"] == "m" and "ts" in rows[-1]
    json.loads(path.read_text().splitlines()[-1])


def test_prompt_version_is_content_hash():
    v = load_prompt("evaluator").version
    assert v.startswith("evaluator@") and len(v.split("@")[1]) == 8
    assert load_prompt("evaluator").version == v
    assert set(all_versions()) == {"interviewer", "evaluator", "coach"}
