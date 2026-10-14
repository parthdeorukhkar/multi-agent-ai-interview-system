"""
Per-call telemetry: every LLM attempt is appended to a JSONL log
(logs/llm_calls.jsonl by default) and kept in memory for session summaries.

Record fields:
  ts, session_id, task, model, tier, route_reason, complexity, prompt_version,
  attempt, status (ok | invalid | error), error, latency_ms,
  input_tokens, output_tokens, est_cost_usd, escalated_from, fallback_from

Env:
  LLM_LOG=0          disable writing the file (in-memory stats still work)
  LLM_LOG_PATH=...   write somewhere else (evals do this)
"""
import json
import math
import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_RECORDS: list[dict] = []
_SESSION_ID = "adhoc"


def log_path() -> Path | None:
    if os.environ.get("LLM_LOG", "1") == "0":
        return None
    custom = os.environ.get("LLM_LOG_PATH")
    return Path(custom) if custom else Path(__file__).parent.parent / "logs" / "llm_calls.jsonl"


def set_session(session_id: str) -> None:
    global _SESSION_ID
    _SESSION_ID = session_id


def get_session() -> str:
    return _SESSION_ID


def record(**fields) -> dict:
    from datetime import datetime, timezone
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "session_id": _SESSION_ID, **fields}
    with _LOCK:
        _RECORDS.append(rec)
        path = log_path()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass  # telemetry must never break the interview
    return rec


def mark() -> int:
    """Index to pass to records_since() to scope stats to one run."""
    return len(_RECORDS)


def records_since(index: int = 0) -> list[dict]:
    return list(_RECORDS[index:])


def load_log(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(1, math.ceil(p / 100 * len(ordered)))
    return float(ordered[k - 1])


def _agg(recs: list[dict]) -> dict:
    lat = [r["latency_ms"] for r in recs if r.get("latency_ms") is not None]
    return {
        "calls": len(recs),
        "input_tokens": sum(r.get("input_tokens", 0) for r in recs),
        "output_tokens": sum(r.get("output_tokens", 0) for r in recs),
        "est_cost_usd": sum(r.get("est_cost_usd", 0.0) for r in recs),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
    }


def summarize(records: list[dict]) -> dict:
    total = _agg(records)
    ok_calls = [r for r in records if r.get("status") in ("ok", "invalid")]
    total.update({
        "errors": sum(1 for r in records if r.get("status") == "error"),
        "invalid": sum(1 for r in records if r.get("status") == "invalid"),
        "escalations": sum(1 for r in records if r.get("escalated_from")),
        "fallbacks": sum(1 for r in records if r.get("fallback_from")),
        "strong_share": (sum(1 for r in ok_calls if r.get("tier") == "strong") / len(ok_calls)) if ok_calls else 0.0,
    })
    by_model, by_task = {}, {}
    for r in records:
        by_model.setdefault(r.get("model", "?"), []).append(r)
        by_task.setdefault(r.get("task", "?"), []).append(r)
    total["by_model"] = {k: _agg(v) for k, v in sorted(by_model.items())}
    total["by_task"] = {k: _agg(v) for k, v in sorted(by_task.items())}
    return total


def format_summary(s: dict) -> str:
    lines = [
        f"LLM calls: {s['calls']}  | tokens in/out: {s['input_tokens']}/{s['output_tokens']}"
        f"  | est. cost: ${s['est_cost_usd']:.5f}",
        f"Latency p50/p95: {s['latency_p50_ms']:.0f}/{s['latency_p95_ms']:.0f} ms"
        f"  | strong-model share: {s['strong_share']:.0%}"
        f"  | escalations: {s['escalations']}  fallbacks: {s['fallbacks']}  errors: {s['errors']}",
        "",
        f"{'model':<28}{'calls':>6}{'in tok':>9}{'out tok':>9}{'cost $':>10}{'p50 ms':>9}{'p95 ms':>9}",
    ]
    for model, a in s["by_model"].items():
        lines.append(f"{model:<28}{a['calls']:>6}{a['input_tokens']:>9}{a['output_tokens']:>9}"
                     f"{a['est_cost_usd']:>10.5f}{a['latency_p50_ms']:>9.0f}{a['latency_p95_ms']:>9.0f}")
    lines += ["", f"{'task':<28}{'calls':>6}{'cost $':>10}{'p50 ms':>9}{'p95 ms':>9}"]
    for task, a in s["by_task"].items():
        lines.append(f"{task:<28}{a['calls']:>6}{a['est_cost_usd']:>10.5f}"
                     f"{a['latency_p50_ms']:>9.0f}{a['latency_p95_ms']:>9.0f}")
    return "\n".join(lines)
