"""
Optional LLM-as-judge (run with --judge). Deterministic checks catch format and
hallucination problems; the judge scores the qualities regex can't (did the
follow-up actually probe the gap? is the coaching specific?).

Always routed to the strong model (see llmops/router.py: task "judge").
Skipped in dry-run mode.
"""
from agents.evaluator import extract_json
from config import chat

_SYSTEM = """You are a strict reviewer of an AI mock-interview system's output.
Score the OUTPUT against the RUBRIC from 1 (fails) to 5 (excellent). Be critical; do not default to 4.
Respond with ONLY JSON: {"score": <1-5>, "reason": "<one sentence>"}"""


def judge(output: str, rubric: str, context: str = "") -> dict:
    user = f"RUBRIC:\n{rubric}\n\nCONTEXT:\n{context}\n\nOUTPUT:\n{output}"
    raw = chat(_SYSTEM, user, temperature=0.0, json_mode=True, task="judge", prompt_version="judge@v1")
    data = extract_json(raw) or {}
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 5:
        return {"score": None, "reason": f"unparseable judge output: {raw[:80]}"}
    return {"score": float(score), "reason": str(data.get("reason", ""))}
