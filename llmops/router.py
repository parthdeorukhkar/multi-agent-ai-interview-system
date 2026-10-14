"""
Automatic model routing.

Each LLM call is classified by task + context and sent to the cheapest model
tier expected to do it well:

  fast   (Llama 3.1 8B)  - low latency / ~12x cheaper
  strong (Llama 3.3 70B) - better reasoning, domain knowledge, long synthesis

The policy is deliberately a small, readable, rule-based scorer rather than an
LLM call: routing must be cheap, fast and deterministic (so it can be unit
tested and evaluated). Every decision carries a human-readable `reason` that is
written to the telemetry log.

The router is only half of the story: client.py adds a safety net (if the fast
model returns invalid / low-confidence output the call is escalated to the
strong model), so routing mistakes cost latency, not correctness.

ROUTING_MODE env var:  auto (default) | fast | strong
  fast / strong force a single model - used by evals to prove that routing
  saves cost without losing accuracy versus the "one big model" baseline.
"""
import os
import re
from dataclasses import dataclass

_IDK = re.compile(
    r"\b(i don'?t know|no idea|not sure|can'?t (think|recall|remember)|i pass|skip (this|that))\b",
    re.IGNORECASE,
)

# score at/above which a task goes to the strong tier
EVALUATOR_STRONG_AT = 2.0
INTERVIEWER_STRONG_AT = 1.5

_DEEP_FOCUS = {"technical": 2.0, "case": 2.0, "mixed": 1.0, "behavioral": 0.0}


@dataclass(frozen=True)
class RoutingDecision:
    tier: str          # "fast" | "strong"
    reason: str        # human-readable, logged
    complexity: float  # the score that produced the decision (0 when forced)


def routing_mode() -> str:
    mode = os.environ.get("ROUTING_MODE", "auto").strip().lower()
    return mode if mode in ("auto", "fast", "strong") else "auto"


def _word_count(text: str) -> int:
    return len((text or "").split())


def route(task: str, ctx: dict | None = None) -> RoutingDecision:
    """
    task: "evaluator" | "interviewer" | "coach" | "judge" | anything else
    ctx keys (all optional): focus_area, answer, turn, total_turns,
                             last_verdict, resume_provided
    """
    ctx = ctx or {}
    mode = routing_mode()
    if mode in ("fast", "strong"):
        return RoutingDecision(mode, f"forced:{mode}", 0.0)

    if task == "coach":
        return RoutingDecision("strong", "long-form synthesis over the whole transcript; runs once", 5.0)
    if task == "judge":
        return RoutingDecision("strong", "LLM-as-judge must be at least as capable as the system under test", 5.0)
    if task == "evaluator":
        return _route_evaluator(ctx)
    if task == "interviewer":
        return _route_interviewer(ctx)
    return RoutingDecision("strong", "unknown task: default to the safest tier", 5.0)


def _route_evaluator(ctx: dict) -> RoutingDecision:
    answer = ctx.get("answer", "")
    words = _word_count(answer)

    if words < 4 or (words < 12 and _IDK.search(answer)):
        return RoutingDecision("fast", "trivial answer (empty / 'I don't know'): verdict is obvious", 0.0)

    score = _DEEP_FOCUS.get(ctx.get("focus_area", "mixed"), 1.0)  # correctness needs domain knowledge
    if words >= 60:
        score += 1.0   # long answer: more claims to check
    elif words >= 25:
        score += 0.5

    if score >= EVALUATOR_STRONG_AT:
        return RoutingDecision("strong", f"needs domain judgement (focus={ctx.get('focus_area')}, {words} words)", score)
    return RoutingDecision("fast", f"low-stakes scoring (focus={ctx.get('focus_area')}, {words} words)", score)


def _route_interviewer(ctx: dict) -> RoutingDecision:
    turn = ctx.get("turn", 1)
    verdict = ctx.get("last_verdict")

    if turn == 1 or verdict is None:
        return RoutingDecision("fast", "scripted warm-up opener", 0.0)

    score = 0.0
    if verdict in ("strong", "adequate"):
        score += 1.0   # must raise difficulty / pick a fitting new topic
    score += _DEEP_FOCUS.get(ctx.get("focus_area", "mixed"), 1.0) * 0.5
    if ctx.get("resume_provided"):
        score += 0.5   # personalised question needs to use the background well

    if score >= INTERVIEWER_STRONG_AT:
        return RoutingDecision("strong", f"adaptive escalation after '{verdict}' answer (focus={ctx.get('focus_area')})", score)
    return RoutingDecision("fast", f"simple probe / pivot after '{verdict}' answer", score)
