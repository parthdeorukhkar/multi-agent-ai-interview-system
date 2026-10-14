"""
A tiny deterministic stand-in for the LLM, used only when INTERVIEW_DRY_RUN=1.
Lets you exercise the full orchestration logic with no API key/network.
"""
import json
import random
import re

_QUESTION_BANK = [
    "Tell me a bit about your background and what draws you to this role.",
    "Walk me through a project you're proud of — what was your role in it?",
    "Describe a time you disagreed with a teammate. How did you handle it?",
    "How would you approach debugging a system that's behaving unexpectedly in production?",
    "What's a trade-off you had to make recently, and how did you decide?",
    "Tell me about a time you had to learn something new quickly to finish a task.",
    "How do you prioritize when you have more work than time?",
    "That's helpful — can you go a bit deeper on the specific outcome there?",
]

_asked = set()


def reset():
    """Clear cross-call state (used by evals for reproducibility)."""
    _asked.clear()


def _mock_interviewer(user_message: str) -> str:
    turn_match = re.search(r"turn (\d+) of (\d+)", user_message)
    turn, total = (int(turn_match.group(1)), int(turn_match.group(2))) if turn_match else (1, 6)

    if "'verdict': 'unclear'" in user_message:
        q = "Sorry, that didn't come through clearly - could you repeat that?"
    elif "weak" in user_message.lower() or '"verdict": "weak"' in user_message:
        q = "Can you walk me through that again with a specific example, step by step?"
    else:
        remaining = [q for q in _QUESTION_BANK if q not in _asked]
        q = remaining[0] if remaining else "Can you tell me more about that?"
        _asked.add(q)

    if turn >= total:
        return q + " ... Great, that'll be our last question today — thanks so much for your time!"
    return q


_IDK = re.compile(r"\b(i don'?t know|no idea|i'?m not sure|i have no clue)\b", re.IGNORECASE)


def _extract_answer(system_prompt: str, user_message: str) -> str:
    # The real evaluator prompt embeds the answer; older call sites put it in the user message.
    m = re.search(r"artifacts\):\s*(.*?)\n\s*\n## What you evaluate", system_prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer.*?:\s*(.*)", user_message, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _mock_evaluator(user_message: str, system_prompt: str = "") -> str:
    answer = _extract_answer(system_prompt, user_message)
    length = len(answer.split())

    if length < 4 or (length < 12 and _IDK.search(answer)):
        verdict, base = "no_answer", 1
    elif length < 15:
        verdict, base = "weak", 2
    elif length < 40:
        verdict, base = "adequate", 3
    else:
        verdict, base = "strong", 4

    jitter = lambda: max(1, min(5, base + random.choice([-1, 0, 0, 1])))
    scores = {
        "correctness": jitter(),
        "depth": jitter(),
        "structure": jitter(),
        "relevance": jitter(),
        "communication": jitter(),
        "verdict": verdict,
        "notes": f"[dry-run mock] heuristic score based on answer length ({length} words).",
    }
    if verdict == "no_answer":
        scores.update(correctness=1, depth=1, communication=3)  # per evaluator prompt: don't punish communication
    return json.dumps(scores)


def _mock_coach(user_message: str, system_prompt: str = "") -> str:
    """Deterministic report built from the real per-turn scores in the prompt,
    so the eval harness can be exercised offline (numbers are real, prose is canned)."""
    import ast
    dims = ("correctness", "depth", "structure", "relevance", "communication")
    scores = []
    for m in re.finditer(r"Evaluator scores: (\{.*\})", system_prompt):
        try:
            scores.append(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            pass
    avgs = {d: (sum(s.get(d, 3) for s in scores) / len(scores)) if scores else 3.0 for d in dims}
    best = max(range(len(scores)), key=lambda i: sum(scores[i].get(d, 3) for d in dims), default=0) + 1
    worst = min(range(len(scores)), key=lambda i: sum(scores[i].get(d, 3) for d in dims), default=0) + 1
    table = "\n".join(f"- {d}: {avgs[d]:.1f}/5" for d in dims)
    return (
        "## Overall Impression\n"
        "[dry-run mock] Placeholder report generated without a real LLM call, to "
        "demonstrate the orchestration flow end to end.\n\n"
        f"## Strengths\n- Turn {best} was your highest-scoring answer.\n"
        f"- Mock strength based on transcript length.\n\n"
        f"## Gaps to address\n- Turn {worst} was your lowest-scoring answer.\n"
        f"- Mock gap based on lower-scored turns.\n\n"
        "## What to practice next\n- Set GROQ_API_KEY and rerun without dry-run mode.\n"
        "- Re-answer your lowest-scoring question out loud.\n"
        "- Add one concrete metric to every example.\n\n"
        f"## Score summary\n{table}\n"
    )


def respond(system_prompt: str, user_message: str, json_mode: bool) -> str:
    if "silent interview evaluator" in system_prompt:
        return _mock_evaluator(user_message, system_prompt)
    if "interview coach" in system_prompt:
        return _mock_coach(user_message, system_prompt)
    return _mock_interviewer(user_message)