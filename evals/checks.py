"""
Deterministic, dependency-free checks on agent outputs. Each returns (passed, detail).
Kept separate from the suites so they are unit-testable and cheap (no LLM needed).
"""
import re

SECTIONS = ("Overall Impression", "Strengths", "Gaps to address", "What to practice next", "Score summary")
DIMENSIONS = ("correctness", "depth", "structure", "relevance", "communication")

_LEAK = re.compile(
    r"evaluator|verdict|private signal|system prompt|\bagents?\b|\b\d(\.\d)?\s*(/|out of)\s*5\b",
    re.IGNORECASE,
)
_MARKDOWN = re.compile(r"(^\s*[-*•]\s)|(\*\*)|(^#+\s)|(`)", re.MULTILINE)
_FEEDBACK = re.compile(
    r"\byou (did|were|are doing|have done) (really |very )?(great|well|good|fantastic|excellent)\b"
    r"|\byour (strengths?|weakness(es)?|score)\b|\bneed to improve\b|\bone area (for|to) improve",
    re.IGNORECASE,
)
_PLATITUDES = re.compile(r"keep up the good work|great job|keep practicing|you('ve| have) got this", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


# ---------------------------------------------------------------- interviewer
def single_question(text: str, _case=None):
    n = text.count("?")
    return n <= 1, f"{n} question marks"


def no_leak(text: str, _case=None):
    m = _LEAK.search(text)
    return m is None, f"leaked '{m.group(0)}'" if m else ""


def no_markdown(text: str, _case=None):
    m = _MARKDOWN.search(text)
    return m is None, "markdown/formatting found" if m else ""


def no_feedback(text: str, _case=None):
    m = _FEEDBACK.search(text)
    return m is None, f"gave feedback: '{m.group(0)}'" if m else ""


def closing(text: str, _case=None):
    ok = bool(re.search(r"\bthank", text, re.IGNORECASE))
    return ok, "" if ok else "no thank-you / closing line on final turn"


def max_words(text: str, limit: int):
    n = len(text.split())
    return n <= limit, f"{n} words (limit {limit})"


def references_any(text: str, keywords: list[str]):
    low = text.lower()
    hit = [k for k in keywords if k.lower() in low]
    return bool(hit), "" if hit else f"mentions none of {keywords}"


def not_repeat(text: str, threshold: float, previous_question: str):
    """Jaccard token overlap with the previous question must stay below threshold."""
    a, b = _tokens(text), _tokens(previous_question)
    if not a or not b:
        return True, ""
    j = len(a & b) / len(a | b)
    return j < threshold, f"token overlap with previous question {j:.2f} (limit {threshold})"


def run_interviewer_checks(text: str, case: dict) -> dict:
    """-> {check_name: (passed, detail)}"""
    results = {}
    prev_q = case["history"][-1]["question"] if case.get("history") else ""
    for spec in case["checks"]:
        if isinstance(spec, str):
            results[spec] = globals()[spec](text, case)
        else:
            (name, arg), = spec.items()
            if name == "not_repeat":
                results[name] = not_repeat(text, arg, prev_q)
            else:
                results[name] = globals()[name](text, arg)
    return results


# ---------------------------------------------------------------- evaluator
def check_evaluation(data: dict | None, expect: dict) -> dict:
    """Compare an evaluator output with a labelled expectation."""
    from agents.evaluator import is_valid_evaluation
    out = {}
    valid = is_valid_evaluation(data)
    out["json_valid"] = (valid, "" if valid else "invalid JSON/schema")
    if not valid:
        return out
    out["verdict"] = (data["verdict"] in expect["verdict"], f"got {data['verdict']!r}, want {expect['verdict']}")
    bad = [f"{d}={data[d]}<{v}" for d, v in expect.get("min", {}).items() if data[d] < v]
    bad += [f"{d}={data[d]}>{v}" for d, v in expect.get("max", {}).items() if data[d] > v]
    out["score_bounds"] = (not bad, "; ".join(bad))
    return out


# ---------------------------------------------------------------- coach
def section_text(report: str, name: str) -> str | None:
    m = re.search(rf"^#+\s*{re.escape(name)}\s*$(.*?)(?=^#+\s|\Z)", report, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return m.group(1) if m else None


def coach_sections(report: str):
    missing = [s for s in SECTIONS if section_text(report, s) is None]
    return not missing, f"missing sections: {missing}" if missing else ""


def coach_bullet_counts(report: str):
    limits = {"Strengths": (2, 4), "Gaps to address": (2, 4), "What to practice next": (3, 5)}
    bad = []
    for name, (lo, hi) in limits.items():
        body = section_text(report, name) or ""
        n = len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", body, re.MULTILINE))
        if not lo <= n <= hi:
            bad.append(f"{name}: {n} bullets (want {lo}-{hi})")
    return not bad, "; ".join(bad)


def coach_turn_references(report: str, minimum: int = 2):
    turns = set(re.findall(r"\b(?:turn|question|q)\s*#?(\d+)\b", report, re.IGNORECASE))
    return len(turns) >= minimum, f"references {len(turns)} distinct turns (want >= {minimum})"


def coach_no_platitudes(report: str):
    m = _PLATITUDES.search(report)
    return m is None, f"platitude: '{m.group(0)}'" if m else ""


def coach_score_accuracy(report: str, scores: list[dict], tol: float = 0.15):
    """Hallucination check (numbers): reported per-dimension averages must match the real scores."""
    body = section_text(report, "Score summary") or ""
    mismatches = []
    for dim in DIMENSIONS:
        real = sum(s[dim] for s in scores) / len(scores)
        m = re.search(rf"{dim}[^0-9\n]*?(\d(?:\.\d+)?)", body, re.IGNORECASE)
        if not m:
            mismatches.append(f"{dim}: not reported")
        elif abs(float(m.group(1)) - real) > tol:
            mismatches.append(f"{dim}: reported {m.group(1)}, real {real:.2f}")
    return not mismatches, "; ".join(mismatches)


def coach_quote_grounding(report: str, history: list[dict], min_words: int = 4):
    """Hallucination check (quotes): every quoted span must appear in the transcript."""
    haystack = normalize(" ".join(h["question"] + " " + h["answer"] for h in history))
    spans = re.findall(r'"([^"\n]+)"|“([^”\n]+)”', report)
    quotes = [a or b for a, b in spans]
    quotes = [q for q in quotes if len(q.split()) >= min_words]
    fabricated = [q for q in quotes if normalize(q) not in haystack]
    return not fabricated, f"{len(fabricated)}/{len(quotes)} quotes not in transcript: {fabricated[:2]}"


def run_coach_checks(report: str, case: dict) -> dict:
    return {
        "sections": coach_sections(report),
        "bullet_counts": coach_bullet_counts(report),
        "turn_references": coach_turn_references(report),
        "no_platitudes": coach_no_platitudes(report),
        "score_accuracy": coach_score_accuracy(report, case["scores"]),
        "quote_grounding": coach_quote_grounding(report, case["history"]),
    }
