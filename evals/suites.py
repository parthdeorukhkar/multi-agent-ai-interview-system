"""
Eval suites. Each returns a SuiteResult with per-case results and flat metrics
("<suite>.<metric>") that the runner compares against a stored baseline.

  evaluator    labelled answers -> verdict accuracy, score bounds, JSON validity, ordering
  interviewer  scenarios -> format / leak / adaptivity checks (+ optional LLM judge)
  coach        fixed transcripts -> structure + hallucination checks (numbers & quotes)
  router       task+context -> expected model tier (the "tool selection" eval)
  e2e          scripted candidate personas through the whole pipeline -> task completion
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from agents.coach import CoachAgent
from agents.evaluator import EvaluatorAgent, extract_json, is_valid_evaluation
from agents.interviewer import InterviewerAgent
from llmops.router import route
from . import checks

DATASETS = Path(__file__).parent / "datasets"
E2E_MAX_COST_USD = 0.05  # generous per-interview cost budget (real runs are ~$0.003-0.01)


@dataclass
class CaseResult:
    id: str
    passed: bool
    checks: dict = field(default_factory=dict)   # name -> (bool, detail)
    extra: dict = field(default_factory=dict)


@dataclass
class SuiteResult:
    name: str
    cases: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _load(name: str) -> dict:
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _reset_state(seed: int = 0):
    random.seed(seed)
    from agents import dry_run_mock
    dry_run_mock.reset()


# ------------------------------------------------------------------ evaluator
def evaluator_suite(use_judge: bool = False) -> SuiteResult:
    _reset_state()
    data = _load("evaluator_cases.json")
    res = SuiteResult("evaluator")
    overall = {}
    for case in data["cases"]:
        agent = EvaluatorAgent(case["role"], case["focus"])
        out = agent.score(case["question"], case["answer"])
        raw_valid = is_valid_evaluation(extract_json(agent.last_raw))
        c = checks.check_evaluation(out if raw_valid else None, case["expect"])
        if raw_valid:
            overall[case["id"]] = sum(out[d] for d in checks.DIMENSIONS) / len(checks.DIMENSIONS)
        res.cases.append(CaseResult(case["id"], all(ok for ok, _ in c.values()), c,
                                    {"output": out if raw_valid else agent.last_raw[:200]}))

    pair_flags = []
    for better, worse in data.get("pairs", []):
        ok = better in overall and worse in overall and overall[better] > overall[worse]
        pair_flags.append(ok)
        res.cases.append(CaseResult(f"pair:{better}>{worse}", ok,
                                    {"ordering": (ok, f"{overall.get(better)} vs {overall.get(worse)}")}))

    single = [c for c in res.cases if not c.id.startswith("pair:")]
    res.metrics = {
        "evaluator.pass_rate": _rate([c.passed for c in single]),
        "evaluator.json_validity": _rate([c.checks["json_valid"][0] for c in single]),
        "evaluator.verdict_accuracy": _rate([c.checks.get("verdict", (False,))[0] for c in single]),
        "evaluator.score_bounds_rate": _rate([c.checks.get("score_bounds", (False,))[0] for c in single]),
        "evaluator.ordering_accuracy": _rate(pair_flags),
    }
    return res


# ---------------------------------------------------------------- interviewer
def interviewer_suite(use_judge: bool = False) -> SuiteResult:
    _reset_state()
    data = _load("interviewer_cases.json")
    res = SuiteResult("interviewer")
    judge_scores = []
    for case in data["cases"]:
        agent = InterviewerAgent(case["role"], case["focus"], "")
        agent.history = list(case["history"])
        text = agent.next_question(case["turn"], case["total"], case["last_eval"])
        c = checks.run_interviewer_checks(text, case)
        extra = {"output": text}
        if use_judge:
            from .judge import judge
            j = judge(text, case["judge"], context=json.dumps({k: case[k] for k in ("role", "focus", "turn", "total", "history", "last_eval")}))
            extra["judge"] = j
            if j["score"] is not None:
                judge_scores.append(j["score"])
        res.cases.append(CaseResult(case["id"], all(ok for ok, _ in c.values()), c, extra))

    res.metrics = {
        "interviewer.pass_rate": _rate([c.passed for c in res.cases]),
        "interviewer.leak_rate": 1 - _rate([c.checks["no_leak"][0] for c in res.cases]),
    }
    if judge_scores:
        res.metrics["interviewer.judge_mean"] = sum(judge_scores) / len(judge_scores)
    return res


# ---------------------------------------------------------------------- coach
def coach_suite(use_judge: bool = False) -> SuiteResult:
    _reset_state()
    data = _load("coach_cases.json")
    res = SuiteResult("coach")
    judge_scores = []
    for case in data["cases"]:
        report = CoachAgent(case["role"], case["focus"]).generate_report(case["history"], case["scores"])
        c = checks.run_coach_checks(report, case)
        extra = {"report": report}
        if use_judge:
            from .judge import judge
            j = judge(report, "Specific, honest, actionable coaching grounded in the transcript; no generic platitudes; "
                              "strengths/gaps tied to real turns.", context=json.dumps(case["history"]))
            extra["judge"] = j
            if j["score"] is not None:
                judge_scores.append(j["score"])
        res.cases.append(CaseResult(case["id"], all(ok for ok, _ in c.values()), c, extra))

    res.metrics = {
        "coach.pass_rate": _rate([c.passed for c in res.cases]),
        "coach.structure_rate": _rate([c.checks["sections"][0] and c.checks["bullet_counts"][0] for c in res.cases]),
        "coach.fabricated_quote_rate": 1 - _rate([c.checks["quote_grounding"][0] for c in res.cases]),
        "coach.score_mismatch_rate": 1 - _rate([c.checks["score_accuracy"][0] for c in res.cases]),
    }
    if judge_scores:
        res.metrics["coach.judge_mean"] = sum(judge_scores) / len(judge_scores)
    return res


# --------------------------------------------------------------------- router
def router_suite(use_judge: bool = False) -> SuiteResult:
    data = _load("router_cases.json")
    res = SuiteResult("router")
    for case in data["cases"]:
        ctx = dict(case["ctx"])
        if "_words" in ctx:
            ctx["answer"] = "word " * ctx.pop("_words")
        # ROUTING_MODE must be 'auto' for the policy itself to be evaluated
        import os
        prev = os.environ.pop("ROUTING_MODE", None)
        try:
            d = route(case["task"], ctx)
        finally:
            if prev is not None:
                os.environ["ROUTING_MODE"] = prev
        ok = d.tier == case["expect"]
        res.cases.append(CaseResult(case["id"], ok, {"tier": (ok, f"got {d.tier} ({d.reason}), want {case['expect']}")}))
    res.metrics = {"router.accuracy": _rate([c.passed for c in res.cases])}
    return res


# ------------------------------------------------------------------------ e2e
PERSONAS = {
    "strong": [
        "I'm a data analyst with four years of experience. I built a churn dashboard in SQL and Tableau that the retention team used weekly, and it helped cut churn by six percent over two quarters.",
        "On a project to migrate our reporting to dbt, I owned the modelling layer. I wrote tests for every model, documented the lineage, and reduced our nightly refresh from three hours to forty minutes.",
        "Once a stakeholder disputed my numbers. I traced the gap to a different definition of active user, wrote a one page metric definition doc, and got both teams to sign off, so the dispute never came back.",
        "I'd prioritize by expected impact on the key metric divided by effort, validate assumptions with a quick query before building anything, and check with stakeholders so the output actually gets used.",
        "I would segment the data by cohort, check for seasonality, run a significance test, and present the finding with a clear recommendation and its limits.",
        "I'd like to learn more about how the team measures the success of its experiments.",
        "Thanks, that covers it.",
    ],
    "weak": [
        "I do analysis stuff.",
        "We made a report. It was okay I think.",
        "I talked to people about it and it was fine.",
        "Um, I'm not sure. Maybe look at the data.",
        "I guess I'd just do what the team says.",
        "Not really.",
        "Okay.",
    ],
    "idk": [
        "I'm not sure.",
        "I don't know.",
        "I don't know, I have no idea.",
        "No idea, sorry.",
        "I don't know.",
        "I don't know.",
        "I don't know.",
    ],
}


def _run_persona(name: str, answers: list[str]) -> tuple[dict, list]:
    from orchestrator import run_interview
    from llmops import telemetry
    answers = list(answers)
    mark = telemetry.mark()
    session = run_interview(
        "Data Analyst", "mixed", "",
        speak_fn=lambda _text: None,
        listen_fn=lambda: answers.pop(0) if answers else "No further comment.",
        total_turns=5, save=False, verbose=False,
    )
    return session, telemetry.records_since(mark)


def e2e_suite(use_judge: bool = False) -> SuiteResult:
    _reset_state()
    res = SuiteResult("e2e")
    means = {}
    for name, answers in PERSONAS.items():
        session, recs = _run_persona(name, answers)
        n_ok = len(session["history"]) == 5 and len(session["scores"]) == 5
        sections_ok = checks.coach_sections(session["report"])
        scores_ok = all(is_valid_evaluation(s) for s in session["scores"])
        cost = session["llm_summary"]["est_cost_usd"]
        c = {
            "completed_all_turns": (n_ok, f"{len(session['history'])} turns, {len(session['scores'])} scores"),
            "report_sections": sections_ok,
            "valid_scores": (scores_ok, "" if scores_ok else "an evaluation failed schema validation"),
            "within_cost_budget": (cost <= E2E_MAX_COST_USD, f"${cost:.4f} (budget ${E2E_MAX_COST_USD})"),
        }
        means[name] = sum(sum(s[d] for d in checks.DIMENSIONS) / 5 for s in session["scores"]) / len(session["scores"])
        res.cases.append(CaseResult(f"persona:{name}", all(ok for ok, _ in c.values()), c,
                                    {"mean_score": round(means[name], 2), "cost_usd": cost,
                                     "calls": session["llm_summary"]["calls"]}))
    ordering_ok = means.get("strong", 0) > means.get("weak", 9) and means.get("strong", 0) > means.get("idk", 9)
    res.cases.append(CaseResult("persona_ordering", ordering_ok,
                                {"strong>weak,idk": (ordering_ok, str({k: round(v, 2) for k, v in means.items()}))}))
    res.metrics = {
        "e2e.completion_rate": _rate([c.passed for c in res.cases if c.id.startswith("persona:")]),
        "e2e.persona_ordering": 1.0 if ordering_ok else 0.0,
    }
    return res


SUITES = {
    "router": router_suite,
    "evaluator": evaluator_suite,
    "interviewer": interviewer_suite,
    "coach": coach_suite,
    "e2e": e2e_suite,
}
