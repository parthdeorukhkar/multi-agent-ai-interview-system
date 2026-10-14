"""The eval checks themselves must be right - a wrong check is worse than none."""
from evals import checks
from evals.run_evals import compare_to_baseline

HISTORY = [{"question": "Why this role?", "answer": "I like owning a problem end to end and cut abandonment by twelve percent."}]
SCORES = [{"correctness": 4, "depth": 2, "structure": 3, "relevance": 5, "communication": 4},
          {"correctness": 2, "depth": 2, "structure": 3, "relevance": 5, "communication": 4}]

GOOD_REPORT = """## Overall Impression
Solid start.
## Strengths
- Turn 1 showed ownership: "cut abandonment by twelve percent".
- Question 2 was structured.
## Gaps to address
- Turn 2 lacked depth.
- Needs concrete metrics.
## What to practice next
- One
- Two
- Three
## Score summary
- correctness: 3.0/5
- depth: 2.0/5
- structure: 3.0
- relevance: 5.0
- communication: 4.0
"""


def test_all_good_report_passes():
    case = {"scores": SCORES, "history": HISTORY}
    results = checks.run_coach_checks(GOOD_REPORT, case)
    assert all(ok for ok, _ in results.values()), results


def test_detects_wrong_score_average():
    bad = GOOD_REPORT.replace("correctness: 3.0/5", "correctness: 4.8/5")
    ok, detail = checks.coach_score_accuracy(bad, SCORES)
    assert not ok and "correctness" in detail


def test_detects_fabricated_quote():
    bad = GOOD_REPORT.replace("cut abandonment by twelve percent", "grew revenue by ninety percent overnight")
    ok, _ = checks.coach_quote_grounding(bad, HISTORY)
    assert not ok


def test_detects_missing_section_and_platitude():
    assert not checks.coach_sections(GOOD_REPORT.replace("## Gaps to address", "## Misc"))[0]
    assert not checks.coach_no_platitudes("Great job, keep practicing!")[0]


def test_interviewer_checks():
    assert checks.single_question("What did you build?")[0]
    assert not checks.single_question("What did you build? And why?")[0]
    assert not checks.no_leak("The evaluator says you're weak")[0]
    assert not checks.no_leak("That's a 3/5 answer")[0]
    assert checks.no_leak("Tell me about a project you led.")[0]
    assert not checks.no_markdown("- first\n- second")[0]
    assert not checks.no_feedback("You did great, your strengths are clear.")[0]
    assert checks.closing("Thanks so much for your time today.")[0]
    assert not checks.not_repeat("How would you design a rate limiter?", 0.6, "How would you design a rate limiter for an API?")[0]
    assert checks.not_repeat("What's a trade-off you made recently?", 0.6, "How would you design a rate limiter?")[0]


def test_check_evaluation_bounds():
    good = {"correctness": 5, "depth": 4, "structure": 4, "relevance": 5, "communication": 4, "verdict": "strong", "notes": ""}
    expect = {"verdict": ["strong"], "min": {"depth": 4}, "max": {"communication": 4}}
    assert all(ok for ok, _ in checks.check_evaluation(good, expect).values())
    assert not checks.check_evaluation({**good, "depth": 2}, expect)["score_bounds"][0]
    assert not checks.check_evaluation({**good, "verdict": "weak"}, expect)["verdict"][0]
    assert not checks.check_evaluation({**good, "depth": 9}, expect)["json_valid"][0]
    assert not checks.check_evaluation(None, expect)["json_valid"][0]


def test_regression_gate_directions_and_tolerance():
    base = {"evaluator.verdict_accuracy": 0.9, "coach.fabricated_quote_rate": 0.0, "ops.est_cost_usd": 0.1}
    # higher-is-better drops, lower-is-better rises, ops is never gated
    cur = {"evaluator.verdict_accuracy": 0.7, "coach.fabricated_quote_rate": 0.5, "ops.est_cost_usd": 9.0}
    problems = compare_to_baseline(cur, base, 0.0)
    assert len(problems) == 2 and not any("ops." in p for p in problems)
    assert compare_to_baseline({"evaluator.verdict_accuracy": 0.85}, base, 0.1) == []
    assert compare_to_baseline({"evaluator.verdict_accuracy": 1.0, "coach.fabricated_quote_rate": 0.0}, base, 0.0) == []
