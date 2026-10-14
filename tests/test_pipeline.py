"""Full pipeline in dry-run mode, plus the evaluator's output handling."""
from agents.evaluator import EvaluatorAgent, extract_json, is_valid_evaluation
from orchestrator import run_interview

GOOD = '{"correctness":3,"depth":3,"structure":3,"relevance":3,"communication":3,"verdict":"adequate","notes":"ok"}'


def test_extract_json_handles_fences_and_garbage():
    assert extract_json("```json\n" + GOOD + "\n```")["verdict"] == "adequate"
    assert extract_json("not json") is None
    assert extract_json("[1,2]") is None


def test_schema_validation():
    assert is_valid_evaluation(extract_json(GOOD))
    assert not is_valid_evaluation({"correctness": 3})
    assert not is_valid_evaluation({**extract_json(GOOD), "verdict": "amazing"})
    assert not is_valid_evaluation({**extract_json(GOOD), "depth": 0})
    assert not is_valid_evaluation({**extract_json(GOOD), "depth": True})


def test_unparseable_output_falls_back_to_default_score():
    out = EvaluatorAgent._parse("garbage")
    assert out["verdict"] == "unclear" and out["correctness"] == 3


def test_full_interview_runs_and_reports_metrics():
    answers = iter(["I built a dashboard in SQL that cut churn by six percent over two quarters."] * 10)
    session = run_interview("Data Analyst", "mixed", "", speak_fn=lambda _t: None,
                            listen_fn=lambda: next(answers), total_turns=5, save=False, verbose=False)
    assert len(session["history"]) == 5 and len(session["scores"]) == 5
    assert "## Score summary" in session["report"]
    s = session["llm_summary"]
    assert s["calls"] == 11 and s["errors"] == 0   # 5 interviewer + 5 evaluator + 1 coach
    assert set(session["prompt_versions"]) == {"interviewer", "evaluator", "coach"}
