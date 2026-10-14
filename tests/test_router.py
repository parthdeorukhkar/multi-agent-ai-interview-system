import pytest

from llmops.router import route


def test_coach_and_judge_always_strong():
    assert route("coach").tier == "strong"
    assert route("judge").tier == "strong"


def test_unknown_task_defaults_to_safe_tier():
    assert route("brand_new_task").tier == "strong"


def test_trivial_answers_go_fast_even_for_technical_focus():
    d = route("evaluator", {"focus_area": "technical", "answer": "I don't know, no idea."})
    assert d.tier == "fast"


def test_technical_claims_need_strong_model():
    d = route("evaluator", {"focus_area": "technical", "answer": "A thread is a faster process with separate memory."})
    assert d.tier == "strong"


def test_behavioral_short_answer_is_fast():
    d = route("evaluator", {"focus_area": "behavioral", "answer": "We talked and it was fine in the end overall."})
    assert d.tier == "fast"


def test_interviewer_opener_is_fast():
    assert route("interviewer", {"turn": 1, "focus_area": "technical"}).tier == "fast"


def test_interviewer_escalates_after_strong_technical_answer():
    d = route("interviewer", {"turn": 4, "focus_area": "technical", "last_verdict": "strong"})
    assert d.tier == "strong"


@pytest.mark.parametrize("mode", ["fast", "strong"])
def test_forced_mode_overrides_policy(monkeypatch, mode):
    monkeypatch.setenv("ROUTING_MODE", mode)
    assert route("coach").tier == mode
    assert route("evaluator", {"focus_area": "technical", "answer": "x " * 100}).tier == mode


def test_invalid_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("ROUTING_MODE", "bogus")
    assert route("coach").reason != "forced:bogus"


def test_every_decision_has_a_reason():
    for task in ("coach", "evaluator", "interviewer", "judge", "x"):
        assert route(task, {"answer": "some words here for a test", "turn": 2, "last_verdict": "weak"}).reason
