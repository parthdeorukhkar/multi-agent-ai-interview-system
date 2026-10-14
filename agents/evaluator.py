import json
from config import chat
from llmops.prompts import load_prompt

DIMENSIONS = ("correctness", "depth", "structure", "relevance", "communication")
VALID_VERDICTS = {"strong", "adequate", "weak", "no_answer", "off_topic", "unclear"}

_DEFAULT_SCORE = {
    "correctness": 3, "depth": 3, "structure": 3, "relevance": 3, "communication": 3,
    "verdict": "unclear", "notes": "Evaluator output could not be parsed; default mid score used.",
}


def extract_json(raw: str) -> dict | None:
    """Parse the model output as a JSON object (tolerating ``` fences). None if impossible."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, IndexError):
        return None


def is_valid_evaluation(data: dict | None) -> bool:
    """Strict schema check: all five dimensions are ints 1-5 and the verdict is known."""
    if not data:
        return False
    for dim in DIMENSIONS:
        v = data.get(dim)
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 5:
            return False
    return data.get("verdict") in VALID_VERDICTS


class EvaluatorAgent:
    """
    Silently scores each answer along multiple dimensions and returns strict JSON.
    Never speaks to the candidate. Output feeds the Interviewer's next decision
    and the Coach's final report.

    LLMOps: routed per answer (see llmops/router.py). If the fast model returns
    malformed output - or "unclear" for an answer that is clearly long enough to
    judge - the call is escalated to the strong model.
    """

    def __init__(self, target_role: str, focus_area: str):
        self.target_role = target_role
        self.focus_area = focus_area
        self.prompt = load_prompt("evaluator")
        self.system_prompt_template = self.prompt.text
        self.last_raw = ""  # raw model output of the latest call (used by evals to measure JSON validity)

    def score(self, question: str, answer: str) -> dict:
        system_prompt = self.system_prompt_template.format(
            target_role=self.target_role,
            focus_area=self.focus_area,
            question=question,
            answer=answer,
        )
        words = len(answer.split())

        def validator(raw: str) -> tuple[bool, str]:
            data = extract_json(raw)
            if not is_valid_evaluation(data):
                return False, "invalid_json_or_schema"
            if data["verdict"] == "unclear" and words >= 25:
                return False, "low_confidence_on_long_answer"
            return True, ""

        raw = chat(
            system_prompt,
            "Evaluate this exchange now and return the JSON object.",
            temperature=0.2,
            json_mode=True,
            task="evaluator",
            ctx={"focus_area": self.focus_area, "answer": answer},
            prompt_version=self.prompt.version,
            validator=validator,
        )
        self.last_raw = raw
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict:
        data = extract_json(raw)
        if data is None:
            return dict(_DEFAULT_SCORE)
        for key in _DEFAULT_SCORE:
            data.setdefault(key, _DEFAULT_SCORE[key])
        return data
