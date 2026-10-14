from config import chat
from llmops.prompts import load_prompt


class CoachAgent:
    """
    Runs once at the end of the interview. Synthesizes the full transcript plus
    all evaluator scores into structured, specific, actionable feedback.

    LLMOps: always routed to the strong model - it runs once per session and
    is the user-facing deliverable, so quality matters more than cost here.
    """

    def __init__(self, target_role: str, focus_area: str):
        self.target_role = target_role
        self.focus_area = focus_area
        self.prompt = load_prompt("coach")
        self.system_prompt_template = self.prompt.text

    def generate_report(self, history: list[dict], scores: list[dict]) -> str:
        blocks = []
        for i, (turn, score) in enumerate(zip(history, scores)):
            blocks.append(
                f"--- Turn {i+1} ---\n"
                f"Q: {turn['question']}\n"
                f"A: {turn['answer']}\n"
                f"Evaluator scores: {score}\n"
            )
        transcript_with_scores = "\n".join(blocks)

        system_prompt = self.system_prompt_template.format(
            target_role=self.target_role,
            focus_area=self.focus_area,
            transcript_with_scores=transcript_with_scores,
        )
        return chat(
            system_prompt,
            "Generate the coaching report now.",
            temperature=0.5,
            task="coach",
            prompt_version=self.prompt.version,
        )
