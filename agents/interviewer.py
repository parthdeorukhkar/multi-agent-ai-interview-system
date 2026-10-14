from config import chat
from llmops.prompts import load_prompt


class InterviewerAgent:
    """
    Drives the interview conversation. Adapts its next question based on the
    Evaluator's private assessment of the previous answer.

    LLMOps: each turn is routed to the fast or strong model depending on how
    much reasoning the next question needs (see llmops/router.py).
    """

    def __init__(self, target_role: str, focus_area: str, resume_snippet: str = ""):
        self.target_role = target_role
        self.focus_area = focus_area
        self.resume_provided = bool(resume_snippet)
        self.resume_snippet = resume_snippet or "Not provided."
        self.prompt = load_prompt("interviewer")
        self.system_prompt = self.prompt.text.format(
            target_role=target_role,
            focus_area=focus_area,
            resume_snippet=self.resume_snippet,
        )
        self.history = []  # list of {"question": ..., "answer": ...}

    def next_question(self, turn: int, total_turns: int, last_eval: dict | None) -> str:
        transcript_so_far = "\n".join(
            f"Q{i+1}: {h['question']}\nCandidate: {h['answer']}"
            for i, h in enumerate(self.history)
        ) or "(no exchanges yet — this is the opening question)"

        eval_signal = (
            f"Private evaluator signal on the last answer (candidate cannot see this): {last_eval}"
            if last_eval else "No prior answer yet."
        )

        user_message = (
            f"This is turn {turn} of {total_turns} in the interview.\n\n"
            f"Conversation so far:\n{transcript_so_far}\n\n"
            f"{eval_signal}\n\n"
            f"Generate the next thing to say to the candidate now."
        )

        question = chat(
            self.system_prompt, user_message, temperature=0.8,
            task="interviewer",
            ctx={
                "turn": turn,
                "total_turns": total_turns,
                "focus_area": self.focus_area,
                "last_verdict": (last_eval or {}).get("verdict"),
                "resume_provided": self.resume_provided,
            },
            prompt_version=self.prompt.version,
        )
        return question.strip()

    def record_answer(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
