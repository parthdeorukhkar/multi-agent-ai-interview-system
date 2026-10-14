"""
Orchestration: Interviewer.next_question() -> speak() -> listen_for_answer()
-> Evaluator.score() (silent) -> feeds back into Interviewer -> loop ->
Coach.generate_report().

Speech I/O is injectable (speak_fn / listen_fn) so the whole pipeline can be
driven by scripted "candidates" in evals and tests without a mic or speaker.
"""
import random
from datetime import datetime
from pathlib import Path

from agents.interviewer import InterviewerAgent
from agents.evaluator import EvaluatorAgent
from agents.coach import CoachAgent
from speech.tts import speak
from speech.stt import listen_for_answer
from llmops import telemetry
from llmops.prompts import all_versions
import config

TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"


def run_interview(target_role: str, focus_area: str, resume_snippet: str = "", *,
                  speak_fn=speak, listen_fn=listen_for_answer,
                  total_turns: int | None = None, save: bool = True,
                  verbose: bool = True) -> dict:
    def say(msg: str):
        if verbose:
            print(msg)

    timestamp = datetime.now().isoformat(timespec="seconds")
    telemetry.set_session(timestamp.replace(":", "-"))
    telemetry_mark = telemetry.mark()

    interviewer = InterviewerAgent(target_role, focus_area, resume_snippet)
    evaluator = EvaluatorAgent(target_role, focus_area)
    coach = CoachAgent(target_role, focus_area)

    if total_turns is None:
        total_turns = random.randint(config.MIN_TURNS, config.MAX_TURNS)
    scores = []

    say(f"\n=== Starting mock interview: {target_role} ({focus_area}) — {total_turns} turns ===\n")

    last_eval = None
    for turn in range(1, total_turns + 1):
        question = interviewer.next_question(turn, total_turns, last_eval)
        speak_fn(question)

        answer = listen_fn()
        say(f"Candidate: {answer}\n")

        interviewer.record_answer(question, answer)

        last_eval = evaluator.score(question, answer)
        scores.append(last_eval)
        say(f"[internal evaluator signal — not shown to candidate: {last_eval}]\n")

    say("=== Interview complete. Generating coaching report... ===\n")
    report = coach.generate_report(interviewer.history, scores)

    llm_summary = telemetry.summarize(telemetry.records_since(telemetry_mark))
    say("=== LLM run metrics ===\n" + telemetry.format_summary(llm_summary) + "\n")

    session = {
        "target_role": target_role,
        "focus_area": focus_area,
        "resume_snippet": resume_snippet,
        "history": interviewer.history,
        "scores": scores,
        "report": report,
        "timestamp": timestamp,
        "llm_summary": llm_summary,
        "prompt_versions": all_versions(),
    }
    if save:
        _save_transcript(session, verbose)
    return session


def _save_transcript(session: dict, verbose: bool = True):
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    fname = TRANSCRIPTS_DIR / f"session_{session['timestamp'].replace(':', '-')}.md"
    lines = [
        f"# Mock Interview Transcript — {session['target_role']} ({session['focus_area']})",
        f"_{session['timestamp']}_\n",
    ]
    for i, (turn, score) in enumerate(zip(session["history"], session["scores"])):
        lines.append(f"### Turn {i+1}")
        lines.append(f"**Interviewer:** {turn['question']}\n")
        lines.append(f"**Candidate:** {turn['answer']}\n")
        lines.append(f"<details><summary>Evaluator signal</summary>\n\n```json\n{score}\n```\n</details>\n")
    lines.append("---\n")
    lines.append(session["report"])

    s = session["llm_summary"]
    lines.append("\n---\n")
    lines.append("<details><summary>Run metrics (LLMOps)</summary>\n")
    lines.append(f"- LLM calls: {s['calls']} (errors {s['errors']}, escalations {s['escalations']}, fallbacks {s['fallbacks']})")
    lines.append(f"- Tokens in/out: {s['input_tokens']}/{s['output_tokens']}; est. cost ${s['est_cost_usd']:.5f}")
    lines.append(f"- Latency p50/p95: {s['latency_p50_ms']:.0f}/{s['latency_p95_ms']:.0f} ms; strong-model share {s['strong_share']:.0%}")
    lines.append(f"- Prompt versions: {session['prompt_versions']}")
    lines.append("\n</details>")
    fname.write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print(f"[transcript saved to {fname}]")
