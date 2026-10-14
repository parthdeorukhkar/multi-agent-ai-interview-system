"""
Usage: python main.py

Env vars:
    GROQ_API_KEY          required (free key: https://console.groq.com/keys)
    INTERVIEW_TEXT_MODE=1 skip mic/speaker, use typed input + printed output
    INTERVIEW_DRY_RUN=1   skip the LLM entirely, use a deterministic mock
"""
from orchestrator import run_interview


def main():
    print("=== AI Mock Interview Coach ===\n")
    target_role = input("Target role (e.g. Product Manager, Data Analyst): ").strip() or "Software Engineer"
    resume_snippet = input("Optional 2-3 line background/resume snippet (or press Enter to skip): ").strip()

    print("\nResume — you can either:")
    print("  1) paste the full path to a resume file (.pdf, .docx, or .txt)")
    print("  2) type a quick 2-3 line background snippet")
    print("  3) press Enter to skip")
    resume_input = input("Resume path or snippet: ").strip()

    resume_snippet = ""
    if resume_input:
        if resume_input.lower().endswith((".pdf", ".docx", ".txt")):
            try:
                from resume_reader import read_resume
                full_text = read_resume(resume_input)
                resume_snippet = full_text[:3000]  # cap length so the prompt stays reasonable
                print(f"[resume loaded: {len(full_text)} characters extracted]")
            except Exception as e:
                print(f"[could not read resume file ({e}), continuing without resume]")
        else:
            resume_snippet = resume_input

    print("\nFocus area: 1) behavioral  2) technical  3) case  4) mixed")
    choice = input("Choose 1-4 [default 4]: ").strip()
    focus_map = {"1": "behavioral", "2": "technical", "3": "case", "4": "mixed"}
    focus_area = focus_map.get(choice, "mixed")

    session = run_interview(target_role, focus_area, resume_snippet)

    print("\n" + "=" * 60)
    print(session["report"])
    print("=" * 60)


if __name__ == "__main__":
    main()