"""
Summarize the LLM call log.

    python -m llmops.report                       # all calls in logs/llm_calls.jsonl
    python -m llmops.report --session 2026-10-13T16-20-01
    python -m llmops.report --log evals/results/x_calls.jsonl
"""
import argparse
import sys

from . import telemetry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="path to a JSONL call log")
    ap.add_argument("--session", default=None, help="only this session id")
    args = ap.parse_args(argv)

    path = args.log or telemetry.log_path()
    if path is None:
        print("Logging is disabled (LLM_LOG=0).")
        return 1
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        print("No call log found yet - run an interview first.")
        return 1
    records = telemetry.load_log(path)
    if args.session:
        records = [r for r in records if r.get("session_id") == args.session]
    if not records:
        print("No matching records.")
        return 1
    print(telemetry.format_summary(telemetry.summarize(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
