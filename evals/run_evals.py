"""
Agent evaluation runner.

  python -m evals.run_evals --dry-run                 # offline: tests the harness + plumbing (mock LLM)
  python -m evals.run_evals                           # live: real models (needs GROQ_API_KEY)
  python -m evals.run_evals --suite evaluator coach   # a subset
  python -m evals.run_evals --judge                   # + LLM-as-judge scores (live only)
  python -m evals.run_evals --compare                 # fail (exit 1) if any metric regressed vs the stored baseline
  python -m evals.run_evals --update-baseline         # accept the current results as the new baseline
  python -m evals.run_evals --compare-routing         # strong-only vs auto routing: accuracy, cost, latency (live)

Baselines are stored per mode (evals/baselines/dry_run.json, live.json) because
dry-run numbers describe the mock, not the model.
"""
import sys
import os

# --dry-run must be applied before `config` is imported
if "--dry-run" in sys.argv:
    os.environ["INTERVIEW_DRY_RUN"] = "1"

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evals" / "results"
BASELINES_DIR = ROOT / "evals" / "baselines"

LOWER_IS_BETTER = {"coach.fabricated_quote_rate", "coach.score_mismatch_rate", "interviewer.leak_rate"}


def compare_to_baseline(current: dict, baseline: dict, tolerance: float) -> list[str]:
    """Return human-readable regressions. Only quality metrics are gated (not ops.*)."""
    problems = []
    for key, base in baseline.items():
        if key.startswith("ops.") or key not in current:
            continue
        cur = current[key]
        worse = (cur > base + tolerance + 1e-9) if key in LOWER_IS_BETTER else (cur < base - tolerance - 1e-9)
        if worse:
            problems.append(f"{key}: {base:.3f} -> {cur:.3f}")
    return problems


def _fmt_checks(c) -> str:
    return "; ".join(f"{n}: {d}" for n, (ok, d) in c.checks.items() if not ok and d) or "; ".join(n for n, (ok, _) in c.checks.items() if not ok)


def write_reports(payload: dict, results: list, stamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(payload, indent=2, default=str))
    lines = [f"# Eval run {stamp} ({payload['mode']})", "",
             f"Routing: `{payload['routing_mode']}`  | models: `{payload['models']}`", f"Prompts: `{payload['prompt_versions']}`", "",
             "## Metrics", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in sorted(payload["metrics"].items())]
    lines += ["", "## Ops", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.5g} |" for k, v in payload["ops"].items()]
    lines += ["", "## Failing cases", ""]
    any_fail = False
    for r in results:
        for c in r.cases:
            if not c.passed:
                any_fail = True
                lines.append(f"- **{r.name}/{c.id}** — {_fmt_checks(c)}")
    if not any_fail:
        lines.append("_none_")
    path = RESULTS_DIR / f"{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_suites(names: list[str], use_judge: bool):
    from evals.suites import SUITES
    return [SUITES[n](use_judge) for n in names]


def ops_metrics(records: list[dict]) -> dict:
    from llmops import telemetry
    s = telemetry.summarize(records)
    return {
        "ops.llm_calls": s["calls"],
        "ops.est_cost_usd": s["est_cost_usd"],
        "ops.latency_p50_ms": s["latency_p50_ms"],
        "ops.latency_p95_ms": s["latency_p95_ms"],
        "ops.strong_share": s["strong_share"],
        "ops.escalation_rate": (s["escalations"] / s["calls"]) if s["calls"] else 0.0,
        "ops.errors": s["errors"],
    }


def compare_routing(use_judge: bool) -> int:
    """Run the same suites with one big model vs the router; print quality / cost / latency."""
    from llmops import telemetry
    rows = {}
    for mode in ("strong", "auto"):
        os.environ["ROUTING_MODE"] = mode
        mark = telemetry.mark()
        results = run_suites(["evaluator", "interviewer"], use_judge)
        metrics = {k: v for r in results for k, v in r.metrics.items()}
        rows[mode] = (metrics, ops_metrics(telemetry.records_since(mark)))
    os.environ.pop("ROUTING_MODE", None)

    keys = ["evaluator.verdict_accuracy", "evaluator.score_bounds_rate", "evaluator.ordering_accuracy", "interviewer.pass_rate"]
    print(f"\n{'':<34}{'strong-only':>14}{'auto-routed':>14}")
    for k in keys:
        print(f"{k:<34}{rows['strong'][0].get(k, 0):>14.3f}{rows['auto'][0].get(k, 0):>14.3f}")
    for k, fmt in (("ops.est_cost_usd", "{:>14.5f}"), ("ops.latency_p50_ms", "{:>14.0f}"),
                   ("ops.latency_p95_ms", "{:>14.0f}"), ("ops.strong_share", "{:>14.0%}"),
                   ("ops.escalation_rate", "{:>14.0%}")):
        print(f"{k:<34}" + fmt.format(rows["strong"][1][k]) + fmt.format(rows["auto"][1][k]))
    cs, ca = rows["strong"][1]["ops.est_cost_usd"], rows["auto"][1]["ops.est_cost_usd"]
    if cs > 0:
        print(f"\nRouting changed estimated cost by {(ca - cs) / cs:+.0%}.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", default=["router", "evaluator", "interviewer", "coach", "e2e"],
                    choices=["router", "evaluator", "interviewer", "coach", "e2e"])
    ap.add_argument("--dry-run", action="store_true", help="use the deterministic mock instead of a real LLM")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge scoring (live only)")
    ap.add_argument("--compare", action="store_true", help="exit 1 if a metric regressed vs the stored baseline")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="allowed metric drop (default 0.0 dry-run, 0.10 live: one flipped case of ~12)")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--compare-routing", action="store_true")
    args = ap.parse_args(argv)

    import config
    from llmops import telemetry
    from llmops.models import get_model
    from llmops.prompts import all_versions
    from llmops.router import routing_mode

    mode = "dry_run" if config.DRY_RUN else "live"
    if args.judge and config.DRY_RUN:
        print("[--judge ignored in dry-run mode]")
        args.judge = False

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LLM_LOG_PATH", str(RESULTS_DIR / f"{stamp}_calls.jsonl"))
    telemetry.set_session(f"eval-{stamp}")

    if args.compare_routing:
        if config.DRY_RUN:
            print("[note: dry-run has no real models, so cost/latency comparison is not meaningful]")
        return compare_routing(args.judge)

    mark = telemetry.mark()
    results = run_suites(args.suite, args.judge)
    metrics = {k: v for r in results for k, v in r.metrics.items()}
    ops = ops_metrics(telemetry.records_since(mark))

    payload = {
        "timestamp": stamp, "mode": mode, "routing_mode": routing_mode(),
        "models": {"fast": get_model("fast").name, "strong": get_model("strong").name},
        "prompt_versions": all_versions(), "metrics": metrics, "ops": ops,
        "suites": {r.name: [{"id": c.id, "passed": c.passed,
                             "checks": {n: {"ok": ok, "detail": d} for n, (ok, d) in c.checks.items()},
                             "extra": c.extra} for c in r.cases] for r in results},
    }
    report_path = write_reports(payload, results, stamp)

    for r in results:
        passed = sum(c.passed for c in r.cases)
        print(f"\n[{r.name}] {passed}/{len(r.cases)} cases passed")
        for c in r.cases:
            if not c.passed:
                print(f"   FAIL {c.id}: {_fmt_checks(c)}")
    print("\nMetrics:")
    for k, v in sorted(metrics.items()):
        print(f"   {k:<34}{v:.3f}")
    print("Ops:   " + "  ".join(f"{k.split('.')[1]}={v:.4g}" for k, v in ops.items()))
    print(f"\nReport: {report_path.relative_to(ROOT)}")

    baseline_path = BASELINES_DIR / f"{mode}.json"
    if args.update_baseline:
        BASELINES_DIR.mkdir(exist_ok=True)
        baseline_path.write_text(json.dumps(
            {"_meta": {"mode": mode, "updated": stamp, "models": payload["models"], "prompt_versions": payload["prompt_versions"],
                       "suites": args.suite},
             "metrics": metrics}, indent=2))
        print(f"Baseline written: {baseline_path.relative_to(ROOT)}")
        return 0

    if args.compare:
        if not baseline_path.exists():
            print(f"No baseline at {baseline_path.relative_to(ROOT)} - run with --update-baseline first.")
            return 2
        tol = args.tolerance if args.tolerance is not None else (0.0 if mode == "dry_run" else 0.10)
        base = json.loads(baseline_path.read_text())["metrics"]
        problems = compare_to_baseline(metrics, base, tol)
        if problems:
            print(f"\nREGRESSIONS vs baseline (tolerance {tol}):")
            for p in problems:
                print("   - " + p)
            return 1
        print(f"\nNo regressions vs baseline (tolerance {tol}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
