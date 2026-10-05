"""Command line: agent-evals run --suite ... --agent module:callable --judge mock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_evals.gate import GateConfig, evaluate_gate
from agent_evals.pricing import load_prices
from agent_evals.report import render_markdown
from agent_evals.results import RunResult
from agent_evals.runner import load_agent, run_suite
from agent_evals.scorers.judge import build_judge
from agent_evals.suite import load_suite

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-evals", description="Run an eval suite against an agent and gate the result."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite, write results + report, apply the gate")
    run.add_argument("--suite", required=True, help="path to the suite YAML")
    run.add_argument("--agent", required=True, help="agent to test, as module:callable")
    run.add_argument("--judge", default="mock", choices=["mock", "anthropic", "openai"])
    run.add_argument("--judge-model", default=None, help="override the pinned judge model id")
    run.add_argument("--pricing", default="config/pricing.yaml", help="price table YAML")
    run.add_argument("--baseline", default=None, help="results.json to compare against")
    run.add_argument("--out", default="runs/latest", help="directory for results.json + report.md")
    run.add_argument(
        "--max-drop",
        type=float,
        default=None,
        help="max pass-rate drop in points vs baseline (default: from suite settings)",
    )
    run.add_argument(
        "--allow-judge-change",
        action="store_true",
        help="compare against a baseline graded by a different judge (not recommended)",
    )
    run.add_argument(
        "--save-baseline", default=None, help="also write this run's results to this path"
    )
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    try:
        suite = load_suite(args.suite)
        prices = load_prices(args.pricing)
        agent = load_agent(args.agent)
        judge = build_judge(args.judge, model=args.judge_model)
        baseline = RunResult.load(args.baseline) if args.baseline else None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    run = run_suite(suite, agent, judge, prices, agent_ref=args.agent)
    max_drop = args.max_drop if args.max_drop is not None else suite.settings.max_pass_rate_drop_pts
    gate = evaluate_gate(
        run,
        baseline,
        GateConfig(max_drop_pts=max_drop, allow_judge_change=args.allow_judge_change),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run.save(out / "results.json")
    (out / "report.md").write_text(render_markdown(run, gate, baseline), encoding="utf-8")
    if args.save_baseline:
        run.save(args.save_baseline)

    s = run.summary
    print(f"suite {suite.name} v{suite.version} | agent {args.agent} | judge {judge.info.label()}")
    line = f"{s.passed}/{s.total} passed ({s.pass_rate:.1f}%)"
    if baseline is not None and gate.pass_rate_delta is not None:
        line += f" | baseline {baseline.summary.pass_rate:.1f}% ({gate.pass_rate_delta:+.1f} pts)"
    print(line)
    print(
        f"cost ${s.cost_total_usd:.6f} total, ${s.cost_per_case_usd:.6f}/case | "
        f"latency p50 {s.latency_p50_ms:.2f} ms, p95 {s.latency_p95_ms:.2f} ms"
    )
    for reason in gate.reasons:
        print(f"  FAIL: {reason}")
    for note in gate.notes:
        print(f"  note: {note}")
    print(f"gate: {'PASS' if gate.passed else 'FAIL'}")
    print(f"wrote {out / 'results.json'} and {out / 'report.md'}")
    return EXIT_OK if gate.passed else EXIT_GATE_FAILED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    return EXIT_USAGE  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
