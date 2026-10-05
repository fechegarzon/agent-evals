"""Markdown report: what passed, what it cost, what changed since the baseline."""

from __future__ import annotations

from agent_evals.gate import GateResult
from agent_evals.results import RunResult


def _usd(value: float) -> str:
    return f"${value:.6f}" if value < 0.01 else f"${value:.4f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _delta(cur: float, base: float | None, unit: str = " pts") -> str:
    if base is None:
        return "n/a"
    return f"{cur - base:+.1f}{unit}"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(run: RunResult, gate: GateResult, baseline: RunResult | None = None) -> str:
    md = run.metadata
    s = run.summary
    bs = baseline.summary if baseline else None
    lines: list[str] = []
    add = lines.append

    add(f"# Eval report: {md.suite_name} v{md.suite_version}")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Run | `{md.run_id}` ({md.started_at}) |")
    add(f"| Agent | `{md.agent_ref}` ({', '.join(md.agent_models) or 'n/a'}) |")
    add(f"| Judge | {md.judge.label()} |")
    add(f"| Judge threshold | score >= {md.judge_min_score} (suite config, not in the prompt) |")
    add(f"| Pricing table | {md.pricing_version} |")
    if baseline:
        add(f"| Baseline | `{baseline.metadata.run_id}` ({baseline.metadata.judge.label()}) |")
    else:
        add("| Baseline | none |")
    add("")

    add(f"## Gate: {'PASS' if gate.passed else 'FAIL'}")
    add("")
    for reason in gate.reasons:
        add(f"- FAIL: {reason}")
    for note in gate.notes:
        add(f"- Note: {note}")
    if not gate.reasons and not gate.notes:
        add("- No critical failures and the pass rate is within the allowed drop.")
    add("")

    add("## Summary")
    add("")
    add("| Metric | This run | Baseline | Change |")
    add("|---|---|---|---|")
    add(
        f"| Pass rate | {_pct(s.pass_rate)} ({s.passed}/{s.total}) | "
        f"{_pct(bs.pass_rate) + f' ({bs.passed}/{bs.total})' if bs else 'n/a'} | "
        f"{_delta(s.pass_rate, bs.pass_rate if bs else None)} |"
    )
    add(
        f"| Critical failures | {len(s.critical_failures)} | "
        f"{len(bs.critical_failures) if bs else 'n/a'} | |"
    )
    add(f"| Total cost | {_usd(s.cost_total_usd)} | {_usd(bs.cost_total_usd) if bs else 'n/a'} | |")
    cost_change = "n/a"
    if bs and bs.cost_per_case_usd > 0:
        cost_change = f"{100 * (s.cost_per_case_usd / bs.cost_per_case_usd - 1):+.1f}%"
    add(
        f"| Cost per case | {_usd(s.cost_per_case_usd)} | "
        f"{_usd(bs.cost_per_case_usd) if bs else 'n/a'} | {cost_change} |"
    )
    add(
        f"| Agent latency p50 / p95 | {s.latency_p50_ms:.2f} / {s.latency_p95_ms:.2f} ms | "
        f"{f'{bs.latency_p50_ms:.2f} / {bs.latency_p95_ms:.2f} ms' if bs else 'n/a'} | |"
    )
    add(
        f"| Tokens in / out | {s.input_tokens:,} / {s.output_tokens:,} | "
        f"{f'{bs.input_tokens:,} / {bs.output_tokens:,}' if bs else 'n/a'} | |"
    )
    judge_mean = f"{s.judge_mean_score:.2f}" if s.judge_mean_score is not None else "n/a"
    base_judge_mean = (
        f"{bs.judge_mean_score:.2f}" if bs and bs.judge_mean_score is not None else "n/a"
    )
    add(f"| Judge mean score | {judge_mean} | {base_judge_mean} | |")
    add(f"| Judge vs code disagreements | {s.judge_disagreements} | | |")
    if s.unpriced_models:
        add("")
        add(f"Models missing from the price table: {', '.join(s.unpriced_models)}.")
    add("")

    add("## Pass rate by tag")
    add("")
    add("| Tag | Cases | This run | Baseline | Change |")
    add("|---|---|---|---|---|")
    for tag, stats in s.by_tag.items():
        base_stats = bs.by_tag.get(tag) if bs else None
        add(
            f"| {tag} | {stats.total} | {_pct(stats.pass_rate)} | "
            f"{_pct(base_stats.pass_rate) if base_stats else 'n/a'} | "
            f"{_delta(stats.pass_rate, base_stats.pass_rate if base_stats else None)} |"
        )
    add("")

    cases = run.case_map()
    add("## Regressions vs baseline")
    add("")
    if baseline is None:
        add("No baseline given.")
    elif not gate.regressions:
        add("None.")
    for case_id in gate.regressions:
        case = cases[case_id]
        flag = " (critical)" if case.critical else ""
        add(f"- `{case_id}`{flag}: " + "; ".join(c.name for c in case.failed_checks()))
    if gate.fixed:
        add("")
        add("Fixed since baseline: " + ", ".join(f"`{c}`" for c in gate.fixed))
    add("")

    failed = [c for c in run.cases if not c.passed]
    add("## Failed cases")
    add("")
    if not failed:
        add("None.")
    for case in failed:
        flag = " **critical**" if case.critical else ""
        add(f"### `{case.id}`{flag}")
        add("")
        add(f"Tags: {', '.join(case.tags) or 'none'}")
        add("")
        if case.error:
            add(f"- error: {_escape(case.error)}")
        for check in case.failed_checks():
            add(f"- `{check.name}`: {_escape(check.detail)}")
        add("")
        add(f"> {_escape(case.response_text[:400])}")
        add("")

    add("## Per-case results")
    add("")
    add("| Case | Result | Latency (ms) | Tokens in / out | Cost | Judge |")
    add("|---|---|---|---|---|---|")
    for case in run.cases:
        result = "pass" if case.passed else ("**FAIL (critical)**" if case.critical else "FAIL")
        judge = str(case.judge_score) if case.judge_score is not None else "-"
        add(
            f"| `{case.id}` | {result} | {case.latency_ms:.2f} | "
            f"{case.agent_usage.input_tokens + case.judge_usage.input_tokens:,} / "
            f"{case.agent_usage.output_tokens + case.judge_usage.output_tokens:,} | "
            f"{_usd(case.cost_usd)} | {judge} |"
        )
    add("")
    return "\n".join(lines)
