"""Regression gate: decide whether a run is allowed to ship."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_evals.results import RunResult


@dataclass
class GateConfig:
    max_drop_pts: float = 5.0
    fail_on_critical: bool = True
    allow_judge_change: bool = False


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)  # why it failed
    notes: list[str] = field(default_factory=list)  # worth reading, not blocking
    regressions: list[str] = field(default_factory=list)  # passed in baseline, fail now
    fixed: list[str] = field(default_factory=list)  # failed in baseline, pass now
    pass_rate_delta: float | None = None
    judge_changed: bool = False


def evaluate_gate(current: RunResult, baseline: RunResult | None, config: GateConfig) -> GateResult:
    result = GateResult(passed=True)

    if config.fail_on_critical and current.summary.critical_failures:
        result.reasons.append(
            "critical case(s) failed: " + ", ".join(current.summary.critical_failures)
        )

    if baseline is None:
        result.notes.append("no baseline: only critical cases were checked")
        result.passed = not result.reasons
        return result

    cur_judge, base_judge = current.metadata.judge, baseline.metadata.judge
    if cur_judge.key() != base_judge.key():
        result.judge_changed = True
        msg = (
            f"judge changed: baseline used {base_judge.label()}, this run used {cur_judge.label()}"
        )
        if config.allow_judge_change:
            result.notes.append(msg + " (allowed; treat the delta with care)")
        else:
            result.reasons.append(
                msg + ". Scores from different judges are not comparable. Re-run the "
                "baseline with the new judge, or pass --allow-judge-change."
            )

    if current.metadata.suite_sha256 != baseline.metadata.suite_sha256:
        result.notes.append(
            f"suite file changed since the baseline (v{baseline.metadata.suite_version} -> "
            f"v{current.metadata.suite_version}); the pass-rate delta mixes both effects"
        )

    base_cases = baseline.case_map()
    cur_cases = current.case_map()
    for case_id, cur in cur_cases.items():
        base = base_cases.get(case_id)
        if base is None:
            continue
        if base.passed and not cur.passed:
            result.regressions.append(case_id)
        elif not base.passed and cur.passed:
            result.fixed.append(case_id)
    missing = sorted(set(base_cases) - set(cur_cases))
    if missing:
        result.notes.append("cases in baseline but not in this run: " + ", ".join(missing))

    delta = current.summary.pass_rate - baseline.summary.pass_rate
    result.pass_rate_delta = delta
    if delta < -config.max_drop_pts:
        result.reasons.append(
            f"pass rate dropped {abs(delta):.1f} pts "
            f"({baseline.summary.pass_rate:.1f}% -> {current.summary.pass_rate:.1f}%), "
            f"limit is {config.max_drop_pts:.1f} pts"
        )

    result.passed = not result.reasons
    return result
