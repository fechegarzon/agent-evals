from __future__ import annotations

from agent_evals.gate import GateConfig, evaluate_gate
from agent_evals.results import JudgeInfo
from tests.conftest import make_run

TEN_CASES = {f"case-{i}": True for i in range(10)}


def with_failures(*failing: str) -> dict[str, bool]:
    return {k: k not in failing for k in TEN_CASES}


def test_no_baseline_passes_when_criticals_pass():
    gate = evaluate_gate(make_run(TEN_CASES), None, GateConfig())
    assert gate.passed
    assert gate.pass_rate_delta is None
    assert any("no baseline" in n for n in gate.notes)


def test_critical_failure_blocks_even_without_baseline():
    run = make_run(with_failures("case-3"), critical={"case-3"})
    gate = evaluate_gate(run, None, GateConfig())
    assert not gate.passed
    assert "case-3" in gate.reasons[0]


def test_critical_failure_blocks_even_when_pass_rate_holds():
    baseline = make_run(with_failures("case-1"))
    current = make_run(with_failures("case-3"), critical={"case-3"})
    gate = evaluate_gate(current, baseline, GateConfig(max_drop_pts=50))
    assert not gate.passed
    assert gate.pass_rate_delta == 0


def test_drop_within_limit_passes():
    baseline = make_run(TEN_CASES)
    current = make_run(with_failures("case-0"))  # -10 pts
    gate = evaluate_gate(current, baseline, GateConfig(max_drop_pts=10))
    assert gate.passed
    assert gate.regressions == ["case-0"]


def test_drop_over_limit_fails():
    baseline = make_run(TEN_CASES)
    current = make_run(with_failures("case-0", "case-1"))  # -20 pts
    gate = evaluate_gate(current, baseline, GateConfig(max_drop_pts=10))
    assert not gate.passed
    assert gate.pass_rate_delta == -20
    assert "dropped 20.0 pts" in gate.reasons[0]


def test_fixed_cases_are_reported():
    baseline = make_run(with_failures("case-5"))
    gate = evaluate_gate(make_run(TEN_CASES), baseline, GateConfig())
    assert gate.passed
    assert gate.fixed == ["case-5"]
    assert gate.pass_rate_delta == 10


def test_judge_change_blocks_comparison():
    new_judge = JudgeInfo(
        provider="anthropic",
        model="claude-sonnet-5-5",
        prompt_version="judge-v1",
        prompt_sha256="a" * 64,
    )
    baseline = make_run(TEN_CASES)
    current = make_run(TEN_CASES, judge=new_judge)
    gate = evaluate_gate(current, baseline, GateConfig())
    assert not gate.passed
    assert gate.judge_changed
    assert "not comparable" in gate.reasons[0]


def test_judge_prompt_change_counts_as_judge_change():
    baseline = make_run(TEN_CASES)
    edited = baseline.metadata.judge.model_copy(update={"prompt_sha256": "b" * 64})
    gate = evaluate_gate(make_run(TEN_CASES, judge=edited), baseline, GateConfig())
    assert gate.judge_changed
    assert not gate.passed


def test_judge_change_can_be_allowed_explicitly():
    new_judge = JudgeInfo(
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_version="judge-v1",
        prompt_sha256="a" * 64,
    )
    gate = evaluate_gate(
        make_run(TEN_CASES, judge=new_judge),
        make_run(TEN_CASES),
        GateConfig(allow_judge_change=True),
    )
    assert gate.passed
    assert gate.judge_changed
    assert any("allowed" in n for n in gate.notes)


def test_suite_change_is_noted():
    gate = evaluate_gate(make_run(TEN_CASES, suite_sha="x" * 64), make_run(TEN_CASES), GateConfig())
    assert gate.passed
    assert any("suite file changed" in n for n in gate.notes)


def test_removed_cases_are_noted():
    smaller = {k: v for k, v in TEN_CASES.items() if k != "case-9"}
    gate = evaluate_gate(make_run(smaller), make_run(TEN_CASES), GateConfig())
    assert any("case-9" in n for n in gate.notes)
