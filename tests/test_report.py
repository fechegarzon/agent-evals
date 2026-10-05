from __future__ import annotations

from agent_evals.gate import GateConfig, evaluate_gate
from agent_evals.report import render_markdown
from agent_evals.results import CheckResult
from tests.conftest import make_run


def test_report_without_baseline():
    run = make_run({"a": True, "b": False})
    md = render_markdown(run, evaluate_gate(run, None, GateConfig()))
    assert md.startswith("# Eval report: test v1")
    assert "## Gate: PASS" in md
    assert "| Pass rate | 50.0% (1/2) | n/a | n/a |" in md
    assert "No baseline given." in md
    assert "## Per-case results" in md


def test_report_with_regression_and_failed_checks():
    baseline = make_run({"a": True, "b": True, "c": True, "d": True})
    current = make_run({"a": True, "b": False, "c": True, "d": True}, critical={"b"})
    current.cases[1].checks = [
        CheckResult(
            name="forbidden:implies-approval",
            kind="forbidden",
            passed=False,
            detail="forbidden claim present in reply",
        )
    ]
    current.cases[1].response_text = "Good news: you're pre-approved!"
    gate = evaluate_gate(current, baseline, GateConfig(max_drop_pts=5))
    md = render_markdown(current, gate, baseline)

    assert "## Gate: FAIL" in md
    assert "critical case(s) failed: b" in md
    assert "| Pass rate | 75.0% (3/4) | 100.0% (4/4) | -25.0 pts |" in md
    assert "- `b` (critical): forbidden:implies-approval" in md
    assert "### `b` **critical**" in md
    assert "> Good news: you're pre-approved!" in md
    assert "**FAIL (critical)**" in md


def test_pass_rate_by_tag_compares_to_baseline():
    baseline = make_run({"a": True, "b": True})  # a -> math, b -> privacy
    current = make_run({"a": True, "b": False})
    md = render_markdown(current, evaluate_gate(current, baseline, GateConfig()), baseline)
    assert "| math | 1 | 100.0% | 100.0% | +0.0 pts |" in md
    assert "| privacy | 1 | 0.0% | 100.0% | -100.0 pts |" in md


def test_cost_per_case_change_is_shown():
    baseline = make_run({"a": True})
    current = make_run({"a": True})
    current.summary.cost_per_case_usd = 0.0012  # +20%
    md = render_markdown(current, evaluate_gate(current, baseline, GateConfig()), baseline)
    assert "| Cost per case | $0.001200 | $0.001000 | +20.0% |" in md


def test_pipe_characters_are_escaped():
    run = make_run({"a": False})
    run.cases[0].checks = [CheckResult(name="fact:x", kind="facts", passed=False, detail="a|b")]
    md = render_markdown(run, evaluate_gate(run, None, GateConfig()))
    assert "a\\|b" in md
