"""End-to-end: the toy agent, the real suite, the mock judge, the CLI and the gate."""

from __future__ import annotations

import json

import pytest

from agent_evals.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main
from agent_evals.pricing import load_prices
from agent_evals.results import RunResult
from agent_evals.runner import load_agent, percentile, run_suite
from agent_evals.scorers.judge import MockJudge
from agent_evals.suite import Suite, load_suite
from examples.toy_agent import monthly_payment, respond


@pytest.fixture
def suite(root) -> Suite:
    return load_suite(root / "suites" / "prequal.yaml")


def test_suite_loads(suite):
    assert len(suite.cases) >= 15
    assert any(c.critical for c in suite.cases)
    assert suite.settings.judge_min_score == 4
    assert len(suite.sha256) == 64


def test_suite_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nversion: '1'\ncases:\n"
        "  - id: a\n    conversation: [{role: user, content: hi}]\n    expect_facts: [oops]\n"
    )
    with pytest.raises(ValueError, match="expect_facts"):
        load_suite(bad)


@pytest.mark.parametrize(
    ("principal", "rate", "years", "expected"),
    [(320_000, 6.5, 30, 2022.62), (250_000, 5.75, 15, 2076.03), (120_000, 0, 10, 1000.00)],
)
def test_amortization(principal, rate, years, expected):
    assert monthly_payment(principal, rate, years) == pytest.approx(expected, abs=0.01)


def test_toy_agent_on_suite(suite, root):
    prices = load_prices(root / "config" / "pricing.yaml")
    run = run_suite(suite, respond, MockJudge(), prices, agent_ref="examples.toy_agent:respond")
    s = run.summary
    assert s.critical_failures == []
    assert [c.id for c in run.cases if not c.passed] == ["lender-comparison"]  # known gap
    assert s.cost_total_usd > 0
    assert s.unpriced_models == []
    assert run.metadata.judge.provider == "mock"
    assert run.metadata.agent_models == ["toy-agent-v1"]


def test_agent_errors_fail_the_case(suite, prices):
    def broken(_messages):
        raise RuntimeError("boom")

    run = run_suite(suite, broken, MockJudge(), prices)
    assert run.summary.passed == 0
    assert "boom" in (run.cases[0].error or "")


def test_dict_responses_are_accepted(suite, prices):
    def dict_agent(_messages):
        return {"text": "hello", "model": "toy-agent-v1"}

    run = run_suite(suite, dict_agent, MockJudge(), prices)
    assert run.cases[0].error is None


def test_unpriced_model_is_reported(suite, prices):
    run = run_suite(suite, lambda _m: {"text": "hi", "model": "mystery-model"}, MockJudge(), prices)
    assert run.summary.unpriced_models == ["mystery-model"]


def test_percentile():
    assert percentile([], 50) == 0.0
    assert percentile([5.0, 1.0, 3.0], 50) == 3.0
    assert percentile(list(map(float, range(1, 101))), 95) == 95.0


def test_load_agent_rejects_bad_refs():
    with pytest.raises(ValueError):
        load_agent("no_colon_here")


def _cli(root, tmp_path, agent: str, *extra: str) -> tuple[int, RunResult, str]:
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--suite", str(root / "suites" / "prequal.yaml"),
            "--agent", agent,
            "--judge", "mock",
            "--pricing", str(root / "config" / "pricing.yaml"),
            "--out", str(out),
            *extra,
        ]
    )  # fmt: skip
    if code == EXIT_USAGE:
        return code, None, ""  # type: ignore[return-value]
    return code, RunResult.load(out / "results.json"), (out / "report.md").read_text()


def test_cli_good_agent_passes_gate(root, tmp_path):
    code, run, report = _cli(
        root,
        tmp_path,
        "examples.toy_agent:respond",
        "--baseline",
        str(root / "baselines/prequal.json"),
    )
    assert code == EXIT_OK
    assert run.summary.pass_rate == pytest.approx(93.75)
    assert "## Gate: PASS" in report


def test_cli_regressed_agent_fails_gate(root, tmp_path):
    code, run, report = _cli(
        root,
        tmp_path,
        "examples.toy_agent:respond_v2",
        "--baseline", str(root / "baselines/prequal.json"),
    )  # fmt: skip
    assert code == EXIT_GATE_FAILED
    assert "no-approval-promise-direct" in run.summary.critical_failures
    assert "## Gate: FAIL" in report
    assert "## Regressions vs baseline" in report


def test_cli_save_baseline(root, tmp_path):
    target = tmp_path / "baseline.json"
    code, _, _ = _cli(root, tmp_path, "examples.toy_agent:respond", "--save-baseline", str(target))
    assert code == EXIT_OK
    assert json.loads(target.read_text())["metadata"]["judge"]["provider"] == "mock"


def test_cli_usage_error(root, tmp_path, capsys):
    code, _, _ = _cli(root, tmp_path, "examples.nope:respond")
    assert code == EXIT_USAGE
    assert "error:" in capsys.readouterr().err
