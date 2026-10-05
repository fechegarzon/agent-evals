from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.pricing import ModelPrice, PriceTable
from agent_evals.results import (
    CaseResult,
    JudgeInfo,
    RunMetadata,
    RunResult,
    Summary,
    TagStats,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable(
        version="test",
        models={
            "toy-agent-v1": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
            "mock-judge": ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0),
        },
    )


MOCK_JUDGE = JudgeInfo(
    provider="mock", model="mock-judge", prompt_version="judge-v1", prompt_sha256="a" * 64
)


def make_run(
    outcomes: dict[str, bool],
    critical: set[str] | None = None,
    judge: JudgeInfo = MOCK_JUDGE,
    suite_sha: str = "s" * 64,
) -> RunResult:
    """Build a small RunResult by hand: {case_id: passed}."""
    critical = critical or set()
    cases = [
        CaseResult(
            id=case_id,
            tags=["math" if i % 2 == 0 else "privacy"],
            critical=case_id in critical,
            passed=passed,
            checks=[],
            agent_cost_usd=0.001,
        )
        for i, (case_id, passed) in enumerate(outcomes.items())
    ]
    by_tag: dict[str, TagStats] = {}
    for c in cases:
        stats = by_tag.setdefault(c.tags[0], TagStats(total=0, passed=0))
        stats.total += 1
        stats.passed += int(c.passed)
    passed = sum(c.passed for c in cases)
    summary = Summary(
        total=len(cases),
        passed=passed,
        pass_rate=100.0 * passed / len(cases),
        by_tag=by_tag,
        critical_failures=[c.id for c in cases if c.critical and not c.passed],
        cost_total_usd=0.001 * len(cases),
        cost_per_case_usd=0.001,
        unpriced_models=[],
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        input_tokens=100,
        output_tokens=50,
        judge_mean_score=None,
        judge_disagreements=0,
    )
    metadata = RunMetadata(
        run_id="test-run",
        started_at="2026-10-01T00:00:00+00:00",
        tool_version="0.1.0",
        suite_name="test",
        suite_version="1",
        suite_sha256=suite_sha,
        agent_ref="tests:agent",
        agent_models=["toy-agent-v1"],
        judge=judge,
        judge_min_score=4,
        pricing_version="test",
    )
    return RunResult(metadata=metadata, summary=summary, cases=cases)
