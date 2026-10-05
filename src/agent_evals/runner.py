"""Run a suite against an agent: call it, score it, time it, price it."""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_evals import __version__
from agent_evals.pricing import PriceTable
from agent_evals.results import CaseResult, CheckResult, RunMetadata, RunResult, Summary, TagStats
from agent_evals.scorers import (
    check_disclaimers,
    check_expected_facts,
    check_forbidden_claims,
    check_forbidden_tools,
    check_json_schema,
    check_max_questions,
    check_numeric,
    check_tool_calls,
)
from agent_evals.scorers.judge import Judge, judge_check
from agent_evals.suite import EvalCase, Suite
from agent_evals.types import AgentResponse

AgentFn = Callable[[list[dict[str, str]]], AgentResponse | dict[str, Any]]


def load_agent(ref: str) -> AgentFn:
    """Import an agent from "package.module:callable". The current directory is importable."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"agent must look like 'module:callable', got {ref!r}")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ValueError(f"{ref!r} is not callable")
    return fn


def deterministic_checks(
    case: EvalCase, response: AgentResponse, suite: Suite
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks += check_expected_facts(case.expected_facts, response)
    checks += check_forbidden_claims(
        suite.settings.global_forbidden_claims + case.forbidden_claims, response
    )
    checks += check_disclaimers(case.required_disclaimers, response)
    if case.max_questions is not None:
        checks.append(check_max_questions(case.max_questions, response))
    checks += check_tool_calls(case.expected_tool_calls, response)
    checks += check_forbidden_tools(case.forbidden_tools, response)
    if case.output_schema is not None:
        checks.append(check_json_schema(case.output_schema, response))
    checks += [check_numeric(n, response) for n in case.numeric]
    return checks


def run_case(
    case: EvalCase, agent: AgentFn, judge: Judge, suite: Suite, prices: PriceTable
) -> CaseResult:
    conversation = [m.model_dump() for m in case.conversation]
    start = time.perf_counter()
    try:
        raw = agent(conversation)
        response = raw if isinstance(raw, AgentResponse) else AgentResponse.model_validate(raw)
    except Exception as exc:  # the agent is untrusted code; record and move on
        return CaseResult(
            id=case.id,
            tags=case.tags,
            critical=case.critical,
            passed=False,
            checks=[],
            error=f"agent error: {type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    checks = deterministic_checks(case, response, suite)

    judge_score = None
    judge_disagreed = False
    judge_latency = 0.0
    judge_usage = None
    if case.judge is not None:
        min_score = case.judge.min_score or suite.settings.judge_min_score
        try:
            outcome = judge.grade(case.judge, case.conversation, response)
        except Exception as exc:  # JudgeError, network, auth: fail closed, never pass
            checks.append(
                CheckResult(name="judge", kind="judge", passed=False, detail=f"judge error: {exc}")
            )
        else:
            check = judge_check(outcome, min_score)
            checks.append(check)
            judge_score = outcome.verdict.score
            judge_disagreed = outcome.verdict.passed != check.passed
            judge_latency = outcome.latency_ms
            judge_usage = outcome.usage

    result = CaseResult(
        id=case.id,
        tags=case.tags,
        critical=case.critical,
        passed=all(c.passed for c in checks),
        checks=checks,
        latency_ms=latency_ms,
        judge_latency_ms=judge_latency,
        agent_model=response.model,
        agent_usage=response.usage,
        agent_cost_usd=prices.cost(response.model, response.usage),
        judge_score=judge_score,
        judge_disagreed=judge_disagreed,
        response_text=response.text,
        tool_calls=response.tool_calls,
    )
    if judge_usage is not None:
        result.judge_usage = judge_usage
        result.judge_cost_usd = prices.cost(judge.info.model, judge_usage)
    return result


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Good enough for small eval sets."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize(cases: list[CaseResult], judge_model: str, prices: PriceTable) -> Summary:
    by_tag: dict[str, TagStats] = {}
    for case in cases:
        for tag in case.tags:
            stats = by_tag.setdefault(tag, TagStats(total=0, passed=0))
            stats.total += 1
            stats.passed += int(case.passed)

    unpriced = sorted(
        {c.agent_model for c in cases if c.agent_cost_usd is None and not c.error}
        | ({judge_model} if judge_model not in prices.models else set())
    )
    scores = [c.judge_score for c in cases if c.judge_score is not None]
    total_cost = sum(c.cost_usd for c in cases)
    passed = sum(c.passed for c in cases)
    return Summary(
        total=len(cases),
        passed=passed,
        pass_rate=100.0 * passed / len(cases) if cases else 0.0,
        by_tag=dict(sorted(by_tag.items())),
        critical_failures=[c.id for c in cases if c.critical and not c.passed],
        cost_total_usd=total_cost,
        cost_per_case_usd=total_cost / len(cases) if cases else 0.0,
        unpriced_models=unpriced,
        latency_p50_ms=percentile([c.latency_ms for c in cases], 50),
        latency_p95_ms=percentile([c.latency_ms for c in cases], 95),
        input_tokens=sum(c.agent_usage.input_tokens + c.judge_usage.input_tokens for c in cases),
        output_tokens=sum(c.agent_usage.output_tokens + c.judge_usage.output_tokens for c in cases),
        judge_mean_score=sum(scores) / len(scores) if scores else None,
        judge_disagreements=sum(c.judge_disagreed for c in cases),
    )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_suite(
    suite: Suite,
    agent: AgentFn,
    judge: Judge,
    prices: PriceTable,
    agent_ref: str = "<callable>",
) -> RunResult:
    started = datetime.now(UTC)
    cases = [run_case(case, agent, judge, suite, prices) for case in suite.cases]
    metadata = RunMetadata(
        run_id=f"{started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        started_at=started.isoformat(timespec="seconds"),
        tool_version=__version__,
        suite_name=suite.name,
        suite_version=suite.version,
        suite_sha256=suite.sha256,
        agent_ref=agent_ref,
        agent_models=sorted({c.agent_model for c in cases if not c.error}),
        judge=judge.info,
        judge_min_score=suite.settings.judge_min_score,
        pricing_version=prices.version,
        git_sha=_git_sha(),
    )
    return RunResult(
        metadata=metadata, summary=summarize(cases, judge.info.model, prices), cases=cases
    )
