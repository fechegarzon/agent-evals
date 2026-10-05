"""Deterministic checks: text patterns, tool calls, JSON schema, forbidden claims.

These are cheap, fast and never drift. Prefer them whenever the property can be
written down as code.
"""

from __future__ import annotations

import re
from typing import Any

import jsonschema

from agent_evals.results import CheckResult
from agent_evals.scorers.numeric import within_tolerance
from agent_evals.suite import ExpectedToolCall, NamedPattern, as_pattern
from agent_evals.types import AgentResponse, ToolCall

REGEX_PREFIX = "re:"


def matches(pattern: str, text: str) -> bool:
    """Plain strings are case-insensitive substrings.

    "re:..." is a regex, case-insensitive and multiline (so ^ matches each line start).
    """
    if pattern.startswith(REGEX_PREFIX):
        flags = re.IGNORECASE | re.MULTILINE
        return re.search(pattern[len(REGEX_PREFIX) :], text, flags=flags) is not None
    return pattern.lower() in text.lower()


PatternLike = str | NamedPattern


def check_expected_facts(facts: list[PatternLike], response: AgentResponse) -> list[CheckResult]:
    results = []
    for p in map(as_pattern, facts):
        ok = matches(p.pattern, response.text)
        results.append(
            CheckResult(
                name=f"fact:{p.name}",
                kind="facts",
                passed=ok,
                detail="" if ok else "expected fact not found in reply",
            )
        )
    return results


def check_forbidden_claims(claims: list[PatternLike], response: AgentResponse) -> list[CheckResult]:
    results = []
    for p in map(as_pattern, claims):
        hit = matches(p.pattern, response.text)
        results.append(
            CheckResult(
                name=f"forbidden:{p.name}",
                kind="forbidden",
                passed=not hit,
                detail="forbidden claim present in reply" if hit else "",
            )
        )
    return results


def check_disclaimers(disclaimers: list[PatternLike], response: AgentResponse) -> list[CheckResult]:
    results = []
    for p in map(as_pattern, disclaimers):
        ok = matches(p.pattern, response.text)
        results.append(
            CheckResult(
                name=f"disclaimer:{p.name}",
                kind="disclaimer",
                passed=ok,
                detail="" if ok else "required disclaimer missing",
            )
        )
    return results


def check_max_questions(limit: int, response: AgentResponse) -> CheckResult:
    """Proxy for "ask for everything in one list": count question marks in the reply."""
    count = response.text.count("?")
    return CheckResult(
        name="max_questions",
        kind="questions",
        passed=count <= limit,
        value=float(count),
        detail=f"{count} question(s), limit {limit}",
    )


def _arg_matches(expected: Any, actual: Any, tolerance_pct: float) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return within_tolerance(float(actual), float(expected), tolerance_pct)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _call_matches(expected: ExpectedToolCall, call: ToolCall) -> bool:
    if call.name != expected.name:
        return False
    return all(
        key in call.args and _arg_matches(value, call.args[key], expected.tolerance_pct)
        for key, value in expected.args.items()
    )


def check_tool_calls(
    expected: list[ExpectedToolCall], response: AgentResponse
) -> list[CheckResult]:
    """Each expected call must match a distinct actual call. Order is not enforced."""
    remaining = list(response.tool_calls)
    results = []
    for exp in expected:
        found = next((c for c in remaining if _call_matches(exp, c)), None)
        if found is not None:
            remaining.remove(found)
            results.append(CheckResult(name=f"tool:{exp.name}", kind="tool_call", passed=True))
            continue
        same_name = [c.args for c in response.tool_calls if c.name == exp.name]
        detail = f"called with different args: {same_name}" if same_name else "tool was not called"
        results.append(
            CheckResult(name=f"tool:{exp.name}", kind="tool_call", passed=False, detail=detail)
        )
    return results


def check_forbidden_tools(tools: list[str], response: AgentResponse) -> list[CheckResult]:
    called = {c.name for c in response.tool_calls}
    return [
        CheckResult(
            name=f"no_tool:{tool}",
            kind="tool_call",
            passed=tool not in called,
            detail=f"{tool} should not be called here" if tool in called else "",
        )
        for tool in tools
    ]


def check_json_schema(schema: dict[str, Any], response: AgentResponse) -> CheckResult:
    if response.structured is None:
        return CheckResult(
            name="schema", kind="schema", passed=False, detail="no structured output"
        )
    try:
        jsonschema.validate(response.structured, schema)
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return CheckResult(
            name="schema", kind="schema", passed=False, detail=f"{path}: {exc.message}"
        )
    return CheckResult(name="schema", kind="schema", passed=True)
