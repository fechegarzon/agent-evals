"""Numeric checks with a relative tolerance (for example, monthly payment within 1%)."""

from __future__ import annotations

import re
from typing import Any

from agent_evals.results import CheckResult
from agent_evals.suite import NumericCheck
from agent_evals.types import AgentResponse


def within_tolerance(actual: float, expected: float, tolerance_pct: float) -> bool:
    """True if `actual` is within `tolerance_pct` percent of `expected`.

    When `expected` is zero the tolerance can't be relative, so we fall back to an
    absolute tolerance of tolerance_pct / 100.
    """
    if tolerance_pct < 0:
        raise ValueError("tolerance_pct must be >= 0")
    if expected == 0:
        return abs(actual) <= tolerance_pct / 100.0 + 1e-9
    return abs(actual - expected) <= abs(expected) * tolerance_pct / 100.0 + 1e-9


def _get_path(data: dict[str, Any] | None, path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _to_float(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_value(check: NumericCheck, response: AgentResponse) -> float | None:
    if check.field is not None:
        return _to_float(_get_path(response.structured, check.field))
    assert check.pattern is not None
    match = re.search(check.pattern, response.text, flags=re.IGNORECASE)
    if not match:
        return None
    return _to_float(match.group(1) if match.groups() else match.group(0))


def check_numeric(check: NumericCheck, response: AgentResponse) -> CheckResult:
    actual = extract_value(check, response)
    source = f"field {check.field}" if check.field else f"pattern {check.pattern!r}"
    if actual is None:
        return CheckResult(
            name=f"numeric:{check.name}",
            kind="numeric",
            passed=False,
            detail=f"no value found ({source})",
        )
    ok = within_tolerance(actual, check.expected, check.tolerance_pct)
    return CheckResult(
        name=f"numeric:{check.name}",
        kind="numeric",
        passed=ok,
        value=actual,
        detail=f"got {actual:,.2f}, expected {check.expected:,.2f} ±{check.tolerance_pct}%",
    )
