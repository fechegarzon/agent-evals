"""Eval suite format (YAML) and loader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Reject unknown keys so a typo in the suite file fails loudly instead of silently."""

    model_config = ConfigDict(extra="forbid")


class NamedPattern(_Strict):
    """A text pattern with a readable name for reports.

    In YAML you can write a bare string (the name defaults to the pattern itself) or
    a mapping: {name: promises-approval, pattern: "re:..."}.
    Plain patterns are case-insensitive substrings; a "re:" prefix means regex.
    """

    name: str
    pattern: str


def as_pattern(item: str | NamedPattern | dict[str, str]) -> NamedPattern:
    if isinstance(item, NamedPattern):
        return item
    if isinstance(item, dict):
        return NamedPattern.model_validate(item)
    label = item if len(item) <= 60 else item[:57] + "..."
    return NamedPattern(name=label, pattern=item)


def _patterns(items: Any) -> list[NamedPattern]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("expected a list of patterns")
    return [as_pattern(i) for i in items]


class Message(_Strict):
    role: Literal["user", "assistant"]
    content: str


class ExpectedToolCall(_Strict):
    """A tool call the agent must make. Extra args on the actual call are allowed."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tolerance_pct: float = 0.0  # applied to numeric args only


class NumericCheck(_Strict):
    """A number the agent must get right, within a relative tolerance.

    Read the actual value from `field` (dot path into the structured output) or from
    the first capture group of `pattern` (regex over the reply text).
    """

    name: str
    expected: float
    tolerance_pct: float = 1.0
    field: str | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> NumericCheck:
        if (self.field is None) == (self.pattern is None):
            raise ValueError(f"numeric check {self.name!r}: set exactly one of field/pattern")
        return self


class JudgeSpec(_Strict):
    """Ask the judge to grade the reply against a rubric.

    `hints` are things a good reply usually covers. Real judges see them as context;
    the offline mock judge uses them to compute a deterministic score.
    `min_score` overrides the suite default. The threshold never goes in the prompt.
    """

    rubric: str
    hints: list[str] = Field(default_factory=list)
    min_score: int | None = Field(default=None, ge=1, le=5)


class EvalCase(_Strict):
    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    critical: bool = False
    conversation: list[Message]

    # Text checks. Plain strings are case-insensitive substrings; "re:" prefix means regex.
    expected_facts: list[NamedPattern] = Field(default_factory=list)
    forbidden_claims: list[NamedPattern] = Field(default_factory=list)
    required_disclaimers: list[NamedPattern] = Field(default_factory=list)
    max_questions: int | None = None

    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    numeric: list[NumericCheck] = Field(default_factory=list)
    judge: JudgeSpec | None = None

    _normalize = field_validator(
        "expected_facts", "forbidden_claims", "required_disclaimers", mode="before"
    )(_patterns)

    @field_validator("conversation")
    @classmethod
    def _ends_with_user(cls, v: list[Message]) -> list[Message]:
        if not v or v[-1].role != "user":
            raise ValueError("conversation must end with a user message")
        return v


class SuiteSettings(_Strict):
    judge_min_score: int = Field(default=4, ge=1, le=5)
    max_pass_rate_drop_pts: float = 5.0
    global_forbidden_claims: list[NamedPattern] = Field(default_factory=list)

    _normalize = field_validator("global_forbidden_claims", mode="before")(_patterns)


class Suite(_Strict):
    name: str
    version: str
    description: str = ""
    settings: SuiteSettings = Field(default_factory=SuiteSettings)
    definitions: dict[str, Any] = Field(default_factory=dict)  # YAML anchors live here
    cases: list[EvalCase]
    sha256: str = ""

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
        return self


def load_suite(path: str | Path) -> Suite:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: suite file must be a YAML mapping")
    suite = Suite.model_validate(data)
    suite.sha256 = hashlib.sha256(raw).hexdigest()
    return suite
