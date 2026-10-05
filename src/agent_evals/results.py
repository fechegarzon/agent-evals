"""Result records written to results.json and read back as baselines."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from agent_evals.types import ToolCall, Usage


class CheckResult(BaseModel):
    name: str
    kind: str  # facts | forbidden | disclaimer | questions | tool_call | schema | numeric | judge
    passed: bool
    detail: str = ""
    value: float | None = None  # numeric value or judge score, when there is one


class JudgeInfo(BaseModel):
    """Everything that defines "the same judge". Compared before trusting a delta."""

    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str

    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.model, self.prompt_sha256)

    def label(self) -> str:
        return (
            f"{self.provider}/{self.model} (prompt {self.prompt_version}, {self.prompt_sha256[:8]})"
        )


class CaseResult(BaseModel):
    id: str
    tags: list[str]
    critical: bool
    passed: bool
    checks: list[CheckResult]
    error: str | None = None
    latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    agent_model: str = "unknown"
    agent_usage: Usage = Field(default_factory=Usage)
    judge_usage: Usage = Field(default_factory=Usage)
    agent_cost_usd: float | None = None
    judge_cost_usd: float | None = None
    judge_score: int | None = None
    judge_disagreed: bool = False  # judge's own pass/fail differs from the code's decision
    response_text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return (self.agent_cost_usd or 0.0) + (self.judge_cost_usd or 0.0)

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


class TagStats(BaseModel):
    total: int
    passed: int

    @property
    def pass_rate(self) -> float:
        return 100.0 * self.passed / self.total if self.total else 0.0


class Summary(BaseModel):
    total: int
    passed: int
    pass_rate: float
    by_tag: dict[str, TagStats]
    critical_failures: list[str]
    cost_total_usd: float
    cost_per_case_usd: float
    unpriced_models: list[str]
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    judge_mean_score: float | None
    judge_disagreements: int


class RunMetadata(BaseModel):
    run_id: str
    started_at: str
    tool_version: str
    suite_name: str
    suite_version: str
    suite_sha256: str
    agent_ref: str
    agent_models: list[str]
    judge: JudgeInfo
    judge_min_score: int
    pricing_version: str
    git_sha: str | None = None


class RunResult(BaseModel):
    metadata: RunMetadata
    summary: Summary
    cases: list[CaseResult]

    def case_map(self) -> dict[str, CaseResult]:
        return {c.id: c for c in self.cases}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
