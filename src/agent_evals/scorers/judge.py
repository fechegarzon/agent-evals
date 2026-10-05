"""LLM-as-judge with typed output and a pinned identity.

Rule: the judge scores, code decides. The judge returns a 1-5 score, a short
rationale and its own pass/fail opinion. Whether the case passes is decided here,
in code, by comparing the score to a threshold from the suite config. The
threshold is never shown to the judge.

Every judge reports a JudgeInfo (provider, model id, prompt version, prompt hash).
The runner stores it in the run metadata and the gate refuses to compare runs
that used different judges.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from agent_evals.results import CheckResult, JudgeInfo
from agent_evals.scorers.deterministic import matches
from agent_evals.suite import JudgeSpec, Message
from agent_evals.types import AgentResponse, Usage

PROMPT_VERSION = "judge-v1"

JUDGE_SYSTEM_PROMPT = """\
You grade replies from a mortgage pre-qualification assistant.
Grade only against the rubric you are given. Do not reward length or politeness.

Scale:
5 = fully meets the rubric
4 = meets it with a minor gap
3 = partly meets it
2 = mostly misses it
1 = misses it or does something harmful

Return JSON with: score (integer 1-5), rationale (one or two sentences that point
to the reply text), passed (your own opinion: does the reply meet the rubric?).\
"""

JUDGE_USER_TEMPLATE = """\
<rubric>
{rubric}
</rubric>

<points_a_good_reply_usually_covers>
{hints}
</points_a_good_reply_usually_covers>

<conversation>
{conversation}
</conversation>

<assistant_reply>
{reply}
</assistant_reply>\
"""

PROMPT_SHA256 = hashlib.sha256(
    (JUDGE_SYSTEM_PROMPT + "\n" + JUDGE_USER_TEMPLATE).encode("utf-8")
).hexdigest()

VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
        "passed": {"type": "boolean"},
    },
    "required": ["score", "rationale", "passed"],
    "additionalProperties": False,
}

# Pinned defaults. Override with env vars, but whatever runs gets written to metadata.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-2025-04-14"


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    passed: bool


class JudgeError(RuntimeError):
    """The judge could not produce a valid verdict. The case fails closed."""


@dataclass
class JudgeOutcome:
    verdict: JudgeVerdict
    usage: Usage
    latency_ms: float


class Judge(Protocol):
    info: JudgeInfo

    def grade(
        self, spec: JudgeSpec, conversation: list[Message], response: AgentResponse
    ) -> JudgeOutcome: ...


def render_prompt(spec: JudgeSpec, conversation: list[Message], response: AgentResponse) -> str:
    hints = "\n".join(f"- {h.removeprefix('re:')}" for h in spec.hints) or "- (none given)"
    transcript = "\n".join(f"{m.role.upper()}: {m.content}" for m in conversation)
    return JUDGE_USER_TEMPLATE.format(
        rubric=spec.rubric.strip(), hints=hints, conversation=transcript, reply=response.text
    )


def parse_verdict(raw: str) -> JudgeVerdict:
    try:
        return JudgeVerdict.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JudgeError(f"judge returned an invalid verdict: {exc}") from exc


class MockJudge:
    """Deterministic, offline judge for tests and CI.

    It is not a quality signal. It scores 1 + 4 * (share of hints found in the reply),
    rounded. With no hints it returns 5 for any non-empty reply. Same input, same score.
    """

    def __init__(self) -> None:
        self.info = JudgeInfo(
            provider="mock",
            model="mock-judge",
            prompt_version=PROMPT_VERSION,
            prompt_sha256=PROMPT_SHA256,
        )

    def grade(
        self, spec: JudgeSpec, conversation: list[Message], response: AgentResponse
    ) -> JudgeOutcome:
        start = time.perf_counter()
        if not response.text.strip():
            verdict = JudgeVerdict(score=1, rationale="Empty reply.", passed=False)
        elif not spec.hints:
            verdict = JudgeVerdict(score=5, rationale="No hints to check.", passed=True)
        else:
            found = [h for h in spec.hints if matches(h, response.text)]
            missing = [h.removeprefix("re:") for h in spec.hints if h not in found]
            share = len(found) / len(spec.hints)
            score = 1 + round(4 * share)
            rationale = f"Covers {len(found)}/{len(spec.hints)} points."
            if missing:
                rationale += f" Missing: {', '.join(missing)}."
            verdict = JudgeVerdict(score=score, rationale=rationale, passed=share >= 0.75)
        return JudgeOutcome(
            verdict=verdict, usage=Usage(), latency_ms=(time.perf_counter() - start) * 1000
        )


class AnthropicJudge:
    """Judge backed by the Anthropic Messages API with a JSON-schema output format.

    No automatic model fallback on purpose: a fallback would silently swap the judge
    mid-run. A refusal or bad output raises JudgeError and the case fails closed.
    """

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or os.environ.get("AGENT_EVALS_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise JudgeError("install the extra: pip install 'agent-evals[anthropic]'") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.info = JudgeInfo(
            provider="anthropic",
            model=self.model,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=PROMPT_SHA256,
        )

    def grade(
        self, spec: JudgeSpec, conversation: list[Message], response: AgentResponse
    ) -> JudgeOutcome:
        start = time.perf_counter()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": render_prompt(spec, conversation, response)}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": VERDICT_JSON_SCHEMA},
            },
        )
        latency = (time.perf_counter() - start) * 1000
        if msg.stop_reason == "refusal":
            raise JudgeError("judge refused to grade this case")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise JudgeError(f"judge returned no text (stop_reason={msg.stop_reason})")
        usage = Usage(input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens)
        return JudgeOutcome(verdict=parse_verdict(text), usage=usage, latency_ms=latency)


class OpenAIJudge:
    """Judge backed by the OpenAI Chat Completions API with strict JSON-schema output."""

    def __init__(self, model: str | None = None, client: Any | None = None) -> None:
        self.model = model or os.environ.get("AGENT_EVALS_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        if client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise JudgeError("install the extra: pip install 'agent-evals[openai]'") from exc
            client = openai.OpenAI()
        self.client = client
        self.info = JudgeInfo(
            provider="openai",
            model=self.model,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=PROMPT_SHA256,
        )

    def grade(
        self, spec: JudgeSpec, conversation: list[Message], response: AgentResponse
    ) -> JudgeOutcome:
        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": render_prompt(spec, conversation, response)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True, "schema": VERDICT_JSON_SCHEMA},
            },
        )
        latency = (time.perf_counter() - start) * 1000
        content = completion.choices[0].message.content
        if not content:
            raise JudgeError("judge returned an empty message")
        usage = Usage(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )
        return JudgeOutcome(verdict=parse_verdict(content), usage=usage, latency_ms=latency)


def build_judge(name: str, model: str | None = None) -> Judge:
    if name == "mock":
        return MockJudge()
    if name == "anthropic":
        return AnthropicJudge(model=model)
    if name == "openai":
        return OpenAIJudge(model=model)
    raise ValueError(f"unknown judge {name!r} (use mock, anthropic or openai)")


def decide(verdict: JudgeVerdict, min_score: int) -> bool:
    """The only place a judge score turns into pass/fail."""
    return verdict.score >= min_score


def judge_check(outcome: JudgeOutcome, min_score: int) -> CheckResult:
    passed = decide(outcome.verdict, min_score)
    return CheckResult(
        name="judge",
        kind="judge",
        passed=passed,
        value=float(outcome.verdict.score),
        detail=f"score {outcome.verdict.score} (need >= {min_score}): {outcome.verdict.rationale}",
    )
