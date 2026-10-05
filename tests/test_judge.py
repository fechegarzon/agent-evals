from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent_evals.scorers.judge import (
    JUDGE_SYSTEM_PROMPT,
    AnthropicJudge,
    JudgeError,
    JudgeVerdict,
    MockJudge,
    OpenAIJudge,
    build_judge,
    decide,
    judge_check,
    render_prompt,
)
from agent_evals.suite import JudgeSpec, Message
from agent_evals.types import AgentResponse

SPEC = JudgeSpec(
    rubric="Teach the user how to compare offers.", hints=["APR", "closing costs", "quotes"]
)
CONVO = [Message(role="user", content="Which lender is best?")]


def test_verdict_is_typed():
    with pytest.raises(ValidationError):
        JudgeVerdict(score=6, rationale="x", passed=True)
    with pytest.raises(ValidationError):
        JudgeVerdict(score=0, rationale="x", passed=False)


def test_mock_judge_is_deterministic():
    judge = MockJudge()
    reply = AgentResponse(text="Compare APR and get quotes.")
    first = judge.grade(SPEC, CONVO, reply).verdict
    second = judge.grade(SPEC, CONVO, reply).verdict
    assert first == second
    assert first.score == 4  # 2 of 3 hints -> 1 + round(4 * 2/3)
    assert "closing costs" in first.rationale


def test_mock_judge_scale():
    judge = MockJudge()
    assert judge.grade(SPEC, CONVO, AgentResponse(text="Nothing useful.")).verdict.score == 1
    full = AgentResponse(text="Compare APR, closing costs and quotes.")
    assert judge.grade(SPEC, CONVO, full).verdict.score == 5
    assert judge.grade(SPEC, CONVO, AgentResponse(text=" ")).verdict.score == 1


def test_code_decides_not_the_judge():
    # The judge says "pass", but the threshold in config is stricter. Code wins.
    verdict = JudgeVerdict(score=4, rationale="Good enough.", passed=True)
    assert decide(verdict, min_score=4)
    assert not decide(verdict, min_score=5)

    # And the other way around: the judge says "fail" but the score clears the bar.
    lenient = JudgeVerdict(score=3, rationale="Meh.", passed=False)
    assert decide(lenient, min_score=3)


def test_judge_check_records_score_and_threshold():
    outcome = MockJudge().grade(SPEC, CONVO, AgentResponse(text="Get quotes."))
    check = judge_check(outcome, min_score=4)
    assert not check.passed
    assert check.value == 2.0
    assert "need >= 4" in check.detail


def test_threshold_never_reaches_the_prompt():
    prompt = JUDGE_SYSTEM_PROMPT + render_prompt(SPEC, CONVO, AgentResponse(text="hi"))
    assert "min_score" not in prompt
    assert ">= 4" not in prompt
    assert "threshold" not in prompt.lower()


def test_judge_identity_is_pinned():
    a, b = MockJudge().info, MockJudge().info
    assert a.key() == b.key()
    assert a.model == "mock-judge"
    assert len(a.prompt_sha256) == 64


def test_build_judge():
    assert isinstance(build_judge("mock"), MockJudge)
    with pytest.raises(ValueError):
        build_judge("gpt-who")


VERDICT_JSON = json.dumps({"score": 5, "rationale": "Covers it.", "passed": True})


class FakeAnthropic:
    def __init__(self, text: str | None = VERDICT_JSON, stop_reason: str = "end_turn"):
        self.calls: list[dict] = []
        content = [SimpleNamespace(type="text", text=text)] if text is not None else []
        self._msg = SimpleNamespace(
            stop_reason=stop_reason,
            content=content,
            usage=SimpleNamespace(input_tokens=900, output_tokens=60),
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._msg


def test_anthropic_judge_parses_typed_output():
    client = FakeAnthropic()
    judge = AnthropicJudge(model="claude-sonnet-5-5", client=client)
    outcome = judge.grade(SPEC, CONVO, AgentResponse(text="Compare APR."))
    assert outcome.verdict.score == 5
    assert outcome.usage.input_tokens == 900
    assert judge.info.model == "claude-sonnet-5-5"
    sent = client.calls[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_judge_fails_closed():
    with pytest.raises(JudgeError):
        AnthropicJudge(client=FakeAnthropic(stop_reason="refusal")).grade(
            SPEC, CONVO, AgentResponse(text="x")
        )
    with pytest.raises(JudgeError):
        AnthropicJudge(client=FakeAnthropic(text='{"score": 9}')).grade(
            SPEC, CONVO, AgentResponse(text="x")
        )


def test_openai_judge_parses_typed_output():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=VERDICT_JSON))],
        usage=SimpleNamespace(prompt_tokens=800, completion_tokens=40),
    )
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return completion

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge = OpenAIJudge(model="gpt-4.1-2025-04-14", client=client)
    outcome = judge.grade(SPEC, CONVO, AgentResponse(text="Compare APR."))
    assert outcome.verdict.passed
    assert outcome.usage.output_tokens == 40
    assert calls[0]["temperature"] == 0
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
