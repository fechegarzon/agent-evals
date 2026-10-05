from __future__ import annotations

from typing import Any, ClassVar

import pytest

from agent_evals.scorers import (
    check_disclaimers,
    check_expected_facts,
    check_forbidden_claims,
    check_forbidden_tools,
    check_json_schema,
    check_max_questions,
    check_numeric,
    check_tool_calls,
    matches,
    within_tolerance,
)
from agent_evals.suite import ExpectedToolCall, NamedPattern, NumericCheck
from agent_evals.types import AgentResponse, ToolCall


def reply(text: str = "", **kwargs) -> AgentResponse:
    return AgentResponse(text=text, **kwargs)


class TestMatches:
    def test_plain_string_is_case_insensitive_substring(self):
        assert matches("Bank Statements", "send bank statements please")
        assert not matches("tax returns", "send bank statements")

    def test_regex_prefix(self):
        assert matches(r"re:can'?t promise", "I cant promise that")
        assert not matches(r"re:\bpromise\b", "promised")

    def test_regex_is_multiline(self):
        assert matches(r"re:^2\.", "Need:\n1. income\n2. debts")


class TestTextChecks:
    def test_expected_facts(self):
        results = check_expected_facts(
            ["income", "re:down ?payment"], reply("Income and down payment")
        )
        assert [r.passed for r in results] == [True, True]

    def test_missing_fact_fails(self):
        [result] = check_expected_facts(["closing costs"], reply("Get quotes."))
        assert not result.passed
        assert result.kind == "facts"

    def test_forbidden_claim_present_fails(self):
        pattern = NamedPattern(name="implies-approval", pattern=r"re:you're\s+(?:pre-?)?approved")
        [result] = check_forbidden_claims([pattern], reply("Good news: you're pre-approved!"))
        assert not result.passed
        assert result.name == "forbidden:implies-approval"

    def test_forbidden_claim_absent_passes(self):
        [result] = check_forbidden_claims([r"re:you're\s+approved"], reply("I can't promise."))
        assert result.passed

    def test_disclaimer_required(self):
        ok, missing = check_disclaimers(
            ["not a loan offer", "lender decides"], reply("This is not a loan offer.")
        )
        assert ok.passed
        assert not missing.passed

    def test_max_questions(self):
        assert check_max_questions(1, reply("What is your income?")).passed
        too_many = check_max_questions(1, reply("Income? Debts? Term?"))
        assert not too_many.passed
        assert too_many.value == 3


class TestToolCalls:
    def test_match_with_numeric_tolerance(self):
        expected = [
            ExpectedToolCall(
                name="calculate_payment",
                args={"principal": 400000, "annual_rate_pct": 7},
                tolerance_pct=0.5,
            )
        ]
        call = ToolCall(
            name="calculate_payment",
            args={"principal": 401000, "annual_rate_pct": 7.0, "term_years": 30},
        )
        [result] = check_tool_calls(expected, reply(tool_calls=[call]))
        assert result.passed  # 0.25% off, extra arg ignored

    def test_wrong_args_fail_with_detail(self):
        expected = [ExpectedToolCall(name="calculate_payment", args={"principal": 320000})]
        call = ToolCall(name="calculate_payment", args={"principal": 400000})
        [result] = check_tool_calls(expected, reply(tool_calls=[call]))
        assert not result.passed
        assert "different args" in result.detail

    def test_missing_call_fails(self):
        [result] = check_tool_calls([ExpectedToolCall(name="calculate_payment")], reply())
        assert not result.passed
        assert result.detail == "tool was not called"

    def test_each_expected_call_needs_its_own_actual_call(self):
        expected = [ExpectedToolCall(name="lookup"), ExpectedToolCall(name="lookup")]
        results = check_tool_calls(expected, reply(tool_calls=[ToolCall(name="lookup")]))
        assert [r.passed for r in results] == [True, False]

    def test_forbidden_tool(self):
        [result] = check_forbidden_tools(
            ["calculate_payment"], reply(tool_calls=[ToolCall(name="calculate_payment")])
        )
        assert not result.passed


class TestJsonSchema:
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "required": ["monthly_payment"],
        "properties": {"monthly_payment": {"type": "number", "exclusiveMinimum": 0}},
    }

    def test_valid(self):
        assert check_json_schema(self.schema, reply(structured={"monthly_payment": 10.5})).passed

    def test_invalid_reports_path(self):
        result = check_json_schema(self.schema, reply(structured={"monthly_payment": -1}))
        assert not result.passed
        assert result.detail.startswith("monthly_payment:")

    def test_no_structured_output(self):
        assert not check_json_schema(self.schema, reply("text only")).passed


class TestNumeric:
    @pytest.mark.parametrize(
        ("actual", "expected", "tol", "ok"),
        [
            (2022.62, 2022.62, 1.0, True),
            (2042.00, 2022.62, 1.0, True),  # +0.96%
            (2043.00, 2022.62, 1.0, False),  # +1.01%
            (2002.50, 2022.62, 1.0, True),  # -0.99%
            (0.004, 0.0, 1.0, True),  # zero expected: absolute tolerance
            (0.02, 0.0, 1.0, False),
        ],
    )
    def test_within_tolerance(self, actual, expected, tol, ok):
        assert within_tolerance(actual, expected, tol) is ok

    def test_negative_tolerance_rejected(self):
        with pytest.raises(ValueError):
            within_tolerance(1, 1, -1)

    def test_from_structured_field(self):
        check = NumericCheck(name="payment", field="monthly_payment", expected=2022.62)
        result = check_numeric(check, reply(structured={"monthly_payment": 2030.0}))
        assert result.passed
        assert result.value == 2030.0

    def test_from_nested_field(self):
        check = NumericCheck(name="dti", field="ratios.dti", expected=30.0)
        assert check_numeric(check, reply(structured={"ratios": {"dti": 30.1}})).passed

    def test_from_text_pattern(self):
        check = NumericCheck(name="payment", pattern=r"\$([\d,]+\.\d{2})", expected=2010.05)
        assert check_numeric(check, reply("Your payment is $2,010.05 a month.")).passed

    def test_out_of_tolerance_fails(self):
        check = NumericCheck(name="payment", field="monthly_payment", expected=2022.62)
        assert not check_numeric(check, reply(structured={"monthly_payment": 2100})).passed

    def test_missing_value_fails(self):
        check = NumericCheck(name="payment", field="monthly_payment", expected=1.0)
        result = check_numeric(check, reply("no numbers"))
        assert not result.passed
        assert "no value found" in result.detail

    def test_needs_exactly_one_source(self):
        with pytest.raises(ValueError):
            NumericCheck(name="x", expected=1.0)
        with pytest.raises(ValueError):
            NumericCheck(name="x", expected=1.0, field="a", pattern="b")
