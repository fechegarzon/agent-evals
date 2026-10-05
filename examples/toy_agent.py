"""A deterministic toy pre-qualification assistant.

No model calls. It parses numbers out of the conversation with regexes, runs a
payment calculator "tool" and answers from templates. It exists so the eval
pipeline has something real to test, offline and for free.

`respond` is the good version. `respond_v2` simulates a prompt change that made
replies "friendlier": it drops disclaimers, hints at approval and asks for an ID.
The gate should block it.

All rules and thresholds here are made up for the demo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent_evals import AgentResponse, ToolCall, Usage

SYSTEM_PROMPT_TOKENS = {"v1": 420, "v2": 680}
DEMO_DTI_REVIEW_PCT = 43.0  # made-up review threshold for the demo

DISCLAIMER = "This is an estimate, not a loan offer. Final terms depend on the lender's review."
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MONEY = r"\$?\s*(\d[\d,]*(?:\.\d+)?\s*[kK]?)"


@dataclass
class Facts:
    price: float | None = None
    down_amount: float | None = None
    down_pct: float | None = None
    loan: float | None = None
    rate_pct: float | None = None
    term_years: int | None = None
    monthly_income: float | None = None
    monthly_debts: float | None = None

    @property
    def principal(self) -> float | None:
        if self.loan is not None:
            return self.loan
        if self.price is None:
            return None
        if self.down_amount is not None:
            return self.price - self.down_amount
        if self.down_pct is not None:
            return self.price * (1 - self.down_pct / 100)
        return None


def _money(raw: str) -> float:
    raw = raw.replace(",", "").replace("$", "").strip()
    if raw.lower().endswith("k"):
        return float(raw[:-1]) * 1000
    return float(raw)


def _first(patterns: list[str], text: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _money(match.group(1))
    return None


def extract_facts(user_text: str) -> Facts:
    t = user_text
    facts = Facts()
    facts.loan = _first(
        [
            rf"loan (?:amount )?(?:of )?{MONEY}",
            rf"{MONEY} loan\b",
            rf"\bon (?:a )?{MONEY}\s+at\b",
        ],
        t,
    )
    facts.price = _first(
        [
            rf"(?:home|house) price (?:is |of )?{MONEY}",
            rf"(?:home|house) (?:is|costs?) {MONEY}",
            rf"{MONEY} (?:home|house)\b",
        ],
        t,
    )
    facts.down_amount = _first([rf"{MONEY} down\b"], t)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%(\s*down)?", t):
        if match.group(2):
            facts.down_pct = float(match.group(1))
        elif facts.rate_pct is None:
            facts.rate_pct = float(match.group(1))

    term = re.search(r"(\d{1,2})[\s-]*(?:years?|yr)\b", t, flags=re.IGNORECASE)
    if term:
        facts.term_years = int(term.group(1))

    income = re.search(
        rf"(?:make|earn|income (?:is |of )?)\s*{MONEY}\s*(?:a|per|/)\s*(month|year)",
        t,
        flags=re.IGNORECASE,
    )
    if income:
        value = _money(income.group(1))
        facts.monthly_income = value / 12 if income.group(2).lower() == "year" else value

    if re.search(r"\bno (?:other )?debts?\b", t, flags=re.IGNORECASE):
        facts.monthly_debts = 0.0
    debts = _first([rf"pay {MONEY}\s*(?:a|per) month", rf"debts? (?:of |are )?{MONEY}"], t)
    if debts is not None:
        facts.monthly_debts = debts
    return facts


def monthly_payment(principal: float, annual_rate_pct: float, term_years: int) -> float:
    """Standard fixed-rate amortization: P * r / (1 - (1 + r)^-n)."""
    n = term_years * 12
    r = annual_rate_pct / 100 / 12
    if r == 0:
        return principal / n
    return principal * r / (1 - (1 + r) ** -n)


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _reply(
    variant: str,
    text: str,
    messages: list[dict[str, str]],
    tool_calls: list[ToolCall] | None = None,
    structured: dict | None = None,
) -> AgentResponse:
    prompt_chars = sum(len(m["content"]) for m in messages)
    calls = tool_calls or []
    usage = Usage(
        input_tokens=SYSTEM_PROMPT_TOKENS[variant] + prompt_chars // 4 + 60 * len(calls),
        output_tokens=len(text) // 4 + 20 * len(calls),
    )
    return AgentResponse(
        text=text,
        tool_calls=calls,
        structured=structured,
        usage=usage,
        model=f"toy-agent-{variant}",
    )


def _answer(messages: list[dict[str, str]], variant: str) -> AgentResponse:
    last = messages[-1]["content"]
    all_user = " ".join(m["content"] for m in messages if m["role"] == "user")
    friendly = variant == "v2"

    # 1. The user pasted an ID number. Don't echo it, don't use it.
    if SSN_RE.search(last) or _has(r"\b(?:SSN|social security|passport)\b", last):
        return _reply(
            variant,
            "Please delete that message. I don't use ID numbers, and you never have to "
            "share them here. For an estimate I only work with numbers like income, "
            "debts and the home price.",
            messages,
        )

    # 2. Approval questions. Never promise.
    if _has(r"\bapprov|\bguarantee", last):
        if friendly:
            text = "With numbers like these you're very likely approved. Let's get going!"
        else:
            text = (
                "I can't promise approval. Only the lender can decide, after a full review "
                "of your application, credit and documents. What I can do is give you an "
                "estimate and show what lenders usually look at: income, monthly debts, "
                "credit history and down payment."
            )
        return _reply(variant, text, messages)

    # 3. Missing documents.
    if _has(r"pay ?stubs?|employment letter|documents?|tax returns?|self-employed", last):
        if _has(r"employment letter", last):
            text = (
                "You don't need to wait. We can start the estimate now with your numbers "
                "and add the employment letter later, when you have it."
            )
        else:
            text = (
                "That's fine. You can still get an estimate now, and documents come later. "
                "If you don't have pay stubs, common alternatives are:\n"
                + _numbered(
                    [
                        "Bank statements from the last 12 months",
                        "Tax returns from the last 2 years",
                        "A profit and loss statement if you are self-employed",
                    ]
                )
                + "\nThe lender decides which documents it accepts, so treat this list as "
                "a starting point."
            )
        return _reply(variant, text, messages)

    # 4. Lender comparisons. Known gap: no APR or closing-cost guidance yet.
    if _has(r"which lender|best lender|best rate", last):
        return _reply(
            variant,
            "I can't recommend a specific lender. Rates change daily and depend on your "
            "profile. The best move is to get quotes from at least three lenders on the "
            "same day.",
            messages,
        )

    # 5. "What do you need from me?"
    if _has(r"what do you need|get started", last):
        items = [
            "Gross monthly income",
            "Monthly debt payments (car, cards, student loans)",
            "Home price you're targeting",
            "Down payment you plan to make",
            "Interest rate and loan term, if you have them",
        ]
        if friendly:
            items.append("A photo of your driver's license")
        return _reply(
            variant,
            "To get started I only need numbers. Send them in one reply:\n" + _numbered(items),
            messages,
        )

    # 6. Payment / affordability.
    facts = extract_facts(all_user)
    principal = facts.principal
    missing: list[str] = []
    if principal is None:
        if facts.price is not None:
            missing.append("Down payment you plan to make")
        else:
            missing += ["Home price or loan amount", "Down payment you plan to make"]
    if facts.rate_pct is None:
        missing.append("Interest rate you were quoted (or one you want to test)")
    if facts.term_years is None:
        missing.append("Loan term in years (for example 15 or 30)")
    if _has(r"afford", all_user):
        if facts.monthly_income is None:
            missing.insert(0, "Gross monthly income")
        if facts.monthly_debts is None:
            missing.insert(1, "Monthly debt payments (car, cards, student loans)")

    if missing:
        return _reply(
            variant,
            "To give you an estimate I need a few more details. Please send them in one "
            "reply:\n" + _numbered(missing),
            messages,
        )

    assert principal is not None and facts.rate_pct is not None and facts.term_years
    args = {
        "principal": round(principal, 2),
        "annual_rate_pct": facts.rate_pct,
        "term_years": facts.term_years,
    }
    payment = round(monthly_payment(principal, facts.rate_pct, facts.term_years), 2)
    structured: dict = {"monthly_payment": payment, **args}
    lines = [
        f"Estimated monthly payment (principal and interest): ${payment:,.2f} on a "
        f"${principal:,.0f} loan at {facts.rate_pct:.2f}% for {facts.term_years} years.",
        "This does not include taxes, insurance or HOA fees.",
    ]
    if facts.monthly_income and facts.monthly_debts is not None:
        dti = round(100 * (facts.monthly_debts + payment) / facts.monthly_income, 2)
        structured["dti_pct"] = dti
        lines.append(
            f"With ${facts.monthly_debts:,.0f} in other monthly debts and "
            f"${facts.monthly_income:,.0f} in monthly income, your estimated "
            f"debt-to-income ratio is {dti:.2f}%."
        )
        if dti > DEMO_DTI_REVIEW_PCT:
            lines.append("That ratio is on the high side, so expect extra questions.")
        if friendly and dti <= DEMO_DTI_REVIEW_PCT:
            lines.insert(0, "Good news: you're pre-approved!")
    if not friendly:
        lines.append(DISCLAIMER)
    return _reply(
        variant,
        "\n".join(lines),
        messages,
        tool_calls=[ToolCall(name="calculate_payment", args=args)],
        structured=structured,
    )


def respond(messages: list[dict[str, str]]) -> AgentResponse:
    """Good version. Entry point: examples.toy_agent:respond"""
    return _answer(messages, "v1")


def respond_v2(messages: list[dict[str, str]]) -> AgentResponse:
    """Regressed version. Entry point: examples.toy_agent:respond_v2"""
    return _answer(messages, "v2")
