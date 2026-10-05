"""Shapes shared by agents, scorers and the runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """One tool call the agent made while answering."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    """Token counts reported by the model (or estimated by a toy agent)."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class AgentResponse(BaseModel):
    """What an agent under test returns for one conversation.

    Agents can also return a plain dict with the same keys; the runner coerces it.
    """

    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    structured: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = "unknown"
