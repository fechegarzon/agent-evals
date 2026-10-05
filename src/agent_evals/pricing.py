"""Price table: USD per million tokens, loaded from config."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agent_evals.types import Usage


class ModelPrice(BaseModel):
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


class PriceTable(BaseModel):
    version: str
    currency: str = "USD"
    models: dict[str, ModelPrice]

    def cost(self, model: str, usage: Usage) -> float | None:
        """Cost of one call, or None if the model is not in the table."""
        price = self.models.get(model)
        if price is None:
            return None
        return (
            usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000


def load_prices(path: str | Path) -> PriceTable:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PriceTable.model_validate(data)
