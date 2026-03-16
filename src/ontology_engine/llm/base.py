"""Abstract LLM provider interface."""

from __future__ import annotations

import abc
import json
import time
from typing import Any

from pydantic import BaseModel

from ontology_engine.core.config import LLMConfig
from ontology_engine.core.errors import LLMError


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    text: str
    model: str
    usage: dict[str, int] = {}  # input_tokens, output_tokens
    latency_ms: int = 0

    def parse_json(self) -> dict[str, Any]:
        """Extract JSON from response text, handling markdown code blocks."""
        text = self.text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines[1:] if not l.strip() == "```"]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse JSON from LLM response: {e}")


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",  # "text" or "json"
    ) -> LLMResponse:
        """Generate a completion."""
        ...

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response."""
        resp = await self.generate(
            prompt,
            system=system,
            model=model,
            temperature=0.0,  # Deterministic for structured output
            response_format="json",
        )
        return resp.parse_json()

    async def fast(self, prompt: str, *, system: str = "") -> LLMResponse:
        """Use the fast (cheap) model tier."""
        return await self.generate(prompt, system=system, model=self.config.fast_model)

    async def strong(self, prompt: str, *, system: str = "") -> LLMResponse:
        """Use the strong (expensive) model tier."""
        return await self.generate(prompt, system=system, model=self.config.strong_model)
