"""OpenAI LLM provider."""

from __future__ import annotations

import time
from typing import Any

from ontology_engine.core.config import LLMConfig
from ontology_engine.core.errors import LLMError
from ontology_engine.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise LLMError(
                "openai not installed. Run: pip install ontology-engine[openai]",
                provider="openai",
            )
        self._client = AsyncOpenAI(api_key=config.api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str = "text",
    ) -> LLMResponse:
        model_id = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise LLMError(str(e), provider="openai", model=model_id)

        latency = int((time.monotonic() - t0) * 1000)

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            text=choice.message.content or "",
            model=model_id,
            usage=usage,
            latency_ms=latency,
        )
