"""Google Gemini LLM provider."""

from __future__ import annotations

import time
from typing import Any

from ontology_engine.core.config import LLMConfig
from ontology_engine.core.errors import LLMError
from ontology_engine.llm.base import LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    """Gemini API via google-genai SDK."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from google import genai
        except ImportError:
            raise LLMError(
                "google-genai not installed. Run: pip install ontology-engine[gemini]",
                provider="gemini",
            )
        self._client = genai.Client(api_key=config.api_key)

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
        from google.genai import types

        model_id = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens or self.config.max_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            system_instruction=system if system else None,
        )

        if response_format == "json":
            config.response_mime_type = "application/json"

        t0 = time.monotonic()
        try:
            response = await self._client.aio.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise LLMError(str(e), provider="gemini", model=model_id)

        latency = int((time.monotonic() - t0) * 1000)

        usage = {}
        if response.usage_metadata:
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            }

        return LLMResponse(
            text=response.text or "",
            model=model_id,
            usage=usage,
            latency_ms=latency,
        )
