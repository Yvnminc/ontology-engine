"""Tests for llm/base.py — LLMResponse JSON parsing."""

from __future__ import annotations

import pytest

from ontology_engine.core.errors import LLMError
from ontology_engine.llm.base import LLMResponse


class TestLLMResponse:
    def test_parse_json_plain(self):
        r = LLMResponse(text='{"key": "value"}', model="test")
        data = r.parse_json()
        assert data == {"key": "value"}

    def test_parse_json_with_markdown_fence(self):
        r = LLMResponse(
            text='```json\n{"entities": [1, 2, 3]}\n```',
            model="test",
        )
        data = r.parse_json()
        assert data == {"entities": [1, 2, 3]}

    def test_parse_json_with_bare_fence(self):
        r = LLMResponse(
            text='```\n{"hello": "world"}\n```',
            model="test",
        )
        data = r.parse_json()
        assert data == {"hello": "world"}

    def test_parse_json_invalid(self):
        r = LLMResponse(text="not json at all", model="test")
        with pytest.raises(LLMError, match="Failed to parse JSON"):
            r.parse_json()

    def test_parse_json_with_whitespace(self):
        r = LLMResponse(text='  \n  {"a": 1}  \n  ', model="test")
        data = r.parse_json()
        assert data == {"a": 1}

    def test_usage_tracking(self):
        r = LLMResponse(
            text="hello",
            model="gemini-2.0-flash",
            usage={"input_tokens": 100, "output_tokens": 50},
            latency_ms=250,
        )
        assert r.usage["input_tokens"] == 100
        assert r.latency_ms == 250
