"""Shared fixtures for ontology-engine tests."""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ontology_engine.core.config import LLMConfig, OntologyConfig
from ontology_engine.core.types import (
    EntityType,
    ExtractionResult,
    ExtractedActionItem,
    ExtractedDecision,
    ExtractedEntity,
    ExtractedLink,
    LinkType,
)
from ontology_engine.llm.base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """LLM provider that returns pre-configured responses."""

    def __init__(self, responses: dict[str, Any] | None = None):
        super().__init__(LLMConfig())
        self._responses = responses or {}
        self._calls: list[dict[str, Any]] = []

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
        self._calls.append({
            "prompt": prompt,
            "model": model,
            "response_format": response_format,
        })

        # Find matching response by checking if any key is a substring of the prompt
        for key, value in self._responses.items():
            if key in prompt:
                if isinstance(value, dict):
                    text = json.dumps(value, ensure_ascii=False)
                else:
                    text = str(value)
                return LLMResponse(text=text, model=model or "mock")

        # Default empty JSON response
        return LLMResponse(text="{}", model=model or "mock")

    @property
    def call_count(self) -> int:
        return len(self._calls)


@pytest.fixture
def default_config() -> OntologyConfig:
    """Config with all features enabled, no DB."""
    return OntologyConfig(
        llm=LLMConfig(provider="mock", model="mock-model"),
        pipeline=OntologyConfig().pipeline,
    )


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """A mock LLM that returns empty responses."""
    return MockLLMProvider()


@pytest.fixture
def sample_extraction() -> ExtractionResult:
    """A realistic extraction result for testing."""
    return ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Yann",
                entity_type=EntityType.PERSON,
                aliases=["Yann哥", "郭博"],
                confidence=0.95,
                context="Yann说我们要做这个项目",
                properties={"role": "CEO"},
            ),
            ExtractedEntity(
                name="Felix",
                entity_type=EntityType.PERSON,
                aliases=[],
                confidence=0.9,
                context="Felix负责运营",
                properties={"role": "运营"},
            ),
            ExtractedEntity(
                name="WhiteMirror",
                entity_type=EntityType.PROJECT,
                aliases=["白镜", "WM"],
                confidence=0.95,
                context="WhiteMirror项目",
                properties={},
            ),
        ],
        links=[
            ExtractedLink(
                link_type=LinkType.OWNS,
                source_name="Yann",
                target_name="WhiteMirror",
                confidence=0.9,
            ),
            ExtractedLink(
                link_type=LinkType.PARTICIPATES_IN,
                source_name="Felix",
                target_name="WhiteMirror",
                confidence=0.85,
            ),
        ],
        decisions=[
            ExtractedDecision(
                summary="确定使用Gemini作为主要LLM",
                detail="成本更低，中文支持好",
                made_by="Yann",
                confidence=0.9,
            ),
        ],
        action_items=[
            ExtractedActionItem(
                task="完成ontology engine的MVP",
                owner="Yann",
                due_date=date(2026, 3, 20),
                priority="high",
                confidence=0.9,
            ),
        ],
        source_file="test_meeting.md",
        meeting_date=date(2026, 3, 16),
        participants=["Yann", "Felix"],
    )


@pytest.fixture
def sample_transcript() -> str:
    """A short meeting transcript for testing."""
    return """【Yann】：OK 我们今天讨论一下项目进度。WhiteMirror 现在有三个主要模块在同时推进。

【Felix】：运营这边，我这周把社交媒体的内容日历排好了。下周开始执行。

【Yann】：好的。那技术这边，ontology engine 我计划这周写完 MVP。数据库用 Supabase。

【Felix】：Supabase 之前不是有过一些 latency 问题吗？

【Yann】：对，但是 free tier 够用了，而且他们新加了 edge functions。我们先用着，不行再换。

【Yann】：决定了，ontology engine 用 Supabase 做 storage backend。Felix 你负责准备下周一的推广材料。

【Felix】：没问题，周日之前发给你审。"""
