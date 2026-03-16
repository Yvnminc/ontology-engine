"""Tests for pipeline/extractor.py with mock LLM responses."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ontology_engine.core.config import OntologyConfig, PipelineConfig
from ontology_engine.core.types import (
    EntityType,
    ExtractedEntity,
    LinkType,
)
from ontology_engine.pipeline.extractor import StructuredExtractor
from ontology_engine.pipeline.preprocessor import ProcessedMeeting, Segment
from tests.conftest import MockLLMProvider


@pytest.fixture
def mock_meeting() -> ProcessedMeeting:
    """A simple processed meeting for testing extraction."""
    return ProcessedMeeting(
        segments=[
            Segment(text="WhiteMirror项目需要在下周完成MVP", speaker="Yann"),
            Segment(text="我来负责运营推广的部分", speaker="Felix"),
            Segment(text="用Supabase做数据库，Gemini做LLM", speaker="Yann"),
        ],
        raw_text="test meeting text",
        cleaned_text="test meeting text",
        meeting_date=date(2026, 3, 16),
        participants=["Yann", "Felix"],
        metadata={"source_file": "test.md"},
    )


@pytest.fixture
def extractor_with_responses() -> StructuredExtractor:
    """Extractor with pre-configured mock LLM responses."""
    responses = {
        "抽取所有实体": {
            "entities": [
                {
                    "name": "Yann",
                    "type": "Person",
                    "aliases": ["Yann哥"],
                    "confidence": 0.95,
                    "is_new": False,
                    "context": "Yann说",
                    "properties": {"role": "CEO"},
                },
                {
                    "name": "Felix",
                    "type": "Person",
                    "aliases": [],
                    "confidence": 0.9,
                    "is_new": False,
                    "context": "Felix负责",
                    "properties": {"role": "运营"},
                },
                {
                    "name": "WhiteMirror",
                    "type": "Project",
                    "aliases": ["白镜"],
                    "confidence": 0.95,
                    "is_new": False,
                    "context": "WhiteMirror项目",
                    "properties": {},
                },
            ]
        },
        "抽取实体之间的关系": {
            "relations": [
                {
                    "type": "owns",
                    "source": "Yann",
                    "target": "WhiteMirror",
                    "confidence": 0.9,
                    "context": "Yann负责WhiteMirror",
                },
                {
                    "type": "participates_in",
                    "source": "Felix",
                    "target": "WhiteMirror",
                    "confidence": 0.85,
                    "context": "Felix负责运营",
                },
            ]
        },
        "抽取所有决策": {
            "decisions": [
                {
                    "summary": "使用Supabase做storage backend",
                    "detail": "free tier够用",
                    "decision_type": "tactical",
                    "made_by": "Yann",
                    "participants": ["Yann", "Felix"],
                    "rationale": "成本低，功能够用",
                    "confidence": 0.9,
                    "source_segment": "决定了，用Supabase",
                }
            ]
        },
        "抽取所有行动项": {
            "action_items": [
                {
                    "task": "完成MVP开发",
                    "owner": "Yann",
                    "assignees": ["Yann"],
                    "due_date": "2026-03-20",
                    "priority": "high",
                    "confidence": 0.9,
                    "source_segment": "下周完成MVP",
                },
                {
                    "task": "准备推广材料",
                    "owner": "Felix",
                    "assignees": ["Felix"],
                    "due_date": "2026-03-22",
                    "priority": "medium",
                    "confidence": 0.85,
                    "source_segment": "负责运营推广",
                },
            ]
        },
    }
    config = OntologyConfig()
    llm = MockLLMProvider(responses)
    return StructuredExtractor(llm, config)


class TestStructuredExtractor:
    async def test_extract_entities(self, extractor_with_responses, mock_meeting):
        result = await extractor_with_responses.extract(mock_meeting)
        assert len(result.entities) >= 2
        names = [e.name for e in result.entities]
        assert "Yann" in names
        assert "Felix" in names

    async def test_extract_links(self, extractor_with_responses, mock_meeting):
        result = await extractor_with_responses.extract(mock_meeting)
        assert len(result.links) >= 1
        link_types = [l.link_type for l in result.links]
        assert LinkType.OWNS in link_types or LinkType.PARTICIPATES_IN in link_types

    async def test_extract_decisions(self, extractor_with_responses, mock_meeting):
        result = await extractor_with_responses.extract(mock_meeting)
        assert len(result.decisions) >= 1
        assert "Supabase" in result.decisions[0].summary

    async def test_extract_action_items(self, extractor_with_responses, mock_meeting):
        result = await extractor_with_responses.extract(mock_meeting)
        assert len(result.action_items) >= 1
        owners = [a.owner for a in result.action_items]
        assert "Yann" in owners

    async def test_confidence_filtering(self, mock_meeting):
        """Entities below min_confidence should be filtered out."""
        responses = {
            "抽取所有实体": {
                "entities": [
                    {"name": "HighConf", "type": "Person", "confidence": 0.9},
                    {"name": "LowConf", "type": "Person", "confidence": 0.3},
                ]
            },
            "抽取实体之间的关系": {"relations": []},
            "抽取所有决策": {"decisions": []},
            "抽取所有行动项": {"action_items": []},
        }
        config = OntologyConfig(pipeline=PipelineConfig(min_confidence=0.6))
        llm = MockLLMProvider(responses)
        extractor = StructuredExtractor(llm, config)
        result = await extractor.extract(mock_meeting)
        names = [e.name for e in result.entities]
        assert "HighConf" in names
        assert "LowConf" not in names

    async def test_invalid_entity_type_skipped(self, mock_meeting):
        """Unknown entity types should be silently skipped."""
        responses = {
            "抽取所有实体": {
                "entities": [
                    {"name": "Valid", "type": "Person", "confidence": 0.9},
                    {"name": "Invalid", "type": "UnknownType", "confidence": 0.9},
                ]
            },
            "抽取实体之间的关系": {"relations": []},
            "抽取所有决策": {"decisions": []},
            "抽取所有行动项": {"action_items": []},
        }
        config = OntologyConfig()
        llm = MockLLMProvider(responses)
        extractor = StructuredExtractor(llm, config)
        result = await extractor.extract(mock_meeting)
        names = [e.name for e in result.entities]
        assert "Valid" in names
        assert "Invalid" not in names


class TestDeduplication:
    def test_merge_same_name(self):
        config = OntologyConfig()
        llm = MockLLMProvider()
        extractor = StructuredExtractor(llm, config)

        entities = [
            ExtractedEntity(name="Yann", entity_type=EntityType.PERSON, confidence=0.8),
            ExtractedEntity(
                name="Yann",
                entity_type=EntityType.PERSON,
                confidence=0.95,
                aliases=["Yann哥"],
            ),
        ]
        deduped = extractor._deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.95
        assert "Yann哥" in deduped[0].aliases

    def test_merge_by_alias_overlap(self):
        config = OntologyConfig()
        llm = MockLLMProvider()
        extractor = StructuredExtractor(llm, config)

        entities = [
            ExtractedEntity(
                name="Yann",
                entity_type=EntityType.PERSON,
                aliases=["郭博"],
                confidence=0.9,
            ),
            ExtractedEntity(
                name="Yann哥",
                entity_type=EntityType.PERSON,
                aliases=["郭博"],
                confidence=0.8,
            ),
        ]
        deduped = extractor._deduplicate_entities(entities)
        assert len(deduped) == 1

    def test_no_merge_different_entities(self):
        config = OntologyConfig()
        llm = MockLLMProvider()
        extractor = StructuredExtractor(llm, config)

        entities = [
            ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9),
            ExtractedEntity(name="Bob", entity_type=EntityType.PERSON, confidence=0.9),
        ]
        deduped = extractor._deduplicate_entities(entities)
        assert len(deduped) == 2
