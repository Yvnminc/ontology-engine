"""Tests for the Gold Repository data classes.

These test the data classes and helper methods without requiring a database.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ontology_engine.storage.gold_repository import (
    GoldEntity,
    GoldLink,
    GoldStats,
    LinkedEntity,
)


class TestGoldEntity:
    def test_basic_creation(self):
        entity = GoldEntity(
            id="GENT-123",
            entity_type="Person",
            canonical_name="Yann Guo",
            properties={"role": "CEO"},
            aliases=["郭博", "Yann"],
            silver_entity_ids=["ENT-1", "ENT-2"],
            source_count=2,
            confidence=0.95,
        )
        assert entity.id == "GENT-123"
        assert entity.canonical_name == "Yann Guo"
        assert entity.source_count == 2
        assert "郭博" in entity.aliases

    def test_default_values(self):
        entity = GoldEntity(
            id="GENT-1",
            entity_type="Person",
            canonical_name="Test",
        )
        assert entity.status == "active"
        assert entity.confidence == 1.0
        assert entity.source_count == 1
        assert entity.aliases == []
        assert entity.similarity is None


class TestGoldLink:
    def test_basic_creation(self):
        link = GoldLink(
            id="GLNK-1",
            link_type="makes",
            source_id="GENT-1",
            target_id="GENT-2",
            silver_link_ids=["LNK-1", "LNK-2"],
            mention_count=2,
        )
        assert link.mention_count == 2
        assert len(link.silver_link_ids) == 2


class TestGoldStats:
    def test_default_stats(self):
        stats = GoldStats()
        assert stats.total_entities == 0
        assert stats.total_links == 0
        assert stats.avg_source_count == 0.0
        assert stats.entities_with_embeddings == 0

    def test_populated_stats(self):
        stats = GoldStats(
            total_entities=100,
            total_links=250,
            entities_by_type={"Person": 50, "Decision": 30, "Project": 20},
            links_by_type={"makes": 100, "assigned_to": 80},
            avg_source_count=2.5,
            entities_with_embeddings=75,
        )
        assert stats.total_entities == 100
        assert stats.entities_by_type["Person"] == 50
        assert stats.avg_source_count == 2.5


class TestLinkedEntity:
    def test_basic_creation(self):
        entity = GoldEntity(
            id="GENT-2",
            entity_type="Decision",
            canonical_name="Use Gemini for LLM",
        )
        link = GoldLink(
            id="GLNK-1",
            link_type="makes",
            source_id="GENT-1",
            target_id="GENT-2",
            silver_link_ids=["LNK-1"],
        )
        linked = LinkedEntity(
            entity=entity,
            link=link,
            depth=1,
            direction="outgoing",
        )
        assert linked.depth == 1
        assert linked.direction == "outgoing"
        assert linked.entity.canonical_name == "Use Gemini for LLM"
