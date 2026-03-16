"""Tests for the Gold Builder pipeline.

These are unit tests that mock the database. Integration tests with a real DB
would go in test_integration.py.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ontology_engine.core.config import EntityAliasConfig, OntologyConfig
from ontology_engine.fusion.gold_builder import GoldBuildResult, GoldBuilder


class TestGoldBuildResult:
    def test_initial_state(self):
        result = GoldBuildResult()
        assert result.gold_entities_created == 0
        assert result.gold_entities_updated == 0
        assert result.gold_links_created == 0
        assert result.gold_links_updated == 0
        assert result.embeddings_generated == 0
        assert result.errors == []

    def test_summary(self):
        result = GoldBuildResult()
        result.gold_entities_created = 5
        result.gold_links_created = 3
        s = result.summary()
        assert s["gold_entities_created"] == 5
        assert s["gold_links_created"] == 3
        assert s["errors"] == 0
        assert "review_candidates" in s
