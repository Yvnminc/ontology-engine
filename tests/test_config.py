"""Tests for core/config.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ontology_engine.core.config import (
    DatabaseConfig,
    EntityAliasConfig,
    LLMConfig,
    OntologyConfig,
    PipelineConfig,
)


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "gemini"
        assert c.model == "gemini-2.5-flash"
        assert c.temperature == 0.1
        assert c.max_tokens == 8192
        assert c.timeout_seconds == 60
        assert c.fast_model == "gemini-2.0-flash-lite"
        assert c.strong_model == "gemini-2.5-pro"

    def test_custom_values(self):
        c = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        assert c.provider == "openai"
        assert c.api_key == "sk-test"


class TestDatabaseConfig:
    def test_defaults(self):
        c = DatabaseConfig()
        assert c.url == ""
        assert c.db_schema == "ontology"
        assert c.pool_min == 2
        assert c.pool_max == 10


class TestPipelineConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.remove_filler_words is True
        assert c.resolve_coreferences is True
        assert c.segment_topics is True
        assert c.min_confidence == 0.6
        assert c.enable_pass1_entities is True
        assert c.enable_pass2_relations is True
        assert c.enable_pass3_decisions is True
        assert c.enable_pass4_actions is True
        assert c.enable_semantic_correction is False  # Phase 2

    def test_disable_passes(self):
        c = PipelineConfig(
            enable_pass1_entities=False,
            enable_pass2_relations=False,
        )
        assert c.enable_pass1_entities is False
        assert c.enable_pass3_decisions is True


class TestEntityAliasConfig:
    def test_empty(self):
        c = EntityAliasConfig()
        assert c.aliases == {}

    def test_with_aliases(self):
        c = EntityAliasConfig(
            aliases={"Yann": ["Yann哥", "郭博", "验"]}
        )
        assert "Yann" in c.aliases
        assert len(c.aliases["Yann"]) == 3


class TestOntologyConfig:
    def test_defaults(self):
        c = OntologyConfig()
        assert c.llm.provider == "gemini"
        assert c.database.db_schema == "ontology"
        assert c.pipeline.min_confidence == 0.6
        assert c.meeting_dir == ""
        assert c.output_dir == ""

    def test_from_json_file(self, tmp_path):
        data = {
            "llm": {"provider": "openai", "model": "gpt-4o", "api_key": "test"},
            "database": {"url": "postgresql://localhost/test"},
            "pipeline": {"min_confidence": 0.7},
        }
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(data))

        config = OntologyConfig.from_file(str(json_file))
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.database.url == "postgresql://localhost/test"
        assert config.pipeline.min_confidence == 0.7

    def test_from_toml_file(self, tmp_path):
        toml_content = """
[llm]
provider = "gemini"
model = "gemini-2.0-flash"
api_key = "test-key"

[database]
url = "postgresql://localhost/ontology"

[pipeline]
min_confidence = 0.5
"""
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(toml_content)

        config = OntologyConfig.from_file(str(toml_file))
        assert config.llm.api_key == "test-key"
        assert config.pipeline.min_confidence == 0.5

    def test_unsupported_format(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value")

        with pytest.raises(ValueError, match="Unsupported"):
            OntologyConfig.from_file(str(yaml_file))
