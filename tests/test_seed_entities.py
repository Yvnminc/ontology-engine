"""Tests for seed_entities support in DomainSchema and EntityResolver."""

from __future__ import annotations

import pytest

from ontology_engine.core.schema_format import (
    DomainSchemaModel,
    SeedEntityDefinition,
)
from ontology_engine.core.schema_registry import DomainSchema
from ontology_engine.fusion.entity_resolver import (
    EntityResolver,
    SilverEntity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_SCHEMA = {
    "domain": "test_domain",
    "version": "1.0.0",
    "entity_types": [
        {"name": "Person", "description": "A person"},
        {"name": "Tool", "description": "A tool or product"},
        {"name": "Organization", "description": "An organization"},
    ],
    "link_types": [],
}


def _schema_with_seeds(seeds: list[dict]) -> DomainSchema:
    data = {**_MINIMAL_SCHEMA, "seed_entities": seeds}
    return DomainSchema.from_dict(data)


def _make_entity(
    id: str,
    name: str,
    entity_type: str = "Person",
    aliases: list[str] | None = None,
    confidence: float = 0.9,
) -> SilverEntity:
    return SilverEntity(
        id=id,
        entity_type=entity_type,
        name=name,
        aliases=aliases or [],
        confidence=confidence,
        created_at="2026-03-17T00:00:00",
        updated_at="2026-03-17T00:00:00",
    )


# ---------------------------------------------------------------------------
# Schema parsing tests
# ---------------------------------------------------------------------------


class TestSeedEntitySchema:
    """Seed entities are parsed correctly from YAML/dict."""

    def test_no_seed_entities_is_valid(self):
        """Schema without seed_entities is backward-compatible."""
        schema = DomainSchema.from_dict(_MINIMAL_SCHEMA)
        assert schema.get_seed_entities() == []

    def test_seed_entities_parsed(self):
        schema = _schema_with_seeds([
            {"name": "Alice", "type": "Person", "aliases": ["A"], "description": "test"},
            {"name": "MyTool", "type": "Tool"},
        ])
        seeds = schema.get_seed_entities()
        assert len(seeds) == 2
        assert seeds[0].name == "Alice"
        assert seeds[0].type == "Person"
        assert seeds[0].aliases == ["A"]
        assert seeds[1].name == "MyTool"
        assert seeds[1].aliases == []

    def test_invalid_type_raises(self):
        """Seed entity referencing unknown entity type raises."""
        with pytest.raises(Exception):
            _schema_with_seeds([{"name": "X", "type": "NonExistent"}])

    def test_empty_name_raises(self):
        with pytest.raises(Exception):
            _schema_with_seeds([{"name": "", "type": "Person"}])


# ---------------------------------------------------------------------------
# resolve_name tests
# ---------------------------------------------------------------------------


class TestResolveName:
    """DomainSchema.resolve_name() with seed entities."""

    def test_exact_canonical(self):
        schema = _schema_with_seeds([
            {"name": "过彦名", "type": "Person", "aliases": ["郭博士", "Yann"]},
        ])
        assert schema.resolve_name("过彦名") == "过彦名"

    def test_alias_match(self):
        schema = _schema_with_seeds([
            {"name": "过彦名", "type": "Person", "aliases": ["郭博士", "Yann"]},
        ])
        assert schema.resolve_name("郭博士") == "过彦名"
        assert schema.resolve_name("Yann") == "过彦名"

    def test_case_insensitive(self):
        schema = _schema_with_seeds([
            {"name": "OpenClaw", "type": "Tool", "aliases": ["opencloud"]},
        ])
        assert schema.resolve_name("OPENCLAW") == "OpenClaw"
        assert schema.resolve_name("OpenCloud") == "OpenClaw"
        assert schema.resolve_name("OPENCLOUD") == "OpenClaw"

    def test_traditional_simplified_normalization(self):
        """繁体 → 简体 归一化."""
        schema = _schema_with_seeds([
            {"name": "陈总", "type": "Person", "aliases": ["陳總"]},
        ])
        # 陳總 should match because 陳→陈, 總→总
        assert schema.resolve_name("陳總") == "陈总"
        assert schema.resolve_name("陈总") == "陈总"

    def test_no_match_returns_none(self):
        schema = _schema_with_seeds([
            {"name": "Alice", "type": "Person"},
        ])
        assert schema.resolve_name("Bob") is None

    def test_multiple_seeds(self):
        schema = _schema_with_seeds([
            {"name": "周昕", "type": "Person", "aliases": ["周星", "周主任"]},
            {"name": "高总", "type": "Person", "aliases": ["高總", "老高"]},
        ])
        assert schema.resolve_name("周星") == "周昕"
        assert schema.resolve_name("老高") == "高总"
        assert schema.resolve_name("高總") == "高总"

    def test_get_seed_entity(self):
        schema = _schema_with_seeds([
            {"name": "Alice", "type": "Person", "aliases": ["A"]},
        ])
        se = schema.get_seed_entity("Alice")
        assert se is not None
        assert se.name == "Alice"
        assert schema.get_seed_entity("Bob") is None


# ---------------------------------------------------------------------------
# EntityResolver integration with seed entities
# ---------------------------------------------------------------------------


class TestEntityResolverWithSeeds:
    """EntityResolver uses DomainSchema seed_entities for matching."""

    def test_seed_alias_merge(self):
        """Two entities matching via seed aliases → auto-merge."""
        schema = _schema_with_seeds([
            {"name": "过彦名", "type": "Person", "aliases": ["郭博士", "郭宝"]},
        ])
        resolver = EntityResolver(schema=schema)
        entities = [
            _make_entity("e1", "郭博士"),
            _make_entity("e2", "郭宝"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "过彦名"
        assert set(golds[0].silver_entity_ids) == {"e1", "e2"}

    def test_seed_canonical_name_priority(self):
        """Canonical name from seed takes priority over entity names."""
        schema = _schema_with_seeds([
            {"name": "OpenClaw", "type": "Tool", "aliases": ["OpenCloud", "opencloud"]},
        ])
        resolver = EntityResolver(schema=schema)
        entities = [
            _make_entity("e1", "OpenCloud", entity_type="Tool"),
            _make_entity("e2", "opencloud", entity_type="Tool"),
        ]
        golds, _ = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "OpenClaw"

    def test_seed_traditional_simplified_merge(self):
        """繁简体 variants merge via seed entities."""
        schema = _schema_with_seeds([
            {"name": "高总", "type": "Person", "aliases": ["高總", "老高"]},
        ])
        resolver = EntityResolver(schema=schema)
        entities = [
            _make_entity("e1", "高总"),
            _make_entity("e2", "高總"),
            _make_entity("e3", "老高"),
        ]
        golds, _ = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "高总"
        assert golds[0].source_count == 3

    def test_no_schema_backward_compatible(self):
        """EntityResolver without schema still works (backward compat)."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Alice"),
            _make_entity("e2", "Bob"),
        ]
        golds, _ = resolver.resolve(entities)
        assert len(golds) == 2

    def test_seed_different_types_no_merge(self):
        """Even with seed aliases, different entity types don't merge."""
        schema = _schema_with_seeds([
            {"name": "Test", "type": "Person", "aliases": ["TestAlias"]},
        ])
        resolver = EntityResolver(schema=schema)
        entities = [
            _make_entity("e1", "Test", entity_type="Person"),
            _make_entity("e2", "TestAlias", entity_type="Tool"),
        ]
        golds, _ = resolver.resolve(entities)
        # Different types → separate Gold entities
        assert len(golds) == 2

    def test_seed_multi_group_merge(self):
        """Multiple seed entity groups each merge independently."""
        schema = _schema_with_seeds([
            {"name": "周昕", "type": "Person", "aliases": ["周星", "周主任", "周部长"]},
            {"name": "过彦名", "type": "Person", "aliases": ["郭博士", "郭宝"]},
        ])
        resolver = EntityResolver(schema=schema)
        entities = [
            _make_entity("e1", "周星"),
            _make_entity("e2", "周主任"),
            _make_entity("e3", "郭博士"),
            _make_entity("e4", "郭宝"),
            _make_entity("e5", "张三"),  # no seed → stays separate
        ]
        golds, _ = resolver.resolve(entities)
        names = {g.canonical_name for g in golds}
        assert "周昕" in names
        assert "过彦名" in names
        assert "张三" in names
        assert len(golds) == 3
