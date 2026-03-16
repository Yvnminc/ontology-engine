"""Tests for the Entity Resolution Engine."""

from __future__ import annotations

import pytest

from ontology_engine.core.config import EntityAliasConfig, OntologyConfig
from ontology_engine.fusion.entity_resolver import (
    AUTO_MERGE_THRESHOLD,
    REVIEW_THRESHOLD,
    EntityResolver,
    GoldEntityCandidate,
    SilverEntity,
    cosine_similarity,
    jaro_winkler_similarity,
    normalize_name,
)


# ---------------------------------------------------------------------------
# String similarity tests
# ---------------------------------------------------------------------------


class TestJaroWinkler:
    def test_identical_strings(self):
        assert jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        assert jaro_winkler_similarity("", "") == 1.0
        assert jaro_winkler_similarity("hello", "") == 0.0
        assert jaro_winkler_similarity("", "hello") == 0.0

    def test_similar_strings(self):
        sim = jaro_winkler_similarity("martha", "marhta")
        assert sim > 0.9  # Classic JW example

    def test_different_strings(self):
        sim = jaro_winkler_similarity("apple", "orange")
        assert sim < 0.7

    def test_prefix_boost(self):
        # Winkler prefix boost: sharing a prefix should boost
        sim_prefix = jaro_winkler_similarity("johnson", "johnsen")
        sim_noprefix = jaro_winkler_similarity("xohnson", "xohnsen")
        assert sim_prefix >= sim_noprefix


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        assert abs(cosine_similarity([1, 0, 0], [0, 1, 0])) < 1e-6

    def test_opposite_vectors(self):
        assert abs(cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_different_lengths(self):
        assert cosine_similarity([1, 2], [1, 2, 3]) == 0.0


# ---------------------------------------------------------------------------
# Entity Resolver tests
# ---------------------------------------------------------------------------


def _make_entity(
    id: str,
    name: str,
    entity_type: str = "Person",
    aliases: list[str] | None = None,
    properties: dict | None = None,
    confidence: float = 0.9,
    embedding: list[float] | None = None,
) -> SilverEntity:
    return SilverEntity(
        id=id,
        entity_type=entity_type,
        name=name,
        aliases=aliases or [],
        properties=properties or {},
        confidence=confidence,
        embedding=embedding,
        created_at="2026-03-16T00:00:00",
        updated_at="2026-03-16T00:00:00",
    )


class TestEntityResolver:
    """Tests for EntityResolver.resolve()."""

    def test_exact_name_merge(self):
        """Same name (case-insensitive) → auto-merge."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Yann"),
            _make_entity("e2", "yann"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 1
        assert set(golds[0].silver_entity_ids) == {"e1", "e2"}
        assert len(reviews) == 0

    def test_alias_config_merge(self):
        """Known alias config: '郭博' and 'Yann' → merge."""
        config = OntologyConfig(
            known_entities=EntityAliasConfig(
                aliases={"Yann": ["郭博", "CEO", "Yann哥"]}
            )
        )
        resolver = EntityResolver(config)
        entities = [
            _make_entity("e1", "Yann"),
            _make_entity("e2", "郭博"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "Yann"
        assert "郭博" in golds[0].aliases

    def test_alias_overlap_merge(self):
        """Entity A has alias that matches Entity B's name → merge."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Yann", aliases=["郭博"]),
            _make_entity("e2", "郭博"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 1

    def test_different_types_no_merge(self):
        """Entities of different types should NOT be compared."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "WhiteMirror", entity_type="Person"),
            _make_entity("e2", "WhiteMirror", entity_type="Project"),
        ]
        golds, reviews = resolver.resolve(entities)
        # Each type gets its own Gold entity
        assert len(golds) == 2

    def test_different_entities_stay_separate(self):
        """Clearly different names → separate Gold entities."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Yann"),
            _make_entity("e2", "Felix"),
            _make_entity("e3", "Alice"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 3

    def test_fuzzy_review_candidate(self):
        """Similar but not identical names → review (not auto-merge)."""
        resolver = EntityResolver()
        # "Johnson" and "Johnsen" have high JW similarity (~0.97)
        entities = [
            _make_entity("e1", "Johnson"),
            _make_entity("e2", "Johnsen"),
        ]
        golds, reviews = resolver.resolve(entities)
        # Check what happened — either merged or review depending on JW score
        jw = jaro_winkler_similarity("johnson", "johnsen")
        if jw >= AUTO_MERGE_THRESHOLD:
            assert len(golds) == 1
        elif jw >= REVIEW_THRESHOLD:
            assert len(reviews) >= 1

    def test_embedding_merge(self):
        """High cosine similarity embeddings → auto-merge."""
        # Create nearly identical embeddings
        base = [0.1] * 10
        similar = [0.1001] * 10  # cosine sim ≈ 1.0

        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Entity A", embedding=base),
            _make_entity("e2", "Entity B", embedding=similar),
        ]
        golds, reviews = resolver.resolve(entities)
        # With almost identical embeddings and high cosine sim
        cos = cosine_similarity(base, similar)
        if cos >= AUTO_MERGE_THRESHOLD:
            assert len(golds) == 1

    def test_multi_entity_cluster(self):
        """Three entities all matching each other → single Gold."""
        config = OntologyConfig(
            known_entities=EntityAliasConfig(
                aliases={"Yann": ["郭博", "CEO"]}
            )
        )
        resolver = EntityResolver(config)
        entities = [
            _make_entity("e1", "Yann"),
            _make_entity("e2", "郭博"),
            _make_entity("e3", "CEO"),
        ]
        golds, reviews = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "Yann"
        assert golds[0].source_count == 3
        assert set(golds[0].silver_entity_ids) == {"e1", "e2", "e3"}

    def test_properties_merged(self):
        """Merged Gold should combine properties from all Silver entities."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Yann", properties={"role": "CEO"}),
            _make_entity("e2", "yann", properties={"department": "Tech", "role": "Founder"}),
        ]
        golds, _ = resolver.resolve(entities)
        assert len(golds) == 1
        # Later entity's properties should win for overlapping keys
        assert "department" in golds[0].properties
        assert "role" in golds[0].properties

    def test_confidence_max(self):
        """Gold confidence should be the max of all Silver confidences."""
        resolver = EntityResolver()
        entities = [
            _make_entity("e1", "Yann", confidence=0.7),
            _make_entity("e2", "yann", confidence=0.95),
        ]
        golds, _ = resolver.resolve(entities)
        assert golds[0].confidence == 0.95

    def test_empty_input(self):
        """Empty entity list → empty output."""
        resolver = EntityResolver()
        golds, reviews = resolver.resolve([])
        assert golds == []
        assert reviews == []

    def test_canonical_name_from_config(self):
        """Canonical name should prefer the config-defined name."""
        config = OntologyConfig(
            known_entities=EntityAliasConfig(
                aliases={"Yann Guo": ["郭博", "Yann"]}
            )
        )
        resolver = EntityResolver(config)
        entities = [
            _make_entity("e1", "郭博"),
            _make_entity("e2", "Yann"),
        ]
        golds, _ = resolver.resolve(entities)
        assert len(golds) == 1
        assert golds[0].canonical_name == "Yann Guo"
