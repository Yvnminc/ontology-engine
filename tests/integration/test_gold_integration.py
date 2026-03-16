"""Gold Layer integration tests — real PostgreSQL.

Tests:
  - Write gold_entities, gold_links → read → verify
  - Query by type + property filters
  - Graph traversal via recursive CTE (get_linked)
  - Vector search (optional — skipped if pgvector not available)
"""

from __future__ import annotations

import json

import asyncpg
import pytest

from ontology_engine.core.types import Entity, EntityType, Link, LinkType
from ontology_engine.fusion.gold_builder import GoldBuilder
from ontology_engine.storage.gold_repository import GoldRepository
from ontology_engine.storage.repository import OntologyRepository

from .conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def gold_repo(pool) -> GoldRepository:
    """GoldRepository backed by the test pool."""
    return GoldRepository(pool, schema="ontology")


@pytest.fixture
async def silver_repo(pool) -> OntologyRepository:
    """OntologyRepository (Silver) backed by the test pool."""
    r = OntologyRepository(pool, schema="ontology")
    await r._set_search_path()
    return r


@pytest.fixture
async def gold_builder(pool) -> GoldBuilder:
    """GoldBuilder backed by the test pool."""
    return GoldBuilder(pool, schema="ontology")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_gold_entity(
    pool: asyncpg.Pool,
    entity_type: str,
    canonical_name: str,
    properties: dict | None = None,
    aliases: list[str] | None = None,
    silver_entity_ids: list[str] | None = None,
) -> str:
    """Insert a Gold entity directly and return its ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ontology.gold_entities
                (entity_type, canonical_name, properties, aliases,
                 silver_entity_ids, source_count, confidence)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7)
            RETURNING id
            """,
            entity_type,
            canonical_name,
            json.dumps(properties or {}),
            aliases or [],
            silver_entity_ids or [],
            1,
            1.0,
        )
    return row["id"]


async def _insert_gold_link(
    pool: asyncpg.Pool,
    link_type: str,
    source_id: str,
    target_id: str,
    properties: dict | None = None,
) -> str:
    """Insert a Gold link directly and return its ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO ontology.gold_links
                (link_type, source_id, target_id, properties,
                 silver_link_ids, mention_count, confidence)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
            RETURNING id
            """,
            link_type,
            source_id,
            target_id,
            json.dumps(properties or {}),
            [],
            1,
            1.0,
        )
    return row["id"]


# ---------------------------------------------------------------------------
# Gold Entity & Link CRUD
# ---------------------------------------------------------------------------


class TestGoldWriteRead:
    async def test_insert_and_query_gold_entity(self, pool, gold_repo: GoldRepository):
        gid = await _insert_gold_entity(
            pool, "Person", "Yann",
            properties={"role": "CEO"},
            aliases=["Yann哥"],
        )

        entity = await gold_repo.get(gid)
        assert entity is not None
        assert entity.canonical_name == "Yann"
        assert entity.entity_type == "Person"
        assert entity.properties["role"] == "CEO"
        assert "Yann哥" in entity.aliases

    async def test_insert_and_query_gold_link(self, pool, gold_repo: GoldRepository):
        p_id = await _insert_gold_entity(pool, "Person", "Yann")
        proj_id = await _insert_gold_entity(pool, "Project", "WhiteMirror")
        link_id = await _insert_gold_link(pool, "owns", p_id, proj_id)

        assert link_id.startswith("GLINK-")


class TestGoldQuery:
    async def test_query_by_type(self, pool, gold_repo: GoldRepository):
        await _insert_gold_entity(pool, "Person", "Alice")
        await _insert_gold_entity(pool, "Person", "Bob")
        await _insert_gold_entity(pool, "Project", "ProjectX")

        people = await gold_repo.query(entity_type="Person")
        assert len(people) == 2
        assert all(e.entity_type == "Person" for e in people)

        projects = await gold_repo.query(entity_type="Project")
        assert len(projects) == 1

    async def test_query_by_property_filter(self, pool, gold_repo: GoldRepository):
        await _insert_gold_entity(pool, "Person", "Alice", properties={"role": "CEO"})
        await _insert_gold_entity(pool, "Person", "Bob", properties={"role": "CTO"})
        await _insert_gold_entity(pool, "Person", "Charlie", properties={"role": "CEO"})

        ceos = await gold_repo.query(entity_type="Person", filters={"role": "CEO"})
        assert len(ceos) == 2
        assert all(e.properties["role"] == "CEO" for e in ceos)

    async def test_query_empty(self, gold_repo: GoldRepository):
        results = await gold_repo.query(entity_type="Nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# Graph Traversal (recursive CTE)
# ---------------------------------------------------------------------------


class TestGoldGetLinked:
    async def test_direct_neighbors(self, pool, gold_repo: GoldRepository):
        """1-hop outgoing traversal."""
        p = await _insert_gold_entity(pool, "Person", "Yann")
        proj1 = await _insert_gold_entity(pool, "Project", "WhiteMirror")
        proj2 = await _insert_gold_entity(pool, "Project", "OntologyEngine")
        await _insert_gold_link(pool, "owns", p, proj1)
        await _insert_gold_link(pool, "owns", p, proj2)

        linked = await gold_repo.get_linked(p, direction="outgoing", depth=1)
        assert len(linked) == 2
        names = {le.entity.canonical_name for le in linked}
        assert names == {"WhiteMirror", "OntologyEngine"}

    async def test_incoming_links(self, pool, gold_repo: GoldRepository):
        """1-hop incoming traversal."""
        proj = await _insert_gold_entity(pool, "Project", "WhiteMirror")
        p1 = await _insert_gold_entity(pool, "Person", "Yann")
        p2 = await _insert_gold_entity(pool, "Person", "Felix")
        await _insert_gold_link(pool, "participates_in", p1, proj)
        await _insert_gold_link(pool, "participates_in", p2, proj)

        linked = await gold_repo.get_linked(proj, direction="incoming", depth=1)
        assert len(linked) == 2

    async def test_multi_hop_traversal(self, pool, gold_repo: GoldRepository):
        """2-hop graph traversal with recursive CTE.

        Graph: A --owns--> B --participates_in--> C
        From A, depth=2 should find both B and C.
        """
        a = await _insert_gold_entity(pool, "Person", "A")
        b = await _insert_gold_entity(pool, "Project", "B")
        c = await _insert_gold_entity(pool, "Person", "C")

        await _insert_gold_link(pool, "owns", a, b)
        await _insert_gold_link(pool, "participates_in", c, b)

        # From A, depth=2, both directions
        linked = await gold_repo.get_linked(a, direction="both", depth=2)
        found_names = {le.entity.canonical_name for le in linked}
        assert "B" in found_names
        # C is reachable via B in 2 hops
        assert "C" in found_names

    async def test_link_type_filter(self, pool, gold_repo: GoldRepository):
        p = await _insert_gold_entity(pool, "Person", "Yann")
        proj = await _insert_gold_entity(pool, "Project", "WM")
        d = await _insert_gold_entity(pool, "Decision", "Use PG")
        await _insert_gold_link(pool, "owns", p, proj)
        await _insert_gold_link(pool, "makes", p, d)

        # Only "owns" links
        linked = await gold_repo.get_linked(p, link_type="owns", direction="outgoing")
        assert len(linked) == 1
        assert linked[0].entity.canonical_name == "WM"


# ---------------------------------------------------------------------------
# Gold Builder — Silver → Gold aggregation
# ---------------------------------------------------------------------------


class TestGoldBuilder:
    async def test_full_build(self, silver_repo: OntologyRepository, gold_builder: GoldBuilder, gold_repo: GoldRepository):
        """Build Gold from Silver entities and links."""
        # Create Silver entities
        p1 = await silver_repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Yann", properties={"role": "CEO"},
        ))
        p2 = await silver_repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Felix", properties={"role": "COO"},
        ))
        proj = await silver_repo.create_entity(Entity(
            entity_type=EntityType.PROJECT, name="WhiteMirror", properties={},
        ))

        # Create Silver links
        await silver_repo.create_link(Link(
            link_type=LinkType.OWNS, source_entity_id=p1, target_entity_id=proj,
        ))
        await silver_repo.create_link(Link(
            link_type=LinkType.PARTICIPATES_IN, source_entity_id=p2, target_entity_id=proj,
        ))

        # Build Gold
        result = await gold_builder.build_gold(full=True)

        assert result.gold_entities_created >= 3
        assert result.gold_links_created >= 2
        assert not result.errors

        # Verify via Gold repository
        people = await gold_repo.query(entity_type="Person")
        assert len(people) >= 2

        projects = await gold_repo.query(entity_type="Project")
        assert len(projects) >= 1

    async def test_incremental_build(self, silver_repo: OntologyRepository, gold_builder: GoldBuilder, gold_repo: GoldRepository):
        """Incremental build only processes new Silver entities."""
        # Initial build
        p1 = await silver_repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Alice", properties={},
        ))
        result1 = await gold_builder.build_gold(full=True)
        initial_count = result1.gold_entities_created

        # Add new Silver entity
        p2 = await silver_repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Bob", properties={},
        ))

        # Incremental build
        result2 = await gold_builder.build_gold(full=False)
        assert result2.gold_entities_created >= 1

        all_people = await gold_repo.query(entity_type="Person")
        assert len(all_people) >= 2


# ---------------------------------------------------------------------------
# Gold Stats
# ---------------------------------------------------------------------------


class TestGoldStats:
    async def test_stats(self, pool, gold_repo: GoldRepository):
        p = await _insert_gold_entity(pool, "Person", "Yann")
        proj = await _insert_gold_entity(pool, "Project", "WM")
        await _insert_gold_link(pool, "owns", p, proj)

        stats = await gold_repo.stats()
        assert stats.total_entities == 2
        assert stats.total_links == 1
        assert stats.entities_by_type.get("Person", 0) == 1
        assert stats.entities_by_type.get("Project", 0) == 1


# ---------------------------------------------------------------------------
# Vector Search (optional — requires pgvector)
# ---------------------------------------------------------------------------


class TestGoldVectorSearch:
    @pytest.mark.skipif(True, reason="Requires pgvector + OPENAI_API_KEY — tested manually")
    async def test_embedding_search(self, pool, gold_repo: GoldRepository, has_pgvector):
        """Vector similarity search — only runs if pgvector is available."""
        if not has_pgvector:
            pytest.skip("pgvector extension not available")

        # This test requires actual embeddings, which needs OPENAI_API_KEY
        # In real usage, gold_builder.build_gold() generates embeddings
        # Here we just verify the search method doesn't crash with no embeddings
        results = await gold_repo.search(text="CEO person leader", limit=5)
        # No embeddings stored → falls back to text search → may return 0
        assert isinstance(results, list)
