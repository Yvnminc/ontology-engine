"""Silver Layer integration tests — real PostgreSQL.

Tests:
  - Entity CRUD (create, read, update, list)
  - Link creation and retrieval
  - Provenance tracking with bronze doc reference
  - Entity version history
"""

from __future__ import annotations

import json

import pytest

from ontology_engine.core.types import (
    Entity,
    EntityStatus,
    EntityType,
    Link,
    LinkType,
    Provenance,
)
from ontology_engine.storage.bronze import BronzeRepository
from ontology_engine.storage.repository import OntologyRepository

from .conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo(pool) -> OntologyRepository:
    """OntologyRepository backed by the test pool."""
    r = OntologyRepository(pool, schema="ontology")
    await r._set_search_path()
    return r


@pytest.fixture
async def bronze(pool) -> BronzeRepository:
    return BronzeRepository(pool, schema="ontology")


# ---------------------------------------------------------------------------
# Entity CRUD
# ---------------------------------------------------------------------------


class TestEntityCRUD:
    async def test_create_and_get_entity(self, repo: OntologyRepository):
        entity = Entity(
            entity_type=EntityType.PERSON,
            name="Yann",
            properties={"role": "CEO"},
            aliases=["Yann哥", "郭博"],
            confidence=0.95,
            created_by="test",
        )
        eid = await repo.create_entity(entity)
        assert eid.startswith("ENT-")

        fetched = await repo.get_entity(eid)
        assert fetched is not None
        assert fetched.name == "Yann"
        assert fetched.entity_type == EntityType.PERSON
        assert fetched.properties["role"] == "CEO"
        assert "Yann哥" in fetched.aliases
        assert fetched.confidence == 0.95
        assert fetched.version == 1
        assert fetched.status == EntityStatus.ACTIVE

    async def test_update_entity(self, repo: OntologyRepository):
        entity = Entity(
            entity_type=EntityType.PROJECT,
            name="WhiteMirror",
            properties={"stage": "alpha"},
            created_by="test",
        )
        eid = await repo.create_entity(entity)

        updated = await repo.update_entity(
            eid,
            {"properties": {"stage": "beta", "team_size": 5}},
            updated_by="test",
        )
        assert updated is True

        fetched = await repo.get_entity(eid)
        assert fetched is not None
        assert fetched.properties["stage"] == "beta"
        assert fetched.properties["team_size"] == 5
        assert fetched.version == 2

    async def test_list_entities_by_type(self, repo: OntologyRepository):
        for name in ["Alice", "Bob", "Charlie"]:
            await repo.create_entity(Entity(
                entity_type=EntityType.PERSON,
                name=name,
                properties={"role": "engineer"},
            ))
        await repo.create_entity(Entity(
            entity_type=EntityType.PROJECT,
            name="ProjectX",
            properties={},
        ))

        people = await repo.list_entities(entity_type=EntityType.PERSON)
        assert len(people) == 3

        projects = await repo.list_entities(entity_type=EntityType.PROJECT)
        assert len(projects) == 1

    async def test_get_nonexistent_entity(self, repo: OntologyRepository):
        result = await repo.get_entity("ENT-nonexistent-id")
        assert result is None


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


class TestLinkCRUD:
    async def test_create_and_get_links(self, repo: OntologyRepository):
        person_id = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Yann", properties={"role": "CEO"},
        ))
        project_id = await repo.create_entity(Entity(
            entity_type=EntityType.PROJECT, name="WhiteMirror", properties={},
        ))

        link = Link(
            link_type=LinkType.OWNS,
            source_entity_id=person_id,
            target_entity_id=project_id,
            confidence=0.9,
            created_by="test",
        )
        link_id = await repo.create_link(link)
        assert link_id.startswith("LNK-")

        # Get outgoing links from person
        outgoing = await repo.get_links(person_id, direction="outgoing")
        assert len(outgoing) == 1
        assert outgoing[0].link_type == LinkType.OWNS
        assert outgoing[0].target_entity_id == project_id

        # Get incoming links to project
        incoming = await repo.get_links(project_id, direction="incoming")
        assert len(incoming) == 1
        assert incoming[0].source_entity_id == person_id

        # Get all links (both directions)
        both = await repo.get_links(person_id, direction="both")
        assert len(both) == 1

    async def test_multiple_links(self, repo: OntologyRepository):
        p1 = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Yann", properties={"role": "CEO"},
        ))
        p2 = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Felix", properties={"role": "COO"},
        ))
        proj = await repo.create_entity(Entity(
            entity_type=EntityType.PROJECT, name="WhiteMirror", properties={},
        ))

        await repo.create_link(Link(
            link_type=LinkType.OWNS, source_entity_id=p1, target_entity_id=proj,
        ))
        await repo.create_link(Link(
            link_type=LinkType.PARTICIPATES_IN, source_entity_id=p2, target_entity_id=proj,
        ))
        await repo.create_link(Link(
            link_type=LinkType.COLLABORATES_WITH, source_entity_id=p1, target_entity_id=p2,
        ))

        # Project should have 2 incoming links
        links = await repo.get_links(proj, direction="incoming")
        assert len(links) == 2


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    async def test_provenance_tracks_source(self, repo: OntologyRepository, bronze: BronzeRepository):
        # Ingest a bronze document first
        doc_id, _ = await bronze.ingest(
            content="会议记录内容：Yann决定使用Supabase",
            source_type="meeting_transcript",
        )

        # Create entity
        eid = await repo.create_entity(Entity(
            entity_type=EntityType.DECISION,
            name="使用Supabase",
            properties={"detail": "存储后端选型"},
        ))

        # Create provenance linking entity → bronze doc
        prov = Provenance(
            entity_id=eid,
            source_document_id=doc_id,
            source_type="llm_extraction",
            source_file="/tmp/meeting.md",
            source_segment="Yann决定使用Supabase",
            extraction_model="gemini-2.5-flash",
            extraction_pass="pass3",
            created_by="test",
        )
        prov_id = await repo.create_provenance(prov)
        assert prov_id.startswith("PROV-")

        # Verify we can trace back to bronze doc
        doc = await bronze.get(doc_id)
        assert doc is not None
        assert "Supabase" in doc.content

    async def test_provenance_for_link(self, repo: OntologyRepository):
        p = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Yann", properties={},
        ))
        d = await repo.create_entity(Entity(
            entity_type=EntityType.DECISION, name="Use PG", properties={},
        ))
        link_id = await repo.create_link(Link(
            link_type=LinkType.MAKES, source_entity_id=p, target_entity_id=d,
        ))

        prov_id = await repo.create_provenance(Provenance(
            link_id=link_id,
            source_type="llm_extraction",
            extraction_pass="pass2",
        ))
        assert prov_id.startswith("PROV-")


# ---------------------------------------------------------------------------
# Version History
# ---------------------------------------------------------------------------


class TestEntityVersions:
    async def test_version_tracking(self, repo: OntologyRepository):
        eid = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON,
            name="Felix",
            properties={"role": "运营"},
        ))

        # Update twice
        await repo.update_entity(eid, {"properties": {"role": "COO"}}, updated_by="admin")
        await repo.update_entity(eid, {"name": "Felix Wang"}, updated_by="admin")

        # Check final state
        e = await repo.get_entity(eid)
        assert e is not None
        assert e.version == 3
        assert e.name == "Felix Wang"

        # Verify version records in DB
        pool = repo._pool
        async with pool.acquire() as conn:
            versions = await conn.fetch(
                "SELECT * FROM ontology.ont_entity_versions WHERE entity_id = $1 ORDER BY version",
                eid,
            )
        assert len(versions) == 2  # 2 updates → 2 version records
        assert versions[0]["change_type"] == "updated"
        assert versions[0]["version"] == 2
        assert versions[1]["version"] == 3


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


class TestQueryHelpers:
    async def test_find_entity_by_name(self, repo: OntologyRepository):
        await repo.create_entity(Entity(
            entity_type=EntityType.PERSON,
            name="Yann",
            aliases=["Yann哥"],
            properties={"role": "CEO"},
        ))

        # Exact match
        results = await repo.find_entity_by_name("Yann", EntityType.PERSON)
        assert len(results) >= 1
        assert results[0].name == "Yann"

    async def test_query_decisions(self, repo: OntologyRepository):
        await repo.create_entity(Entity(
            entity_type=EntityType.DECISION,
            name="采用 Supabase",
            properties={"detail": "数据库选型", "status": "active"},
        ))
        await repo.create_entity(Entity(
            entity_type=EntityType.DECISION,
            name="使用 Gemini API",
            properties={"detail": "LLM选型", "status": "active"},
        ))

        results = await repo.query_decisions(keyword="Supabase")
        assert len(results) >= 1
        assert "Supabase" in results[0].name

    async def test_stats(self, repo: OntologyRepository):
        await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="A", properties={},
        ))
        await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="B", properties={},
        ))
        p3 = await repo.create_entity(Entity(
            entity_type=EntityType.PROJECT, name="X", properties={},
        ))

        s = await repo.stats()
        assert s["entities"] == 3
