"""End-to-end pipeline tests — real Gemini API + real PostgreSQL.

Validates the complete Bronze → Silver → Gold chain:
  1. ingest() writes Bronze doc + extracts Silver entities/links
  2. Verify Bronze document exists
  3. Verify Silver entities and links
  4. Verify provenance chain
  5. Gold build aggregates Silver → Gold

Requires:
  - ONTOLOGY_TEST_DB_URL  (PostgreSQL connection)
  - GEMINI_API_KEY         (Gemini API access)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from ontology_engine.core.config import LLMConfig, OntologyConfig, PipelineConfig
from ontology_engine.fusion.gold_builder import GoldBuilder
from ontology_engine.pipeline.engine import PipelineEngine
from ontology_engine.storage.bronze import BronzeRepository
from ontology_engine.storage.gold_repository import GoldRepository
from ontology_engine.storage.repository import OntologyRepository

from .conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

requires_gemini = pytest.mark.skipif(
    not GEMINI_KEY,
    reason="GEMINI_API_KEY not set — skipping end-to-end pipeline tests",
)

# ---------------------------------------------------------------------------
# Test fixture: meeting transcript
# ---------------------------------------------------------------------------

MEETING_CONTENT = """\
# 周会记录 — 2026-03-15

参会人：Yann、Felix、Leo

---

【Yann】：大家好，今天主要聊两件事。第一，ontology engine 的进度。第二，下周的产品发布。

【Felix】：运营这边，我已经准备好了发布前的推广素材，包括社交媒体帖子和邮件模板。

【Yann】：很好。技术这边，我这周完成了 Gold 层的聚合逻辑。现在 Bronze → Silver → Gold 全链路打通了。

【Leo】：我在做前端的 dashboard，预计下周三完成。

【Yann】：好的。那我们做一个决定：产品发布日期定在3月25日。Felix 你来协调发布流程。

【Felix】：没问题。不过有一个风险——第三方 API 的 rate limit 可能会影响发布当天的稳定性。

【Yann】：这个风险需要重视。Leo，你负责在发布前做一次压力测试。截止日期是3月22日。

【Leo】：收到，我来安排。
"""


@pytest.fixture
def meeting_file(tmp_path: Path) -> Path:
    """Write the test meeting transcript to a temp file."""
    f = tmp_path / "20260315_meeting.md"
    f.write_text(MEETING_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def pipeline_config() -> OntologyConfig:
    """Config with Gemini + conservative pipeline settings."""
    return OntologyConfig(
        llm=LLMConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key=GEMINI_KEY,
            fast_model="gemini-2.0-flash-lite",
        ),
        pipeline=PipelineConfig(
            remove_filler_words=True,
            segment_topics=False,   # Save API calls
            resolve_coreferences=False,
            min_confidence=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


@requires_gemini
class TestE2EPipeline:
    """Full Bronze → Silver → Gold pipeline with real APIs."""

    async def test_ingest_creates_bronze_and_silver(
        self,
        meeting_file: Path,
        pipeline_config: OntologyConfig,
        db_url: str,
        pool,
    ):
        """Step 1-4: Ingest → Bronze → Silver → verify provenance."""
        engine = await PipelineEngine.create(pipeline_config, db_url=db_url)
        try:
            result = await engine.ingest(str(meeting_file))

            # --- Basic success ---
            assert result.success, f"Ingestion failed: {result.error}"
            assert result.extraction is not None
            assert result.bronze_doc_id is not None
            assert not result.skipped_duplicate

            # --- Step 2: Bronze document exists ---
            bronze = BronzeRepository(pool, schema="ontology")
            doc = await bronze.get(result.bronze_doc_id)
            assert doc is not None
            assert "Yann" in doc.content
            assert doc.source_type == "meeting_transcript"

            # --- Step 3: Silver entities & links ---
            assert len(result.extraction.entities) >= 2, (
                f"Expected ≥2 entities, got {len(result.extraction.entities)}"
            )
            entity_names = {e.name for e in result.extraction.entities}
            # Should extract at least some of these
            expected_names = {"Yann", "Felix", "Leo"}
            found = entity_names & expected_names
            assert len(found) >= 2, f"Expected ≥2 of {expected_names}, got {found}"

            # Should have some links
            assert len(result.extraction.links) >= 1 or len(result.extraction.decisions) >= 1, (
                "Expected at least 1 link or decision"
            )

            # --- Step 4: Verify stored IDs ---
            assert len(result.stored_ids.get("entities", [])) >= 2
            assert len(result.stored_ids.get("provenance", [])) >= 1

            # Verify entities in DB
            repo = OntologyRepository(pool, schema="ontology")
            await repo._set_search_path()
            for eid in result.stored_ids["entities"][:3]:
                entity = await repo.get_entity(eid)
                assert entity is not None, f"Entity {eid} not found in DB"

            # Verify provenance chain → bronze doc
            async with pool.acquire() as conn:
                provs = await conn.fetch(
                    "SELECT * FROM ontology.ont_provenance WHERE source_document_id = $1",
                    result.bronze_doc_id,
                )
            assert len(provs) >= 1, "No provenance records linked to bronze doc"

            print(f"\n  ✓ Ingest successful")
            print(f"    Bronze doc: {result.bronze_doc_id}")
            print(f"    Entities: {len(result.extraction.entities)}")
            print(f"    Links: {len(result.extraction.links)}")
            print(f"    Decisions: {len(result.extraction.decisions)}")
            print(f"    Action Items: {len(result.extraction.action_items)}")
            print(f"    Stored entities: {len(result.stored_ids.get('entities', []))}")
            print(f"    Stored links: {len(result.stored_ids.get('links', []))}")
            print(f"    Provenance records: {len(provs)}")

        finally:
            await engine.close()

    async def test_duplicate_ingest_skipped(
        self,
        meeting_file: Path,
        pipeline_config: OntologyConfig,
        db_url: str,
    ):
        """Ingesting the same file twice should be deduplicated at Bronze layer."""
        engine = await PipelineEngine.create(pipeline_config, db_url=db_url)
        try:
            result1 = await engine.ingest(str(meeting_file))
            assert result1.success
            assert not result1.skipped_duplicate

            result2 = await engine.ingest(str(meeting_file))
            assert result2.success
            assert result2.skipped_duplicate
            assert result2.bronze_doc_id == result1.bronze_doc_id
        finally:
            await engine.close()

    async def test_gold_build_after_ingest(
        self,
        meeting_file: Path,
        pipeline_config: OntologyConfig,
        db_url: str,
        pool,
    ):
        """Step 5: After ingesting, build Gold layer from Silver."""
        engine = await PipelineEngine.create(pipeline_config, db_url=db_url)
        try:
            result = await engine.ingest(str(meeting_file))
            assert result.success, f"Ingestion failed: {result.error}"
        finally:
            await engine.close()

        # Build Gold
        builder = GoldBuilder(pool, schema="ontology")
        build_result = await builder.build_gold(full=True)

        assert build_result.gold_entities_created >= 1, (
            f"Expected ≥1 Gold entities, got {build_result.gold_entities_created}"
        )
        assert not build_result.errors, f"Gold build errors: {build_result.errors}"

        # Query Gold layer
        gold_repo = GoldRepository(pool, schema="ontology")
        all_gold = await gold_repo.query()
        assert len(all_gold) >= 1

        print(f"\n  ✓ Gold build successful")
        print(f"    Gold entities created: {build_result.gold_entities_created}")
        print(f"    Gold links created: {build_result.gold_links_created}")
        print(f"    Review candidates: {len(build_result.review_candidates)}")


# ---------------------------------------------------------------------------
# Without Gemini — test DB-only flows
# ---------------------------------------------------------------------------


class TestE2EDBOnly:
    """End-to-end tests that only need the database (no LLM)."""

    async def test_bronze_to_silver_to_gold_manual(self, pool, db_url: str):
        """Manually write Bronze → Silver → build Gold — no LLM needed."""
        from datetime import date

        from ontology_engine.core.types import Entity, EntityType, Link, LinkType, Provenance

        # 1. Bronze
        bronze = BronzeRepository(pool, schema="ontology")
        doc_id, _ = await bronze.ingest(
            content=MEETING_CONTENT,
            source_type="meeting_transcript",
            metadata={"meeting_date": "2026-03-15"},
        )

        # 2. Silver
        repo = OntologyRepository(pool, schema="ontology")
        await repo._set_search_path()

        p1 = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Yann",
            properties={"role": "CEO"}, created_by="test",
        ))
        p2 = await repo.create_entity(Entity(
            entity_type=EntityType.PERSON, name="Felix",
            properties={"role": "运营"}, created_by="test",
        ))
        proj = await repo.create_entity(Entity(
            entity_type=EntityType.PROJECT, name="OntologyEngine",
            properties={}, created_by="test",
        ))
        dec = await repo.create_entity(Entity(
            entity_type=EntityType.DECISION, name="发布日期定在3月25日",
            properties={"detail": "产品发布日期"}, created_by="test",
        ))

        await repo.create_link(Link(
            link_type=LinkType.OWNS, source_entity_id=p1,
            target_entity_id=proj, created_by="test",
        ))
        await repo.create_link(Link(
            link_type=LinkType.MAKES, source_entity_id=p1,
            target_entity_id=dec, created_by="test",
        ))

        # Provenance
        await repo.create_provenance(Provenance(
            entity_id=p1, source_document_id=doc_id,
            source_type="llm_extraction",
        ))
        await repo.create_provenance(Provenance(
            entity_id=dec, source_document_id=doc_id,
            source_type="llm_extraction",
        ))

        # 3. Gold build
        builder = GoldBuilder(pool, schema="ontology")
        build_result = await builder.build_gold(full=True)

        assert build_result.gold_entities_created >= 4
        assert build_result.gold_links_created >= 2

        # 4. Query Gold
        gold_repo = GoldRepository(pool, schema="ontology")
        people = await gold_repo.query(entity_type="Person")
        assert len(people) >= 2

        # 5. Verify full provenance chain
        async with pool.acquire() as conn:
            provs = await conn.fetch(
                "SELECT * FROM ontology.ont_provenance WHERE source_document_id = $1",
                doc_id,
            )
        assert len(provs) >= 2

        print(f"\n  ✓ Manual Bronze → Silver → Gold chain verified")
        print(f"    Bronze doc: {doc_id}")
        print(f"    Silver entities: 4")
        print(f"    Gold entities: {build_result.gold_entities_created}")
        print(f"    Gold links: {build_result.gold_links_created}")
