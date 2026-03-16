"""Bronze Layer integration tests — real PostgreSQL.

Tests:
  - Write → Read round-trip
  - SHA-256 hash deduplication
  - List with source_type filter
"""

from __future__ import annotations

import pytest

from ontology_engine.storage.bronze import BronzeRepository

from .conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def bronze(pool) -> BronzeRepository:
    """BronzeRepository backed by the test pool."""
    return BronzeRepository(pool, schema="ontology")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBronzeWriteRead:
    """Write documents → read back → verify round-trip."""

    async def test_ingest_and_get(self, bronze: BronzeRepository):
        content = "这是一份测试会议记录。参会人：Yann、Felix。"
        doc_id, is_new = await bronze.ingest(
            content=content,
            source_type="meeting_transcript",
            source_uri="/tmp/test_meeting.md",
            metadata={"participants": ["Yann", "Felix"]},
        )

        assert is_new is True
        assert doc_id.startswith("DOC-")

        doc = await bronze.get(doc_id)
        assert doc is not None
        assert doc.content == content
        assert doc.source_type == "meeting_transcript"
        assert doc.source_uri == "/tmp/test_meeting.md"
        assert doc.metadata["participants"] == ["Yann", "Felix"]
        assert doc.ingested_at is not None

    async def test_ingest_returns_correct_hash(self, bronze: BronzeRepository):
        content = "Hash verification test content 哈希测试"
        expected_hash = BronzeRepository.compute_hash(content)

        doc_id, _ = await bronze.ingest(content=content, source_type="test")
        doc = await bronze.get(doc_id)
        assert doc is not None
        assert doc.source_hash == expected_hash


class TestBronzeDedup:
    """SHA-256 content deduplication."""

    async def test_duplicate_content_returns_same_id(self, bronze: BronzeRepository):
        content = "重复内容测试 - 这段话只应存储一次。"

        doc_id_1, is_new_1 = await bronze.ingest(content=content, source_type="test")
        doc_id_2, is_new_2 = await bronze.ingest(content=content, source_type="test")

        assert is_new_1 is True
        assert is_new_2 is False
        assert doc_id_1 == doc_id_2

    async def test_different_content_creates_new_doc(self, bronze: BronzeRepository):
        id1, _ = await bronze.ingest(content="文档 A", source_type="test")
        id2, _ = await bronze.ingest(content="文档 B", source_type="test")

        assert id1 != id2

    async def test_exists_check(self, bronze: BronzeRepository):
        content = "Existence check test"
        hash_val = BronzeRepository.compute_hash(content)

        assert await bronze.exists(hash_val) is None

        doc_id, _ = await bronze.ingest(content=content, source_type="test")
        assert await bronze.exists(hash_val) == doc_id


class TestBronzeList:
    """List and filter queries."""

    async def test_list_all(self, bronze: BronzeRepository):
        await bronze.ingest(content="Doc 1", source_type="meeting_transcript")
        await bronze.ingest(content="Doc 2", source_type="manual_input")
        await bronze.ingest(content="Doc 3", source_type="meeting_transcript")

        docs = await bronze.list()
        assert len(docs) == 3

    async def test_list_with_source_type_filter(self, bronze: BronzeRepository):
        await bronze.ingest(content="Meeting 1", source_type="meeting_transcript")
        await bronze.ingest(content="Meeting 2", source_type="meeting_transcript")
        await bronze.ingest(content="Manual 1", source_type="manual_input")

        meetings = await bronze.list(source_type="meeting_transcript")
        assert len(meetings) == 2
        assert all(d.source_type == "meeting_transcript" for d in meetings)

        manual = await bronze.list(source_type="manual_input")
        assert len(manual) == 1

    async def test_list_pagination(self, bronze: BronzeRepository):
        for i in range(5):
            await bronze.ingest(content=f"Paginated doc {i}", source_type="test")

        page1 = await bronze.list(limit=2, offset=0)
        page2 = await bronze.list(limit=2, offset=2)
        page3 = await bronze.list(limit=2, offset=4)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

        # No overlap
        all_ids = {d.id for d in page1} | {d.id for d in page2} | {d.id for d in page3}
        assert len(all_ids) == 5

    async def test_list_empty(self, bronze: BronzeRepository):
        docs = await bronze.list()
        assert docs == []

        docs = await bronze.list(source_type="nonexistent")
        assert docs == []
