"""Tests for storage/bronze.py — BronzeRepository and BronzeDocument."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ontology_engine.storage.bronze import BronzeDocument, BronzeRepository


# =============================================================================
# BronzeDocument unit tests
# =============================================================================


class TestBronzeDocument:
    def test_create_document(self):
        doc = BronzeDocument(
            id="DOC-123",
            source_type="meeting_transcript",
            source_uri="/path/to/file.md",
            source_hash="abc123",
            content="Hello world",
            content_format="text",
            language="en",
            metadata={"key": "value"},
            ingested_at=datetime(2026, 3, 17, 10, 0, 0),
            ingested_by="pipeline",
        )
        assert doc.id == "DOC-123"
        assert doc.source_type == "meeting_transcript"
        assert doc.source_uri == "/path/to/file.md"
        assert doc.source_hash == "abc123"
        assert doc.content == "Hello world"
        assert doc.content_format == "text"
        assert doc.language == "en"
        assert doc.metadata == {"key": "value"}
        assert doc.ingested_by == "pipeline"

    def test_to_dict(self):
        now = datetime(2026, 3, 17, 10, 0, 0)
        doc = BronzeDocument(
            id="DOC-456",
            source_type="document",
            source_uri=None,
            source_hash="def456",
            content="Some content",
            ingested_at=now,
        )
        d = doc.to_dict()
        assert d["id"] == "DOC-456"
        assert d["source_type"] == "document"
        assert d["source_uri"] is None
        assert d["source_hash"] == "def456"
        assert d["content"] == "Some content"
        assert d["content_format"] == "text"
        assert d["language"] == "auto"
        assert d["metadata"] == {}
        assert d["ingested_at"] == now.isoformat()
        assert d["ingested_by"] == "system"

    def test_defaults(self):
        doc = BronzeDocument(
            id="DOC-789",
            source_type="agent_log",
            source_uri=None,
            source_hash="ghi789",
            content="Log entry",
        )
        assert doc.content_format == "text"
        assert doc.language == "auto"
        assert doc.metadata == {}
        assert doc.ingested_by == "system"
        assert doc.ingested_at is None


# =============================================================================
# BronzeRepository unit tests (mocked asyncpg)
# =============================================================================


class _MockAcquireCtx:
    """Sync-returning async context manager to mimic pool.acquire()."""

    def __init__(self, conn: AsyncMock):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        return False


def _make_mock_pool():
    """Create a mock asyncpg pool with acquire() context manager."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value = _MockAcquireCtx(conn)
    return pool, conn


class TestComputeHash:
    def test_sha256_hash(self):
        content = "Hello, Bronze Layer!"
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert BronzeRepository.compute_hash(content) == expected

    def test_deterministic(self):
        content = "Same content twice"
        h1 = BronzeRepository.compute_hash(content)
        h2 = BronzeRepository.compute_hash(content)
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = BronzeRepository.compute_hash("content A")
        h2 = BronzeRepository.compute_hash("content B")
        assert h1 != h2

    def test_empty_content(self):
        h = BronzeRepository.compute_hash("")
        expected = hashlib.sha256(b"").hexdigest()
        assert h == expected


class TestBronzeRepositoryIngest:
    async def test_ingest_new_document(self):
        """New document should be inserted and return (doc_id, True)."""
        pool, conn = _make_mock_pool()
        # exists() returns None (no existing doc)
        conn.fetchrow.side_effect = [
            None,  # exists() check
            {"id": "DOC-new-123"},  # INSERT RETURNING id
        ]

        repo = BronzeRepository(pool, schema="ontology")
        doc_id, is_new = await repo.ingest(
            content="New meeting content",
            source_type="meeting_transcript",
            source_uri="/meetings/2026-03-17.md",
            metadata={"meeting_date": "2026-03-17"},
            ingested_by="pipeline",
        )

        assert doc_id == "DOC-new-123"
        assert is_new is True

    async def test_ingest_duplicate_document(self):
        """Duplicate document should return existing id and is_new=False."""
        pool, conn = _make_mock_pool()
        # exists() returns existing doc_id
        conn.fetchrow.return_value = {"id": "DOC-existing-456"}

        repo = BronzeRepository(pool, schema="ontology")
        doc_id, is_new = await repo.ingest(
            content="Already ingested content",
            source_type="meeting_transcript",
        )

        assert doc_id == "DOC-existing-456"
        assert is_new is False


class TestBronzeRepositoryGet:
    async def test_get_existing(self):
        """Get should return a BronzeDocument for existing id."""
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {
            "id": "DOC-abc",
            "source_type": "document",
            "source_uri": "/docs/test.md",
            "source_hash": "hash123",
            "content": "Document content",
            "content_format": "markdown",
            "language": "en",
            "metadata": {"key": "val"},
            "ingested_at": datetime(2026, 3, 17),
            "ingested_by": "system",
        }

        repo = BronzeRepository(pool, schema="ontology")
        doc = await repo.get("DOC-abc")

        assert doc is not None
        assert doc.id == "DOC-abc"
        assert doc.source_type == "document"
        assert doc.content == "Document content"
        assert doc.content_format == "markdown"
        assert doc.metadata == {"key": "val"}

    async def test_get_nonexistent(self):
        """Get should return None for missing id."""
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None

        repo = BronzeRepository(pool, schema="ontology")
        doc = await repo.get("DOC-nonexistent")

        assert doc is None


class TestBronzeRepositoryList:
    async def test_list_all(self):
        """List without filter should return all documents."""
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = [
            {
                "id": "DOC-1",
                "source_type": "meeting_transcript",
                "source_uri": "/m1.md",
                "source_hash": "h1",
                "content": "Content 1",
                "content_format": "text",
                "language": "auto",
                "metadata": {},
                "ingested_at": datetime(2026, 3, 17),
                "ingested_by": "system",
            },
            {
                "id": "DOC-2",
                "source_type": "document",
                "source_uri": "/d1.md",
                "source_hash": "h2",
                "content": "Content 2",
                "content_format": "markdown",
                "language": "en",
                "metadata": {},
                "ingested_at": datetime(2026, 3, 16),
                "ingested_by": "pipeline",
            },
        ]

        repo = BronzeRepository(pool, schema="ontology")
        docs = await repo.list()

        assert len(docs) == 2
        assert docs[0].id == "DOC-1"
        assert docs[1].id == "DOC-2"

    async def test_list_with_source_type_filter(self):
        """List with source_type filter should pass it to query."""
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []

        repo = BronzeRepository(pool, schema="ontology")
        docs = await repo.list(source_type="meeting_transcript", limit=10, offset=5)

        assert len(docs) == 0
        # Verify the query was called with source_type parameter
        call_args = conn.fetch.call_args
        assert "meeting_transcript" in call_args[0]


class TestBronzeRepositoryExists:
    async def test_exists_found(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = {"id": "DOC-found"}

        repo = BronzeRepository(pool, schema="ontology")
        result = await repo.exists("somehash")

        assert result == "DOC-found"

    async def test_exists_not_found(self):
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None

        repo = BronzeRepository(pool, schema="ontology")
        result = await repo.exists("nonexistent-hash")

        assert result is None
