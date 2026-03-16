"""Tests for the OntologyClient SDK (mocked asyncpg)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ontology_engine.sdk.client import DescribeResult, IngestResult, OntologyClient


class MockRecord(dict):
    def keys(self):
        return super().keys()


class AsyncCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def make_entity(**kw: Any) -> MockRecord:
    d = {
        "id": "ENT-test-123", "entity_type": "Decision", "name": "Test",
        "properties": json.dumps({"summary": "test", "status": "active"}),
        "aliases": [], "confidence": 0.9, "version": 1,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc), "created_by": "test",
    }
    d.update(kw)
    return MockRecord(d)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value = AsyncCM(conn)
    pool.close = AsyncMock()
    return pool, conn


@pytest.fixture
def client(mock_pool):
    pool, _ = mock_pool
    c = OntologyClient(db_url="postgresql://mock", schema="ontology")
    c._pool = pool
    return c


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect(self):
        with patch("ontology_engine.sdk.client.asyncpg") as m:
            p = MagicMock(); p.acquire.return_value = AsyncCM(AsyncMock()); p.close = AsyncMock()
            m.create_pool = AsyncMock(return_value=p)
            c = OntologyClient("postgresql://test")
            await c.connect()
            m.create_pool.assert_called_once()
            await c.close()

    @pytest.mark.asyncio
    async def test_close(self, client, mock_pool):
        await client.close()
        mock_pool[0].close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        with patch("ontology_engine.sdk.client.asyncpg") as m:
            p = MagicMock(); p.acquire.return_value = AsyncCM(AsyncMock()); p.close = AsyncMock()
            m.create_pool = AsyncMock(return_value=p)
            async with OntologyClient("postgresql://test") as c:
                assert c._pool is not None


class TestAssertEntity:
    @pytest.mark.asyncio
    async def test_create_new(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.side_effect = [None, MockRecord({"id": "ENT-new"}), None]
        conn.execute = AsyncMock()
        assert await client.assert_entity("Decision", {"summary": "x"}, "src") == "ENT-new"

    @pytest.mark.asyncio
    async def test_update_existing(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = MockRecord({"id": "ENT-old", "properties": "{}", "version": 1})
        conn.execute = AsyncMock()
        assert await client.assert_entity("Decision", {"summary": "x"}, "src", name="n") == "ENT-old"

    @pytest.mark.asyncio
    async def test_name_from_properties(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.side_effect = [None, MockRecord({"id": "ENT-auto"}), None]
        conn.execute = AsyncMock()
        assert await client.assert_entity("Person", {"name": "Alice"}, "m") == "ENT-auto"

    @pytest.mark.asyncio
    async def test_confidence(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.side_effect = [None, MockRecord({"id": "ENT-c"}), None]
        conn.execute = AsyncMock()
        await client.assert_entity("Risk", {"description": "x"}, "a", confidence=0.75)
        assert any(0.75 == a for a in conn.fetchrow.call_args_list[1].args if isinstance(a, float))


class TestAssertLink:
    @pytest.mark.asyncio
    async def test_create(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.side_effect = [None, MockRecord({"id": "LNK-new"})]
        assert await client.assert_link("ENT-a", "makes", "ENT-b") == "LNK-new"

    @pytest.mark.asyncio
    async def test_existing(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = MockRecord({"id": "LNK-ex"})
        assert await client.assert_link("ENT-a", "makes", "ENT-b") == "LNK-ex"

    @pytest.mark.asyncio
    async def test_with_props(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.side_effect = [None, MockRecord({"id": "LNK-p"})]
        assert await client.assert_link("ENT-a", "assigned_to", "ENT-b", {"priority": "high"}) == "LNK-p"


class TestQuery:
    @pytest.mark.asyncio
    async def test_basic(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [make_entity(id="ENT-1"), make_entity(id="ENT-2")]
        r = await client.query("Decision")
        assert len(r) == 2 and r[0]["id"] == "ENT-1"

    @pytest.mark.asyncio
    async def test_with_filters(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [make_entity()]
        assert len(await client.query("Decision", {"status": "active"})) == 1

    @pytest.mark.asyncio
    async def test_limit_offset(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await client.query("Decision", limit=5, offset=10)
        assert 5 in conn.fetch.call_args.args and 10 in conn.fetch.call_args.args

    @pytest.mark.asyncio
    async def test_empty(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        assert await client.query("X") == []


class TestSearch:
    @pytest.mark.asyncio
    async def test_results(self, client, mock_pool):
        _, conn = mock_pool
        rec = make_entity(); rec["relevance"] = 0.85
        conn.fetch.return_value = [rec]
        r = await client.search("delayed")
        assert len(r) == 1 and r[0]["relevance"] == 0.85

    @pytest.mark.asyncio
    async def test_type_filter(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await client.search("test", entity_type="Decision")
        assert "Decision" in conn.fetch.call_args.args


class TestGetEntity:
    @pytest.mark.asyncio
    async def test_found(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = make_entity()
        assert (await client.get_entity("ENT-test-123"))["id"] == "ENT-test-123"

    @pytest.mark.asyncio
    async def test_not_found(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        assert await client.get_entity("x") is None


class TestDescribe:
    @pytest.mark.asyncio
    async def test_with_type_def(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = MockRecord({
            "id": "Decision", "display_name": "决策", "description": "会议中做出的决策",
            "schema": json.dumps({"summary": {"type": "string"}, "decision_type": {"type": "enum"}}),
            "required_fields": ["summary", "decision_type"],
        })
        conn.fetch.side_effect = [
            [MockRecord({"link_type": "makes", "direction": "incoming"})],
            [make_entity()],
        ]
        r = await client.describe("Decision")
        assert r.entity_type == "Decision" and len(r.properties) == 2

    @pytest.mark.asyncio
    async def test_infer_from_data(self, client, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        conn.fetch.side_effect = [[], [make_entity(properties=json.dumps({"summary": "t", "status": "a"}))]]
        r = await client.describe("Decision")
        assert {p["name"] for p in r.properties} >= {"summary", "status"}


class TestModels:
    def test_ingest_result(self):
        r = IngestResult()
        assert r.document_id == "" and r.errors == []

    def test_ingest_result_data(self):
        r = IngestResult(document_id="DOC-1", entities_created=5, entity_ids=["E1", "E2"])
        assert r.entities_created == 5 and len(r.entity_ids) == 2

    def test_describe_result(self):
        r = DescribeResult(entity_type="Person")
        assert r.properties == [] and r.examples == []
