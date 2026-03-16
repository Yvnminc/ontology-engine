"""Tests for the Agent Registry."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ontology_engine.sdk.registry import AgentInfo, AgentRegistry, AGENT_REGISTRY_DDL


class MockRecord(dict):
    def keys(self):
        return super().keys()

    def get(self, key, default=None):
        return super().get(key, default)


class AsyncCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def make_agent(**kw: Any) -> MockRecord:
    now = datetime.now(timezone.utc)
    d = {
        "id": "test-bot", "display_name": "Test Bot", "description": "A test agent",
        "produces": ["Decision"], "consumes": ["Person"], "capabilities": ["extraction"],
        "version": "1.0.0", "status": "active", "metadata": "{}",
        "registered_at": now, "last_seen_at": now,
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
def registry(mock_pool):
    pool, _ = mock_pool
    return AgentRegistry(pool, schema="ontology")


class TestAgentInfo:
    def test_defaults(self):
        info = AgentInfo(id="bot-1")
        assert info.id == "bot-1" and info.produces == [] and info.status == "active"

    def test_to_dict(self):
        info = AgentInfo(id="bot-1", display_name="My Bot", produces=["Decision"])
        d = info.to_dict()
        assert d["id"] == "bot-1" and d["produces"] == ["Decision"]

    def test_metadata(self):
        info = AgentInfo(id="bot-2", metadata={"owner": "team-a"})
        assert info.metadata["owner"] == "team-a"


class TestRegistration:
    @pytest.mark.asyncio
    async def test_register(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = make_agent()
        r = await registry.register_agent(id="test-bot", display_name="Test Bot", produces=["Decision"])
        assert r.id == "test-bot" and "Decision" in r.produces

    @pytest.mark.asyncio
    async def test_upsert(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = make_agent(version="2.0.0")
        r = await registry.register_agent(id="test-bot", version="2.0.0")
        assert r.version == "2.0.0"


class TestDiscovery:
    @pytest.mark.asyncio
    async def test_list(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [make_agent(id="b1"), make_agent(id="b2")]
        agents = await registry.list_agents()
        assert len(agents) == 2 and agents[0].id == "b1"

    @pytest.mark.asyncio
    async def test_list_filter(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [make_agent()]
        assert len(await registry.list_agents(produces="Decision")) == 1

    @pytest.mark.asyncio
    async def test_get(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = make_agent()
        assert (await registry.get_agent("test-bot")).id == "test-bot"

    @pytest.mark.asyncio
    async def test_get_missing(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        assert await registry.get_agent("x") is None


class TestManagement:
    @pytest.mark.asyncio
    async def test_heartbeat(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = MockRecord({"id": "test-bot"})
        assert await registry.heartbeat("test-bot") is True

    @pytest.mark.asyncio
    async def test_heartbeat_missing(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = None
        assert await registry.heartbeat("x") is False

    @pytest.mark.asyncio
    async def test_deactivate(self, registry, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = MockRecord({"id": "test-bot"})
        assert await registry.deactivate_agent("test-bot") is True


class TestDDL:
    def test_contains_table(self):
        assert "agent_registry" in AGENT_REGISTRY_DDL
        assert "produces" in AGENT_REGISTRY_DDL and "consumes" in AGENT_REGISTRY_DDL
