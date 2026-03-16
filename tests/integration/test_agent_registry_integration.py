"""Agent Registry integration tests — real PostgreSQL.

Tests:
  - Register agent → list → heartbeat → deactivate
  - Filtering by produces / consumes
"""

from __future__ import annotations

import pytest

from ontology_engine.sdk.registry import AgentRegistry

from .conftest import requires_pg

pytestmark = [pytest.mark.integration, requires_pg]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def registry(pool) -> AgentRegistry:
    """AgentRegistry backed by the test pool."""
    reg = AgentRegistry(pool, schema="ontology")
    await reg._ensure_table()
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentLifecycle:
    async def test_register_and_get(self, registry: AgentRegistry):
        agent = await registry.register_agent(
            id="meeting-bot",
            display_name="Meeting Bot",
            description="Extracts entities from meeting transcripts",
            produces=["Decision", "Person", "ActionItem"],
            consumes=[],
            capabilities=["extraction", "preprocessing"],
            version="1.0.0",
            metadata={"model": "gemini-2.5-flash"},
        )

        assert agent.id == "meeting-bot"
        assert agent.display_name == "Meeting Bot"
        assert "Decision" in agent.produces
        assert agent.status == "active"
        assert agent.registered_at is not None

        # Retrieve by ID
        fetched = await registry.get_agent("meeting-bot")
        assert fetched is not None
        assert fetched.id == "meeting-bot"
        assert fetched.metadata.get("model") == "gemini-2.5-flash"

    async def test_list_active_agents(self, registry: AgentRegistry):
        await registry.register_agent(id="bot-1", display_name="Bot 1")
        await registry.register_agent(id="bot-2", display_name="Bot 2")
        await registry.register_agent(id="bot-3", display_name="Bot 3")

        agents = await registry.list_agents(status="active")
        assert len(agents) == 3

    async def test_heartbeat(self, registry: AgentRegistry):
        await registry.register_agent(id="worker-1")

        # Heartbeat updates last_seen_at
        result = await registry.heartbeat("worker-1")
        assert result is True

        agent = await registry.get_agent("worker-1")
        assert agent is not None
        assert agent.last_seen_at is not None

        # Heartbeat for nonexistent agent
        result = await registry.heartbeat("nonexistent")
        assert result is False

    async def test_deactivate_agent(self, registry: AgentRegistry):
        await registry.register_agent(id="temp-agent")

        result = await registry.deactivate_agent("temp-agent")
        assert result is True

        agent = await registry.get_agent("temp-agent")
        assert agent is not None
        assert agent.status == "inactive"

        # Should not appear in active list
        active = await registry.list_agents(status="active")
        assert not any(a.id == "temp-agent" for a in active)

    async def test_reregister_reactivates(self, registry: AgentRegistry):
        """Re-registering a deactivated agent should reactivate it."""
        await registry.register_agent(id="flaky-agent")
        await registry.deactivate_agent("flaky-agent")

        # Re-register
        agent = await registry.register_agent(
            id="flaky-agent",
            display_name="Flaky Agent v2",
            version="2.0.0",
        )
        assert agent.status == "active"
        assert agent.version == "2.0.0"


class TestAgentDiscovery:
    async def test_filter_by_produces(self, registry: AgentRegistry):
        await registry.register_agent(
            id="extractor",
            produces=["Decision", "Person"],
        )
        await registry.register_agent(
            id="summarizer",
            produces=["Summary"],
        )

        decision_producers = await registry.list_agents(produces="Decision")
        assert len(decision_producers) == 1
        assert decision_producers[0].id == "extractor"

    async def test_filter_by_consumes(self, registry: AgentRegistry):
        await registry.register_agent(
            id="pm-agent",
            consumes=["Decision"],
            produces=["ActionItem"],
        )
        await registry.register_agent(
            id="cto-agent",
            consumes=["Risk"],
        )

        decision_consumers = await registry.list_agents(consumes="Decision")
        assert len(decision_consumers) == 1
        assert decision_consumers[0].id == "pm-agent"

    async def test_get_nonexistent_agent(self, registry: AgentRegistry):
        result = await registry.get_agent("ghost-agent")
        assert result is None
