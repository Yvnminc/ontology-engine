"""Tests for the Event Notifier system."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from ontology_engine.events.notifier import (
    CHANNEL_PREFIX, EVENT_TYPES, EventNotifier, OntologyEvent,
)


class AsyncCM:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


class TestOntologyEvent:
    def test_creation(self):
        e = OntologyEvent("entity.created", "Decision", "ENT-1", "bot", {"k": "v"})
        assert e.event_type == "entity.created" and e.entity_id == "ENT-1"

    def test_to_dict(self):
        d = OntologyEvent("entity.updated", "Person", "ENT-2").to_dict()
        assert d["event_type"] == "entity.updated" and "timestamp" in d

    def test_from_dict(self):
        e = OntologyEvent.from_dict({
            "event_type": "link.created", "entity_type": "makes", "entity_id": "LNK-1",
            "source_agent": "pm", "payload": {"x": 1}, "timestamp": "2026-03-15T10:00:00+00:00",
        })
        assert e.source_agent == "pm" and e.payload["x"] == 1

    def test_roundtrip(self):
        orig = OntologyEvent("gold.fused", "Decision", "ENT-g", "resolver", {"merged": ["a", "b"]})
        restored = OntologyEvent.from_dict(orig.to_dict())
        assert restored.event_type == orig.event_type and restored.payload == orig.payload

    def test_repr(self):
        r = repr(OntologyEvent("entity.created", "Risk", "ENT-r"))
        assert "entity.created" in r and "Risk" in r

    def test_default_timestamp(self):
        e = OntologyEvent("entity.created", "Person", "ENT-1")
        assert isinstance(e.timestamp, datetime)


class TestEventNotifier:
    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value = AsyncCM(conn)
        pool.close = AsyncMock()
        return pool, conn

    @pytest.fixture
    def notifier(self, mock_pool):
        return EventNotifier(mock_pool[0])

    @pytest.mark.asyncio
    async def test_emit_valid(self, notifier, mock_pool):
        _, conn = mock_pool
        await notifier.emit("entity.created", "Decision", "ENT-1", "bot")
        conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_invalid_type(self, notifier):
        with pytest.raises(ValueError, match="Invalid event_type"):
            await notifier.emit("bad.type", "Decision", "ENT-1")

    @pytest.mark.asyncio
    async def test_emit_with_payload(self, notifier, mock_pool):
        _, conn = mock_pool
        await notifier.emit("entity.updated", "Person", "ENT-p", payload={"name": "Alice"})
        assert "NOTIFY" in conn.execute.call_args.args[0]

    @pytest.mark.asyncio
    async def test_subscribe(self, notifier, mock_pool):
        pool, _ = mock_pool
        listen_conn = AsyncMock()
        listen_conn.add_listener = MagicMock()
        pool.acquire = AsyncMock(return_value=listen_conn)
        cb = AsyncMock()
        await notifier.subscribe("Decision", cb)
        assert "Decision" in notifier._subscriptions and cb in notifier._subscriptions["Decision"]

    def test_channel_name(self):
        assert EventNotifier._channel_name("Decision") == f"{CHANNEL_PREFIX}_decision"
        assert EventNotifier._channel_name("ActionItem") == f"{CHANNEL_PREFIX}_actionitem"


class TestConstants:
    def test_event_types(self):
        for t in ("entity.created", "entity.updated", "link.created", "gold.fused", "conflict.detected"):
            assert t in EVENT_TYPES
        # Kinetic Layer adds: action.started, action.completed, action.failed
        for t in ("action.started", "action.completed", "action.failed"):
            assert t in EVENT_TYPES
        assert len(EVENT_TYPES) == 8
