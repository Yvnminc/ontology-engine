"""Event system based on PostgreSQL LISTEN/NOTIFY.

Provides pub/sub for ontology changes:
  - entity.created, entity.updated
  - link.created
  - gold.fused
  - conflict.detected

Usage:
    notifier = await EventNotifier.create(db_url)
    await notifier.emit("entity.created", "Decision", "ENT-123", "meeting-bot", {...})
    await notifier.subscribe("Decision", callback)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import asyncpg

logger = logging.getLogger(__name__)

# Valid event types
EVENT_TYPES = frozenset({
    "entity.created",
    "entity.updated",
    "link.created",
    "gold.fused",
    "conflict.detected",
})

# PG channel prefix
CHANNEL_PREFIX = "ontology_events"


class OntologyEvent:
    """An event emitted by the ontology engine."""

    __slots__ = (
        "event_type", "entity_type", "entity_id",
        "source_agent", "payload", "timestamp",
    )

    def __init__(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        source_agent: str = "system",
        payload: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ):
        self.event_type = event_type
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.source_agent = source_agent
        self.payload = payload or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "source_agent": self.source_agent,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OntologyEvent:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            event_type=data["event_type"],
            entity_type=data["entity_type"],
            entity_id=data["entity_id"],
            source_agent=data.get("source_agent", "system"),
            payload=data.get("payload", {}),
            timestamp=ts,
        )

    def __repr__(self) -> str:
        return (
            f"OntologyEvent({self.event_type}, {self.entity_type}, "
            f"{self.entity_id}, agent={self.source_agent})"
        )


# Callback type: async function that receives an OntologyEvent
EventCallback = Callable[[OntologyEvent], Coroutine[Any, Any, None]]


class EventNotifier:
    """Pub/sub event system using PostgreSQL LISTEN/NOTIFY."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool
        self._subscriptions: dict[str, list[EventCallback]] = {}
        self._listen_conn: asyncpg.Connection | None = None
        self._listening_channels: set[str] = set()
        self._running = False

    @classmethod
    async def create(cls, db_url: str) -> EventNotifier:
        """Create an EventNotifier with a connection pool."""
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        return cls(pool)

    async def emit(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        source_agent: str = "system",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Publish an event via PG NOTIFY."""
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{event_type}'. Must be one of: {EVENT_TYPES}"
            )

        event = OntologyEvent(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            source_agent=source_agent,
            payload=payload,
        )

        channel = self._channel_name(entity_type)
        message = json.dumps(event.to_dict(), ensure_ascii=False)

        async with self._pool.acquire() as conn:
            if len(message.encode("utf-8")) > 7900:
                event.payload = {"truncated": True}
                message = json.dumps(event.to_dict(), ensure_ascii=False)
            await conn.execute(f"NOTIFY {channel}, $1", message)

        logger.debug("Emitted %s on channel %s", event, channel)

    async def subscribe(
        self,
        entity_type: str,
        callback: EventCallback,
    ) -> None:
        """Subscribe to events for a given entity type."""
        channel = self._channel_name(entity_type)

        if entity_type not in self._subscriptions:
            self._subscriptions[entity_type] = []
        self._subscriptions[entity_type].append(callback)

        if channel not in self._listening_channels:
            await self._ensure_listener()
            if self._listen_conn:
                await self._listen_conn.execute(f"LISTEN {channel}")
                self._listening_channels.add(channel)
                logger.info("Listening on PG channel: %s", channel)

    async def unsubscribe(self, entity_type: str) -> None:
        """Remove all subscriptions for an entity type."""
        channel = self._channel_name(entity_type)
        self._subscriptions.pop(entity_type, None)
        if channel in self._listening_channels and self._listen_conn:
            await self._listen_conn.execute(f"UNLISTEN {channel}")
            self._listening_channels.discard(channel)

    async def _ensure_listener(self) -> None:
        """Ensure we have a dedicated connection for LISTEN."""
        if self._listen_conn is not None:
            return
        self._listen_conn = await self._pool.acquire()
        self._running = True

    async def start(self) -> None:
        """Start the event listener."""
        await self._ensure_listener()
        if self._listen_conn and CHANNEL_PREFIX not in self._listening_channels:
            await self._listen_conn.execute(f"LISTEN {CHANNEL_PREFIX}")
            self._listening_channels.add(CHANNEL_PREFIX)

    async def stop(self) -> None:
        """Stop listening and release the connection."""
        self._running = False
        if self._listen_conn:
            for ch in list(self._listening_channels):
                try:
                    await self._listen_conn.execute(f"UNLISTEN {ch}")
                except Exception:
                    pass
            self._listening_channels.clear()
            try:
                await self._pool.release(self._listen_conn)
            except Exception:
                pass
            self._listen_conn = None
        self._subscriptions.clear()

    async def close(self) -> None:
        """Stop listener and close the pool."""
        await self.stop()
        await self._pool.close()

    @staticmethod
    def _channel_name(entity_type: str) -> str:
        """Convert entity type to PG channel name."""
        safe = entity_type.lower().replace(" ", "_").replace("-", "_")
        return f"{CHANNEL_PREFIX}_{safe}"
