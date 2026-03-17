"""Audit Trail — immutable, queryable log of every action execution.

Every action execution (success, failure, or rollback) produces an
:class:`AuditEntry` that is stored in the :class:`AuditTrail`. This
provides a complete operational history answering *who did what, when,
and what happened*.

Two storage backends:

- **In-memory** (default): fast, suitable for tests and single-process use.
- **PostgreSQL**: production backend; see :meth:`PgAuditTrail.create`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data model
# =============================================================================


@dataclass
class AuditEntry:
    """A single audit record for one action execution.

    Attributes:
        id: Unique execution identifier (e.g. ``EXEC-abc123``).
        action_name: The action that was executed.
        params: Input parameters supplied by the caller.
        result: Output returned by the handler (empty dict on failure).
        actor: Who or what triggered the action (agent ID, user ID, ``"system"``).
        timestamp: When the action completed (UTC).
        status: ``"success"``, ``"failed"``, ``"validation_error"``, or ``"rolled_back"``.
        duration_ms: Wall-clock execution time in milliseconds.
        error_message: Human-readable error detail (empty on success).
    """

    id: str
    action_name: str
    params: dict[str, Any]
    result: dict[str, Any]
    actor: str
    timestamp: datetime
    status: str  # success | failed | validation_error | rolled_back
    duration_ms: int
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_name": self.action_name,
            "params": self.params,
            "result": self.result,
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
        }


# =============================================================================
# In-memory backend
# =============================================================================


class AuditTrail:
    """In-memory audit trail — stores entries in a list.

    Suitable for tests and single-process applications.
    For production use, see :class:`PgAuditTrail`.
    """

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        """Append an audit entry."""
        self._entries.append(entry)
        logger.debug(
            "Audit: %s %s by %s -> %s (%dms)",
            entry.action_name, entry.id, entry.actor,
            entry.status, entry.duration_ms,
        )

    def query(self, filters: dict[str, Any] | None = None) -> list[AuditEntry]:
        """Return entries matching *filters*.

        Supported filter keys:

        - ``id``: exact match on execution ID
        - ``action_name``: exact match
        - ``actor``: exact match
        - ``status``: exact match
        - ``limit``: max number of results (default: all)

        Returns entries in reverse chronological order (newest first).
        """
        filters = filters or {}
        results = list(reversed(self._entries))

        if "id" in filters:
            results = [e for e in results if e.id == filters["id"]]
        if "action_name" in filters:
            results = [e for e in results if e.action_name == filters["action_name"]]
        if "actor" in filters:
            results = [e for e in results if e.actor == filters["actor"]]
        if "status" in filters:
            results = [e for e in results if e.status == filters["status"]]
        if "limit" in filters:
            results = results[: int(filters["limit"])]

        return results

    def get_lineage(self, entity_id: str) -> list[AuditEntry]:
        """Return all entries whose params or result reference *entity_id*.

        This is a simple substring search over serialised params/result.
        Production backends should use indexed queries instead.
        """
        matching: list[AuditEntry] = []
        for entry in reversed(self._entries):
            params_str = json.dumps(entry.params)
            result_str = json.dumps(entry.result)
            if entity_id in params_str or entity_id in result_str:
                matching.append(entry)
        return matching

    @property
    def entries(self) -> list[AuditEntry]:
        """All entries in chronological order (oldest first)."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries (useful in tests)."""
        self._entries.clear()


# =============================================================================
# PostgreSQL backend
# =============================================================================


class PgAuditTrail:
    """PostgreSQL-backed audit trail.

    Writes entries to ``ontology.action_audit_trail`` and supports
    indexed queries by action name, actor, and timestamp.

    Usage::

        audit = await PgAuditTrail.create(pool, schema="ontology")
        audit.record(entry)
        entries = await audit.async_query({"action_name": "create_entity", "limit": 20})
    """

    def __init__(self, pool: Any, schema: str = "ontology") -> None:
        self._pool = pool
        self._schema = schema

    @classmethod
    async def create(cls, pool: Any, schema: str = "ontology") -> PgAuditTrail:
        """Create a :class:`PgAuditTrail` and ensure the table exists."""
        instance = cls(pool, schema)
        await instance._ensure_table()
        return instance

    async def _ensure_table(self) -> None:
        """Create the audit trail table if it does not exist."""
        async with self._pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._schema}.action_audit_trail (
                    id              TEXT PRIMARY KEY,
                    action_name     TEXT NOT NULL,
                    params          JSONB NOT NULL DEFAULT '{{}}',
                    result          JSONB DEFAULT '{{}}',
                    actor           TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    duration_ms     INTEGER DEFAULT 0,
                    error_message   TEXT DEFAULT '',
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_audit_trail_action
                    ON {self._schema}.action_audit_trail(action_name);
                CREATE INDEX IF NOT EXISTS idx_audit_trail_actor
                    ON {self._schema}.action_audit_trail(actor);
                CREATE INDEX IF NOT EXISTS idx_audit_trail_created
                    ON {self._schema}.action_audit_trail(created_at);
                CREATE INDEX IF NOT EXISTS idx_audit_trail_status
                    ON {self._schema}.action_audit_trail(status);
            """)

    def record(self, entry: AuditEntry) -> None:
        """Synchronous record — schedules an async insert.

        For fully async usage, call :meth:`async_record` directly.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_record(entry))
        except RuntimeError:
            # No running loop — skip (tests / sync context)
            logger.debug("No event loop; skipping PG audit write for %s", entry.id)

    async def async_record(self, entry: AuditEntry) -> None:
        """Insert an audit entry into PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""INSERT INTO {self._schema}.action_audit_trail
                    (id, action_name, params, result, actor, status, duration_ms, error_message, created_at)
                    VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7, $8, $9)
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        duration_ms = EXCLUDED.duration_ms,
                        error_message = EXCLUDED.error_message
                """,
                entry.id,
                entry.action_name,
                json.dumps(entry.params, ensure_ascii=False),
                json.dumps(entry.result, ensure_ascii=False),
                entry.actor,
                entry.status,
                entry.duration_ms,
                entry.error_message,
                entry.timestamp,
            )

    async def async_query(self, filters: dict[str, Any] | None = None) -> list[AuditEntry]:
        """Query audit entries with indexed lookups."""
        filters = filters or {}
        conditions: list[str] = []
        params: list[Any] = []

        if "id" in filters:
            params.append(filters["id"])
            conditions.append(f"id = ${len(params)}")
        if "action_name" in filters:
            params.append(filters["action_name"])
            conditions.append(f"action_name = ${len(params)}")
        if "actor" in filters:
            params.append(filters["actor"])
            conditions.append(f"actor = ${len(params)}")
        if "status" in filters:
            params.append(filters["status"])
            conditions.append(f"status = ${len(params)}")

        where = " AND ".join(conditions) if conditions else "TRUE"
        limit = int(filters.get("limit", 100))
        params.append(limit)

        sql = f"""
            SELECT id, action_name, params, result, actor, status,
                   duration_ms, error_message, created_at
            FROM {self._schema}.action_audit_trail
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ${len(params)}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [self._row_to_entry(r) for r in rows]

    async def async_get_lineage(self, entity_id: str) -> list[AuditEntry]:
        """Return entries whose params or result contain *entity_id*."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT id, action_name, params, result, actor, status,
                           duration_ms, error_message, created_at
                    FROM {self._schema}.action_audit_trail
                    WHERE params::text ILIKE $1 OR result::text ILIKE $1
                    ORDER BY created_at DESC""",
                f"%{entity_id}%",
            )
        return [self._row_to_entry(r) for r in rows]

    @staticmethod
    def _row_to_entry(row: Any) -> AuditEntry:
        """Convert a DB row to an :class:`AuditEntry`."""
        params = row["params"]
        if isinstance(params, str):
            params = json.loads(params)
        result = row["result"]
        if isinstance(result, str):
            result = json.loads(result)
        return AuditEntry(
            id=row["id"],
            action_name=row["action_name"],
            params=params,
            result=result,
            actor=row["actor"],
            timestamp=row["created_at"],
            status=row["status"],
            duration_ms=row["duration_ms"] or 0,
            error_message=row["error_message"] or "",
        )
