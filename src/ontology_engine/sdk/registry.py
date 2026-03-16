"""Agent Registration & Discovery.

Agents register themselves so other agents can discover who produces/consumes
what entity types. Storage: PostgreSQL table `ontology.agent_registry`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentInfo(BaseModel):
    """Registered agent metadata."""

    id: str
    display_name: str = ""
    description: str = ""
    produces: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    registered_at: datetime | None = None
    last_seen_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        for k in ("registered_at", "last_seen_at"):
            if d.get(k):
                d[k] = d[k].isoformat()
        return d


AGENT_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.agent_registry (
    id              TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    produces        TEXT[] DEFAULT '{{}}',
    consumes        TEXT[] DEFAULT '{{}}',
    capabilities    TEXT[] DEFAULT '{{}}',
    version         TEXT DEFAULT '1.0.0',
    status          TEXT DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'deprecated')),
    metadata        JSONB DEFAULT '{{}}',
    registered_at   TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_registry_status
    ON {schema}.agent_registry(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_agent_registry_produces
    ON {schema}.agent_registry USING gin(produces);
CREATE INDEX IF NOT EXISTS idx_agent_registry_consumes
    ON {schema}.agent_registry USING gin(consumes);
"""


class AgentRegistry:
    """Manages agent registration and discovery via PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool, schema: str = "ontology"):
        self._pool = pool
        self._schema = schema

    @classmethod
    async def create(cls, db_url: str, schema: str = "ontology") -> AgentRegistry:
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        registry = cls(pool, schema)
        await registry._ensure_table()
        return registry

    async def _ensure_table(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(AGENT_REGISTRY_DDL.format(schema=self._schema))

    async def register_agent(
        self,
        id: str,
        display_name: str = "",
        description: str = "",
        produces: list[str] | None = None,
        consumes: list[str] | None = None,
        capabilities: list[str] | None = None,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> AgentInfo:
        """Register or update an agent (upsert)."""
        sql = f"""
            INSERT INTO {self._schema}.agent_registry
                (id, display_name, description, produces, consumes,
                 capabilities, version, status, metadata, registered_at, last_seen_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8::jsonb, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                produces = EXCLUDED.produces,
                consumes = EXCLUDED.consumes,
                capabilities = EXCLUDED.capabilities,
                version = EXCLUDED.version,
                status = 'active',
                metadata = EXCLUDED.metadata,
                last_seen_at = NOW()
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql, id, display_name or id, description,
                produces or [], consumes or [], capabilities or [],
                version, json.dumps(metadata or {}),
            )
            return self._row_to_agent(row)

    async def list_agents(
        self, status: str = "active",
        produces: str | None = None,
        consumes: str | None = None,
    ) -> list[AgentInfo]:
        conditions = ["status = $1"]
        params: list[Any] = [status]
        if produces:
            params.append(produces)
            conditions.append(f"${len(params)} = ANY(produces)")
        if consumes:
            params.append(consumes)
            conditions.append(f"${len(params)} = ANY(consumes)")
        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM {self._schema}.agent_registry
            WHERE {where} ORDER BY display_name
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_agent(r) for r in rows]

    async def get_agent(self, agent_id: str) -> AgentInfo | None:
        sql = f"SELECT * FROM {self._schema}.agent_registry WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, agent_id)
            return self._row_to_agent(row) if row else None

    async def heartbeat(self, agent_id: str) -> bool:
        sql = f"""
            UPDATE {self._schema}.agent_registry SET last_seen_at = NOW()
            WHERE id = $1 RETURNING id
        """
        async with self._pool.acquire() as conn:
            return (await conn.fetchrow(sql, agent_id)) is not None

    async def deactivate_agent(self, agent_id: str) -> bool:
        sql = f"""
            UPDATE {self._schema}.agent_registry SET status = 'inactive'
            WHERE id = $1 RETURNING id
        """
        async with self._pool.acquire() as conn:
            return (await conn.fetchrow(sql, agent_id)) is not None

    async def close(self) -> None:
        await self._pool.close()

    @staticmethod
    def _row_to_agent(row: asyncpg.Record) -> AgentInfo:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return AgentInfo(
            id=row["id"], display_name=row["display_name"],
            description=row["description"],
            produces=list(row.get("produces") or []),
            consumes=list(row.get("consumes") or []),
            capabilities=list(row.get("capabilities") or []),
            version=row["version"], status=row["status"],
            metadata=metadata or {},
            registered_at=row["registered_at"],
            last_seen_at=row["last_seen_at"],
        )
