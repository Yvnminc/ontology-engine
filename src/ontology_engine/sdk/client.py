"""OntologyClient — the primary SDK for AI agents to interact with the knowledge graph.

Usage:
    async with OntologyClient("postgresql://...", schema="ontology") as client:
        await client.register_agent(id="meeting-bot", produces=["Decision"])
        eid = await client.assert_entity("Decision", {"summary": "..."}, source="meeting:...")
        results = await client.query("Decision", filters={"status": "active"})
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import asyncpg
from pydantic import BaseModel, Field

from ontology_engine.core.errors import StorageError
from ontology_engine.events.notifier import EventCallback, EventNotifier, OntologyEvent
from ontology_engine.sdk.registry import AgentInfo, AgentRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Result models
# =============================================================================


class IngestResult(BaseModel):
    """Result of a full pipeline ingestion."""
    document_id: str = ""
    entities_created: int = 0
    links_created: int = 0
    entity_ids: list[str] = Field(default_factory=list)
    link_ids: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class DescribeResult(BaseModel):
    """Schema introspection result for an entity type."""
    entity_type: str
    description: str = ""
    properties: list[dict[str, Any]] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    link_types: list[dict[str, Any]] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# OntologyClient
# =============================================================================


class OntologyClient:
    """Primary SDK for AI agents to interact with the Ontology Engine."""

    def __init__(self, db_url: str, schema: str = "ontology"):
        self._db_url = db_url
        self._schema = schema
        self._pool: asyncpg.Pool | None = None
        self._notifier: EventNotifier | None = None
        self._registry: AgentRegistry | None = None
        self._agent_id: str | None = None

    # ---- Lifecycle ----

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._db_url, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(f"SET search_path TO {self._schema}, public")

    async def close(self) -> None:
        if self._notifier:
            await self._notifier.close()
            self._notifier = None
        if self._registry:
            await self._registry.close()
            self._registry = None
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> OntologyClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise StorageError("Client not connected. Call connect() first.")
        return self._pool

    # ---- Agent Registration ----

    async def _ensure_registry(self) -> AgentRegistry:
        if self._registry is None:
            self._registry = AgentRegistry(self._ensure_pool(), self._schema)
            await self._registry._ensure_table()
        return self._registry

    async def register_agent(self, id: str, **kwargs: Any) -> AgentInfo:
        registry = await self._ensure_registry()
        self._agent_id = id
        return await registry.register_agent(id=id, **kwargs)

    async def list_agents(self, **kwargs: Any) -> list[AgentInfo]:
        return await (await self._ensure_registry()).list_agents(**kwargs)

    async def get_agent(self, agent_id: str) -> AgentInfo | None:
        return await (await self._ensure_registry()).get_agent(agent_id)

    # ---- Write: assert_entity ----

    async def assert_entity(
        self,
        entity_type: str,
        properties: dict[str, Any],
        source: str,
        confidence: float = 0.9,
        name: str | None = None,
    ) -> str:
        """Create or update an entity, returning its entity_id."""
        pool = self._ensure_pool()
        agent_id = self._agent_id or "sdk"
        entity_name = (
            name or properties.get("name") or properties.get("summary")
            or properties.get("title") or f"{entity_type}-{uuid.uuid4().hex[:8]}"
        )

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                f"""SELECT id, properties, version FROM {self._schema}.ont_entities
                    WHERE entity_type = $1 AND name = $2 AND status = 'active' LIMIT 1""",
                entity_type, entity_name,
            )
            if existing:
                entity_id = existing["id"]
                old_props = (
                    json.loads(existing["properties"])
                    if isinstance(existing["properties"], str)
                    else (existing["properties"] or {})
                )
                merged = {**old_props, **properties}
                await conn.execute(
                    f"""UPDATE {self._schema}.ont_entities
                        SET properties = $1::jsonb, confidence = GREATEST(confidence, $2),
                            updated_at = NOW(), updated_by = $3, version = version + 1
                        WHERE id = $4""",
                    json.dumps(merged), confidence, agent_id, entity_id,
                )
                await self._emit_event("entity.updated", entity_type, entity_id, merged)
            else:
                row = await conn.fetchrow(
                    f"""INSERT INTO {self._schema}.ont_entities
                        (entity_type, name, properties, status, confidence, created_by)
                        VALUES ($1, $2, $3::jsonb, 'active', $4, $5) RETURNING id""",
                    entity_type, entity_name, json.dumps(properties), confidence, agent_id,
                )
                if row is None:
                    raise StorageError("Failed to create entity")
                entity_id = row["id"]
                await conn.execute(
                    f"""INSERT INTO {self._schema}.ont_provenance
                        (entity_id, source_type, source_file, created_by)
                        VALUES ($1, $2, $3, $4)""",
                    entity_id, "agent_report", source, agent_id,
                )
                await self._emit_event("entity.created", entity_type, entity_id, properties)
            return entity_id

    # ---- Write: assert_link ----

    async def assert_link(
        self,
        source_id: str,
        link_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
        confidence: float = 0.9,
    ) -> str:
        """Create a relationship between two entities."""
        pool = self._ensure_pool()
        agent_id = self._agent_id or "sdk"

        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                f"""SELECT id FROM {self._schema}.ont_links
                    WHERE link_type = $1 AND source_entity_id = $2
                      AND target_entity_id = $3 AND status = 'active'""",
                link_type, source_id, target_id,
            )
            if existing:
                link_id = existing["id"]
                if properties:
                    await conn.execute(
                        f"""UPDATE {self._schema}.ont_links
                            SET properties = $1::jsonb, confidence = GREATEST(confidence, $2),
                                updated_at = NOW() WHERE id = $3""",
                        json.dumps(properties), confidence, link_id,
                    )
                return link_id

            row = await conn.fetchrow(
                f"""INSERT INTO {self._schema}.ont_links
                    (link_type, source_entity_id, target_entity_id,
                     properties, confidence, created_by)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6) RETURNING id""",
                link_type, source_id, target_id,
                json.dumps(properties or {}), confidence, agent_id,
            )
            if row is None:
                raise StorageError("Failed to create link")
            link_id = row["id"]
            await self._emit_event("link.created", link_type, link_id, {
                "source_id": source_id, "target_id": target_id,
            })
            return link_id

    # ---- Write: ingest ----

    async def ingest(self, text: str, source: str, source_type: str = "document") -> IngestResult:
        """Store a document in Bronze layer. Full pipeline requires PipelineEngine."""
        import hashlib
        import time

        pool = self._ensure_pool()
        agent_id = self._agent_id or "sdk"
        t0 = time.monotonic()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""INSERT INTO {self._schema}.bronze_documents
                    (source_type, source_uri, source_hash, content, metadata, ingested_by)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                    ON CONFLICT (source_hash) DO UPDATE SET
                        source_uri = EXCLUDED.source_uri, ingested_by = EXCLUDED.ingested_by
                    RETURNING id""",
                source_type, source, content_hash, text,
                json.dumps({"agent_id": agent_id}), agent_id,
            )
            return IngestResult(
                document_id=row["id"] if row else "",
                processing_time_ms=int((time.monotonic() - t0) * 1000),
            )

    # ---- Read: query ----

    async def query(
        self, entity_type: str, filters: dict[str, Any] | None = None,
        limit: int = 20, offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query Gold entities by type and optional filters."""
        pool = self._ensure_pool()
        conditions = ["entity_type = $1", "status = 'active'"]
        params: list[Any] = [entity_type]

        if filters:
            for key, value in filters.items():
                params.append(str(value))
                idx = len(params)
                conditions.append(
                    f"(properties->>'{key}' = ${idx} OR ('{key}' = 'name' AND name = ${idx}))"
                )
        where = " AND ".join(conditions)
        params.extend([limit, offset])

        sql = f"""
            SELECT id, entity_type, name, properties, aliases,
                   confidence, version, created_at, updated_at, created_by
            FROM {self._schema}.ont_entities WHERE {where}
            ORDER BY updated_at DESC LIMIT ${len(params) - 1} OFFSET ${len(params)}
        """
        async with pool.acquire() as conn:
            return [self._row_to_dict(r) for r in await conn.fetch(sql, *params)]

    # ---- Read: search ----

    async def search(
        self, text: str, limit: int = 10, entity_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Text search over Gold entities using PG trigram similarity."""
        pool = self._ensure_pool()
        conditions = ["status = 'active'"]
        params: list[Any] = [text]
        if entity_type:
            params.append(entity_type)
            conditions.append(f"entity_type = ${len(params)}")
        where = " AND ".join(conditions)
        params.append(limit)

        sql = f"""
            SELECT id, entity_type, name, properties, aliases,
                   confidence, version, created_at, updated_at, created_by,
                   GREATEST(
                       similarity(name, $1),
                       similarity(COALESCE(properties->>'summary', ''), $1),
                       similarity(COALESCE(properties->>'description', ''), $1),
                       similarity(COALESCE(properties->>'task', ''), $1)
                   ) AS relevance
            FROM {self._schema}.ont_entities WHERE {where}
              AND (name % $1 OR properties->>'summary' % $1
                   OR properties->>'description' % $1 OR properties->>'task' % $1
                   OR name ILIKE '%' || $1 || '%')
            ORDER BY relevance DESC LIMIT ${len(params)}
        """
        async with pool.acquire() as conn:
            results = []
            for r in await conn.fetch(sql, *params):
                d = self._row_to_dict(r)
                d["relevance"] = float(r["relevance"])
                results.append(d)
            return results

    # ---- Read: get_entity ----

    async def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        pool = self._ensure_pool()
        return await self._get_entity_by_id(pool, entity_id)

    # ---- Read: get_linked (graph traversal) ----

    async def get_linked(
        self, entity_id: str, link_type: str | None = None,
        direction: str = "both", depth: int = 1,
    ) -> list[dict[str, Any]]:
        """Traverse the knowledge graph from an entity."""
        pool = self._ensure_pool()
        if depth < 1:
            return []
        results: list[dict[str, Any]] = []
        visited: set[str] = {entity_id}
        frontier: list[str] = [entity_id]

        for d in range(depth):
            if not frontier:
                break
            next_frontier: list[str] = []
            for eid in frontier:
                for link_data in await self._get_links_for_entity(pool, eid, link_type, direction):
                    nid = link_data["neighbor_id"]
                    if nid in visited:
                        continue
                    visited.add(nid)
                    next_frontier.append(nid)
                    entity = await self._get_entity_by_id(pool, nid)
                    if entity:
                        entity["_link"] = {
                            "link_id": link_data["link_id"],
                            "link_type": link_data["link_type"],
                            "direction": link_data["direction"],
                            "depth": d + 1,
                        }
                        results.append(entity)
            frontier = next_frontier
        return results

    # ---- Read: describe (schema introspection) ----

    async def describe(self, entity_type: str) -> DescribeResult:
        """Return schema definition for an entity type."""
        pool = self._ensure_pool()
        async with pool.acquire() as conn:
            type_def = await conn.fetchrow(
                f"""SELECT id, display_name, description, schema, required_fields
                    FROM {self._schema}.ont_type_definitions WHERE id = $1""",
                entity_type,
            )
            link_rows = await conn.fetch(
                f"""SELECT DISTINCT link_type, 'outgoing' as direction
                    FROM {self._schema}.ont_links l
                    JOIN {self._schema}.ont_entities e ON l.source_entity_id = e.id
                    WHERE e.entity_type = $1 AND l.status = 'active'
                    UNION
                    SELECT DISTINCT link_type, 'incoming' as direction
                    FROM {self._schema}.ont_links l
                    JOIN {self._schema}.ont_entities e ON l.target_entity_id = e.id
                    WHERE e.entity_type = $1 AND l.status = 'active'""",
                entity_type,
            )
            examples = await conn.fetch(
                f"""SELECT id, name, properties FROM {self._schema}.ont_entities
                    WHERE entity_type = $1 AND status = 'active'
                    ORDER BY created_at DESC LIMIT 3""",
                entity_type,
            )

        properties: list[dict[str, Any]] = []
        required_fields: list[str] = []
        description = ""

        if type_def:
            description = type_def["description"] or ""
            required_fields = list(type_def.get("required_fields") or [])
            schema_json = type_def.get("schema")
            if schema_json:
                if isinstance(schema_json, str):
                    schema_json = json.loads(schema_json)
                if isinstance(schema_json, dict):
                    for pname, pdef in schema_json.items():
                        properties.append({
                            "name": pname, "type": pdef.get("type", "string"),
                            "required": pname in required_fields,
                            "description": pdef.get("description", ""),
                            "enum_values": pdef.get("enum_values"),
                        })

        if not properties and examples:
            seen: set[str] = set()
            for ex in examples:
                props = ex["properties"]
                if isinstance(props, str):
                    props = json.loads(props)
                if isinstance(props, dict):
                    for k in props:
                        if k not in seen:
                            seen.add(k)
                            properties.append({
                                "name": k, "type": "string",
                                "required": k in required_fields,
                                "description": "", "enum_values": None,
                            })

        example_dicts = []
        for ex in examples:
            props = ex["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            example_dicts.append({"id": ex["id"], "name": ex["name"], "properties": props})

        return DescribeResult(
            entity_type=entity_type, description=description,
            properties=properties, required_fields=required_fields,
            link_types=[{"link_type": r["link_type"], "direction": r["direction"]} for r in link_rows],
            examples=example_dicts,
        )

    # ---- Kinetic Layer: Action Execution ----

    async def execute_action(
        self,
        action_name: str,
        params: dict[str, Any],
        actor: str | None = None,
    ) -> dict[str, Any]:
        """Execute a registered Kinetic Layer action.

        This is a convenience wrapper that:
        1. Creates (or reuses) an ActionExecutor with an in-memory AuditTrail.
        2. Validates the params and runs the handler.
        3. Emits action.completed / action.failed events.
        4. Returns the ActionResult as a dict.

        For advanced usage (custom audit backends, rollback, etc.), use
        :class:`ontology_engine.kinetic.ActionExecutor` directly.

        Args:
            action_name: Registered action type name.
            params: Input parameters (validated against JSON Schema).
            actor: Override for the actor identity (defaults to the registered agent id).

        Returns:
            Dict with keys: execution_id, action_name, status, result, error, duration_ms.
        """
        from ontology_engine.kinetic.action_executor import ActionExecutor, ExecutionContext

        executor = self._get_executor()
        ctx = ExecutionContext(actor=actor or self._agent_id or "sdk")
        result = await executor.execute(action_name, params, ctx)

        # Emit events
        event_type = "action.completed" if result.status == "success" else "action.failed"
        await self._emit_event(event_type, action_name, result.execution_id, {
            "action_name": result.action_name,
            "status": result.status,
            "actor": ctx.actor,
            "duration_ms": result.duration_ms,
        })

        return {
            "execution_id": result.execution_id,
            "action_name": result.action_name,
            "status": result.status,
            "result": result.result,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }

    async def list_actions(self) -> list[dict[str, Any]]:
        """List all registered action types.

        Returns a list of dicts with action type metadata.
        """
        from ontology_engine.kinetic.action_types import ActionType

        registry = self._get_registry()
        return [
            {
                "name": a.name,
                "description": a.description,
                "idempotent": a.idempotent,
                "reversible": a.reversible,
                "preconditions": a.preconditions,
                "side_effects": a.side_effects,
            }
            for a in registry.list()
        ]

    async def get_audit_trail(
        self,
        entity_id: str | None = None,
        action_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Query the audit trail.

        Args:
            entity_id: If provided, returns lineage for this entity.
            action_name: Filter by action name.
            limit: Max results.

        Returns:
            List of audit entry dicts.
        """
        audit = self._get_audit()
        if entity_id:
            entries = audit.get_lineage(entity_id)[:limit]
        else:
            filters: dict[str, Any] = {"limit": limit}
            if action_name:
                filters["action_name"] = action_name
            entries = audit.query(filters)
        return [e.to_dict() for e in entries]

    def _get_registry(self) -> "ActionRegistry":
        """Return the shared ActionRegistry, creating one if needed."""
        from ontology_engine.kinetic.action_types import ActionRegistry

        if not hasattr(self, "_action_registry") or self._action_registry is None:
            self._action_registry: ActionRegistry = ActionRegistry()
        return self._action_registry

    def _get_audit(self) -> "AuditTrail":
        """Return the shared AuditTrail, creating one if needed."""
        from ontology_engine.kinetic.audit_trail import AuditTrail

        if not hasattr(self, "_audit_trail") or self._audit_trail is None:
            self._audit_trail: AuditTrail = AuditTrail()
        return self._audit_trail

    def _get_executor(self) -> "ActionExecutor":
        """Return the shared ActionExecutor, creating one if needed."""
        from ontology_engine.kinetic.action_executor import ActionExecutor

        if not hasattr(self, "_action_executor") or self._action_executor is None:
            self._action_executor: ActionExecutor = ActionExecutor(
                self._get_registry(), self._get_audit()
            )
        return self._action_executor

    # ---- Subscribe ----

    async def _ensure_notifier(self) -> EventNotifier:
        if self._notifier is None:
            self._notifier = EventNotifier(self._ensure_pool())
        return self._notifier

    async def subscribe(self, entity_type: str, callback: EventCallback) -> None:
        """Subscribe to entity change events."""
        await (await self._ensure_notifier()).subscribe(entity_type, callback)

    async def _emit_event(
        self, event_type: str, entity_type: str, entity_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            notifier = await self._ensure_notifier()
            await notifier.emit(
                event_type=event_type, entity_type=entity_type,
                entity_id=entity_id, source_agent=self._agent_id or "sdk",
                payload=payload,
            )
        except Exception:
            logger.debug("Failed to emit event %s", event_type, exc_info=True)

    # ---- Internal helpers ----

    async def _get_links_for_entity(
        self, pool: asyncpg.Pool, entity_id: str,
        link_type: str | None, direction: str,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        async with pool.acquire() as conn:
            if direction in ("outgoing", "both"):
                conds = ["source_entity_id = $1", "status = 'active'"]
                params: list[Any] = [entity_id]
                if link_type:
                    params.append(link_type)
                    conds.append(f"link_type = ${len(params)}")
                for r in await conn.fetch(
                    f"SELECT id, link_type, target_entity_id FROM {self._schema}.ont_links WHERE {' AND '.join(conds)}",
                    *params,
                ):
                    results.append({"link_id": r["id"], "link_type": r["link_type"],
                                    "neighbor_id": r["target_entity_id"], "direction": "outgoing"})
            if direction in ("incoming", "both"):
                conds = ["target_entity_id = $1", "status = 'active'"]
                params = [entity_id]
                if link_type:
                    params.append(link_type)
                    conds.append(f"link_type = ${len(params)}")
                for r in await conn.fetch(
                    f"SELECT id, link_type, source_entity_id FROM {self._schema}.ont_links WHERE {' AND '.join(conds)}",
                    *params,
                ):
                    results.append({"link_id": r["id"], "link_type": r["link_type"],
                                    "neighbor_id": r["source_entity_id"], "direction": "incoming"})
        return results

    async def _get_entity_by_id(self, pool: asyncpg.Pool, entity_id: str) -> dict[str, Any] | None:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""SELECT id, entity_type, name, properties, aliases,
                       confidence, version, created_at, updated_at, created_by
                    FROM {self._schema}.ont_entities WHERE id = $1 AND status != 'deleted'""",
                entity_id,
            )
            return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for key in row.keys():
            val = row[key]
            if isinstance(val, datetime):
                d[key] = val.isoformat()
            elif key == "properties":
                d[key] = json.loads(val) if isinstance(val, str) else (val or {})
            elif key == "aliases":
                d[key] = list(val or [])
            else:
                d[key] = val
        return d
