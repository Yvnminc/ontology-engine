"""CRUD operations for the ontology graph — PostgreSQL via asyncpg."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg

from ontology_engine.core.errors import StorageError
from ontology_engine.core.types import (
    Entity,
    EntityStatus,
    EntityType,
    Link,
    LinkType,
    Provenance,
)


class OntologyRepository:
    """Async repository for ontology entities, links, and provenance."""

    def __init__(self, pool: asyncpg.Pool, schema: str = "ontology"):
        self._pool = pool
        self._schema = schema

    @classmethod
    async def create(cls, db_url: str, schema: str = "ontology") -> OntologyRepository:
        """Create a repository with a connection pool."""
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        repo = cls(pool, schema)
        await repo._set_search_path()
        return repo

    async def _set_search_path(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(f"SET search_path TO {self._schema}, public")

    async def close(self) -> None:
        await self._pool.close()

    # =========================================================================
    # Entity CRUD
    # =========================================================================

    async def create_entity(self, entity: Entity) -> str:
        """Insert a new entity and return its ID."""
        sql = f"""
            INSERT INTO {self._schema}.ont_entities
                (entity_type, name, properties, aliases, status, confidence, created_by)
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7)
            RETURNING id
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                entity.entity_type.value,
                entity.name,
                json.dumps(entity.properties),
                entity.aliases,
                entity.status.value,
                entity.confidence,
                entity.created_by,
            )
            if row is None:
                raise StorageError("Failed to create entity")
            return row["id"]

    async def get_entity(self, entity_id: str) -> Entity | None:
        """Fetch a single entity by ID."""
        sql = f"""
            SELECT * FROM {self._schema}.ont_entities
            WHERE id = $1 AND status != 'deleted'
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, entity_id)
            if row is None:
                return None
            return self._row_to_entity(row)

    async def find_entity_by_name(
        self, name: str, entity_type: EntityType | None = None
    ) -> list[Entity]:
        """Find entities by name (fuzzy match via pg_trgm) or alias."""
        conditions = ["status = 'active'"]
        params: list[Any] = []

        # Fuzzy name match OR alias match
        params.append(name)
        conditions.append(f"(name % ${len(params)} OR ${len(params)} = ANY(aliases))")

        if entity_type:
            params.append(entity_type.value)
            conditions.append(f"entity_type = ${len(params)}")

        where = " AND ".join(conditions)
        sql = f"""
            SELECT *, similarity(name, $1) AS sim
            FROM {self._schema}.ont_entities
            WHERE {where}
            ORDER BY sim DESC
            LIMIT 10
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_entity(r) for r in rows]

    async def update_entity(
        self, entity_id: str, updates: dict[str, Any], updated_by: str = "system"
    ) -> bool:
        """Update entity fields. Creates a version record."""
        # Get current state for version history
        current = await self.get_entity(entity_id)
        if current is None:
            return False

        set_clauses = []
        params: list[Any] = [entity_id]

        if "name" in updates:
            params.append(updates["name"])
            set_clauses.append(f"name = ${len(params)}")
        if "properties" in updates:
            params.append(json.dumps(updates["properties"]))
            set_clauses.append(f"properties = ${len(params)}::jsonb")
        if "aliases" in updates:
            params.append(updates["aliases"])
            set_clauses.append(f"aliases = ${len(params)}")
        if "status" in updates:
            params.append(updates["status"])
            set_clauses.append(f"status = ${len(params)}")
        if "confidence" in updates:
            params.append(updates["confidence"])
            set_clauses.append(f"confidence = ${len(params)}")

        params.append(updated_by)
        set_clauses.append(f"updated_by = ${len(params)}")
        set_clauses.append("updated_at = NOW()")
        set_clauses.append("version = version + 1")

        sql = f"""
            UPDATE {self._schema}.ont_entities
            SET {', '.join(set_clauses)}
            WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql, *params)
                # Record version
                await conn.execute(
                    f"""
                    INSERT INTO {self._schema}.ont_entity_versions
                        (entity_id, version, change_type, old_values, new_values, changed_by)
                    VALUES ($1, $2, 'updated', $3::jsonb, $4::jsonb, $5)
                    """,
                    entity_id,
                    current.version + 1,
                    json.dumps({"name": current.name, "properties": current.properties}),
                    json.dumps(updates),
                    updated_by,
                )
        return True

    async def list_entities(
        self,
        entity_type: EntityType | None = None,
        status: EntityStatus = EntityStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entity]:
        """List entities with optional filters."""
        conditions = [f"status = '{status.value}'"]
        params: list[Any] = []

        if entity_type:
            params.append(entity_type.value)
            conditions.append(f"entity_type = ${len(params)}")

        where = " AND ".join(conditions)
        params.extend([limit, offset])
        sql = f"""
            SELECT * FROM {self._schema}.ont_entities
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ${len(params) - 1} OFFSET ${len(params)}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_entity(r) for r in rows]

    # =========================================================================
    # Link CRUD
    # =========================================================================

    async def create_link(self, link: Link) -> str:
        """Insert a new relationship."""
        sql = f"""
            INSERT INTO {self._schema}.ont_links
                (link_type, source_entity_id, target_entity_id, properties,
                 confidence, valid_from, valid_to, created_by)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8)
            RETURNING id
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                link.link_type.value,
                link.source_entity_id,
                link.target_entity_id,
                json.dumps(link.properties),
                link.confidence,
                link.valid_from,
                link.valid_to,
                link.created_by,
            )
            if row is None:
                raise StorageError("Failed to create link")
            return row["id"]

    async def get_links(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        link_type: LinkType | None = None,
    ) -> list[Link]:
        """Get all links for an entity."""
        conditions = ["status = 'active'"]
        params: list[Any] = [entity_id]

        if direction == "outgoing":
            conditions.append("source_entity_id = $1")
        elif direction == "incoming":
            conditions.append("target_entity_id = $1")
        else:
            conditions.append("(source_entity_id = $1 OR target_entity_id = $1)")

        if link_type:
            params.append(link_type.value)
            conditions.append(f"link_type = ${len(params)}")

        where = " AND ".join(conditions)
        sql = f"SELECT * FROM {self._schema}.ont_links WHERE {where}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_link(r) for r in rows]

    # =========================================================================
    # Provenance
    # =========================================================================

    async def create_provenance(self, prov: Provenance) -> str:
        """Record provenance for an entity or link."""
        sql = f"""
            INSERT INTO {self._schema}.ont_provenance
                (entity_id, link_id, source_type, source_file, source_meeting_date,
                 source_participants, source_segment, extraction_model,
                 extraction_pass, raw_extraction, created_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11)
            RETURNING id
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                prov.entity_id,
                prov.link_id,
                prov.source_type,
                prov.source_file,
                prov.source_meeting_date,
                prov.source_participants,
                prov.source_segment,
                prov.extraction_model,
                prov.extraction_pass,
                json.dumps(prov.raw_extraction) if prov.raw_extraction else None,
                prov.created_by,
            )
            if row is None:
                raise StorageError("Failed to create provenance")
            return row["id"]

    # =========================================================================
    # Query Helpers (for agents)
    # =========================================================================

    async def query_decisions(
        self,
        keyword: str | None = None,
        made_by: str | None = None,
        status: str = "active",
        limit: int = 20,
    ) -> list[Entity]:
        """Query decisions — primary interface for agents."""
        conditions = ["entity_type = 'Decision'", f"e.status = '{status}'"]
        params: list[Any] = []

        if keyword:
            params.append(f"%{keyword}%")
            conditions.append(
                f"(e.name ILIKE ${len(params)} OR e.properties->>'detail' ILIKE ${len(params)})"
            )
        if made_by:
            params.append(made_by)
            conditions.append(
                f"EXISTS (SELECT 1 FROM {self._schema}.ont_links l "
                f"WHERE l.link_type = 'makes' AND l.target_entity_id = e.id "
                f"AND l.source_entity_id IN "
                f"(SELECT id FROM {self._schema}.ont_entities WHERE name = ${len(params)}))"
            )

        where = " AND ".join(conditions)
        sql = f"""
            SELECT e.* FROM {self._schema}.ont_entities e
            WHERE {where}
            ORDER BY e.created_at DESC
            LIMIT {limit}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_entity(r) for r in rows]

    async def query_action_items(
        self,
        owner: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Query action items — primary interface for agents."""
        conditions = ["entity_type = 'ActionItem'", "e.status = 'active'"]
        params: list[Any] = []

        if status:
            params.append(status)
            conditions.append(f"e.properties->>'status' = ${len(params)}")
        if owner:
            params.append(owner)
            conditions.append(
                f"EXISTS (SELECT 1 FROM {self._schema}.ont_links l "
                f"WHERE l.link_type = 'assigned_to' AND l.source_entity_id = e.id "
                f"AND l.target_entity_id IN "
                f"(SELECT id FROM {self._schema}.ont_entities WHERE name = ${len(params)}))"
            )

        where = " AND ".join(conditions)
        sql = f"""
            SELECT e.* FROM {self._schema}.ont_entities e
            WHERE {where}
            ORDER BY e.created_at DESC
            LIMIT {limit}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_entity(r) for r in rows]

    # =========================================================================
    # Stats
    # =========================================================================

    async def stats(self) -> dict[str, int]:
        """Get entity/link counts by type."""
        sql = f"""
            SELECT
                (SELECT COUNT(*) FROM {self._schema}.ont_entities WHERE status = 'active') as entities,
                (SELECT COUNT(*) FROM {self._schema}.ont_links WHERE status = 'active') as links,
                (SELECT COUNT(*) FROM {self._schema}.ont_provenance) as provenances,
                (SELECT COUNT(*) FROM {self._schema}.ont_conflicts WHERE status = 'open') as open_conflicts
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql)
            return dict(row) if row else {}

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _row_to_entity(row: asyncpg.Record) -> Entity:
        return Entity(
            id=row["id"],
            entity_type=EntityType(row["entity_type"]),
            name=row["name"],
            properties=json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"],
            aliases=list(row.get("aliases") or []),
            status=EntityStatus(row["status"]),
            confidence=row["confidence"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row.get("created_by", "system"),
        )

    @staticmethod
    def _row_to_link(row: asyncpg.Record) -> Link:
        return Link(
            id=row["id"],
            link_type=LinkType(row["link_type"]),
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            properties=json.loads(row["properties"]) if isinstance(row["properties"], str) else row["properties"],
            confidence=row["confidence"],
            status=EntityStatus(row["status"]),
            valid_from=row.get("valid_from"),
            valid_to=row.get("valid_to"),
            created_at=row["created_at"],
            created_by=row.get("created_by", "system"),
        )
