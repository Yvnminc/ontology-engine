"""Gold Layer Query API — read-only queries on the Gold layer.

Provides:
  - query(): filter by entity_type, status, properties
  - search(): vector similarity search via pgvector
  - get_linked(): graph traversal via recursive CTE (up to N hops)
  - stats(): Gold layer statistics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GoldEntity:
    """A Gold-layer entity returned by queries."""

    id: str
    entity_type: str
    canonical_name: str
    properties: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    silver_entity_ids: list[str] = field(default_factory=list)
    source_count: int = 1
    confidence: float = 1.0
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_seen_at: datetime | None = None
    # Optional: similarity score from vector search
    similarity: float | None = None


@dataclass
class GoldLink:
    """A Gold-layer link returned by queries."""

    id: str
    link_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    silver_link_ids: list[str] = field(default_factory=list)
    mention_count: int = 1
    confidence: float = 1.0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    status: str = "active"


@dataclass
class LinkedEntity:
    """An entity found via graph traversal, with path info."""

    entity: GoldEntity
    link: GoldLink
    depth: int
    direction: str  # "outgoing" or "incoming"


@dataclass
class GoldStats:
    """Gold layer statistics."""

    total_entities: int = 0
    total_links: int = 0
    entities_by_type: dict[str, int] = field(default_factory=dict)
    links_by_type: dict[str, int] = field(default_factory=dict)
    avg_source_count: float = 0.0
    entities_with_embeddings: int = 0
    review_pending: int = 0


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class GoldRepository:
    """Read-only query interface for the Gold layer."""

    def __init__(self, pool: asyncpg.Pool, schema: str = "ontology"):
        self._pool = pool
        self._schema = schema

    @classmethod
    async def create(cls, db_url: str, schema: str = "ontology") -> GoldRepository:
        """Create a repository with a connection pool."""
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        async with pool.acquire() as conn:
            await conn.execute(f"SET search_path TO {schema}, public")
        return cls(pool, schema)

    async def close(self) -> None:
        await self._pool.close()

    # =========================================================================
    # Query
    # =========================================================================

    async def query(
        self,
        entity_type: str | None = None,
        status: str = "active",
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[GoldEntity]:
        """Query Gold entities by type, status, and optional property filters.

        Args:
            entity_type: Filter by entity type (e.g. "Person", "Decision").
            status: Filter by status (default "active").
            filters: JSONB property filters, e.g. {"role": "CEO"}.
            limit: Max results.
            offset: Pagination offset.
        """
        conditions = [f"status = $1"]
        params: list[Any] = [status]

        if entity_type:
            params.append(entity_type)
            conditions.append(f"entity_type = ${len(params)}")

        if filters:
            params.append(json.dumps(filters))
            conditions.append(f"properties @> ${len(params)}::jsonb")

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        sql = f"""
            SELECT * FROM {self._schema}.gold_entities
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ${len(params) - 1} OFFSET ${len(params)}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [self._row_to_entity(r) for r in rows]

    # =========================================================================
    # Vector Search
    # =========================================================================

    async def search(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> list[GoldEntity]:
        """Search Gold entities by vector similarity.

        Provide either `text` (will generate embedding) or a pre-computed `embedding`.
        Falls back to trigram text search if no embeddings are available.
        """
        if embedding:
            return await self._vector_search(embedding, limit, min_similarity)

        if text:
            # Try vector search first (requires embedding generation)
            try:
                from ontology_engine.fusion.embeddings import EmbeddingGenerator

                gen = EmbeddingGenerator()
                if gen.available:
                    emb = await gen.embed_single(text)
                    results = await self._vector_search(emb, limit, min_similarity)
                    if results:
                        return results
            except Exception as e:
                logger.debug("Vector search failed, falling back to text: %s", e)

            # Fallback: trigram text search
            return await self._text_search(text, limit)

        return []

    async def _vector_search(
        self,
        embedding: list[float],
        limit: int,
        min_similarity: float,
    ) -> list[GoldEntity]:
        """Cosine similarity search using pgvector."""
        emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
        sql = f"""
            SELECT *,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM {self._schema}.gold_entities
            WHERE status = 'active'
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> $1::vector) >= $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, emb_str, min_similarity, limit)
        return [self._row_to_entity(r, similarity=r.get("similarity")) for r in rows]

    async def _text_search(self, text: str, limit: int) -> list[GoldEntity]:
        """Trigram-based text search (fallback when embeddings not available)."""
        sql = f"""
            SELECT *,
                   similarity(canonical_name, $1) AS sim
            FROM {self._schema}.gold_entities
            WHERE status = 'active'
              AND (
                  canonical_name % $1
                  OR $1 = ANY(aliases)
                  OR canonical_name ILIKE '%' || $1 || '%'
              )
            ORDER BY sim DESC
            LIMIT $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, text, limit)
        return [self._row_to_entity(r, similarity=r.get("sim")) for r in rows]

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    async def get_linked(
        self,
        entity_id: str,
        link_type: str | None = None,
        direction: str = "both",
        depth: int = 1,
    ) -> list[LinkedEntity]:
        """Get linked entities via graph traversal (recursive CTE).

        Args:
            entity_id: Starting Gold entity ID.
            link_type: Optional link type filter.
            direction: "outgoing", "incoming", or "both".
            depth: Max hops (1 = direct neighbors, 2 = 2-hop, etc.).
        """
        # Build direction filter
        if direction == "outgoing":
            anchor_join = "l.source_id = $1"
            next_col = "target_id"
            prev_col = "source_id"
        elif direction == "incoming":
            anchor_join = "l.target_id = $1"
            next_col = "source_id"
            prev_col = "target_id"
        else:
            # "both" — we union outgoing and incoming
            anchor_join = "(l.source_id = $1 OR l.target_id = $1)"
            next_col = None  # handled specially
            prev_col = None

        link_filter = ""
        params: list[Any] = [entity_id, depth]
        if link_type:
            params.append(link_type)
            link_filter = f"AND l.link_type = ${len(params)}"

        if direction in ("outgoing", "incoming"):
            assert next_col and prev_col
            sql = f"""
                WITH RECURSIVE traverse AS (
                    -- Anchor: direct neighbors
                    SELECT
                        l.id AS link_id,
                        l.link_type,
                        l.source_id,
                        l.target_id,
                        l.properties AS link_properties,
                        l.silver_link_ids,
                        l.mention_count,
                        l.confidence AS link_confidence,
                        l.first_seen,
                        l.last_seen,
                        l.{next_col} AS neighbor_id,
                        '{direction}' AS direction,
                        1 AS depth
                    FROM {self._schema}.gold_links l
                    WHERE {anchor_join}
                      AND l.status = 'active'
                      {link_filter}

                    UNION ALL

                    -- Recursive: follow links from previously found neighbors
                    SELECT
                        l.id,
                        l.link_type,
                        l.source_id,
                        l.target_id,
                        l.properties,
                        l.silver_link_ids,
                        l.mention_count,
                        l.confidence,
                        l.first_seen,
                        l.last_seen,
                        l.{next_col},
                        '{direction}',
                        t.depth + 1
                    FROM {self._schema}.gold_links l
                    JOIN traverse t ON l.{prev_col} = t.neighbor_id
                    WHERE t.depth < $2
                      AND l.status = 'active'
                      AND l.{next_col} != $1
                      {link_filter}
                )
                SELECT DISTINCT ON (t.neighbor_id, t.link_id)
                    t.*,
                    e.id AS ent_id,
                    e.entity_type,
                    e.canonical_name,
                    e.properties AS ent_properties,
                    e.aliases,
                    e.silver_entity_ids,
                    e.source_count,
                    e.confidence AS ent_confidence,
                    e.status AS ent_status,
                    e.created_at AS ent_created_at,
                    e.updated_at AS ent_updated_at,
                    e.last_seen_at AS ent_last_seen_at
                FROM traverse t
                JOIN {self._schema}.gold_entities e ON e.id = t.neighbor_id
                WHERE e.status = 'active'
                ORDER BY t.neighbor_id, t.link_id, t.depth
            """
        else:
            # Both directions
            sql = f"""
                WITH RECURSIVE traverse AS (
                    SELECT
                        l.id AS link_id,
                        l.link_type,
                        l.source_id,
                        l.target_id,
                        l.properties AS link_properties,
                        l.silver_link_ids,
                        l.mention_count,
                        l.confidence AS link_confidence,
                        l.first_seen,
                        l.last_seen,
                        CASE WHEN l.source_id = $1 THEN l.target_id
                             ELSE l.source_id END AS neighbor_id,
                        CASE WHEN l.source_id = $1 THEN 'outgoing'
                             ELSE 'incoming' END AS direction,
                        1 AS depth
                    FROM {self._schema}.gold_links l
                    WHERE {anchor_join}
                      AND l.status = 'active'
                      {link_filter}

                    UNION ALL

                    SELECT
                        l.id,
                        l.link_type,
                        l.source_id,
                        l.target_id,
                        l.properties,
                        l.silver_link_ids,
                        l.mention_count,
                        l.confidence,
                        l.first_seen,
                        l.last_seen,
                        CASE WHEN l.source_id = t.neighbor_id THEN l.target_id
                             ELSE l.source_id END,
                        CASE WHEN l.source_id = t.neighbor_id THEN 'outgoing'
                             ELSE 'incoming' END,
                        t.depth + 1
                    FROM {self._schema}.gold_links l
                    JOIN traverse t ON (l.source_id = t.neighbor_id OR l.target_id = t.neighbor_id)
                    WHERE t.depth < $2
                      AND l.status = 'active'
                      AND CASE WHEN l.source_id = t.neighbor_id THEN l.target_id
                               ELSE l.source_id END != $1
                      {link_filter}
                )
                SELECT DISTINCT ON (t.neighbor_id, t.link_id)
                    t.*,
                    e.id AS ent_id,
                    e.entity_type,
                    e.canonical_name,
                    e.properties AS ent_properties,
                    e.aliases,
                    e.silver_entity_ids,
                    e.source_count,
                    e.confidence AS ent_confidence,
                    e.status AS ent_status,
                    e.created_at AS ent_created_at,
                    e.updated_at AS ent_updated_at,
                    e.last_seen_at AS ent_last_seen_at
                FROM traverse t
                JOIN {self._schema}.gold_entities e ON e.id = t.neighbor_id
                WHERE e.status = 'active'
                ORDER BY t.neighbor_id, t.link_id, t.depth
            """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        results: list[LinkedEntity] = []
        for r in rows:
            entity = GoldEntity(
                id=r["ent_id"],
                entity_type=r["entity_type"],
                canonical_name=r["canonical_name"],
                properties=(
                    json.loads(r["ent_properties"])
                    if isinstance(r["ent_properties"], str)
                    else dict(r["ent_properties"] or {})
                ),
                aliases=list(r.get("aliases") or []),
                silver_entity_ids=list(r.get("silver_entity_ids") or []),
                source_count=r.get("source_count", 1),
                confidence=r.get("ent_confidence", 1.0),
                status=r.get("ent_status", "active"),
                created_at=r.get("ent_created_at"),
                updated_at=r.get("ent_updated_at"),
                last_seen_at=r.get("ent_last_seen_at"),
            )
            link = GoldLink(
                id=r["link_id"],
                link_type=r["link_type"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                properties=(
                    json.loads(r["link_properties"])
                    if isinstance(r["link_properties"], str)
                    else dict(r["link_properties"] or {})
                ),
                silver_link_ids=list(r.get("silver_link_ids") or []),
                mention_count=r.get("mention_count", 1),
                confidence=r.get("link_confidence", 1.0),
                first_seen=r.get("first_seen"),
                last_seen=r.get("last_seen"),
            )
            results.append(
                LinkedEntity(
                    entity=entity,
                    link=link,
                    depth=r["depth"],
                    direction=r["direction"],
                )
            )

        return results

    # =========================================================================
    # Stats
    # =========================================================================

    async def stats(self) -> GoldStats:
        """Get Gold layer statistics."""
        async with self._pool.acquire() as conn:
            # Totals
            totals = await conn.fetchrow(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM {self._schema}.gold_entities
                     WHERE status = 'active') AS total_entities,
                    (SELECT COUNT(*) FROM {self._schema}.gold_links
                     WHERE status = 'active') AS total_links,
                    (SELECT COALESCE(AVG(source_count), 0)
                     FROM {self._schema}.gold_entities
                     WHERE status = 'active') AS avg_source_count,
                    (SELECT COUNT(*) FROM {self._schema}.gold_entities
                     WHERE status = 'active' AND embedding IS NOT NULL
                    ) AS entities_with_embeddings
                """
            )

            # By type — entities
            etype_rows = await conn.fetch(
                f"""
                SELECT entity_type, COUNT(*) AS cnt
                FROM {self._schema}.gold_entities
                WHERE status = 'active'
                GROUP BY entity_type
                ORDER BY cnt DESC
                """
            )

            # By type — links
            ltype_rows = await conn.fetch(
                f"""
                SELECT link_type, COUNT(*) AS cnt
                FROM {self._schema}.gold_links
                WHERE status = 'active'
                GROUP BY link_type
                ORDER BY cnt DESC
                """
            )

        return GoldStats(
            total_entities=totals["total_entities"],
            total_links=totals["total_links"],
            entities_by_type={r["entity_type"]: r["cnt"] for r in etype_rows},
            links_by_type={r["link_type"]: r["cnt"] for r in ltype_rows},
            avg_source_count=float(totals["avg_source_count"]),
            entities_with_embeddings=totals["entities_with_embeddings"],
        )

    # =========================================================================
    # Get single entity
    # =========================================================================

    async def get(self, entity_id: str) -> GoldEntity | None:
        """Fetch a single Gold entity by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self._schema}.gold_entities WHERE id = $1",
                entity_id,
            )
        if row is None:
            return None
        return self._row_to_entity(row)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _row_to_entity(
        row: asyncpg.Record, similarity: float | None = None
    ) -> GoldEntity:
        props = row["properties"]
        if isinstance(props, str):
            props = json.loads(props)
        return GoldEntity(
            id=row["id"],
            entity_type=row["entity_type"],
            canonical_name=row["canonical_name"],
            properties=dict(props or {}),
            aliases=list(row.get("aliases") or []),
            silver_entity_ids=list(row.get("silver_entity_ids") or []),
            source_count=row.get("source_count", 1),
            confidence=row.get("confidence", 1.0),
            status=row.get("status", "active"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            last_seen_at=row.get("last_seen_at"),
            similarity=similarity,
        )
