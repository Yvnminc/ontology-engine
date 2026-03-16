"""Gold Aggregation Pipeline — build Gold layer from Silver.

The Gold layer is a *computed view* — it can always be fully rebuilt from Silver.

Modes:
  - Full rebuild: clear Gold → process all Silver entities/links
  - Incremental:  process only Silver entities updated since last Gold build

Pipeline:
  1. Fetch Silver entities (all or incremental)
  2. Run entity resolution → merge duplicates
  3. Upsert Gold entities
  4. Aggregate Silver links → Gold links (deduplicate, count mentions)
  5. Optionally generate embeddings
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg

from ontology_engine.core.config import OntologyConfig
from ontology_engine.fusion.embeddings import EmbeddingGenerator
from ontology_engine.fusion.entity_resolver import (
    EntityResolver,
    GoldEntityCandidate,
    MergeCandidate,
    SilverEntity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build result
# ---------------------------------------------------------------------------


class GoldBuildResult:
    """Summary of a Gold build run."""

    def __init__(self) -> None:
        self.gold_entities_created: int = 0
        self.gold_entities_updated: int = 0
        self.gold_links_created: int = 0
        self.gold_links_updated: int = 0
        self.review_candidates: list[MergeCandidate] = []
        self.embeddings_generated: int = 0
        self.errors: list[str] = []

    def summary(self) -> dict[str, Any]:
        return {
            "gold_entities_created": self.gold_entities_created,
            "gold_entities_updated": self.gold_entities_updated,
            "gold_links_created": self.gold_links_created,
            "gold_links_updated": self.gold_links_updated,
            "review_candidates": len(self.review_candidates),
            "embeddings_generated": self.embeddings_generated,
            "errors": len(self.errors),
        }


# ---------------------------------------------------------------------------
# Gold Builder
# ---------------------------------------------------------------------------


class GoldBuilder:
    """Builds the Gold layer from Silver data."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        config: OntologyConfig | None = None,
        schema: str = "ontology",
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        self._pool = pool
        self._config = config or OntologyConfig()
        self._schema = schema
        self._resolver = EntityResolver(self._config)
        self._embeddings = embedding_generator

    @classmethod
    async def create(
        cls,
        db_url: str,
        config: OntologyConfig | None = None,
        schema: str = "ontology",
    ) -> GoldBuilder:
        """Create a GoldBuilder with a connection pool."""
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        # Set search path
        async with pool.acquire() as conn:
            await conn.execute(f"SET search_path TO {schema}, public")

        emb = EmbeddingGenerator()
        if not emb.available:
            logger.info("No OPENAI_API_KEY — embeddings will be skipped")
            emb = None

        return cls(pool, config, schema, emb)

    async def build_gold(self, full: bool = False) -> GoldBuildResult:
        """Build or incrementally update the Gold layer.

        Args:
            full: If True, clear Gold tables and rebuild from scratch.
                  If False, only process new/updated Silver entities.
        """
        result = GoldBuildResult()

        if full:
            logger.info("Starting full Gold rebuild...")
            await self._clear_gold()

        # Step 1: Fetch Silver entities
        silver_entities = await self._fetch_silver_entities(full=full)
        logger.info("Fetched %d Silver entities", len(silver_entities))

        if not silver_entities:
            logger.info("No Silver entities to process")
            return result

        # Step 2: Entity resolution
        gold_candidates, reviews = self._resolver.resolve(silver_entities)
        result.review_candidates = reviews
        logger.info(
            "Resolved to %d Gold candidates, %d need review",
            len(gold_candidates), len(reviews),
        )

        # Step 3: Upsert Gold entities
        gold_id_map = await self._upsert_gold_entities(gold_candidates, result)

        # Step 4: Aggregate Silver links → Gold links
        await self._aggregate_links(gold_id_map, result)

        # Step 5: Generate embeddings (if available)
        if self._embeddings and self._embeddings.available:
            await self._generate_embeddings(result)

        logger.info("Gold build complete: %s", result.summary())
        return result

    # =========================================================================
    # Internal steps
    # =========================================================================

    async def _clear_gold(self) -> None:
        """Drop all Gold data for full rebuild."""
        async with self._pool.acquire() as conn:
            await conn.execute(f"DELETE FROM {self._schema}.gold_links")
            await conn.execute(f"DELETE FROM {self._schema}.gold_entities")
        logger.info("Cleared Gold tables")

    async def _fetch_silver_entities(
        self, full: bool = False
    ) -> list[SilverEntity]:
        """Fetch Silver entities (from ont_entities)."""
        sql = f"""
            SELECT id, entity_type, name, properties, aliases,
                   confidence, created_at, updated_at
            FROM {self._schema}.ont_entities
            WHERE status = 'active'
        """
        if not full:
            # Incremental: only entities not yet in Gold
            sql += f"""
                AND id NOT IN (
                    SELECT unnest(silver_entity_ids)
                    FROM {self._schema}.gold_entities
                    WHERE status = 'active'
                )
            """
        sql += " ORDER BY created_at"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)

        return [
            SilverEntity(
                id=r["id"],
                entity_type=r["entity_type"],
                name=r["name"],
                aliases=list(r.get("aliases") or []),
                properties=(
                    json.loads(r["properties"])
                    if isinstance(r["properties"], str)
                    else dict(r["properties"] or {})
                ),
                confidence=r["confidence"],
                created_at=str(r["created_at"]) if r["created_at"] else None,
                updated_at=str(r["updated_at"]) if r["updated_at"] else None,
            )
            for r in rows
        ]

    async def _upsert_gold_entities(
        self,
        candidates: list[GoldEntityCandidate],
        result: GoldBuildResult,
    ) -> dict[str, str]:
        """Insert or update Gold entities. Returns silver_id → gold_id mapping."""
        silver_to_gold: dict[str, str] = {}

        async with self._pool.acquire() as conn:
            for c in candidates:
                # Check if a Gold entity already covers any of these silver IDs
                existing_id = await self._find_existing_gold(conn, c.silver_entity_ids)

                if existing_id:
                    # Update existing Gold entity
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.gold_entities
                        SET canonical_name = $1,
                            properties = $2::jsonb,
                            aliases = $3,
                            silver_entity_ids = (
                                SELECT ARRAY(
                                    SELECT DISTINCT unnest(
                                        silver_entity_ids || $4::TEXT[]
                                    )
                                )
                            ),
                            source_count = $5,
                            confidence = $6,
                            updated_at = NOW(),
                            last_seen_at = NOW()
                        WHERE id = $7
                        """,
                        c.canonical_name,
                        json.dumps(c.properties),
                        c.aliases,
                        c.silver_entity_ids,
                        c.source_count,
                        c.confidence,
                        existing_id,
                    )
                    for sid in c.silver_entity_ids:
                        silver_to_gold[sid] = existing_id
                    result.gold_entities_updated += 1
                else:
                    # Insert new Gold entity
                    row = await conn.fetchrow(
                        f"""
                        INSERT INTO {self._schema}.gold_entities
                            (entity_type, canonical_name, properties, aliases,
                             silver_entity_ids, source_count, confidence, last_seen_at)
                        VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, NOW())
                        RETURNING id
                        """,
                        c.entity_type,
                        c.canonical_name,
                        json.dumps(c.properties),
                        c.aliases,
                        c.silver_entity_ids,
                        c.source_count,
                        c.confidence,
                    )
                    gold_id = row["id"]
                    for sid in c.silver_entity_ids:
                        silver_to_gold[sid] = gold_id
                    result.gold_entities_created += 1

        return silver_to_gold

    async def _find_existing_gold(
        self, conn: asyncpg.Connection, silver_ids: list[str]
    ) -> str | None:
        """Find a Gold entity that already contains any of the given silver IDs."""
        row = await conn.fetchrow(
            f"""
            SELECT id FROM {self._schema}.gold_entities
            WHERE silver_entity_ids && $1
              AND status = 'active'
            LIMIT 1
            """,
            silver_ids,
        )
        return row["id"] if row else None

    async def _aggregate_links(
        self,
        silver_to_gold: dict[str, str],
        result: GoldBuildResult,
    ) -> None:
        """Aggregate Silver links into Gold links."""
        # If we don't have any silver→gold mapping, also load existing mappings
        if not silver_to_gold:
            silver_to_gold = await self._load_silver_gold_map()

        if not silver_to_gold:
            return

        # Fetch all active Silver links
        async with self._pool.acquire() as conn:
            silver_links = await conn.fetch(
                f"""
                SELECT id, link_type, source_entity_id, target_entity_id,
                       properties, confidence, created_at
                FROM {self._schema}.ont_links
                WHERE status = 'active'
                """
            )

        # Group by (gold_source, gold_target, link_type)
        link_groups: dict[tuple[str, str, str], list[dict]] = {}
        for sl in silver_links:
            gold_source = silver_to_gold.get(sl["source_entity_id"])
            gold_target = silver_to_gold.get(sl["target_entity_id"])
            if not gold_source or not gold_target:
                continue
            if gold_source == gold_target:
                continue  # Skip self-links in Gold

            key = (gold_source, gold_target, sl["link_type"])
            link_groups.setdefault(key, []).append(dict(sl))

        # Upsert Gold links
        async with self._pool.acquire() as conn:
            for (src, tgt, ltype), links in link_groups.items():
                silver_ids = [l["id"] for l in links]
                merged_props = {}
                for l in links:
                    props = l["properties"]
                    if isinstance(props, str):
                        props = json.loads(props)
                    if props:
                        merged_props.update(props)

                confidence = max(l["confidence"] for l in links)
                first_seen = min(
                    (l["created_at"] for l in links if l["created_at"]),
                    default=None,
                )
                last_seen = max(
                    (l["created_at"] for l in links if l["created_at"]),
                    default=None,
                )

                # Check if Gold link already exists
                existing = await conn.fetchrow(
                    f"""
                    SELECT id, silver_link_ids, mention_count
                    FROM {self._schema}.gold_links
                    WHERE source_id = $1 AND target_id = $2
                      AND link_type = $3 AND status = 'active'
                    """,
                    src, tgt, ltype,
                )

                if existing:
                    # Update: merge silver_link_ids, increment mention_count
                    all_ids = list(
                        set(list(existing["silver_link_ids"]) + silver_ids)
                    )
                    await conn.execute(
                        f"""
                        UPDATE {self._schema}.gold_links
                        SET silver_link_ids = $1,
                            mention_count = $2,
                            confidence = GREATEST(confidence, $3),
                            properties = properties || $4::jsonb,
                            last_seen = $5,
                            updated_at = NOW()
                        WHERE id = $6
                        """,
                        all_ids,
                        len(all_ids),
                        confidence,
                        json.dumps(merged_props),
                        last_seen,
                        existing["id"],
                    )
                    result.gold_links_updated += 1
                else:
                    await conn.execute(
                        f"""
                        INSERT INTO {self._schema}.gold_links
                            (link_type, source_id, target_id, properties,
                             silver_link_ids, mention_count, confidence,
                             first_seen, last_seen)
                        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9)
                        """,
                        ltype, src, tgt,
                        json.dumps(merged_props),
                        silver_ids,
                        len(silver_ids),
                        confidence,
                        first_seen,
                        last_seen,
                    )
                    result.gold_links_created += 1

    async def _load_silver_gold_map(self) -> dict[str, str]:
        """Load the full silver→gold ID mapping from Gold entities."""
        mapping: dict[str, str] = {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, silver_entity_ids
                FROM {self._schema}.gold_entities
                WHERE status = 'active'
                """
            )
        for r in rows:
            for sid in r["silver_entity_ids"]:
                mapping[sid] = r["id"]
        return mapping

    async def _generate_embeddings(self, result: GoldBuildResult) -> None:
        """Generate embeddings for Gold entities that don't have one yet."""
        if not self._embeddings:
            return

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, canonical_name, aliases, properties
                FROM {self._schema}.gold_entities
                WHERE status = 'active' AND embedding IS NULL
                ORDER BY created_at
                """
            )

        if not rows:
            return

        logger.info("Generating embeddings for %d Gold entities...", len(rows))

        texts = []
        ids = []
        for r in rows:
            props = r["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            text = self._embeddings.build_text(
                r["canonical_name"],
                list(r.get("aliases") or []),
                props,
            )
            texts.append(text)
            ids.append(r["id"])

        try:
            embeddings = await self._embeddings.embed_texts(texts)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            result.errors.append(f"Embedding generation failed: {e}")
            return

        # Store embeddings
        async with self._pool.acquire() as conn:
            for eid, emb in zip(ids, embeddings):
                await conn.execute(
                    f"""
                    UPDATE {self._schema}.gold_entities
                    SET embedding = $1, updated_at = NOW()
                    WHERE id = $2
                    """,
                    str(emb),  # pgvector accepts string representation
                    eid,
                )
                result.embeddings_generated += 1

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()
        if self._embeddings:
            await self._embeddings.close()
