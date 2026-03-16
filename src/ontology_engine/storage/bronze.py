"""Bronze Layer — immutable, append-only raw document storage.

All ingested documents are stored here before any Silver-layer extraction.
Documents are deduplicated by SHA-256 content hash.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import asyncpg

from ontology_engine.core.errors import StorageError


class BronzeDocument:
    """In-memory representation of a bronze_documents row."""

    __slots__ = (
        "id",
        "source_type",
        "source_uri",
        "source_hash",
        "content",
        "content_format",
        "language",
        "metadata",
        "ingested_at",
        "ingested_by",
    )

    def __init__(
        self,
        *,
        id: str,
        source_type: str,
        source_uri: str | None,
        source_hash: str,
        content: str,
        content_format: str = "text",
        language: str = "auto",
        metadata: dict[str, Any] | None = None,
        ingested_at: datetime | None = None,
        ingested_by: str = "system",
    ):
        self.id = id
        self.source_type = source_type
        self.source_uri = source_uri
        self.source_hash = source_hash
        self.content = content
        self.content_format = content_format
        self.language = language
        self.metadata = metadata or {}
        self.ingested_at = ingested_at
        self.ingested_by = ingested_by

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_uri": self.source_uri,
            "source_hash": self.source_hash,
            "content": self.content,
            "content_format": self.content_format,
            "language": self.language,
            "metadata": self.metadata,
            "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
            "ingested_by": self.ingested_by,
        }


class BronzeRepository:
    """Async repository for bronze_documents — append-only, no UPDATE/DELETE."""

    def __init__(self, pool: asyncpg.Pool, schema: str = "ontology"):
        self._pool = pool
        self._schema = schema

    @classmethod
    async def create(cls, db_url: str, schema: str = "ontology") -> BronzeRepository:
        """Create a repository with a connection pool."""
        pool = await asyncpg.create_pool(db_url, min_size=2, max_size=10)
        repo = cls(pool, schema)
        return repo

    async def close(self) -> None:
        await self._pool.close()

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def ingest(
        self,
        content: str,
        source_type: str,
        *,
        source_uri: str | None = None,
        content_format: str = "text",
        language: str = "auto",
        metadata: dict[str, Any] | None = None,
        ingested_by: str = "system",
    ) -> tuple[str, bool]:
        """Ingest a document into the bronze layer.

        Computes SHA-256 hash and performs dedup insert.

        Returns:
            (doc_id, is_new) — doc_id of existing or newly created document,
            and whether it was newly inserted.
        """
        source_hash = self.compute_hash(content)

        # Check for existing document first
        existing_id = await self.exists(source_hash)
        if existing_id is not None:
            return existing_id, False

        sql = f"""
            INSERT INTO {self._schema}.bronze_documents
                (source_type, source_uri, source_hash, content,
                 content_format, language, metadata, ingested_by)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
            ON CONFLICT (source_hash) DO NOTHING
            RETURNING id
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                source_type,
                source_uri,
                source_hash,
                content,
                content_format,
                language,
                json.dumps(metadata or {}),
                ingested_by,
            )
            if row is not None:
                return row["id"], True

            # Race condition: another process inserted between our check and insert
            existing_id = await self.exists(source_hash)
            if existing_id is not None:
                return existing_id, False

            raise StorageError("Failed to ingest bronze document")

    async def get(self, doc_id: str) -> BronzeDocument | None:
        """Fetch a single document by ID."""
        sql = f"SELECT * FROM {self._schema}.bronze_documents WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, doc_id)
            if row is None:
                return None
            return self._row_to_document(row)

    async def list(
        self,
        source_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[BronzeDocument]:
        """List documents with optional source_type filter."""
        conditions: list[str] = []
        params: list[Any] = []

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"source_type = ${len(params)}")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])
        sql = f"""
            SELECT * FROM {self._schema}.bronze_documents
            {where}
            ORDER BY ingested_at DESC
            LIMIT ${len(params) - 1} OFFSET ${len(params)}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_document(r) for r in rows]

    async def exists(self, source_hash: str) -> str | None:
        """Check if a document with the given hash already exists.

        Returns the doc_id if found, None otherwise.
        """
        sql = f"""
            SELECT id FROM {self._schema}.bronze_documents
            WHERE source_hash = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, source_hash)
            return row["id"] if row else None

    @staticmethod
    def _row_to_document(row: asyncpg.Record) -> BronzeDocument:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return BronzeDocument(
            id=row["id"],
            source_type=row["source_type"],
            source_uri=row["source_uri"],
            source_hash=row["source_hash"],
            content=row["content"],
            content_format=row["content_format"],
            language=row["language"],
            metadata=metadata,
            ingested_at=row["ingested_at"],
            ingested_by=row["ingested_by"],
        )
