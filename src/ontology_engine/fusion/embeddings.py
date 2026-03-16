"""Embedding generation for Gold entities.

Uses OpenAI text-embedding-3-small (1536 dimensions) by default.
Falls back to skipping if no API key is available.

Entity text = canonical_name + aliases + key property values, concatenated.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.5  # seconds between batches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EmbeddingGenerator:
    """Generate embeddings for Gold entities via OpenAI API."""

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    @property
    def available(self) -> bool:
        """Check if embedding generation is available (has API key)."""
        return bool(self._api_key)

    async def _get_client(self) -> Any:
        """Lazy-init the OpenAI async client."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "No OPENAI_API_KEY set. Embedding generation requires an API key."
                )
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def build_text(
        self,
        canonical_name: str,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Build the text string to embed for an entity.

        Format: "canonical_name | alias1, alias2 | key: value, key: value"
        """
        parts = [canonical_name]

        if aliases:
            parts.append(", ".join(aliases))

        if properties:
            # Pick key properties (skip internal/meta fields)
            skip = {"status", "created_at", "updated_at", "id", "embedding"}
            prop_parts = []
            for k, v in properties.items():
                if k in skip or v is None or v == "":
                    continue
                prop_parts.append(f"{k}: {v}")
            if prop_parts:
                parts.append(", ".join(prop_parts))

        return " | ".join(parts)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts. Handles batching + rate limiting."""
        if not texts:
            return []

        client = await self._get_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]

            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Rate limiting between batches
            if i + MAX_BATCH_SIZE < len(texts):
                await asyncio.sleep(RATE_LIMIT_DELAY)

            logger.info(
                "Generated embeddings batch %d–%d of %d",
                i, min(i + MAX_BATCH_SIZE, len(texts)), len(texts),
            )

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        results = await self.embed_texts([text])
        return results[0]

    async def close(self) -> None:
        """Close the client if needed."""
        if self._client is not None:
            await self._client.close()
            self._client = None
