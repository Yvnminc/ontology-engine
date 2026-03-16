"""Entity Resolution Engine — merge duplicate entities across sources.

Strategies (conservative — prefer false negatives over false positives):
  1. Exact name match (case-insensitive)
  2. Alias match (known_entities config)
  3. Fuzzy name match (Jaro-Winkler similarity)
  4. Embedding cosine similarity (if embeddings available)

Merge thresholds:
  - similarity > 0.95 → auto-merge
  - 0.80–0.95      → flag for review (no auto-merge)
  - < 0.80          → treat as distinct
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ontology_engine.core.config import OntologyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

AUTO_MERGE_THRESHOLD = 0.95
REVIEW_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SilverEntity:
    """Lightweight representation of a Silver-layer entity for resolution."""

    id: str
    entity_type: str
    name: str
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    embedding: list[float] | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class MergeCandidate:
    """A pair of entities considered for merging."""

    entity_a: SilverEntity
    entity_b: SilverEntity
    similarity: float
    match_reason: str  # "exact_name", "alias", "fuzzy_name", "embedding"
    auto_merge: bool = False
    needs_review: bool = False


@dataclass
class GoldEntityCandidate:
    """A merged Gold entity built from one or more Silver entities."""

    entity_type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    silver_entity_ids: list[str] = field(default_factory=list)
    source_count: int = 1
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# String similarity functions
# ---------------------------------------------------------------------------


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity between two strings (0.0–1.0)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Jaro distance
    max_dist = max(len(s1), len(s2)) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)

    matches = 0
    transpositions = 0

    for i in range(len(s1)):
        lo = max(0, i - max_dist)
        hi = min(len(s2), i + max_dist + 1)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len(s1)):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (
        matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches
    ) / 3

    # Winkler boost for common prefix (up to 4 chars)
    prefix = 0
    for i in range(min(4, min(len(s1), len(s2)))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_name(name: str) -> str:
    """Normalize a name for comparison."""
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Entity Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolves and merges duplicate entities from the Silver layer."""

    def __init__(self, config: OntologyConfig | None = None):
        self._config = config or OntologyConfig()
        # Build alias lookup: alias → canonical name
        self._alias_map: dict[str, str] = {}
        if self._config.known_entities.aliases:
            for canonical, aliases in self._config.known_entities.aliases.items():
                canonical_lower = canonical.lower()
                self._alias_map[canonical_lower] = canonical
                for alias in aliases:
                    self._alias_map[alias.lower()] = canonical

    def resolve(
        self, entities: list[SilverEntity]
    ) -> tuple[list[GoldEntityCandidate], list[MergeCandidate]]:
        """Resolve a list of Silver entities into Gold candidates.

        Returns:
            (gold_candidates, review_candidates) — auto-merged golds and
            pairs flagged for human review.
        """
        if not entities:
            return [], []

        # Group by entity_type — only compare within same type
        by_type: dict[str, list[SilverEntity]] = {}
        for e in entities:
            by_type.setdefault(e.entity_type, []).append(e)

        all_golds: list[GoldEntityCandidate] = []
        all_reviews: list[MergeCandidate] = []

        for etype, ents in by_type.items():
            golds, reviews = self._resolve_group(ents)
            all_golds.extend(golds)
            all_reviews.extend(reviews)

        return all_golds, all_reviews

    def _resolve_group(
        self, entities: list[SilverEntity]
    ) -> tuple[list[GoldEntityCandidate], list[MergeCandidate]]:
        """Resolve entities of the same type."""
        # Union-Find for clustering
        parent: dict[str, str] = {e.id: e.id for e in entities}
        reviews: list[MergeCandidate] = []

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        entity_map = {e.id: e for e in entities}

        # Compare all pairs
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                if find(a.id) == find(b.id):
                    continue  # Already in same cluster

                sim, reason = self._compute_similarity(a, b)

                if sim >= AUTO_MERGE_THRESHOLD:
                    union(a.id, b.id)
                    logger.info(
                        "Auto-merge: '%s' ↔ '%s' (sim=%.3f, reason=%s)",
                        a.name, b.name, sim, reason,
                    )
                elif sim >= REVIEW_THRESHOLD:
                    reviews.append(
                        MergeCandidate(
                            entity_a=a,
                            entity_b=b,
                            similarity=sim,
                            match_reason=reason,
                            auto_merge=False,
                            needs_review=True,
                        )
                    )
                    logger.info(
                        "Review needed: '%s' ↔ '%s' (sim=%.3f, reason=%s)",
                        a.name, b.name, sim, reason,
                    )

        # Build Gold candidates from clusters
        clusters: dict[str, list[SilverEntity]] = {}
        for e in entities:
            root = find(e.id)
            clusters.setdefault(root, []).append(e)

        golds = [self._build_gold_candidate(cluster) for cluster in clusters.values()]
        return golds, reviews

    def _compute_similarity(
        self, a: SilverEntity, b: SilverEntity
    ) -> tuple[float, str]:
        """Compute similarity between two entities. Returns (score, reason)."""
        na = normalize_name(a.name)
        nb = normalize_name(b.name)

        # 1. Exact name match
        if na == nb:
            return 1.0, "exact_name"

        # 2. Alias match (via known_entities config)
        canonical_a = self._alias_map.get(na)
        canonical_b = self._alias_map.get(nb)
        if canonical_a and canonical_b and canonical_a == canonical_b:
            return 1.0, "alias_config"

        # Check if one name appears in the other's aliases
        a_names = {na} | {alias.lower() for alias in a.aliases}
        b_names = {nb} | {alias.lower() for alias in b.aliases}
        if a_names & b_names:
            return 1.0, "alias_overlap"

        # Check if one entity's name is in the other's alias list
        if na in b_names or nb in a_names:
            return 1.0, "alias_match"

        # 3. Embedding cosine similarity
        if a.embedding and b.embedding:
            cos_sim = cosine_similarity(a.embedding, b.embedding)
            if cos_sim >= AUTO_MERGE_THRESHOLD:
                return cos_sim, "embedding"
            if cos_sim >= REVIEW_THRESHOLD:
                return cos_sim, "embedding"

        # 4. Fuzzy name match (Jaro-Winkler)
        jw = jaro_winkler_similarity(na, nb)
        if jw >= REVIEW_THRESHOLD:
            return jw, "fuzzy_name"

        return jw, "fuzzy_name"

    def _build_gold_candidate(
        self, cluster: list[SilverEntity]
    ) -> GoldEntityCandidate:
        """Build a Gold entity candidate from a cluster of Silver entities."""
        # Pick canonical name: prefer known_entities canonical, else most frequent / longest
        canonical = self._pick_canonical_name(cluster)

        # Merge aliases: union of all names + aliases, minus the canonical
        all_names: set[str] = set()
        for e in cluster:
            all_names.add(e.name)
            all_names.update(e.aliases)
        # Don't include canonical in aliases
        aliases = sorted(n for n in all_names if n.lower() != canonical.lower())

        # Merge properties: later wins for same key
        merged_props: dict[str, Any] = {}
        for e in sorted(cluster, key=lambda x: x.updated_at or x.created_at or ""):
            merged_props.update(e.properties)

        # Confidence: max of cluster
        confidence = max(e.confidence for e in cluster)

        return GoldEntityCandidate(
            entity_type=cluster[0].entity_type,
            canonical_name=canonical,
            aliases=aliases,
            properties=merged_props,
            silver_entity_ids=[e.id for e in cluster],
            source_count=len(cluster),
            confidence=confidence,
        )

    def _pick_canonical_name(self, cluster: list[SilverEntity]) -> str:
        """Pick the best canonical name for a cluster."""
        # Check known_entities config first
        for e in cluster:
            canonical = self._alias_map.get(e.name.lower())
            if canonical:
                return canonical
            for alias in e.aliases:
                canonical = self._alias_map.get(alias.lower())
                if canonical:
                    return canonical

        # Fallback: pick the name with highest confidence, then longest
        best = max(cluster, key=lambda e: (e.confidence, len(e.name)))
        return best.name
