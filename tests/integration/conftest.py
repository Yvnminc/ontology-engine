"""Shared fixtures for PostgreSQL integration tests.

Reads the test database URL from ONTOLOGY_TEST_DB_URL.
If not set → all integration tests are skipped automatically.

Each test gets a clean schema (TRUNCATE CASCADE after every test).
"""

from __future__ import annotations

import os

import asyncpg
import pytest

# ---------------------------------------------------------------------------
# Skip-logic
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("ONTOLOGY_TEST_DB_URL", "")

requires_pg = pytest.mark.skipif(
    not DB_URL,
    reason="ONTOLOGY_TEST_DB_URL not set — skipping PostgreSQL integration tests",
)

# ---------------------------------------------------------------------------
# Gold-layer DDL (not part of SCHEMA_DDL but required for gold tests)
# ---------------------------------------------------------------------------

GOLD_DDL = """
CREATE TABLE IF NOT EXISTS ontology.gold_entities (
    id                  TEXT PRIMARY KEY DEFAULT 'GOLD-' || gen_random_uuid()::TEXT,
    entity_type         TEXT NOT NULL,
    canonical_name      TEXT NOT NULL,
    properties          JSONB NOT NULL DEFAULT '{}',
    aliases             TEXT[] DEFAULT '{}',
    silver_entity_ids   TEXT[] DEFAULT '{}',
    source_count        INTEGER DEFAULT 1,
    confidence          FLOAT DEFAULT 1.0,
    status              TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    embedding           vector(768),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gold_entities_type
    ON ontology.gold_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_gold_entities_status
    ON ontology.gold_entities(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_gold_entities_silver
    ON ontology.gold_entities USING gin(silver_entity_ids);

CREATE TABLE IF NOT EXISTS ontology.gold_links (
    id                  TEXT PRIMARY KEY DEFAULT 'GLINK-' || gen_random_uuid()::TEXT,
    link_type           TEXT NOT NULL,
    source_id           TEXT NOT NULL REFERENCES ontology.gold_entities(id),
    target_id           TEXT NOT NULL REFERENCES ontology.gold_entities(id),
    properties          JSONB DEFAULT '{}',
    silver_link_ids     TEXT[] DEFAULT '{}',
    mention_count       INTEGER DEFAULT 1,
    confidence          FLOAT DEFAULT 1.0,
    first_seen          TIMESTAMPTZ,
    last_seen           TIMESTAMPTZ,
    status              TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gold_links_type
    ON ontology.gold_links(link_type);
CREATE INDEX IF NOT EXISTS idx_gold_links_source
    ON ontology.gold_links(source_id);
CREATE INDEX IF NOT EXISTS idx_gold_links_target
    ON ontology.gold_links(target_id);
"""

# Same DDL but without embedding column (for DBs without pgvector)
GOLD_DDL_NO_VECTOR = GOLD_DDL.replace(
    "    embedding           vector(768),\n", ""
)

# ---------------------------------------------------------------------------
# All tables we need to TRUNCATE between tests
# ---------------------------------------------------------------------------

ALL_TABLES = [
    "ontology.gold_links",
    "ontology.gold_entities",
    "ontology.ont_entity_versions",
    "ontology.ont_conflicts",
    "ontology.ont_provenance",
    "ontology.ont_links",
    "ontology.ont_processing_log",
    "ontology.ont_entities",
    "ontology.bronze_documents",
    "ontology.ont_type_definitions",
    "ontology.agent_registry",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_url() -> str:
    """Return the test database URL (session-scoped)."""
    return DB_URL


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def _init_schema(db_url: str):
    """One-time schema bootstrap (session scope).

    Creates the ontology schema + all tables, including Gold + agent_registry.
    Runs once per test session.
    """
    from ontology_engine.storage.schema import initialize_schema

    # Initialize core schema (Bronze, Silver, etc.)
    await initialize_schema(db_url)

    # Add Gold tables + agent_registry
    conn = await asyncpg.connect(db_url)
    try:
        # Check if pgvector is available
        has_vector = False
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            has_vector = True
        except Exception:
            pass

        gold_ddl = GOLD_DDL if has_vector else GOLD_DDL_NO_VECTOR
        await conn.execute(gold_ddl)

        # Agent registry (uses its own DDL)
        from ontology_engine.sdk.registry import AGENT_REGISTRY_DDL

        await conn.execute(AGENT_REGISTRY_DDL.format(schema="ontology"))
    finally:
        await conn.close()

    yield has_vector  # downstream tests can know if pgvector is available


@pytest.fixture(autouse=True)
async def _clean_tables(db_url: str, _init_schema):
    """Truncate all tables before each test (clean slate)."""
    yield  # run the test first, then clean up

    conn = await asyncpg.connect(db_url)
    try:
        for table in ALL_TABLES:
            try:
                await conn.execute(f"TRUNCATE {table} CASCADE")
            except Exception:
                pass  # Table may not exist yet
    finally:
        await conn.close()


@pytest.fixture
async def pool(db_url: str, _init_schema) -> asyncpg.Pool:
    """Create a connection pool for the test database."""
    p = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
    yield p
    await p.close()


@pytest.fixture
def has_pgvector(_init_schema) -> bool:
    """Whether pgvector extension is available."""
    return _init_schema
