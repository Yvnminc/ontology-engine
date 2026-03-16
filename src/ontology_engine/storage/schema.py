"""Database schema DDL and migration support.

Supports dynamic seeding of ont_type_definitions from domain schema YAML.
When no schema is provided, uses default types (backward compatible).
"""

from __future__ import annotations

from typing import Any

# Default type definitions (backward compatible with original 6 entity types)
DEFAULT_TYPE_SEEDS: list[tuple[str, str, str, str, list[str]]] = [
    ("Person", "entity", "人物", "组织内外的人物实体", ["name", "role"]),
    ("Decision", "entity", "决策", "会议中做出的决策", ["summary", "decision_type"]),
    ("ActionItem", "entity", "行动项", "需要执行的任务", ["task"]),
    ("Project", "entity", "项目", "公司项目", ["name"]),
    ("Risk", "entity", "风险", "已识别的风险", ["description"]),
    ("Deadline", "entity", "截止日期", "关键时间节点", ["date", "description"]),
]


def generate_seed_sql(domain_schema: Any | None = None) -> str:
    """Generate INSERT SQL for ont_type_definitions from a domain schema.

    Args:
        domain_schema: A DomainSchema object, raw dict, or None for defaults.

    Returns:
        SQL INSERT statement with ON CONFLICT DO NOTHING.
    """
    rows = _rows_from_schema(domain_schema) if domain_schema else list(DEFAULT_TYPE_SEEDS)
    if not rows:
        rows = list(DEFAULT_TYPE_SEEDS)

    values_parts: list[str] = []
    for type_id, category, display_name, description, required_fields in rows:
        fields_sql = "ARRAY[" + ", ".join(f"'{f}'" for f in required_fields) + "]"
        esc_id = type_id.replace("'", "''")
        esc_dn = display_name.replace("'", "''")
        esc_desc = description.replace("'", "''")
        values_parts.append(
            f"    ('{esc_id}', '{category}', '{esc_dn}', '{esc_desc}', {fields_sql})"
        )

    return (
        "INSERT INTO ont_type_definitions "
        "(id, category, display_name, description, required_fields)\nVALUES\n"
        + ",\n".join(values_parts)
        + "\nON CONFLICT (id) DO NOTHING;"
    )


def _rows_from_schema(schema: Any) -> list[tuple[str, str, str, str, list[str]]]:
    """Extract type definition rows from a DomainSchema or dict."""
    rows: list[tuple[str, str, str, str, list[str]]] = []

    if isinstance(schema, dict):
        for et in schema.get("entity_types", []):
            name = et.get("name", "")
            desc = et.get("description", "")
            req = [
                p.get("name", "")
                for p in et.get("properties", [])
                if p.get("required")
            ]
            rows.append((name, "entity", name, desc, req or ["name"]))
        for lt in schema.get("link_types", []):
            rows.append((
                lt.get("name", ""), "link",
                lt.get("name", ""), lt.get("description", ""), [],
            ))
        return rows

    # DomainSchema object
    if hasattr(schema, "entity_types"):
        for et in schema.entity_types:
            req = [p.name for p in et.properties if p.required]
            rows.append((et.name, "entity", et.name, et.description, req or ["name"]))
    if hasattr(schema, "link_types"):
        for lt in schema.link_types:
            rows.append((lt.name, "link", lt.name, lt.description, []))
    return rows

# Full DDL for Ontology Engine schema (PostgreSQL + pgvector)
# Designed for Supabase but works with any PG 15+ instance.

SCHEMA_DDL = """
-- ============================================================
-- Ontology Engine Schema v1.0
-- ============================================================

CREATE SCHEMA IF NOT EXISTS ontology;
SET search_path TO ontology;

-- Extensions (run as superuser / via Supabase dashboard)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";


-- 1. Type Definitions
CREATE TABLE IF NOT EXISTS ont_type_definitions (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL CHECK (category IN ('entity', 'link')),
    display_name    TEXT NOT NULL,
    description     TEXT,
    schema          JSONB NOT NULL DEFAULT '{}',
    required_fields TEXT[] DEFAULT '{}',
    icon            TEXT,
    color           TEXT,
    is_active       BOOLEAN DEFAULT true,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);


-- 2. Entities (core node table)
CREATE TABLE IF NOT EXISTS ont_entities (
    id              TEXT PRIMARY KEY DEFAULT 'ENT-' || gen_random_uuid()::TEXT,
    entity_type     TEXT NOT NULL REFERENCES ont_type_definitions(id),
    name            TEXT NOT NULL,
    properties      JSONB NOT NULL DEFAULT '{}',
    aliases         TEXT[] DEFAULT '{}',
    status          TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    confidence      FLOAT DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    version         INTEGER DEFAULT 1,
    is_current      BOOLEAN DEFAULT true,
    previous_version_id TEXT REFERENCES ont_entities(id),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    created_by      TEXT DEFAULT 'system',
    updated_by      TEXT DEFAULT 'system'
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON ont_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON ont_entities USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_properties ON ont_entities USING gin(properties);
CREATE INDEX IF NOT EXISTS idx_entities_status ON ont_entities(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_entities_aliases ON ont_entities USING gin(aliases);


-- 3. Links (relationships between entities)
CREATE TABLE IF NOT EXISTS ont_links (
    id                  TEXT PRIMARY KEY DEFAULT 'LNK-' || gen_random_uuid()::TEXT,
    link_type           TEXT NOT NULL,
    source_entity_id    TEXT NOT NULL REFERENCES ont_entities(id),
    target_entity_id    TEXT NOT NULL REFERENCES ont_entities(id),
    properties          JSONB DEFAULT '{}',
    confidence          FLOAT DEFAULT 1.0,
    status              TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    valid_from          TIMESTAMPTZ,
    valid_to            TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    created_by          TEXT DEFAULT 'system',
    CONSTRAINT no_self_link CHECK (source_entity_id != target_entity_id),
    CONSTRAINT unique_active_link UNIQUE (link_type, source_entity_id, target_entity_id, status)
);

CREATE INDEX IF NOT EXISTS idx_links_type ON ont_links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_source ON ont_links(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON ont_links(target_entity_id);


-- 4. Provenance (where knowledge came from)
CREATE TABLE IF NOT EXISTS ont_provenance (
    id                  TEXT PRIMARY KEY DEFAULT 'PROV-' || gen_random_uuid()::TEXT,
    entity_id           TEXT REFERENCES ont_entities(id),
    link_id             TEXT REFERENCES ont_links(id),
    source_document_id  TEXT REFERENCES bronze_documents(id),
    source_type         TEXT NOT NULL CHECK (source_type IN (
                            'meeting_transcript', 'manual_input', 'llm_extraction',
                            'agent_report', 'external_import', 'system_inference'
                        )),
    source_file         TEXT,
    source_meeting_date DATE,
    source_participants TEXT[],
    source_segment      TEXT,
    source_offset       INTEGER,
    extraction_model    TEXT,
    extraction_schema   TEXT,
    extraction_pass     TEXT,
    raw_extraction      JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    created_by          TEXT DEFAULT 'system',
    CONSTRAINT has_target CHECK (entity_id IS NOT NULL OR link_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_prov_entity ON ont_provenance(entity_id);
CREATE INDEX IF NOT EXISTS idx_prov_link ON ont_provenance(link_id);
CREATE INDEX IF NOT EXISTS idx_prov_source ON ont_provenance(source_file);
CREATE INDEX IF NOT EXISTS idx_prov_date ON ont_provenance(source_meeting_date);


-- 5. Version History
CREATE TABLE IF NOT EXISTS ont_entity_versions (
    id              TEXT PRIMARY KEY DEFAULT 'VER-' || gen_random_uuid()::TEXT,
    entity_id       TEXT NOT NULL REFERENCES ont_entities(id),
    version         INTEGER NOT NULL,
    change_type     TEXT NOT NULL CHECK (change_type IN (
                        'created', 'updated', 'merged', 'archived', 'deleted', 'restored'
                    )),
    old_values      JSONB,
    new_values      JSONB,
    change_reason   TEXT,
    changed_by      TEXT DEFAULT 'system',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_versions_entity ON ont_entity_versions(entity_id);


-- 6. Conflicts (detected contradictions)
CREATE TABLE IF NOT EXISTS ont_conflicts (
    id              TEXT PRIMARY KEY DEFAULT 'CONF-' || gen_random_uuid()::TEXT,
    conflict_type   TEXT NOT NULL CHECK (conflict_type IN (
                        'contradiction', 'duplicate', 'stale', 'missing_ref'
                    )),
    entity_a_id     TEXT REFERENCES ont_entities(id),
    entity_b_id     TEXT REFERENCES ont_entities(id),
    description     TEXT NOT NULL,
    severity        TEXT DEFAULT 'warning' CHECK (severity IN ('error', 'warning', 'info')),
    status          TEXT DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'ignored')),
    resolution      TEXT,
    resolved_by     TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status ON ont_conflicts(status) WHERE status = 'open';


-- 7. Processing Log (audit trail for pipeline runs)
CREATE TABLE IF NOT EXISTS ont_processing_log (
    id              TEXT PRIMARY KEY DEFAULT 'LOG-' || gen_random_uuid()::TEXT,
    source_file     TEXT NOT NULL,
    meeting_date    DATE,
    status          TEXT DEFAULT 'pending' CHECK (status IN (
                        'pending', 'processing', 'completed', 'failed'
                    )),
    entities_extracted  INTEGER DEFAULT 0,
    links_extracted     INTEGER DEFAULT 0,
    decisions_extracted INTEGER DEFAULT 0,
    actions_extracted   INTEGER DEFAULT 0,
    errors              JSONB DEFAULT '[]',
    processing_time_ms  INTEGER,
    model_used          TEXT,
    cost_estimate       FLOAT,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);


-- 8. Bronze Documents (immutable raw document store)
CREATE TABLE IF NOT EXISTS bronze_documents (
    id              TEXT PRIMARY KEY DEFAULT 'DOC-' || gen_random_uuid()::TEXT,
    source_type     TEXT NOT NULL,
    source_uri      TEXT,
    source_hash     TEXT NOT NULL,
    content         TEXT NOT NULL,
    content_format  TEXT DEFAULT 'text',
    language        TEXT DEFAULT 'auto',
    metadata        JSONB DEFAULT '{}',
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    ingested_by     TEXT DEFAULT 'system',
    CONSTRAINT unique_content UNIQUE (source_hash)
);

CREATE INDEX IF NOT EXISTS idx_bronze_source_type ON bronze_documents(source_type);
CREATE INDEX IF NOT EXISTS idx_bronze_ingested ON bronze_documents(ingested_at);
CREATE INDEX IF NOT EXISTS idx_bronze_metadata ON bronze_documents USING gin(metadata);


-- Seed type definitions
INSERT INTO ont_type_definitions (id, category, display_name, description, required_fields)
VALUES
    ('Person',     'entity', '人物', '组织内外的人物实体', ARRAY['name', 'role']),
    ('Decision',   'entity', '决策', '会议中做出的决策',  ARRAY['summary', 'decision_type']),
    ('ActionItem', 'entity', '行动项', '需要执行的任务',  ARRAY['task']),
    ('Project',    'entity', '项目', '公司项目',          ARRAY['name']),
    ('Risk',       'entity', '风险', '已识别的风险',      ARRAY['description']),
    ('Deadline',   'entity', '截止日期', '关键时间节点',   ARRAY['date', 'description'])
ON CONFLICT (id) DO NOTHING;
"""


async def initialize_schema(
    db_url: str, domain_schema: Any | None = None
) -> None:
    """Create the ontology schema. Optionally seed extra types from a domain schema."""
    import asyncpg

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(SCHEMA_DDL)
        if domain_schema is not None:
            await conn.execute(generate_seed_sql(domain_schema))
    finally:
        await conn.close()
