-- ============================================================
-- Kinetic Layer Schema — Action Types + Audit Trail
-- Migration: 002_kinetic_layer
-- ============================================================

SET search_path TO ontology, public;

-- 1. Action Types — declarative definitions of executable operations
CREATE TABLE IF NOT EXISTS action_types (
    name            TEXT PRIMARY KEY,
    description     TEXT DEFAULT '',
    input_schema    JSONB DEFAULT '{}',
    output_schema   JSONB DEFAULT '{}',
    preconditions   JSONB DEFAULT '[]',
    postconditions  JSONB DEFAULT '[]',
    side_effects    JSONB DEFAULT '[]',
    idempotent      BOOLEAN DEFAULT false,
    reversible      BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Audit Trail — immutable log of every action execution
CREATE TABLE IF NOT EXISTS action_audit_trail (
    id              TEXT PRIMARY KEY,
    action_name     TEXT NOT NULL,
    params          JSONB NOT NULL DEFAULT '{}',
    result          JSONB DEFAULT '{}',
    actor           TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    duration_ms     INTEGER DEFAULT 0,
    error_message   TEXT DEFAULT '',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_audit_trail_action
    ON action_audit_trail(action_name);
CREATE INDEX IF NOT EXISTS idx_audit_trail_actor
    ON action_audit_trail(actor);
CREATE INDEX IF NOT EXISTS idx_audit_trail_created
    ON action_audit_trail(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_trail_status
    ON action_audit_trail(status);

-- GIN index on params for entity lineage queries
CREATE INDEX IF NOT EXISTS idx_audit_trail_params
    ON action_audit_trail USING gin(params);
