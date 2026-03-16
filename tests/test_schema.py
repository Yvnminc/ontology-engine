"""Tests for storage/schema.py — DDL validation."""

from __future__ import annotations

import pytest

from ontology_engine.storage.schema import SCHEMA_DDL


class TestSchemaDDL:
    def test_ddl_is_nonempty(self):
        assert len(SCHEMA_DDL) > 100

    def test_creates_schema(self):
        assert "CREATE SCHEMA IF NOT EXISTS ontology" in SCHEMA_DDL

    def test_creates_all_tables(self):
        expected_tables = [
            "ont_type_definitions",
            "ont_entities",
            "ont_links",
            "ont_provenance",
            "ont_entity_versions",
            "ont_conflicts",
            "ont_processing_log",
            "bronze_documents",
        ]
        for table in expected_tables:
            assert table in SCHEMA_DDL, f"Missing table: {table}"

    def test_entity_table_has_columns(self):
        assert "entity_type" in SCHEMA_DDL
        assert "properties" in SCHEMA_DDL
        assert "aliases" in SCHEMA_DDL
        assert "confidence" in SCHEMA_DDL

    def test_link_table_constraints(self):
        assert "no_self_link" in SCHEMA_DDL
        assert "unique_active_link" in SCHEMA_DDL

    def test_provenance_constraint(self):
        assert "has_target" in SCHEMA_DDL

    def test_seeds_type_definitions(self):
        expected_types = ["Person", "Decision", "ActionItem", "Project", "Risk", "Deadline"]
        for t in expected_types:
            assert f"'{t}'" in SCHEMA_DDL

    def test_creates_indexes(self):
        assert "idx_entities_type" in SCHEMA_DDL
        assert "idx_entities_name" in SCHEMA_DDL
        assert "idx_links_source" in SCHEMA_DDL
        assert "idx_prov_entity" in SCHEMA_DDL
        assert "idx_bronze_source_type" in SCHEMA_DDL
        assert "idx_bronze_ingested" in SCHEMA_DDL
        assert "idx_bronze_metadata" in SCHEMA_DDL

    def test_bronze_table_has_columns(self):
        assert "source_type" in SCHEMA_DDL
        assert "source_hash" in SCHEMA_DDL
        assert "content_format" in SCHEMA_DDL
        assert "unique_content" in SCHEMA_DDL

    def test_provenance_has_source_document_id(self):
        assert "source_document_id" in SCHEMA_DDL
