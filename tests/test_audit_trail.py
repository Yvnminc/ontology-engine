"""Tests for Kinetic Layer — AuditTrail (in-memory backend)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ontology_engine.kinetic.audit_trail import AuditEntry, AuditTrail


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def audit() -> AuditTrail:
    return AuditTrail()


def _make_entry(
    id: str = "EXEC-001",
    action_name: str = "create_entity",
    params: dict | None = None,
    result: dict | None = None,
    actor: str = "agent-1",
    status: str = "success",
    duration_ms: int = 10,
    error_message: str = "",
) -> AuditEntry:
    return AuditEntry(
        id=id,
        action_name=action_name,
        params=params or {},
        result=result or {},
        actor=actor,
        timestamp=datetime.now(timezone.utc),
        status=status,
        duration_ms=duration_ms,
        error_message=error_message,
    )


# =============================================================================
# AuditEntry
# =============================================================================


class TestAuditEntry:
    def test_to_dict(self) -> None:
        entry = _make_entry(
            params={"name": "Test"},
            result={"entity_id": "E1"},
            error_message="",
        )
        d = entry.to_dict()
        assert d["id"] == "EXEC-001"
        assert d["action_name"] == "create_entity"
        assert d["params"] == {"name": "Test"}
        assert d["result"] == {"entity_id": "E1"}
        assert d["actor"] == "agent-1"
        assert d["status"] == "success"
        assert d["duration_ms"] == 10
        assert isinstance(d["timestamp"], str)

    def test_error_entry(self) -> None:
        entry = _make_entry(status="failed", error_message="Something broke")
        assert entry.error_message == "Something broke"
        d = entry.to_dict()
        assert d["error_message"] == "Something broke"


# =============================================================================
# AuditTrail — record and query
# =============================================================================


class TestAuditTrailRecordQuery:
    def test_record_and_len(self, audit: AuditTrail) -> None:
        assert len(audit) == 0
        audit.record(_make_entry())
        assert len(audit) == 1

    def test_multiple_records(self, audit: AuditTrail) -> None:
        for i in range(5):
            audit.record(_make_entry(id=f"EXEC-{i:03d}"))
        assert len(audit) == 5

    def test_query_all(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="A"))
        audit.record(_make_entry(id="B"))
        results = audit.query()
        assert len(results) == 2
        # Newest first
        assert results[0].id == "B"
        assert results[1].id == "A"

    def test_query_by_id(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="X"))
        audit.record(_make_entry(id="Y"))
        results = audit.query({"id": "X"})
        assert len(results) == 1
        assert results[0].id == "X"

    def test_query_by_action_name(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", action_name="create"))
        audit.record(_make_entry(id="2", action_name="update"))
        audit.record(_make_entry(id="3", action_name="create"))
        results = audit.query({"action_name": "create"})
        assert len(results) == 2

    def test_query_by_actor(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", actor="alice"))
        audit.record(_make_entry(id="2", actor="bob"))
        results = audit.query({"actor": "bob"})
        assert len(results) == 1
        assert results[0].actor == "bob"

    def test_query_by_status(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", status="success"))
        audit.record(_make_entry(id="2", status="failed"))
        audit.record(_make_entry(id="3", status="success"))
        results = audit.query({"status": "failed"})
        assert len(results) == 1

    def test_query_with_limit(self, audit: AuditTrail) -> None:
        for i in range(10):
            audit.record(_make_entry(id=f"E{i}"))
        results = audit.query({"limit": 3})
        assert len(results) == 3

    def test_query_combined_filters(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", actor="alice", status="success"))
        audit.record(_make_entry(id="2", actor="alice", status="failed"))
        audit.record(_make_entry(id="3", actor="bob", status="success"))
        results = audit.query({"actor": "alice", "status": "success"})
        assert len(results) == 1
        assert results[0].id == "1"

    def test_query_no_match(self, audit: AuditTrail) -> None:
        audit.record(_make_entry())
        results = audit.query({"actor": "nobody"})
        assert len(results) == 0


# =============================================================================
# AuditTrail — get_lineage
# =============================================================================


class TestGetLineage:
    def test_lineage_by_params(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", params={"entity_id": "ENT-001"}))
        audit.record(_make_entry(id="2", params={"entity_id": "ENT-002"}))
        audit.record(_make_entry(id="3", params={"target": "ENT-001"}))

        lineage = audit.get_lineage("ENT-001")
        assert len(lineage) == 2
        ids = {e.id for e in lineage}
        assert "1" in ids
        assert "3" in ids

    def test_lineage_by_result(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", result={"created_id": "ENT-ABC"}))
        audit.record(_make_entry(id="2", result={"updated_id": "ENT-XYZ"}))

        lineage = audit.get_lineage("ENT-ABC")
        assert len(lineage) == 1
        assert lineage[0].id == "1"

    def test_lineage_empty(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="1", params={"x": "y"}))
        lineage = audit.get_lineage("ENT-MISSING")
        assert len(lineage) == 0


# =============================================================================
# AuditTrail — entries property and clear
# =============================================================================


class TestEntriesAndClear:
    def test_entries_chronological(self, audit: AuditTrail) -> None:
        audit.record(_make_entry(id="A"))
        audit.record(_make_entry(id="B"))
        audit.record(_make_entry(id="C"))
        entries = audit.entries
        assert [e.id for e in entries] == ["A", "B", "C"]

    def test_clear(self, audit: AuditTrail) -> None:
        audit.record(_make_entry())
        audit.record(_make_entry(id="X"))
        assert len(audit) == 2
        audit.clear()
        assert len(audit) == 0
        assert audit.entries == []
