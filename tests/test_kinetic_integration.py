"""Integration tests for the Kinetic Layer — end-to-end action flow.

Tests the full stack: YAML loading → Registry → Executor → Audit → SDK.
Does NOT require PostgreSQL (uses in-memory backends).
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from ontology_engine.kinetic.action_executor import ActionExecutor, ExecutionContext
from ontology_engine.kinetic.action_types import ActionRegistry, ActionType, load_actions_from_yaml
from ontology_engine.kinetic.audit_trail import AuditTrail


# =============================================================================
# End-to-end: YAML → Registry → Execute → Audit
# =============================================================================


class TestKineticE2E:
    """Full round-trip test for the Kinetic Layer."""

    @pytest.fixture
    def yaml_data(self) -> dict[str, Any]:
        return yaml.safe_load("""
actions:
  - name: enrich_entity
    description: "Add computed properties to an entity"
    input:
      entity_id: { type: string, required: true }
      properties: { type: object, required: true }
    output:
      updated_entity: { type: object }
    preconditions:
      - "entity_exists(entity_id)"
    idempotent: true

  - name: enroll_student
    description: "Enroll a student in a course"
    input:
      student_id: { type: string, required: true }
      course_id: { type: string, required: true }
      semester: { type: string, required: true }
    output:
      enrollment_id: { type: string }
    reversible: true
""")

    @pytest.fixture
    def setup(self, yaml_data: dict[str, Any]) -> tuple[ActionExecutor, AuditTrail]:
        registry = ActionRegistry()
        for action_type in load_actions_from_yaml(yaml_data):
            registry.register(action_type)

        audit = AuditTrail()
        executor = ActionExecutor(registry, audit)

        # Register handlers
        async def enrich_handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"updated_entity": {"id": params["entity_id"], **params["properties"]}}

        async def enroll_handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"enrollment_id": f"ENR-{params['student_id']}-{params['course_id']}"}

        async def enroll_rollback(
            exec_id: str, params: dict[str, Any], result: dict[str, Any], ctx: ExecutionContext,
        ) -> dict[str, Any]:
            return {"cancelled_enrollment": result["enrollment_id"]}

        executor.register_handler("enrich_entity", enrich_handler)
        executor.register_handler("enroll_student", enroll_handler)
        executor.register_rollback_handler("enroll_student", enroll_rollback)

        return executor, audit

    @pytest.mark.asyncio
    async def test_full_cycle(self, setup: tuple[ActionExecutor, AuditTrail]) -> None:
        executor, audit = setup

        # 1. Execute enrich_entity
        r1 = await executor.execute(
            "enrich_entity",
            {"entity_id": "ENT-001", "properties": {"mastery": 0.85}},
            ExecutionContext(actor="tutor-agent"),
        )
        assert r1.status == "success"
        assert r1.result["updated_entity"]["mastery"] == 0.85

        # 2. Execute enroll_student
        r2 = await executor.execute(
            "enroll_student",
            {"student_id": "S001", "course_id": "CS101", "semester": "2026-S1"},
            ExecutionContext(actor="admin"),
        )
        assert r2.status == "success"
        assert r2.result["enrollment_id"] == "ENR-S001-CS101"

        # 3. Rollback enroll_student
        rb = await executor.rollback(r2.execution_id, ExecutionContext(actor="admin"))
        assert rb.status == "rolled_back"
        assert rb.result["cancelled_enrollment"] == "ENR-S001-CS101"

        # 4. Verify audit trail
        all_entries = audit.query()
        # 3 entries: enrich, enroll, rollback
        assert len(all_entries) == 3

        # Check actor filter
        admin_entries = audit.query({"actor": "admin"})
        assert len(admin_entries) == 2

        # Check lineage for S001
        lineage = audit.get_lineage("S001")
        assert len(lineage) >= 1

    @pytest.mark.asyncio
    async def test_validation_rejects_bad_input(self, setup: tuple[ActionExecutor, AuditTrail]) -> None:
        executor, audit = setup

        # Missing required fields
        result = await executor.execute(
            "enrich_entity",
            {"entity_id": "ENT-001"},  # missing "properties"
        )
        assert result.status == "validation_error"
        assert len(audit) == 1

    @pytest.mark.asyncio
    async def test_idempotent_execution(self, setup: tuple[ActionExecutor, AuditTrail]) -> None:
        executor, audit = setup
        params = {"entity_id": "ENT-001", "properties": {"x": 1}}
        ctx = ExecutionContext(actor="bot")

        r1 = await executor.execute("enrich_entity", params, ctx)
        r2 = await executor.execute("enrich_entity", params, ctx)

        assert r1.status == "success"
        assert r2.status == "success"
        assert r1.result == r2.result
        assert len(audit) == 2  # Both recorded


# =============================================================================
# Multiple domains
# =============================================================================


class TestMultipleDomains:
    """Verify that actions from different domains coexist in one registry."""

    @pytest.mark.asyncio
    async def test_mixed_domains(self) -> None:
        edtech_yaml = {
            "actions": [
                {"name": "enroll_student", "description": "Enroll", "input": {"sid": {"type": "string", "required": True}}},
            ],
        }
        finance_yaml = {
            "actions": [
                {"name": "freeze_account", "description": "Freeze", "input": {"aid": {"type": "string", "required": True}}},
            ],
        }

        registry = ActionRegistry()
        for a in load_actions_from_yaml(edtech_yaml):
            registry.register(a)
        for a in load_actions_from_yaml(finance_yaml):
            registry.register(a)

        assert len(registry) == 2
        assert registry.has("enroll_student")
        assert registry.has("freeze_account")

        # Each validates independently
        assert registry.validate_input("enroll_student", {"sid": "S1"}).valid
        assert not registry.validate_input("enroll_student", {}).valid
        assert registry.validate_input("freeze_account", {"aid": "A1"}).valid


# =============================================================================
# YAML file round-trip
# =============================================================================


class TestYAMLFileRoundTrip:
    """Load from actual YAML file and verify."""

    def test_load_edtech_example(self, tmp_path: Any) -> None:
        yaml_content = """
actions:
  - name: enrich_entity
    description: "Add computed properties"
    input:
      entity_id: { type: string, required: true }
      properties: { type: object, required: true }
    output:
      updated_entity: { type: object }
    preconditions:
      - "entity_exists(entity_id)"
    idempotent: true

  - name: enroll_student
    description: "Enroll a student"
    input:
      student_id: { type: string, required: true }
      course_id: { type: string, required: true }
      semester: { type: string, required: true }
    output:
      enrollment_id: { type: string }
    reversible: true
"""
        yaml_file = tmp_path / "actions_edtech.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        raw = yaml.safe_load(yaml_file.read_text())
        actions = load_actions_from_yaml(raw)

        assert len(actions) == 2
        registry = ActionRegistry()
        for a in actions:
            registry.register(a)

        # Validate
        assert registry.validate_input(
            "enroll_student", {"student_id": "S1", "course_id": "C1", "semester": "2026-S1"}
        ).valid
        assert not registry.validate_input("enroll_student", {"student_id": "S1"}).valid
