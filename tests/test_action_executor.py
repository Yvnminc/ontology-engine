"""Tests for Kinetic Layer — ActionExecutor."""

from __future__ import annotations

from typing import Any

import pytest

from ontology_engine.kinetic.action_executor import (
    ActionExecutor,
    ActionResult,
    ExecutionContext,
    RollbackResult,
)
from ontology_engine.kinetic.action_types import ActionRegistry, ActionType
from ontology_engine.kinetic.audit_trail import AuditTrail


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> ActionRegistry:
    r = ActionRegistry()
    r.register(ActionType(
        name="create_entity",
        description="Create an entity",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "entity_type": {"type": "string"},
            },
            "required": ["name", "entity_type"],
        },
        reversible=True,
    ))
    r.register(ActionType(
        name="update_property",
        description="Update a property",
        input_schema={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["entity_id", "key"],
        },
    ))
    r.register(ActionType(
        name="no_schema",
        description="Action with no schema",
    ))
    return r


@pytest.fixture
def audit() -> AuditTrail:
    return AuditTrail()


@pytest.fixture
def executor(registry: ActionRegistry, audit: AuditTrail) -> ActionExecutor:
    return ActionExecutor(registry, audit)


# =============================================================================
# Execution
# =============================================================================


class TestExecution:
    @pytest.mark.asyncio
    async def test_successful_execution(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"entity_id": "ENT-001"}

        executor.register_handler("create_entity", handler)
        result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
            ExecutionContext(actor="agent-1"),
        )

        assert result.status == "success"
        assert result.result == {"entity_id": "ENT-001"}
        assert result.action_name == "create_entity"
        assert result.execution_id.startswith("EXEC-")
        assert result.duration_ms >= 0
        assert result.error == ""

    @pytest.mark.asyncio
    async def test_validation_error(self, executor: ActionExecutor) -> None:
        result = await executor.execute(
            "create_entity",
            {"name": "Test"},  # missing entity_type
        )
        assert result.status == "validation_error"
        assert "entity_type" in result.error.lower() or "required" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_handler(self, executor: ActionExecutor) -> None:
        result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        assert result.status == "failed"
        assert "no handler" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handler_exception(self, executor: ActionExecutor) -> None:
        async def bad_handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            raise RuntimeError("Something went wrong")

        executor.register_handler("create_entity", bad_handler)
        result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        assert result.status == "failed"
        assert "RuntimeError" in result.error
        assert "Something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_default_context(self, executor: ActionExecutor) -> None:
        captured_ctx: list[ExecutionContext] = []

        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            captured_ctx.append(ctx)
            return {}

        executor.register_handler("no_schema", handler)
        await executor.execute("no_schema", {})
        assert captured_ctx[0].actor == "system"

    @pytest.mark.asyncio
    async def test_custom_context(self, executor: ActionExecutor) -> None:
        captured_ctx: list[ExecutionContext] = []

        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            captured_ctx.append(ctx)
            return {}

        executor.register_handler("no_schema", handler)
        ctx = ExecutionContext(actor="user-42", metadata={"request_id": "req-1"})
        await executor.execute("no_schema", {}, ctx)
        assert captured_ctx[0].actor == "user-42"
        assert captured_ctx[0].metadata["request_id"] == "req-1"

    @pytest.mark.asyncio
    async def test_no_schema_accepts_any_params(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"received": params}

        executor.register_handler("no_schema", handler)
        result = await executor.execute("no_schema", {"any": "thing"})
        assert result.status == "success"
        assert result.result["received"] == {"any": "thing"}


# =============================================================================
# Audit recording
# =============================================================================


class TestAuditRecording:
    @pytest.mark.asyncio
    async def test_success_audit(self, executor: ActionExecutor, audit: AuditTrail) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"id": "ENT-001"}

        executor.register_handler("create_entity", handler)
        result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
            ExecutionContext(actor="bot-1"),
        )

        entries = audit.query({"id": result.execution_id})
        assert len(entries) == 1
        entry = entries[0]
        assert entry.action_name == "create_entity"
        assert entry.actor == "bot-1"
        assert entry.status == "success"
        assert entry.result == {"id": "ENT-001"}

    @pytest.mark.asyncio
    async def test_failure_audit(self, executor: ActionExecutor, audit: AuditTrail) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            raise ValueError("bad value")

        executor.register_handler("create_entity", handler)
        result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )

        entries = audit.query({"status": "failed"})
        assert len(entries) == 1
        assert "bad value" in entries[0].error_message

    @pytest.mark.asyncio
    async def test_validation_error_audit(self, executor: ActionExecutor, audit: AuditTrail) -> None:
        await executor.execute("create_entity", {})  # missing required fields
        entries = audit.query({"status": "validation_error"})
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_multiple_executions_audit(self, executor: ActionExecutor, audit: AuditTrail) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {}

        executor.register_handler("no_schema", handler)

        for i in range(5):
            await executor.execute("no_schema", {"i": i}, ExecutionContext(actor=f"a{i}"))

        assert len(audit) == 5
        entries = audit.query({"actor": "a3"})
        assert len(entries) == 1


# =============================================================================
# Validation helper
# =============================================================================


class TestValidate:
    def test_validate_valid(self, executor: ActionExecutor) -> None:
        result = executor.validate("create_entity", {"name": "x", "entity_type": "Person"})
        assert result.valid is True

    def test_validate_invalid(self, executor: ActionExecutor) -> None:
        result = executor.validate("create_entity", {"name": 123})
        assert result.valid is False


# =============================================================================
# Rollback
# =============================================================================


class TestRollback:
    @pytest.mark.asyncio
    async def test_successful_rollback(self, executor: ActionExecutor, audit: AuditTrail) -> None:
        created_ids: list[str] = []

        async def create_handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            eid = "ENT-test"
            created_ids.append(eid)
            return {"entity_id": eid}

        async def rollback_handler(
            exec_id: str, params: dict[str, Any], result: dict[str, Any], ctx: ExecutionContext,
        ) -> dict[str, Any]:
            eid = result["entity_id"]
            created_ids.remove(eid)
            return {"deleted_entity_id": eid}

        executor.register_handler("create_entity", create_handler)
        executor.register_rollback_handler("create_entity", rollback_handler)

        # Execute
        exec_result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        assert exec_result.status == "success"
        assert len(created_ids) == 1

        # Rollback
        rb_result = await executor.rollback(exec_result.execution_id)
        assert rb_result.status == "rolled_back"
        assert rb_result.result == {"deleted_entity_id": "ENT-test"}
        assert len(created_ids) == 0

    @pytest.mark.asyncio
    async def test_rollback_nonexistent(self, executor: ActionExecutor) -> None:
        result = await executor.rollback("EXEC-nonexistent")
        assert result.status == "failed"
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rollback_non_reversible(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {}

        executor.register_handler("update_property", handler)
        exec_result = await executor.execute(
            "update_property",
            {"entity_id": "E1", "key": "k"},
        )
        assert exec_result.status == "success"

        rb_result = await executor.rollback(exec_result.execution_id)
        assert rb_result.status == "failed"
        assert "not reversible" in rb_result.error.lower()

    @pytest.mark.asyncio
    async def test_rollback_failed_execution(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            raise RuntimeError("fail")

        executor.register_handler("create_entity", handler)
        exec_result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        assert exec_result.status == "failed"

        rb_result = await executor.rollback(exec_result.execution_id)
        assert rb_result.status == "failed"
        assert "cannot rollback" in rb_result.error.lower()

    @pytest.mark.asyncio
    async def test_rollback_handler_exception(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"id": "E1"}

        async def bad_rollback(
            exec_id: str, params: dict[str, Any], result: dict[str, Any], ctx: ExecutionContext,
        ) -> dict[str, Any]:
            raise RuntimeError("rollback failed")

        executor.register_handler("create_entity", handler)
        executor.register_rollback_handler("create_entity", bad_rollback)

        exec_result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        rb_result = await executor.rollback(exec_result.execution_id)
        assert rb_result.status == "failed"
        assert "rollback failed" in rb_result.error

    @pytest.mark.asyncio
    async def test_rollback_no_handler(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {"id": "E1"}

        executor.register_handler("create_entity", handler)
        # No rollback handler registered

        exec_result = await executor.execute(
            "create_entity",
            {"name": "Test", "entity_type": "Person"},
        )
        rb_result = await executor.rollback(exec_result.execution_id)
        assert rb_result.status == "failed"
        assert "no rollback handler" in rb_result.error.lower()


# =============================================================================
# Handler registration
# =============================================================================


class TestHandlerRegistration:
    def test_register_handler_for_unknown_action_raises(self, executor: ActionExecutor) -> None:
        async def handler(params: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
            return {}

        with pytest.raises(KeyError):
            executor.register_handler("nonexistent", handler)
