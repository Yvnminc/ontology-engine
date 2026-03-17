"""Action Executor — validates, executes, and optionally rolls back actions.

The executor is the runtime heart of the Kinetic Layer. It:

1. Validates input parameters against the registered JSON Schema.
2. Checks preconditions (pluggable, caller-supplied logic).
3. Invokes the handler function inside an execution context.
4. Records the result in the :class:`AuditTrail`.
5. Supports rollback for reversible actions.

Handler functions are registered per action name and receive
``(params, context)`` — they are pure domain logic, intentionally
decoupled from database transactions so that callers can wrap them
in their own transaction boundaries.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

from ontology_engine.kinetic.action_types import ActionRegistry, ValidationResult
from ontology_engine.kinetic.audit_trail import AuditEntry, AuditTrail

logger = logging.getLogger(__name__)


# Type alias for async handler functions
ActionHandler = Callable[
    [dict[str, Any], "ExecutionContext"],
    Coroutine[Any, Any, dict[str, Any]],
]
RollbackHandler = Callable[
    [str, dict[str, Any], dict[str, Any], "ExecutionContext"],
    Coroutine[Any, Any, dict[str, Any]],
]


# =============================================================================
# Data models
# =============================================================================


@dataclass
class ExecutionContext:
    """Ambient context passed to every action handler.

    Attributes:
        actor: Identifier of the agent or user triggering the action.
        metadata: Free-form dict for caller-specific context (e.g. request_id).
    """

    actor: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Outcome of an action execution.

    Attributes:
        execution_id: Unique ID for this execution (same as the audit entry ID).
        action_name: The action that was executed.
        status: ``"success"``, ``"failed"``, or ``"validation_error"``.
        result: The return value from the handler (or empty dict on failure).
        error: Error message on failure (empty string on success).
        duration_ms: Wall-clock time in milliseconds.
    """

    execution_id: str
    action_name: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_ms: int = 0


@dataclass
class RollbackResult:
    """Outcome of rolling back a previous execution.

    Attributes:
        execution_id: The original execution that was rolled back.
        status: ``"rolled_back"`` or ``"failed"``.
        result: Data returned by the rollback handler.
        error: Error message on failure.
    """

    execution_id: str
    status: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# =============================================================================
# Executor
# =============================================================================


class ActionExecutor:
    """Validates and dispatches action executions, recording audit entries.

    Usage::

        registry = ActionRegistry()
        audit = AuditTrail()
        executor = ActionExecutor(registry, audit)

        # Register a handler for "create_entity"
        async def handle_create(params, ctx):
            return {"entity_id": "ENT-123"}

        executor.register_handler("create_entity", handle_create)

        result = await executor.execute(
            "create_entity", {"name": "Foo"}, ExecutionContext(actor="agent-1")
        )
    """

    def __init__(self, registry: ActionRegistry, audit: AuditTrail) -> None:
        self._registry = registry
        self._audit = audit
        self._handlers: dict[str, ActionHandler] = {}
        self._rollback_handlers: dict[str, RollbackHandler] = {}

    # ---- Handler registration ------------------------------------------------

    def register_handler(self, action_name: str, handler: ActionHandler) -> None:
        """Bind *handler* to *action_name*.

        Raises:
            KeyError: If *action_name* is not in the registry.
        """
        # Verify the action type exists
        self._registry.get(action_name)
        self._handlers[action_name] = handler
        logger.debug("Handler registered for action: %s", action_name)

    def register_rollback_handler(self, action_name: str, handler: RollbackHandler) -> None:
        """Bind a rollback handler for *action_name*.

        The rollback handler receives ``(execution_id, params, result, context)``
        and should undo the effects of the original execution.
        """
        self._rollback_handlers[action_name] = handler
        logger.debug("Rollback handler registered for action: %s", action_name)

    # ---- Validate ------------------------------------------------------------

    def validate(self, action_name: str, params: dict[str, Any]) -> ValidationResult:
        """Validate *params* against the action's input schema.

        Convenience wrapper around :meth:`ActionRegistry.validate_input`.
        """
        return self._registry.validate_input(action_name, params)

    # ---- Execute -------------------------------------------------------------

    async def execute(
        self,
        action_name: str,
        params: dict[str, Any],
        context: ExecutionContext | None = None,
    ) -> ActionResult:
        """Execute an action end-to-end.

        Steps:
          1. Validate input.
          2. Look up handler.
          3. Run handler.
          4. Record audit entry.

        Returns an :class:`ActionResult` — never raises for domain errors,
        only for programming bugs (e.g. missing handler).
        """
        ctx = context or ExecutionContext()
        execution_id = f"EXEC-{uuid.uuid4().hex[:12]}"
        t0 = time.monotonic()

        # 1. Validate
        validation = self._registry.validate_input(action_name, params)
        if not validation.valid:
            duration_ms = int((time.monotonic() - t0) * 1000)
            error_msg = "; ".join(validation.errors)
            await self._record_audit(
                execution_id, action_name, params, {}, ctx.actor,
                "validation_error", duration_ms, error_msg,
            )
            return ActionResult(
                execution_id=execution_id,
                action_name=action_name,
                status="validation_error",
                error=error_msg,
                duration_ms=duration_ms,
            )

        # 2. Look up handler
        handler = self._handlers.get(action_name)
        if handler is None:
            duration_ms = int((time.monotonic() - t0) * 1000)
            error_msg = f"No handler registered for action '{action_name}'"
            await self._record_audit(
                execution_id, action_name, params, {}, ctx.actor,
                "failed", duration_ms, error_msg,
            )
            return ActionResult(
                execution_id=execution_id,
                action_name=action_name,
                status="failed",
                error=error_msg,
                duration_ms=duration_ms,
            )

        # 3. Run handler
        try:
            result_data = await handler(params, ctx)
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Action '%s' failed: %s", action_name, error_msg, exc_info=True)
            await self._record_audit(
                execution_id, action_name, params, {}, ctx.actor,
                "failed", duration_ms, error_msg,
            )
            return ActionResult(
                execution_id=execution_id,
                action_name=action_name,
                status="failed",
                error=error_msg,
                duration_ms=duration_ms,
            )

        duration_ms = int((time.monotonic() - t0) * 1000)

        # 4. Audit
        await self._record_audit(
            execution_id, action_name, params, result_data, ctx.actor,
            "success", duration_ms, "",
        )

        return ActionResult(
            execution_id=execution_id,
            action_name=action_name,
            status="success",
            result=result_data,
            duration_ms=duration_ms,
        )

    # ---- Rollback ------------------------------------------------------------

    async def rollback(
        self,
        execution_id: str,
        context: ExecutionContext | None = None,
    ) -> RollbackResult:
        """Roll back a previously executed action.

        Looks up the original audit entry, checks that the action type is
        reversible, and delegates to the registered rollback handler.
        """
        ctx = context or ExecutionContext()

        # Find the original audit entry
        entries = self._audit.query({"id": execution_id})
        if not entries:
            return RollbackResult(
                execution_id=execution_id,
                status="failed",
                error=f"Execution not found: {execution_id}",
            )
        entry = entries[0]

        if entry.status != "success":
            return RollbackResult(
                execution_id=execution_id,
                status="failed",
                error=f"Cannot rollback execution with status '{entry.status}'",
            )

        # Check the action type is reversible
        action_type = self._registry.get(entry.action_name)
        if not action_type.reversible:
            return RollbackResult(
                execution_id=execution_id,
                status="failed",
                error=f"Action '{entry.action_name}' is not reversible",
            )

        # Look up rollback handler
        rb_handler = self._rollback_handlers.get(entry.action_name)
        if rb_handler is None:
            return RollbackResult(
                execution_id=execution_id,
                status="failed",
                error=f"No rollback handler for action '{entry.action_name}'",
            )

        try:
            rb_result = await rb_handler(execution_id, entry.params, entry.result, ctx)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Rollback failed for %s: %s", execution_id, error_msg, exc_info=True)
            return RollbackResult(
                execution_id=execution_id,
                status="failed",
                error=error_msg,
            )

        # Update the original audit entry status
        entry.status = "rolled_back"
        # Record a new audit entry for the rollback itself
        rollback_id = f"RBCK-{uuid.uuid4().hex[:12]}"
        await self._record_audit(
            rollback_id, f"rollback:{entry.action_name}", entry.params,
            rb_result, ctx.actor, "success", 0,
            f"Rolled back execution {execution_id}",
        )

        return RollbackResult(
            execution_id=execution_id,
            status="rolled_back",
            result=rb_result,
        )

    # ---- Internal ------------------------------------------------------------

    async def _record_audit(
        self,
        execution_id: str,
        action_name: str,
        params: dict[str, Any],
        result: dict[str, Any],
        actor: str,
        status: str,
        duration_ms: int,
        error_message: str = "",
    ) -> None:
        """Create and record an :class:`AuditEntry`."""
        entry = AuditEntry(
            id=execution_id,
            action_name=action_name,
            params=params,
            result=result,
            actor=actor,
            timestamp=datetime.now(timezone.utc),
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        self._audit.record(entry)
