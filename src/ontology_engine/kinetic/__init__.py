"""Kinetic Layer — Action Types, Executor, and Audit Trail.

The Kinetic Layer transforms the Ontology Engine from a passive knowledge graph
into an active operations system. It defines "verbs" (actions and mutations)
on top of the Semantic Layer's "nouns" (entities and relationships).

Core components:
  - ActionType / ActionRegistry: declare and register executable operations
  - ActionExecutor: validate, execute, and rollback actions
  - AuditTrail / AuditEntry: immutable record of every operation
"""

from ontology_engine.kinetic.action_types import (
    ActionRegistry,
    ActionType,
    ValidationResult,
)
from ontology_engine.kinetic.action_executor import (
    ActionExecutor,
    ActionResult,
    ExecutionContext,
    RollbackResult,
)
from ontology_engine.kinetic.audit_trail import (
    AuditEntry,
    AuditTrail,
)

__all__ = [
    "ActionType",
    "ActionRegistry",
    "ValidationResult",
    "ActionExecutor",
    "ActionResult",
    "ExecutionContext",
    "RollbackResult",
    "AuditEntry",
    "AuditTrail",
]
