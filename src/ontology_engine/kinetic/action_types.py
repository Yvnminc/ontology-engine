"""Action Type definitions and registry for the Kinetic Layer.

ActionType is the fundamental unit: a declarative, schema-validated definition
of an executable operation on the ontology. ActionRegistry holds all known
action types and provides input validation via JSON Schema.

Action types can be registered programmatically or loaded from YAML domain
schema files (see `load_actions_from_yaml`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jsonschema  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# =============================================================================
# Data models
# =============================================================================


@dataclass
class ActionType:
    """A declarative definition of an executable operation on the ontology.

    Attributes:
        name: Machine-readable identifier (e.g. ``"create_entity"``).
        description: Human-readable purpose of the action.
        input_schema: JSON Schema dict describing the expected input parameters.
        output_schema: JSON Schema dict describing the output shape.
        preconditions: List of human-readable conditions that must hold
            before the action can execute (evaluated externally).
        postconditions: List of conditions guaranteed after successful execution.
        side_effects: External effects the action may trigger (e.g. notifications).
        idempotent: Whether executing twice with the same params is safe.
        reversible: Whether the action can be rolled back.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    idempotent: bool = False
    reversible: bool = False


@dataclass
class ValidationResult:
    """Outcome of validating action input parameters against the schema.

    Attributes:
        valid: ``True`` if validation passed.
        errors: Human-readable error messages (empty when valid).
    """

    valid: bool
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Registry
# =============================================================================


class ActionRegistry:
    """In-memory registry of :class:`ActionType` definitions.

    Supports programmatic registration and YAML bulk-loading.
    Thread-safe for read-heavy workloads (single writer assumed).
    """

    def __init__(self) -> None:
        self._actions: dict[str, ActionType] = {}

    # ---- CRUD -----------------------------------------------------------------

    def register(self, action_type: ActionType) -> None:
        """Register (or replace) an action type.

        Raises:
            ValueError: If the action type has an empty name.
        """
        if not action_type.name:
            raise ValueError("ActionType.name must not be empty")
        if action_type.name in self._actions:
            logger.info("Replacing existing action type: %s", action_type.name)
        self._actions[action_type.name] = action_type
        logger.debug("Registered action type: %s", action_type.name)

    def get(self, name: str) -> ActionType:
        """Return the action type for *name*.

        Raises:
            KeyError: If the name is not registered.
        """
        try:
            return self._actions[name]
        except KeyError:
            raise KeyError(f"Action type not found: '{name}'") from None

    def list(self) -> list[ActionType]:
        """Return all registered action types in registration order."""
        return list(self._actions.values())

    def has(self, name: str) -> bool:
        """Check whether *name* is registered."""
        return name in self._actions

    def unregister(self, name: str) -> None:
        """Remove an action type.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._actions:
            raise KeyError(f"Action type not found: '{name}'")
        del self._actions[name]

    def clear(self) -> None:
        """Remove all registered action types."""
        self._actions.clear()

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    # ---- Validation -----------------------------------------------------------

    def validate_input(self, action_name: str, params: dict[str, Any]) -> ValidationResult:
        """Validate *params* against the registered input schema for *action_name*.

        If the action type has no ``input_schema`` (empty dict), any params
        are accepted. Validation uses JSON Schema Draft-7.

        Raises:
            KeyError: If *action_name* is not registered.
        """
        action_type = self.get(action_name)
        schema = action_type.input_schema
        if not schema:
            return ValidationResult(valid=True)

        # Ensure the schema has a type if not specified
        if "type" not in schema:
            schema = {**schema, "type": "object"}

        errors: list[str] = []
        try:
            jsonschema.validate(instance=params, schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(exc.message)
        except jsonschema.SchemaError as exc:
            errors.append(f"Invalid JSON Schema in action '{action_name}': {exc.message}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)


# =============================================================================
# YAML loading helpers
# =============================================================================


def _yaml_field_to_json_schema_type(field_def: dict[str, Any]) -> dict[str, Any]:
    """Convert a simplified YAML field definition to a JSON Schema property."""
    type_map = {
        "string": "string",
        "integer": "integer",
        "int": "integer",
        "float": "number",
        "number": "number",
        "boolean": "boolean",
        "bool": "boolean",
        "object": "object",
        "array": "array",
    }
    raw_type = str(field_def.get("type", "string")).lower()
    json_type = type_map.get(raw_type, "string")
    prop: dict[str, Any] = {"type": json_type}
    if "description" in field_def:
        prop["description"] = field_def["description"]
    if "enum" in field_def:
        prop["enum"] = field_def["enum"]
    if "default" in field_def:
        prop["default"] = field_def["default"]
    return prop


def _build_json_schema(fields: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON Schema ``object`` from a mapping of field names to defs."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, fdef in fields.items():
        if isinstance(fdef, dict):
            properties[name] = _yaml_field_to_json_schema_type(fdef)
            if fdef.get("required"):
                required.append(name)
        else:
            # Shorthand: field_name: type_string
            properties[name] = {"type": str(fdef)}
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def load_actions_from_yaml(data: dict[str, Any]) -> list[ActionType]:
    """Parse the ``actions`` section of a domain schema YAML dict.

    Expected format::

        actions:
          - name: enrich_entity
            description: "..."
            input:
              entity_id: { type: string, required: true }
            output:
              updated_entity: { type: object }
            preconditions:
              - "entity_exists(entity_id)"
            idempotent: true

    Returns:
        A list of :class:`ActionType` instances.
    """
    actions: list[ActionType] = []
    for item in data.get("actions", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if not name:
            logger.warning("Skipping action definition with empty name")
            continue

        input_schema = _build_json_schema(item.get("input", {}))
        output_schema = _build_json_schema(item.get("output", {}))

        actions.append(ActionType(
            name=name,
            description=item.get("description", ""),
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=item.get("preconditions", []),
            postconditions=item.get("postconditions", []),
            side_effects=item.get("side_effects", []),
            idempotent=bool(item.get("idempotent", False)),
            reversible=bool(item.get("reversible", False)),
        ))
    return actions
