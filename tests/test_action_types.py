"""Tests for Kinetic Layer — ActionType and ActionRegistry."""

from __future__ import annotations

import pytest

from ontology_engine.kinetic.action_types import (
    ActionRegistry,
    ActionType,
    ValidationResult,
    load_actions_from_yaml,
)


# =============================================================================
# ActionType dataclass
# =============================================================================


class TestActionType:
    def test_defaults(self) -> None:
        at = ActionType(name="test", description="A test action")
        assert at.name == "test"
        assert at.description == "A test action"
        assert at.input_schema == {}
        assert at.output_schema == {}
        assert at.preconditions == []
        assert at.postconditions == []
        assert at.side_effects == []
        assert at.idempotent is False
        assert at.reversible is False

    def test_full_construction(self) -> None:
        at = ActionType(
            name="create_entity",
            description="Create a new entity",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            output_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            preconditions=["user_authenticated"],
            postconditions=["entity_exists"],
            side_effects=["notify_admin"],
            idempotent=True,
            reversible=True,
        )
        assert at.idempotent is True
        assert at.reversible is True
        assert len(at.preconditions) == 1
        assert at.input_schema["required"] == ["name"]


# =============================================================================
# ActionRegistry
# =============================================================================


class TestActionRegistry:
    @pytest.fixture
    def registry(self) -> ActionRegistry:
        return ActionRegistry()

    @pytest.fixture
    def sample_action(self) -> ActionType:
        return ActionType(
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
        )

    def test_register_and_get(self, registry: ActionRegistry, sample_action: ActionType) -> None:
        registry.register(sample_action)
        assert registry.get("create_entity") is sample_action

    def test_register_empty_name_raises(self, registry: ActionRegistry) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            registry.register(ActionType(name="", description="bad"))

    def test_get_missing_raises(self, registry: ActionRegistry) -> None:
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list(self, registry: ActionRegistry) -> None:
        a1 = ActionType(name="a", description="first")
        a2 = ActionType(name="b", description="second")
        registry.register(a1)
        registry.register(a2)
        actions = registry.list()
        assert len(actions) == 2
        assert actions[0].name == "a"
        assert actions[1].name == "b"

    def test_has(self, registry: ActionRegistry, sample_action: ActionType) -> None:
        assert registry.has("create_entity") is False
        registry.register(sample_action)
        assert registry.has("create_entity") is True
        assert "create_entity" in registry

    def test_unregister(self, registry: ActionRegistry, sample_action: ActionType) -> None:
        registry.register(sample_action)
        registry.unregister("create_entity")
        assert registry.has("create_entity") is False
        assert len(registry) == 0

    def test_unregister_missing_raises(self, registry: ActionRegistry) -> None:
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_clear(self, registry: ActionRegistry) -> None:
        registry.register(ActionType(name="a", description=""))
        registry.register(ActionType(name="b", description=""))
        registry.clear()
        assert len(registry) == 0

    def test_replace_existing(self, registry: ActionRegistry) -> None:
        v1 = ActionType(name="act", description="v1")
        v2 = ActionType(name="act", description="v2")
        registry.register(v1)
        registry.register(v2)
        assert registry.get("act").description == "v2"
        assert len(registry) == 1

    def test_len_and_contains(self, registry: ActionRegistry) -> None:
        assert len(registry) == 0
        registry.register(ActionType(name="x", description=""))
        assert len(registry) == 1
        assert "x" in registry
        assert "y" not in registry


# =============================================================================
# Input validation
# =============================================================================


class TestValidation:
    @pytest.fixture
    def registry(self) -> ActionRegistry:
        r = ActionRegistry()
        r.register(ActionType(
            name="create_entity",
            description="Create an entity",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["name"],
            },
        ))
        r.register(ActionType(
            name="no_schema",
            description="Action with no input schema",
        ))
        return r

    def test_valid_input(self, registry: ActionRegistry) -> None:
        result = registry.validate_input("create_entity", {"name": "Foo", "count": 5})
        assert result.valid is True
        assert result.errors == []

    def test_missing_required(self, registry: ActionRegistry) -> None:
        result = registry.validate_input("create_entity", {"count": 5})
        assert result.valid is False
        assert len(result.errors) == 1
        assert "name" in result.errors[0].lower() or "required" in result.errors[0].lower()

    def test_wrong_type(self, registry: ActionRegistry) -> None:
        result = registry.validate_input("create_entity", {"name": "Foo", "count": "not_int"})
        assert result.valid is False

    def test_no_schema_accepts_anything(self, registry: ActionRegistry) -> None:
        result = registry.validate_input("no_schema", {"anything": "goes"})
        assert result.valid is True

    def test_validate_nonexistent_action_raises(self, registry: ActionRegistry) -> None:
        with pytest.raises(KeyError):
            registry.validate_input("nonexistent", {})

    def test_empty_params_with_no_required(self, registry: ActionRegistry) -> None:
        r = ActionRegistry()
        r.register(ActionType(
            name="optional",
            description="All optional",
            input_schema={
                "type": "object",
                "properties": {"note": {"type": "string"}},
            },
        ))
        result = r.validate_input("optional", {})
        assert result.valid is True


# =============================================================================
# YAML loading
# =============================================================================


class TestYAMLLoading:
    def test_load_basic(self) -> None:
        yaml_data = {
            "actions": [
                {
                    "name": "enrich_entity",
                    "description": "Add computed properties",
                    "input": {
                        "entity_id": {"type": "string", "required": True},
                        "properties": {"type": "object", "required": True},
                    },
                    "output": {
                        "updated_entity": {"type": "object"},
                    },
                    "preconditions": ["entity_exists(entity_id)"],
                    "idempotent": True,
                },
            ],
        }
        actions = load_actions_from_yaml(yaml_data)
        assert len(actions) == 1
        a = actions[0]
        assert a.name == "enrich_entity"
        assert a.idempotent is True
        assert a.reversible is False
        assert len(a.preconditions) == 1
        assert a.input_schema["required"] == ["entity_id", "properties"]

    def test_load_multiple(self) -> None:
        yaml_data = {
            "actions": [
                {"name": "a1", "description": "first"},
                {"name": "a2", "description": "second", "reversible": True},
            ],
        }
        actions = load_actions_from_yaml(yaml_data)
        assert len(actions) == 2
        assert actions[1].reversible is True

    def test_load_empty(self) -> None:
        assert load_actions_from_yaml({}) == []
        assert load_actions_from_yaml({"actions": []}) == []

    def test_skip_invalid_entries(self) -> None:
        yaml_data = {
            "actions": [
                "not_a_dict",
                {"description": "missing name"},
                {"name": "", "description": "empty name"},
                {"name": "valid", "description": "ok"},
            ],
        }
        actions = load_actions_from_yaml(yaml_data)
        assert len(actions) == 1
        assert actions[0].name == "valid"

    def test_load_into_registry(self) -> None:
        yaml_data = {
            "actions": [
                {
                    "name": "freeze_account",
                    "description": "Freeze an account",
                    "input": {
                        "account_id": {"type": "string", "required": True},
                        "reason": {"type": "string", "required": True},
                    },
                    "reversible": True,
                },
            ],
        }
        actions = load_actions_from_yaml(yaml_data)
        registry = ActionRegistry()
        for a in actions:
            registry.register(a)

        assert registry.has("freeze_account")
        result = registry.validate_input("freeze_account", {
            "account_id": "ACC-001",
            "reason": "suspicious activity",
        })
        assert result.valid is True

    def test_yaml_file_loading(self, tmp_path: Any) -> None:
        """Test loading from an actual YAML file."""
        import yaml as pyyaml

        yaml_content = {
            "actions": [
                {
                    "name": "test_action",
                    "description": "A test action from file",
                    "input": {"name": {"type": "string", "required": True}},
                    "idempotent": True,
                },
            ],
        }
        yaml_file = tmp_path / "actions_test.yaml"
        yaml_file.write_text(pyyaml.dump(yaml_content), encoding="utf-8")

        raw = pyyaml.safe_load(yaml_file.read_text())
        actions = load_actions_from_yaml(raw)
        assert len(actions) == 1
        assert actions[0].name == "test_action"
