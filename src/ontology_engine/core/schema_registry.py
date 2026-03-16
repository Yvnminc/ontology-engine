"""Domain Schema loader and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ontology_engine.core.errors import ConfigError
from ontology_engine.core.schema_format import (
    DomainSchemaModel, EntityTypeDefinition, ExtractionConfig,
    LinkTypeDefinition, ValidationRule,
)


class DomainSchema:
    """Loaded domain schema with convenience accessors."""

    def __init__(self, model: DomainSchemaModel):
        self._model = model
        self._entity_map = {et.name: et for et in model.entity_types}
        self._link_map = {lt.name: lt for lt in model.link_types}

    @classmethod
    def from_yaml(cls, path: str | Path) -> DomainSchema:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Schema file not found: {p}")
        if p.suffix not in (".yaml", ".yml"):
            raise ConfigError(f"Expected YAML file, got: {p.suffix}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Schema YAML must be a mapping, got {type(raw).__name__}")
        try:
            model = DomainSchemaModel.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Schema validation failed for {p}: {exc}") from exc
        return cls(model)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainSchema:
        return cls(DomainSchemaModel.model_validate(data))

    @property
    def domain(self) -> str: return self._model.domain
    @property
    def version(self) -> str: return self._model.version
    @property
    def description(self) -> str: return self._model.description
    @property
    def model(self) -> DomainSchemaModel: return self._model
    @property
    def entity_types(self) -> list[EntityTypeDefinition]: return self._model.entity_types
    @property
    def link_types(self) -> list[LinkTypeDefinition]: return self._model.link_types
    @property
    def extraction(self) -> ExtractionConfig: return self._model.extraction
    @property
    def validation_rules(self) -> list[ValidationRule]: return self._model.validation_rules

    def get_entity_type(self, name: str) -> EntityTypeDefinition | None:
        return self._entity_map.get(name)

    def get_link_type(self, name: str) -> LinkTypeDefinition | None:
        return self._link_map.get(name)

    def entity_type_names(self) -> list[str]:
        return [et.name for et in self._model.entity_types]

    def link_type_names(self) -> list[str]:
        return [lt.name for lt in self._model.link_types]

    def has_entity_type(self, name: str) -> bool: return name in self._entity_map
    def has_link_type(self, name: str) -> bool: return name in self._link_map

    def __repr__(self) -> str:
        return (f"DomainSchema(domain='{self.domain}', version='{self.version}', "
                f"entity_types={len(self.entity_types)}, link_types={len(self.link_types)})")


class SchemaRegistry:
    """Singleton registry for domain schemas."""
    _instance: SchemaRegistry | None = None
    _initialized: bool = False

    def __new__(cls) -> SchemaRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not SchemaRegistry._initialized:
            self._schemas: dict[str, DomainSchema] = {}
            self._active: str | None = None
            SchemaRegistry._initialized = True

    def register(self, schema: DomainSchema) -> None:
        self._schemas[schema.domain] = schema
        if self._active is None:
            self._active = schema.domain

    def register_from_yaml(self, path: str | Path) -> DomainSchema:
        s = DomainSchema.from_yaml(path)
        self.register(s)
        return s

    def get(self, domain: str) -> DomainSchema | None: return self._schemas.get(domain)

    def get_active(self) -> DomainSchema | None:
        return self._schemas.get(self._active) if self._active else None

    def set_active(self, domain: str) -> None:
        if domain not in self._schemas:
            raise ConfigError(f"Domain '{domain}' not registered")
        self._active = domain

    def list_domains(self) -> list[dict[str, Any]]:
        return [{"domain": n, "version": s.version, "description": s.description,
                 "entity_types": len(s.entity_types), "link_types": len(s.link_types),
                 "active": n == self._active} for n, s in self._schemas.items()]

    def has_domain(self, d: str) -> bool: return d in self._schemas

    def unregister(self, domain: str) -> None:
        if domain in self._schemas:
            del self._schemas[domain]
            if self._active == domain:
                self._active = next(iter(self._schemas), None)

    def clear(self) -> None:
        self._schemas.clear(); self._active = None

    @classmethod
    def reset(cls) -> None:
        cls._instance = None; cls._initialized = False

    def __len__(self) -> int: return len(self._schemas)
