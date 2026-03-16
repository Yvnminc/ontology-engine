"""Domain Schema YAML format definitions (Pydantic v2 models)."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PropertyType(str, enum.Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    ENUM = "enum"
    LIST_STRING = "list[string]"


class PropertyDefinition(BaseModel):
    name: str
    type: PropertyType
    required: bool = False
    description: str = ""
    default: Any = None
    enum_values: list[str] | None = None

    @model_validator(mode="after")
    def enum_must_have_values(self) -> PropertyDefinition:
        if self.type == PropertyType.ENUM and not self.enum_values:
            raise ValueError("Property of type 'enum' must define 'enum_values'")
        if self.type != PropertyType.ENUM and self.enum_values:
            raise ValueError("'enum_values' should only be set for type 'enum'")
        return self


class ValidationRule(BaseModel):
    field: str
    rule: str
    params: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class EntityTypeDefinition(BaseModel):
    name: str
    description: str = ""
    properties: list[PropertyDefinition] = Field(default_factory=list)
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    extraction_hint: str = ""

    @field_validator("name")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if not v or not v[0].isupper():
            raise ValueError(f"Entity type name must start with uppercase: '{v}'")
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Entity type name must be alphanumeric (with underscores): '{v}'")
        return v


class LinkTypeDefinition(BaseModel):
    name: str
    description: str = ""
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    properties: list[PropertyDefinition] = Field(default_factory=list)
    directional: bool = True

    @field_validator("name")
    @classmethod
    def name_must_be_snake_case(cls, v: str) -> str:
        if not v or not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(f"Link type name must be lowercase snake_case: '{v}'")
        return v


class ExtractionConfig(BaseModel):
    system_prompt: str = ""
    entity_prompt_template: str = ""
    relation_prompt_template: str = ""
    decision_prompt_template: str = ""
    action_prompt_template: str = ""


class DomainSchemaModel(BaseModel):
    domain: str
    version: str = "1.0.0"
    description: str = ""
    entity_types: list[EntityTypeDefinition] = Field(min_length=1)
    link_types: list[LinkTypeDefinition] = Field(default_factory=list)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    validation_rules: list[ValidationRule] = Field(default_factory=list)

    @field_validator("domain")
    @classmethod
    def domain_must_be_valid(cls, v: str) -> str:
        if not v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid domain name: '{v}'")
        return v

    @model_validator(mode="after")
    def link_types_reference_valid_entities(self) -> DomainSchemaModel:
        entity_names = {et.name for et in self.entity_types}
        for lt in self.link_types:
            for src in lt.source_types:
                if src not in entity_names:
                    raise ValueError(f"Link '{lt.name}' source_type '{src}' not in entity types: {entity_names}")
            for tgt in lt.target_types:
                if tgt not in entity_names:
                    raise ValueError(f"Link '{lt.name}' target_type '{tgt}' not in entity types: {entity_names}")
        return self
