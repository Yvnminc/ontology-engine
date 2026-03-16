"""Pydantic models for the HTTP API request/response serialization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Response Models ---

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    db_schema: str = "ontology"


class EntityResponse(BaseModel):
    id: str
    entity_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    version: int = 1
    created_at: str | None = None
    updated_at: str | None = None
    created_by: str = "system"


class EntityListResponse(BaseModel):
    entities: list[EntityResponse]
    total: int
    limit: int
    offset: int


class SearchResultResponse(BaseModel):
    id: str
    entity_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    relevance: float = 0.0


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
    query: str
    total: int


class LinkedEntityResponse(BaseModel):
    id: str
    entity_type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    link: dict[str, Any] = Field(default_factory=dict)


class LinkedResponse(BaseModel):
    entity_id: str
    linked: list[LinkedEntityResponse]
    total: int


class PropertySchema(BaseModel):
    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    enum_values: list[str] | None = None


class LinkTypeSchema(BaseModel):
    link_type: str
    direction: str


class ExampleEntity(BaseModel):
    id: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class DescribeResponse(BaseModel):
    entity_type: str
    description: str = ""
    properties: list[PropertySchema] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    link_types: list[LinkTypeSchema] = Field(default_factory=list)
    examples: list[ExampleEntity] = Field(default_factory=list)


class AgentResponse(BaseModel):
    id: str
    display_name: str = ""
    description: str = ""
    produces: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    registered_at: str | None = None
    last_seen_at: str | None = None


class AgentListResponse(BaseModel):
    agents: list[AgentResponse]
    total: int


# --- Request Models ---

class IngestRequest(BaseModel):
    text: str
    source: str
    source_type: str = "document"


class IngestResponse(BaseModel):
    document_id: str = ""
    entities_created: int = 0
    links_created: int = 0
    entity_ids: list[str] = Field(default_factory=list)
    link_ids: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0


class AssertEntityRequest(BaseModel):
    entity_type: str
    properties: dict[str, Any]
    source: str
    confidence: float = 0.9
    name: str | None = None


class AssertLinkRequest(BaseModel):
    source_id: str
    link_type: str
    target_id: str
    properties: dict[str, Any] | None = None
    confidence: float = 0.9


class AssertResponse(BaseModel):
    id: str
    type: str  # "entity" or "link"
    action: str  # "created" or "updated"


class RegisterAgentRequest(BaseModel):
    id: str
    display_name: str = ""
    description: str = ""
    produces: list[str] = Field(default_factory=list)
    consumes: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
