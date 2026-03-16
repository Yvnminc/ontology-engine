"""API route definitions — all routes under /api/v1/."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ontology_engine.api.models import (
    AgentListResponse, AgentResponse, AssertEntityRequest, AssertLinkRequest,
    AssertResponse, DescribeResponse, EntityListResponse, EntityResponse,
    ExampleEntity, HealthResponse, IngestRequest, IngestResponse,
    LinkedEntityResponse, LinkedResponse, LinkTypeSchema, PropertySchema,
    RegisterAgentRequest, SearchResponse, SearchResultResponse,
)
from ontology_engine.sdk.client import OntologyClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_client: OntologyClient | None = None


def set_client(client: OntologyClient) -> None:
    global _client
    _client = client


def get_client() -> OntologyClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="OntologyClient not initialized")
    return _client


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/query", response_model=EntityListResponse)
async def query_entities(
    entity_type: str = Query(...),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
    keyword: str | None = Query(None),
    client: OntologyClient = Depends(get_client),
) -> EntityListResponse:
    filters: dict[str, Any] = {}
    if status:
        filters["status"] = status
    if keyword:
        filters["keyword"] = keyword
    entities = await client.query(entity_type, filters=filters, limit=limit, offset=offset)
    return EntityListResponse(
        entities=[EntityResponse(**_norm_entity(e)) for e in entities],
        total=len(entities), limit=limit, offset=offset,
    )


@router.get("/search", response_model=SearchResponse)
async def search_entities(
    q: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
    entity_type: str | None = Query(None),
    client: OntologyClient = Depends(get_client),
) -> SearchResponse:
    results = await client.search(q, limit=limit, entity_type=entity_type)
    return SearchResponse(
        results=[SearchResultResponse(
            id=r["id"], entity_type=r["entity_type"], name=r["name"],
            properties=r.get("properties", {}), relevance=r.get("relevance", 0.0),
        ) for r in results],
        query=q, total=len(results),
    )


@router.get("/entity/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: str, client: OntologyClient = Depends(get_client)) -> EntityResponse:
    entity = await client.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return EntityResponse(**_norm_entity(entity))


@router.get("/linked/{entity_id}", response_model=LinkedResponse)
async def get_linked(
    entity_id: str,
    link_type: str | None = Query(None),
    direction: str = Query("both", pattern="^(outgoing|incoming|both)$"),
    depth: int = Query(1, ge=1, le=5),
    client: OntologyClient = Depends(get_client),
) -> LinkedResponse:
    linked = await client.get_linked(entity_id, link_type=link_type, direction=direction, depth=depth)
    return LinkedResponse(
        entity_id=entity_id,
        linked=[LinkedEntityResponse(
            id=e["id"], entity_type=e["entity_type"], name=e["name"],
            properties=e.get("properties", {}), link=e.get("_link", {}),
        ) for e in linked],
        total=len(linked),
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, client: OntologyClient = Depends(get_client)) -> IngestResponse:
    result = await client.ingest(req.text, source=req.source, source_type=req.source_type)
    return IngestResponse(
        document_id=result.document_id, entities_created=result.entities_created,
        links_created=result.links_created, entity_ids=result.entity_ids,
        link_ids=result.link_ids, processing_time_ms=result.processing_time_ms,
    )


@router.post("/assert/entity", response_model=AssertResponse)
async def assert_entity(req: AssertEntityRequest, client: OntologyClient = Depends(get_client)) -> AssertResponse:
    entity_id = await client.assert_entity(
        entity_type=req.entity_type, properties=req.properties,
        source=req.source, confidence=req.confidence, name=req.name,
    )
    return AssertResponse(id=entity_id, type="entity", action="created")


@router.post("/assert/link", response_model=AssertResponse)
async def assert_link(req: AssertLinkRequest, client: OntologyClient = Depends(get_client)) -> AssertResponse:
    link_id = await client.assert_link(
        source_id=req.source_id, link_type=req.link_type, target_id=req.target_id,
        properties=req.properties, confidence=req.confidence,
    )
    return AssertResponse(id=link_id, type="link", action="created")


@router.get("/describe/{entity_type}", response_model=DescribeResponse)
async def describe(entity_type: str, client: OntologyClient = Depends(get_client)) -> DescribeResponse:
    result = await client.describe(entity_type)
    return DescribeResponse(
        entity_type=result.entity_type, description=result.description,
        properties=[PropertySchema(**p) for p in result.properties],
        required_fields=result.required_fields,
        link_types=[LinkTypeSchema(**lt) for lt in result.link_types],
        examples=[ExampleEntity(**ex) for ex in result.examples],
    )


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    status: str = Query("active"),
    produces: str | None = Query(None),
    consumes: str | None = Query(None),
    client: OntologyClient = Depends(get_client),
) -> AgentListResponse:
    agents = await client.list_agents(status=status, produces=produces, consumes=consumes)
    return AgentListResponse(
        agents=[AgentResponse(**_norm_agent(a)) for a in agents],
        total=len(agents),
    )


@router.post("/agents/register", response_model=AgentResponse)
async def register_agent(req: RegisterAgentRequest, client: OntologyClient = Depends(get_client)) -> AgentResponse:
    agent = await client.register_agent(
        id=req.id, display_name=req.display_name, description=req.description,
        produces=req.produces, consumes=req.consumes, capabilities=req.capabilities,
        version=req.version, metadata=req.metadata,
    )
    return AgentResponse(**_norm_agent(agent))


def _norm_entity(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": d.get("id", ""), "entity_type": d.get("entity_type", ""),
        "name": d.get("name", ""), "properties": d.get("properties", {}),
        "aliases": d.get("aliases", []), "confidence": d.get("confidence", 1.0),
        "version": d.get("version", 1), "created_at": d.get("created_at"),
        "updated_at": d.get("updated_at"), "created_by": d.get("created_by", "system"),
    }


def _norm_agent(agent: Any) -> dict[str, Any]:
    d = agent.model_dump() if hasattr(agent, "model_dump") else (agent if isinstance(agent, dict) else vars(agent))
    for key in ("registered_at", "last_seen_at"):
        val = d.get(key)
        if val and hasattr(val, "isoformat"):
            d[key] = val.isoformat()
    return d
