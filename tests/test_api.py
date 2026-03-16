"""Tests for the HTTP API routes (FastAPI TestClient with mocked OntologyClient)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ontology_engine.sdk.client import DescribeResult, IngestResult
from ontology_engine.sdk.registry import AgentInfo

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")


@pytest.fixture
def mock_client():
    """MagicMock base avoids AsyncMock intercepting assert_* as test assertions."""
    c = MagicMock()
    for m in ("connect", "close", "query", "search", "get_entity", "get_linked",
              "ingest", "assert_entity", "assert_link", "describe",
              "list_agents", "register_agent"):
        setattr(c, m, AsyncMock())
    return c


@pytest.fixture
def app(mock_client):
    from fastapi import FastAPI
    from ontology_engine.api.routes import router, set_client
    app = FastAPI()
    app.include_router(router)
    set_client(mock_client)
    return app


@pytest.fixture
def tc(app):
    return TestClient(app)


class TestHealth:
    def test_ok(self, tc):
        r = tc.get("/api/v1/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


class TestQuery:
    def test_query(self, tc, mock_client):
        mock_client.query.return_value = [{
            "id": "ENT-1", "entity_type": "Decision", "name": "Test",
            "properties": {}, "aliases": [], "confidence": 0.9, "version": 1,
            "created_at": "2026-03-15T10:00:00", "updated_at": "2026-03-15T10:00:00",
            "created_by": "test",
        }]
        r = tc.get("/api/v1/query?entity_type=Decision")
        assert r.status_code == 200 and r.json()["total"] == 1

    def test_missing_type(self, tc):
        assert tc.get("/api/v1/query").status_code == 422

    def test_with_filters(self, tc, mock_client):
        mock_client.query.return_value = []
        assert tc.get("/api/v1/query?entity_type=Decision&status=active").status_code == 200


class TestSearch:
    def test_search(self, tc, mock_client):
        mock_client.search.return_value = [
            {"id": "E1", "entity_type": "Decision", "name": "x", "properties": {}, "relevance": 0.85}
        ]
        r = tc.get("/api/v1/search?q=launch")
        assert r.status_code == 200 and r.json()["results"][0]["relevance"] == 0.85

    def test_missing_q(self, tc):
        assert tc.get("/api/v1/search").status_code == 422


class TestEntity:
    def test_found(self, tc, mock_client):
        mock_client.get_entity.return_value = {
            "id": "ENT-1", "entity_type": "Person", "name": "Alice",
            "properties": {}, "aliases": [], "confidence": 0.95, "version": 1,
            "created_at": None, "updated_at": None, "created_by": "system",
        }
        assert tc.get("/api/v1/entity/ENT-1").json()["name"] == "Alice"

    def test_not_found(self, tc, mock_client):
        mock_client.get_entity.return_value = None
        assert tc.get("/api/v1/entity/X").status_code == 404


class TestLinked:
    def test_linked(self, tc, mock_client):
        mock_client.get_linked.return_value = [{
            "id": "ENT-l", "entity_type": "Project", "name": "WM", "properties": {},
            "_link": {"link_id": "L1", "link_type": "owns", "direction": "outgoing", "depth": 1},
        }]
        r = tc.get("/api/v1/linked/ENT-1")
        assert r.status_code == 200 and r.json()["total"] == 1

    def test_params(self, tc, mock_client):
        mock_client.get_linked.return_value = []
        tc.get("/api/v1/linked/ENT-1?link_type=makes&direction=outgoing&depth=2")
        mock_client.get_linked.assert_called_once_with("ENT-1", link_type="makes", direction="outgoing", depth=2)


class TestIngest:
    def test_ingest(self, tc, mock_client):
        mock_client.ingest.return_value = IngestResult(document_id="DOC-1", entities_created=3)
        r = tc.post("/api/v1/ingest", json={"text": "...", "source": "m:2026"})
        assert r.status_code == 200 and r.json()["document_id"] == "DOC-1"


class TestAssert:
    def test_entity(self, tc, mock_client):
        mock_client.assert_entity.return_value = "ENT-new"
        r = tc.post("/api/v1/assert/entity", json={
            "entity_type": "Decision", "properties": {"summary": "x"}, "source": "m:2026",
        })
        assert r.status_code == 200 and r.json()["id"] == "ENT-new"

    def test_link(self, tc, mock_client):
        mock_client.assert_link.return_value = "LNK-new"
        r = tc.post("/api/v1/assert/link", json={
            "source_id": "ENT-a", "link_type": "makes", "target_id": "ENT-b",
        })
        assert r.status_code == 200 and r.json()["id"] == "LNK-new"


class TestDescribe:
    def test_describe(self, tc, mock_client):
        mock_client.describe.return_value = DescribeResult(
            entity_type="Decision", description="决策",
            properties=[{"name": "summary", "type": "string", "required": True}],
            required_fields=["summary"],
            link_types=[{"link_type": "makes", "direction": "incoming"}],
            examples=[{"id": "E1", "name": "T", "properties": {}}],
        )
        r = tc.get("/api/v1/describe/Decision")
        assert r.status_code == 200 and r.json()["entity_type"] == "Decision"


class TestAgents:
    def test_list(self, tc, mock_client):
        now = datetime.now(timezone.utc)
        mock_client.list_agents.return_value = [
            AgentInfo(id="b1", display_name="Bot", produces=["Decision"],
                      registered_at=now, last_seen_at=now)
        ]
        r = tc.get("/api/v1/agents")
        assert r.status_code == 200 and r.json()["total"] == 1

    def test_register(self, tc, mock_client):
        now = datetime.now(timezone.utc)
        mock_client.register_agent.return_value = AgentInfo(
            id="new", display_name="New", registered_at=now, last_seen_at=now)
        r = tc.post("/api/v1/agents/register", json={"id": "new", "display_name": "New"})
        assert r.status_code == 200 and r.json()["id"] == "new"
