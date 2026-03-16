"""Integration tests simulating multi-agent collaboration.

3 agents:
  1. meeting-bot: writes Decision entities
  2. pm-agent: reads Decisions → writes ActionItems
  3. cto-agent: queries Risks

Uses an in-memory store to test cross-agent knowledge sharing without PG.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from ontology_engine.events.notifier import OntologyEvent
from ontology_engine.sdk.client import DescribeResult, IngestResult, OntologyClient


class InMemoryStore:
    """Minimal in-memory ontology store for integration testing."""

    def __init__(self):
        self.entities: dict[str, dict[str, Any]] = {}
        self.links: dict[str, dict[str, Any]] = {}
        self.agents: dict[str, dict[str, Any]] = {}
        self.events: list[OntologyEvent] = []
        self._ec = self._lc = 0

    def create_entity(self, etype: str, name: str, props: dict, created_by: str = "system") -> str:
        self._ec += 1
        eid = f"ENT-{self._ec:04d}"
        now = datetime.now(timezone.utc).isoformat()
        self.entities[eid] = {
            "id": eid, "entity_type": etype, "name": name, "properties": props,
            "status": "active", "created_by": created_by, "created_at": now,
        }
        self.events.append(OntologyEvent("entity.created", etype, eid, created_by, props))
        return eid

    def create_link(self, src: str, ltype: str, tgt: str, created_by: str = "system") -> str:
        self._lc += 1
        lid = f"LNK-{self._lc:04d}"
        self.links[lid] = {
            "id": lid, "link_type": ltype, "source_entity_id": src,
            "target_entity_id": tgt, "status": "active", "created_by": created_by,
        }
        self.events.append(OntologyEvent("link.created", ltype, lid, created_by))
        return lid

    def query(self, etype: str, filters: dict | None = None, limit: int = 20) -> list[dict]:
        results = []
        for e in self.entities.values():
            if e["entity_type"] != etype or e["status"] != "active":
                continue
            if filters and not all(e.get("properties", {}).get(k) == v for k, v in filters.items()):
                continue
            results.append(e)
            if len(results) >= limit:
                break
        return results

    def get_linked(self, eid: str, ltype: str | None = None, direction: str = "both") -> list[dict]:
        results = []
        for link in self.links.values():
            if link["status"] != "active":
                continue
            if ltype and link["link_type"] != ltype:
                continue
            nid = None
            if direction in ("outgoing", "both") and link["source_entity_id"] == eid:
                nid = link["target_entity_id"]
            elif direction in ("incoming", "both") and link["target_entity_id"] == eid:
                nid = link["source_entity_id"]
            if nid and nid in self.entities:
                results.append(self.entities[nid])
        return results

    def register_agent(self, aid: str, **kw: Any) -> dict:
        self.agents[aid] = {"id": aid, "status": "active", **kw}
        return self.agents[aid]


@pytest.fixture
def store():
    return InMemoryStore()


class TestMultiAgent:
    def test_meeting_bot_creates_decisions(self, store):
        store.register_agent("meeting-bot", produces=["Decision", "Person"])
        pid = store.create_entity("Person", "Yann", {"role": "CEO"}, "meeting-bot")
        did = store.create_entity("Decision", "Delay launch", {
            "summary": "Delay by 2 weeks", "decision_type": "tactical", "status": "active",
        }, "meeting-bot")
        store.create_link(pid, "makes", did, "meeting-bot")
        assert pid.startswith("ENT-") and did.startswith("ENT-")
        assert len(store.events) == 3

    def test_pm_reads_decisions_creates_actions(self, store):
        store.register_agent("meeting-bot", produces=["Decision"])
        store.register_agent("pm-agent", consumes=["Decision"], produces=["ActionItem"])
        did = store.create_entity("Decision", "Launch delayed", {
            "summary": "Delay by 2 weeks", "decision_type": "tactical", "status": "active",
        }, "meeting-bot")

        decisions = store.query("Decision", {"status": "active"})
        assert len(decisions) >= 1

        aid = store.create_entity("ActionItem", "Update QA plan", {
            "task": "Update QA plan", "priority": "high", "status": "pending",
        }, "pm-agent")
        store.create_link(did, "generates", aid, "pm-agent")
        linked = store.get_linked(did, "generates")
        assert len(linked) == 1 and linked[0]["entity_type"] == "ActionItem"

    def test_cto_queries_risks(self, store):
        store.register_agent("meeting-bot", produces=["Risk"])
        store.register_agent("cto-agent", consumes=["Risk"])
        rid = store.create_entity("Risk", "API latency", {
            "description": "Response times degrading", "impact": "high", "status": "active",
        }, "meeting-bot")
        aid = store.create_entity("ActionItem", "Add caching", {
            "task": "Redis caching", "priority": "critical", "status": "pending",
        }, "meeting-bot")
        store.create_link(aid, "mitigates", rid, "meeting-bot")

        risks = store.query("Risk")
        assert len(risks) == 1 and risks[0]["properties"]["impact"] == "high"
        mits = store.get_linked(rid, "mitigates", "incoming")
        assert len(mits) == 1

    def test_full_cross_agent_flow(self, store):
        store.register_agent("meeting-bot", produces=["Decision", "Person", "Risk"])
        store.register_agent("pm-agent", consumes=["Decision"], produces=["ActionItem"])
        store.register_agent("cto-agent", consumes=["Risk"])

        yid = store.create_entity("Person", "Yann", {"role": "CEO"}, "meeting-bot")
        store.create_entity("Person", "Felix", {"role": "COO"}, "meeting-bot")
        did = store.create_entity("Decision", "Pivot B2B", {
            "summary": "Pivot to B2B", "decision_type": "strategic", "status": "active",
        }, "meeting-bot")
        rid = store.create_entity("Risk", "Revenue gap", {
            "description": "6-month gap", "impact": "high", "status": "active",
        }, "meeting-bot")
        store.create_link(yid, "makes", did, "meeting-bot")
        store.create_link(did, "relates_to", rid, "meeting-bot")

        for dec in store.query("Decision", {"status": "active"}):
            aid = store.create_entity("ActionItem", f"Plan: {dec['name']}", {
                "task": f"Plan: {dec['name']}", "priority": "critical", "status": "pending",
            }, "pm-agent")
            store.create_link(dec["id"], "generates", aid, "pm-agent")

        assert len(store.entities) == 5  # 2 persons + 1 decision + 1 risk + 1 action
        assert len(store.links) == 3
        assert len(store.agents) == 3

    def test_agent_discovery(self, store):
        store.register_agent("meeting-bot", produces=["Decision"])
        store.register_agent("pm-agent", consumes=["Decision"])
        producers = [a for a in store.agents.values() if "Decision" in a.get("produces", [])]
        assert len(producers) == 1 and producers[0]["id"] == "meeting-bot"

    def test_event_tracking(self, store):
        store.create_entity("Decision", "T", {"status": "active"}, "bot-1")
        store.create_entity("Person", "A", {"role": "CTO"}, "bot-2")
        store.create_link("ENT-0001", "makes", "ENT-0002", "bot-1")
        ent_ev = [e for e in store.events if e.event_type == "entity.created"]
        lnk_ev = [e for e in store.events if e.event_type == "link.created"]
        assert len(ent_ev) == 2 and len(lnk_ev) == 1


class TestSDKContracts:
    def test_has_all_methods(self):
        c = OntologyClient(db_url="postgresql://mock")
        for m in ("assert_entity", "assert_link", "ingest", "query", "search",
                   "get_linked", "describe", "get_entity", "register_agent",
                   "list_agents", "get_agent", "subscribe", "connect", "close"):
            assert hasattr(c, m)

    def test_async_context_manager(self):
        c = OntologyClient(db_url="postgresql://mock")
        assert hasattr(c, "__aenter__") and hasattr(c, "__aexit__")

    def test_ingest_result(self):
        r = IngestResult(document_id="DOC-1", entities_created=5)
        assert r.document_id == "DOC-1" and r.link_ids == []

    def test_describe_result(self):
        r = DescribeResult(entity_type="Decision", properties=[{"name": "summary", "type": "string"}])
        assert r.entity_type == "Decision" and len(r.properties) == 1
