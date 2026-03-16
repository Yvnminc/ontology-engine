"""Tests for core/types.py — Entity, Link, Provenance, ExtractionResult, Validation."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from ontology_engine.core.types import (
    ActionItemStatus,
    DecisionStatus,
    DecisionType,
    Entity,
    EntityStatus,
    EntityType,
    ExtractionResult,
    ExtractedActionItem,
    ExtractedDecision,
    ExtractedEntity,
    ExtractedLink,
    Link,
    LinkType,
    Provenance,
    ProjectStatus,
    RiskCategory,
    RiskImpact,
    ValidationError as VError,
    ValidationResult,
)


# =============================================================================
# Enums
# =============================================================================


class TestEntityType:
    def test_all_six_types(self):
        assert len(EntityType) == 6
        expected = {"Person", "Decision", "ActionItem", "Project", "Risk", "Deadline"}
        assert {e.value for e in EntityType} == expected

    def test_str_enum(self):
        assert EntityType.PERSON == "Person"
        assert isinstance(EntityType.PERSON, str)


class TestLinkType:
    def test_all_thirteen_types(self):
        assert len(LinkType) == 13

    def test_values(self):
        assert LinkType.PARTICIPATES_IN == "participates_in"
        assert LinkType.MAKES == "makes"
        assert LinkType.ASSIGNED_TO == "assigned_to"
        assert LinkType.DEADLINE_FOR == "deadline_for"


class TestDecisionType:
    def test_values(self):
        assert DecisionType.STRATEGIC == "strategic"
        assert DecisionType.TACTICAL == "tactical"
        assert DecisionType.OPERATIONAL == "operational"


class TestStatusEnums:
    def test_entity_status(self):
        assert EntityStatus.ACTIVE == "active"
        assert EntityStatus.ARCHIVED == "archived"
        assert EntityStatus.DELETED == "deleted"

    def test_action_item_status(self):
        assert len(ActionItemStatus) == 5

    def test_decision_status(self):
        assert len(DecisionStatus) == 4

    def test_project_status(self):
        assert len(ProjectStatus) == 5

    def test_risk_enums(self):
        assert len(RiskImpact) == 3
        assert len(RiskCategory) == 5


# =============================================================================
# Entity Model
# =============================================================================


class TestEntity:
    def test_minimal_entity(self):
        e = Entity(entity_type=EntityType.PERSON, name="Alice")
        assert e.id is None
        assert e.entity_type == EntityType.PERSON
        assert e.name == "Alice"
        assert e.properties == {}
        assert e.aliases == []
        assert e.status == EntityStatus.ACTIVE
        assert e.confidence == 1.0
        assert e.version == 1
        assert e.created_by == "system"

    def test_full_entity(self):
        e = Entity(
            id="ENT-123",
            entity_type=EntityType.PROJECT,
            name="WhiteMirror",
            properties={"description": "AI startup"},
            aliases=["白镜", "WM"],
            status=EntityStatus.ACTIVE,
            confidence=0.95,
            version=3,
            created_by="pipeline",
        )
        assert e.id == "ENT-123"
        assert len(e.aliases) == 2
        assert e.properties["description"] == "AI startup"

    def test_confidence_bounds(self):
        # Valid bounds
        Entity(entity_type=EntityType.PERSON, name="X", confidence=0.0)
        Entity(entity_type=EntityType.PERSON, name="X", confidence=1.0)

        # Out of bounds
        with pytest.raises(ValidationError):
            Entity(entity_type=EntityType.PERSON, name="X", confidence=1.5)
        with pytest.raises(ValidationError):
            Entity(entity_type=EntityType.PERSON, name="X", confidence=-0.1)


# =============================================================================
# Link Model
# =============================================================================


class TestLink:
    def test_minimal_link(self):
        link = Link(
            link_type=LinkType.PARTICIPATES_IN,
            source_entity_id="ENT-1",
            target_entity_id="ENT-2",
        )
        assert link.id is None
        assert link.confidence == 1.0
        assert link.status == EntityStatus.ACTIVE

    def test_self_link_rejected(self):
        with pytest.raises(ValidationError, match="Self-referencing"):
            Link(
                link_type=LinkType.COLLABORATES_WITH,
                source_entity_id="ENT-1",
                target_entity_id="ENT-1",
            )

    def test_temporal_validity(self):
        now = datetime.now()
        link = Link(
            link_type=LinkType.OWNS,
            source_entity_id="ENT-1",
            target_entity_id="ENT-2",
            valid_from=now,
        )
        assert link.valid_from == now
        assert link.valid_to is None


# =============================================================================
# Provenance Model
# =============================================================================


class TestProvenance:
    def test_entity_provenance(self):
        p = Provenance(
            entity_id="ENT-1",
            source_type="meeting_transcript",
            source_file="meeting.md",
            source_meeting_date=date(2026, 3, 16),
            source_participants=["Alice", "Bob"],
            extraction_model="gemini-2.0-flash",
            extraction_pass="pass1",
        )
        assert p.entity_id == "ENT-1"
        assert p.link_id is None

    def test_link_provenance(self):
        p = Provenance(
            link_id="LNK-1",
            source_type="llm_extraction",
        )
        assert p.link_id == "LNK-1"
        assert p.entity_id is None

    def test_must_have_target(self):
        with pytest.raises(ValidationError, match="must reference"):
            Provenance(source_type="meeting_transcript")


# =============================================================================
# Extraction Models
# =============================================================================


class TestExtractedEntity:
    def test_defaults(self):
        e = ExtractedEntity(name="Alice", entity_type=EntityType.PERSON)
        assert e.confidence == 0.8
        assert e.is_new is False
        assert e.aliases == []


class TestExtractedLink:
    def test_defaults(self):
        link = ExtractedLink(
            link_type=LinkType.OWNS,
            source_name="Alice",
            target_name="Project X",
        )
        assert link.confidence == 0.8


class TestExtractedDecision:
    def test_defaults(self):
        d = ExtractedDecision(summary="Use Gemini for LLM")
        assert d.decision_type == DecisionType.OPERATIONAL
        assert d.confidence == 0.8


class TestExtractedActionItem:
    def test_with_due_date(self):
        a = ExtractedActionItem(
            task="Write tests",
            owner="Alice",
            due_date=date(2026, 3, 20),
            priority="high",
        )
        assert a.due_date == date(2026, 3, 20)
        assert a.priority == "high"


class TestExtractionResult:
    def test_empty_result(self):
        r = ExtractionResult()
        assert r.entities == []
        assert r.links == []
        assert r.decisions == []
        assert r.action_items == []

    def test_populated_result(self, sample_extraction):
        r = sample_extraction
        assert len(r.entities) == 3
        assert len(r.links) == 2
        assert len(r.decisions) == 1
        assert len(r.action_items) == 1
        assert r.meeting_date == date(2026, 3, 16)


# =============================================================================
# Validation Models
# =============================================================================


class TestValidationResult:
    def test_valid(self):
        v = ValidationResult(is_valid=True)
        assert v.errors == []
        assert v.warnings == []
        assert v.auto_fixes_applied == 0

    def test_with_errors(self):
        v = ValidationResult(
            is_valid=False,
            errors=[
                VError(
                    layer="factual",
                    severity="error",
                    message="Missing name",
                )
            ],
        )
        assert not v.is_valid
        assert len(v.errors) == 1
        assert v.errors[0].layer == "factual"
