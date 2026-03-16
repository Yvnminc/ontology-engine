"""Tests for pipeline/validator.py — 3-layer validation."""

from __future__ import annotations

from datetime import date

import pytest

from ontology_engine.core.config import OntologyConfig, PipelineConfig
from ontology_engine.core.types import (
    EntityType,
    ExtractionResult,
    ExtractedActionItem,
    ExtractedDecision,
    ExtractedEntity,
    ExtractedLink,
    LinkType,
)
from ontology_engine.pipeline.validator import ExtractionValidator


@pytest.fixture
def validator_with_aliases():
    """Validator with known aliases."""
    config = OntologyConfig(
        pipeline=PipelineConfig(min_confidence=0.6),
    )
    known = {
        "Yann": ["Yann哥", "郭博", "验"],
        "Felix": ["Felix运营"],
    }
    return ExtractionValidator(config, known_entities=known)


@pytest.fixture
def validator_no_aliases():
    """Validator without aliases."""
    config = OntologyConfig()
    return ExtractionValidator(config)


# =============================================================================
# Layer 1: Factual Correction
# =============================================================================


class TestFactualCorrection:
    def test_name_normalization(self, validator_with_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Yann哥", entity_type=EntityType.PERSON, confidence=0.9),
                ExtractedEntity(name="Felix运营", entity_type=EntityType.PERSON, confidence=0.9),
            ],
        )
        validation = validator_with_aliases.validate(result)
        # Names should be normalized to canonical forms
        assert validation.extraction is not None
        entity_names = [e.name for e in validation.extraction.entities]
        assert "Yann" in entity_names
        assert "Felix" in entity_names
        assert validation.auto_fixes_applied >= 2

    def test_decision_maker_normalization(self, validator_with_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Yann", entity_type=EntityType.PERSON, confidence=0.9),
            ],
            decisions=[
                ExtractedDecision(summary="Use Gemini", made_by="郭博", confidence=0.9),
            ],
        )
        validation = validator_with_aliases.validate(result)
        assert validation.extraction is not None
        assert validation.extraction.decisions[0].made_by == "Yann"

    def test_action_owner_normalization(self, validator_with_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Felix", entity_type=EntityType.PERSON, confidence=0.9),
            ],
            action_items=[
                ExtractedActionItem(task="Do stuff", owner="Felix运营", confidence=0.9),
            ],
        )
        validation = validator_with_aliases.validate(result)
        assert validation.extraction is not None
        assert validation.extraction.action_items[0].owner == "Felix"

    def test_missing_role_warning(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice",
                    entity_type=EntityType.PERSON,
                    confidence=0.9,
                    properties={},  # No role
                ),
            ],
        )
        validation = validator_no_aliases.validate(result)
        warnings = [w for w in validation.warnings if "role" in w.message]
        assert len(warnings) >= 1

    def test_low_confidence_warning(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="MaybeEntity",
                    entity_type=EntityType.PERSON,
                    confidence=0.5,  # Below min_confidence
                    properties={"role": "test"},
                ),
            ],
        )
        validation = validator_no_aliases.validate(result)
        warnings = [w for w in validation.warnings if "confidence" in w.message.lower()]
        assert len(warnings) >= 1


# =============================================================================
# Layer 3: Consistency Checks
# =============================================================================


class TestConsistencyChecks:
    def test_orphan_link_source(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={"role": "x"}),
            ],
            links=[
                ExtractedLink(
                    link_type=LinkType.OWNS,
                    source_name="Unknown",  # Not in entities
                    target_name="Alice",
                    confidence=0.9,
                ),
            ],
        )
        validation = validator_no_aliases.validate(result)
        link_warnings = [w for w in validation.warnings if "source" in w.field or ""]
        assert len(link_warnings) >= 1

    def test_orphan_link_target(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={"role": "x"}),
            ],
            links=[
                ExtractedLink(
                    link_type=LinkType.OWNS,
                    source_name="Alice",
                    target_name="GhostProject",
                    confidence=0.9,
                ),
            ],
        )
        validation = validator_no_aliases.validate(result)
        warnings = [w for w in validation.warnings if "target" in (w.field or "")]
        assert len(warnings) >= 1

    def test_orphan_action_owner(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={"role": "x"}),
            ],
            action_items=[
                ExtractedActionItem(task="Do stuff", owner="Unknown", confidence=0.9),
            ],
        )
        validation = validator_no_aliases.validate(result)
        warnings = [w for w in validation.warnings if "owner" in (w.field or "")]
        assert len(warnings) >= 1

    def test_orphan_decision_maker(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={"role": "x"}),
            ],
            decisions=[
                ExtractedDecision(summary="Decision X", made_by="Ghost", confidence=0.9),
            ],
        )
        validation = validator_no_aliases.validate(result)
        warnings = [w for w in validation.warnings if "made_by" in (w.field or "")]
        assert len(warnings) >= 1

    def test_duplicate_entity_names(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={"role": "x"}),
                ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.8, properties={"role": "y"}),
            ],
        )
        validation = validator_no_aliases.validate(result)
        errors = [e for e in validation.errors if "Duplicate" in e.message]
        assert len(errors) >= 1
        assert not validation.is_valid  # Duplicates are errors

    def test_valid_extraction_passes(self, validator_no_aliases):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice",
                    entity_type=EntityType.PERSON,
                    confidence=0.9,
                    properties={"role": "engineer"},
                ),
                ExtractedEntity(
                    name="Project X",
                    entity_type=EntityType.PROJECT,
                    confidence=0.9,
                ),
            ],
            links=[
                ExtractedLink(
                    link_type=LinkType.PARTICIPATES_IN,
                    source_name="Alice",
                    target_name="Project X",
                    confidence=0.9,
                ),
            ],
            decisions=[
                ExtractedDecision(summary="Use Python", made_by="Alice", confidence=0.9),
            ],
            action_items=[
                ExtractedActionItem(task="Write code", owner="Alice", confidence=0.9),
            ],
        )
        validation = validator_no_aliases.validate(result)
        assert validation.is_valid
