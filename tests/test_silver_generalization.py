"""Tests for M3: Silver Layer Generalization — schema-driven extraction.

Covers: dynamic prompt generation, schema-aware validation, dynamic seeding,
rerun capability, extraction model/schema tracking, cross-domain extraction.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ontology_engine.core.config import OntologyConfig, PipelineConfig
from ontology_engine.core.types import (
    EntityType,
    ExtractionResult,
    ExtractedEntity,
    ExtractedLink,
    LinkType,
    Provenance,
)
from ontology_engine.pipeline.engine import PipelineEngine
from ontology_engine.pipeline.extractor import StructuredExtractor
from ontology_engine.pipeline.preprocessor import ProcessedMeeting, Segment
from ontology_engine.pipeline.validator import ExtractionValidator
from ontology_engine.storage.schema import generate_seed_sql, DEFAULT_TYPE_SEEDS
from tests.conftest import MockLLMProvider

SCHEMAS_DIR = Path(__file__).parent.parent / "domain_schemas"


# ========================================================================
# Helpers
# ========================================================================

def _mock_meeting(text: str = "测试内容", speaker: str = "Alice") -> ProcessedMeeting:
    return ProcessedMeeting(
        segments=[Segment(text=text, speaker=speaker)],
        raw_text=text,
        cleaned_text=text,
        meeting_date=date(2026, 3, 17),
        participants=[speaker],
        metadata={"source_file": "test.md"},
    )


def _edtech_responses() -> dict[str, Any]:
    """Mock LLM responses for edtech schema extraction."""
    return {
        "抽取所有实体": {
            "entities": [
                {"name": "张三", "type": "Student", "confidence": 0.9,
                 "properties": {"student_id": "S001"}, "aliases": [], "context": "张三同学"},
                {"name": "CS101", "type": "Course", "confidence": 0.9,
                 "properties": {"course_code": "CS101"}, "aliases": ["计算机入门"], "context": "CS101课程"},
                {"name": "Python基础", "type": "KnowledgeUnit", "confidence": 0.85,
                 "properties": {"topic": "Python"}, "aliases": [], "context": "学习Python基础"},
            ]
        },
        "抽取实体之间的关系": {
            "relations": [
                {"type": "enrolled_in", "source": "张三", "target": "CS101", "confidence": 0.85},
            ]
        },
        "抽取所有决策": {"decisions": []},
        "抽取所有行动项": {"action_items": []},
    }


def _finance_responses() -> dict[str, Any]:
    """Mock LLM responses for finance schema extraction."""
    return {
        "抽取所有实体": {
            "entities": [
                {"name": "TXN-001", "type": "Transaction", "confidence": 0.9,
                 "properties": {"transaction_id": "TXN-001", "amount": 50000.0}, "aliases": []},
                {"name": "客户A", "type": "Client", "confidence": 0.9,
                 "properties": {"client_id": "C001"}, "aliases": []},
                {"name": "网络攻击事件", "type": "RiskEvent", "confidence": 0.85,
                 "properties": {"risk_type": "cyber", "severity": "high"}, "aliases": []},
            ]
        },
        "抽取实体之间的关系": {
            "relations": [
                {"type": "affects", "source": "网络攻击事件", "target": "客户A", "confidence": 0.8},
            ]
        },
        "抽取所有决策": {"decisions": []},
        "抽取所有行动项": {"action_items": []},
    }


# ========================================================================
# 1. Edtech schema produces Student/Course/KnowledgeUnit entities
# ========================================================================

class TestEdtechExtraction:
    async def test_edtech_schema_entities(self):
        """Edtech schema → Student, Course, KnowledgeUnit entity types."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        llm = MockLLMProvider(_edtech_responses())
        extractor = StructuredExtractor(llm, OntologyConfig(), domain_schema=schema)

        result = await extractor.extract(_mock_meeting("张三选了CS101课程，学习Python基础"))
        types = {e.entity_type for e in result.entities}
        assert "Student" in types
        assert "Course" in types
        assert "KnowledgeUnit" in types
        assert result.extraction_schema == "edtech"


# ========================================================================
# 2. Finance schema produces Transaction/Client/RiskEvent entities
# ========================================================================

class TestFinanceExtraction:
    async def test_finance_schema_entities(self):
        """Finance schema → Transaction, Client, RiskEvent entity types."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "finance.yaml")
        llm = MockLLMProvider(_finance_responses())
        extractor = StructuredExtractor(llm, OntologyConfig(), domain_schema=schema)

        result = await extractor.extract(_mock_meeting("TXN-001交易，客户A受网络攻击事件影响"))
        types = {e.entity_type for e in result.entities}
        assert "Transaction" in types
        assert "Client" in types
        assert "RiskEvent" in types
        assert result.extraction_schema == "finance"


# ========================================================================
# 3. Same text, different schema → different entity types
# ========================================================================

class TestCrossSchemaExtraction:
    async def test_same_text_different_schemas(self):
        """Same text with edtech vs finance schema produces different entity types."""
        from ontology_engine.core.schema_registry import DomainSchema

        meeting = _mock_meeting("张三进行了一笔交易，金额50000元")

        edtech_schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        finance_schema = DomainSchema.from_yaml(SCHEMAS_DIR / "finance.yaml")

        edtech_result = await StructuredExtractor(
            MockLLMProvider(_edtech_responses()), OntologyConfig(), domain_schema=edtech_schema
        ).extract(meeting)

        finance_result = await StructuredExtractor(
            MockLLMProvider(_finance_responses()), OntologyConfig(), domain_schema=finance_schema
        ).extract(meeting)

        edtech_types = {e.entity_type for e in edtech_result.entities}
        finance_types = {e.entity_type for e in finance_result.entities}
        assert edtech_types != finance_types
        assert edtech_result.extraction_schema != finance_result.extraction_schema


# ========================================================================
# 4. Default mode backward compatibility
# ========================================================================

class TestDefaultModeCompat:
    async def test_no_schema_uses_enum(self):
        """No schema → falls back to EntityType enum, unknown types skipped."""
        responses = {
            "抽取所有实体": {
                "entities": [
                    {"name": "Alice", "type": "Person", "confidence": 0.9, "properties": {"role": "dev"}},
                    {"name": "BadType", "type": "UnknownCustom", "confidence": 0.9},
                ]
            },
            "抽取实体之间的关系": {"relations": []},
            "抽取所有决策": {"decisions": []},
            "抽取所有行动项": {"action_items": []},
        }
        extractor = StructuredExtractor(MockLLMProvider(responses), OntologyConfig())
        result = await extractor.extract(_mock_meeting())

        assert len(result.entities) == 1
        assert result.entities[0].entity_type == EntityType.PERSON
        assert result.extraction_schema == "default"

    async def test_no_schema_extracts_decisions_and_actions(self):
        """Default mode still runs pass 3 and 4 (Decision/ActionItem in EntityType)."""
        responses = {
            "抽取所有实体": {"entities": [
                {"name": "Bob", "type": "Person", "confidence": 0.9, "properties": {"role": "PM"}},
            ]},
            "抽取实体之间的关系": {"relations": []},
            "抽取所有决策": {"decisions": [
                {"summary": "用Python", "decision_type": "tactical", "made_by": "Bob", "confidence": 0.9},
            ]},
            "抽取所有行动项": {"action_items": [
                {"task": "写代码", "owner": "Bob", "confidence": 0.9},
            ]},
        }
        extractor = StructuredExtractor(MockLLMProvider(responses), OntologyConfig())
        result = await extractor.extract(_mock_meeting())

        assert len(result.decisions) == 1
        assert len(result.action_items) == 1


# ========================================================================
# 5. Schema without Decision/ActionItem skips pass 3/4
# ========================================================================

class TestConditionalPasses:
    async def test_schema_without_decision_skips_pass3(self):
        """Schema without Decision type → pass 3 not executed."""
        schema_dict = {
            "domain": "minimal",
            "entity_types": [{"name": "Widget", "description": "A widget"}],
        }
        responses = {
            "抽取所有实体": {"entities": [{"name": "W1", "type": "Widget", "confidence": 0.9}]},
            "抽取实体之间的关系": {"relations": []},
            # These should NOT be called:
            "抽取所有决策": {"decisions": [{"summary": "GHOST", "confidence": 0.99}]},
            "抽取所有行动项": {"action_items": [{"task": "GHOST", "confidence": 0.99}]},
        }
        extractor = StructuredExtractor(MockLLMProvider(responses), OntologyConfig(), domain_schema=schema_dict)
        result = await extractor.extract(_mock_meeting())

        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "Widget"
        assert len(result.decisions) == 0  # Pass 3 skipped
        assert len(result.action_items) == 0  # Pass 4 skipped


# ========================================================================
# 6-7. Schema-aware validation
# ========================================================================

class TestSchemaValidation:
    def test_required_property_missing_warns(self):
        """Schema validation warns when a required property is missing."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        validator = ExtractionValidator(OntologyConfig(), domain_schema=schema)

        result = ExtractionResult(entities=[
            ExtractedEntity(name="张三", entity_type="Student", confidence=0.9,
                            properties={}),  # Missing student_id (required)
        ])
        validation = validator.validate(result)
        schema_warnings = [w for w in validation.warnings if w.layer == "schema"]
        assert any("student_id" in w.message for w in schema_warnings)

    def test_enum_value_out_of_range_warns(self):
        """Schema validation warns when enum property has invalid value."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        validator = ExtractionValidator(OntologyConfig(), domain_schema=schema)

        result = ExtractionResult(entities=[
            ExtractedEntity(name="张三", entity_type="Student", confidence=0.9,
                            properties={"student_id": "S001", "enrollment_status": "expelled"}),
        ])
        validation = validator.validate(result)
        enum_warnings = [w for w in validation.warnings if "enrollment_status" in (w.field or "")]
        assert len(enum_warnings) >= 1

    def test_no_schema_default_person_role_check(self):
        """Without schema, Person without 'role' still generates warning."""
        validator = ExtractionValidator(OntologyConfig())  # No schema
        result = ExtractionResult(entities=[
            ExtractedEntity(name="Alice", entity_type=EntityType.PERSON, confidence=0.9, properties={}),
        ])
        validation = validator.validate(result)
        role_warnings = [w for w in validation.warnings if "role" in w.message]
        assert len(role_warnings) >= 1


# ========================================================================
# 8-9. Dynamic seeding (generate_seed_sql)
# ========================================================================

class TestDynamicSeeding:
    def test_default_seed_sql(self):
        """No schema → default 6 entity types seeded."""
        sql = generate_seed_sql()
        for name in ["Person", "Decision", "ActionItem", "Project", "Risk", "Deadline"]:
            assert f"'{name}'" in sql

    def test_schema_driven_seed_sql(self):
        """With edtech schema → Student, Course, enrolled_in etc. seeded."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        sql = generate_seed_sql(schema)
        assert "'Student'" in sql
        assert "'Course'" in sql
        assert "'KnowledgeUnit'" in sql
        assert "'enrolled_in'" in sql
        assert "'teaches'" in sql

    def test_dict_schema_seed_sql(self):
        """Dict-based schema also works for seeding."""
        schema_dict = {
            "entity_types": [
                {"name": "Widget", "description": "A widget", "properties": [{"name": "wid", "required": True}]},
            ],
            "link_types": [{"name": "part_of", "description": "Part of"}],
        }
        sql = generate_seed_sql(schema_dict)
        assert "'Widget'" in sql
        assert "'part_of'" in sql


# ========================================================================
# 10. Extraction model/schema tracking
# ========================================================================

class TestExtractionTracking:
    def test_provenance_has_extraction_schema(self):
        """Provenance model accepts extraction_schema field."""
        prov = Provenance(
            entity_id="ENT-1",
            source_type="meeting_transcript",
            extraction_model="gemini-2.5-flash",
            extraction_schema="edtech",
        )
        assert prov.extraction_schema == "edtech"
        assert prov.extraction_model == "gemini-2.5-flash"

    def test_extraction_result_has_schema(self):
        """ExtractionResult tracks schema name."""
        result = ExtractionResult(
            extraction_model="gemini-2.5-flash",
            extraction_schema="finance",
        )
        assert result.extraction_schema == "finance"


# ========================================================================
# 11. Rerun without Bronze raises
# ========================================================================

class TestRerunCapability:
    async def test_rerun_without_bronze_raises(self):
        """Rerun fails gracefully when no Bronze repo is available."""
        from ontology_engine.core.errors import ExtractionError

        engine = PipelineEngine(
            llm=MockLLMProvider(), repo=None, config=OntologyConfig(), bronze=None
        )
        with pytest.raises(ExtractionError, match="Bronze repository not available"):
            await engine.rerun("DOC-nonexistent")


# ========================================================================
# 12. Engine with schema passes it to extractor/validator
# ========================================================================

class TestEngineSchemaWiring:
    def test_engine_passes_schema_to_components(self):
        """Engine passes domain_schema to extractor and validator."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        engine = PipelineEngine(
            llm=MockLLMProvider(), repo=None,
            config=OntologyConfig(pipeline=PipelineConfig(
                segment_topics=False, resolve_coreferences=False,
            )),
            domain_schema=schema,
        )
        assert engine.extractor.schema_name == "edtech"
        assert engine.extractor._schema is not None
        assert engine.validator._schema is not None


# ========================================================================
# 13. Schema loading
# ========================================================================

class TestSchemaLoading:
    def test_load_all_three_schemas(self):
        """All three YAML schemas load without error."""
        from ontology_engine.core.schema_registry import DomainSchema

        for name in ["default", "edtech", "finance"]:
            s = DomainSchema.from_yaml(SCHEMAS_DIR / f"{name}.yaml")
            assert s.domain == name
            assert len(s.entity_types) >= 1

    def test_default_schema_matches_original_types(self):
        """Default schema has the same 6 entity types as the original enum."""
        from ontology_engine.core.schema_registry import DomainSchema

        s = DomainSchema.from_yaml(SCHEMAS_DIR / "default.yaml")
        names = set(s.entity_type_names())
        expected = {"Person", "Decision", "ActionItem", "Project", "Risk", "Deadline"}
        assert names == expected

    def test_from_dict(self):
        """DomainSchema.from_dict works for inline schemas."""
        from ontology_engine.core.schema_registry import DomainSchema

        s = DomainSchema.from_dict({
            "domain": "test",
            "entity_types": [{"name": "Thing", "description": "A thing"}],
        })
        assert s.domain == "test"
        assert s.has_entity_type("Thing")


# ========================================================================
# 14. Schema-driven relation types
# ========================================================================

class TestSchemaRelations:
    async def test_schema_link_types_accepted(self):
        """Schema-defined link types are accepted as strings (not LinkType enum)."""
        from ontology_engine.core.schema_registry import DomainSchema

        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        llm = MockLLMProvider(_edtech_responses())
        extractor = StructuredExtractor(llm, OntologyConfig(), domain_schema=schema)

        result = await extractor.extract(_mock_meeting("张三选了CS101"))
        assert len(result.links) >= 1
        assert result.links[0].link_type == "enrolled_in"
        # Confirm it's a string, not a LinkType enum
        assert isinstance(result.links[0].link_type, str)
        assert not isinstance(result.links[0].link_type, LinkType)


# ========================================================================
# 15. Full pipeline with edtech schema end-to-end
# ========================================================================

class TestEngineEndToEnd:
    async def test_ingest_with_edtech_schema(self, tmp_path):
        """Full pipeline with edtech schema produces domain-specific entities."""
        from ontology_engine.core.schema_registry import DomainSchema

        meeting_file = tmp_path / "20260317_meeting.md"
        meeting_file.write_text("【王老师】：张三选了CS101课程，要学Python基础。")

        responses = {
            "话题": {"topics": []},
            "代词消解": {"resolved": []},
            **_edtech_responses(),
        }
        schema = DomainSchema.from_yaml(SCHEMAS_DIR / "edtech.yaml")
        config = OntologyConfig(pipeline=PipelineConfig(
            segment_topics=False, resolve_coreferences=False,
        ))
        engine = PipelineEngine(
            llm=MockLLMProvider(responses), repo=None, config=config, domain_schema=schema,
        )
        result = await engine.ingest(str(meeting_file))

        assert result.success
        assert result.extraction is not None
        types = {e.entity_type for e in result.extraction.entities}
        assert "Student" in types or "Course" in types
        assert result.extraction.extraction_schema == "edtech"
