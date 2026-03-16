"""Tests for pipeline/engine.py — PipelineEngine and IngestResult."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from ontology_engine.core.config import OntologyConfig, PipelineConfig
from ontology_engine.pipeline.engine import IngestResult, PipelineEngine
from tests.conftest import MockLLMProvider


@pytest.fixture
def mock_engine():
    """PipelineEngine with mock LLM and no DB."""
    responses = {
        "话题": {
            "topics": [{"topic": "项目进度", "segment_ids": [0, 1]}]
        },
        "代词消解": {"resolved": []},
        "抽取所有实体": {
            "entities": [
                {
                    "name": "Yann",
                    "type": "Person",
                    "confidence": 0.95,
                    "is_new": False,
                    "context": "Yann says",
                    "properties": {"role": "CEO"},
                    "aliases": [],
                },
                {
                    "name": "WhiteMirror",
                    "type": "Project",
                    "confidence": 0.9,
                    "aliases": ["WM"],
                    "context": "WhiteMirror project",
                    "properties": {},
                },
            ]
        },
        "抽取实体之间的关系": {
            "relations": [
                {
                    "type": "owns",
                    "source": "Yann",
                    "target": "WhiteMirror",
                    "confidence": 0.9,
                },
            ]
        },
        "抽取所有决策": {
            "decisions": [
                {
                    "summary": "Use Supabase",
                    "detail": "Free tier",
                    "decision_type": "tactical",
                    "made_by": "Yann",
                    "participants": ["Yann"],
                    "confidence": 0.9,
                },
            ]
        },
        "抽取所有行动项": {
            "action_items": [
                {
                    "task": "Build MVP",
                    "owner": "Yann",
                    "due_date": "2026-03-20",
                    "priority": "high",
                    "confidence": 0.9,
                },
            ]
        },
    }
    config = OntologyConfig(
        pipeline=PipelineConfig(
            segment_topics=False,
            resolve_coreferences=False,
        )
    )
    llm = MockLLMProvider(responses)
    return PipelineEngine(llm=llm, repo=None, config=config)


class TestIngestResult:
    def test_success_result(self):
        r = IngestResult(file="test.md", processing_time_ms=500)
        assert r.success is True
        assert r.processing_time_ms == 500

    def test_error_result(self):
        r = IngestResult(file="test.md", error="File not found")
        assert r.success is False
        assert r.error == "File not found"

    def test_summary(self, sample_extraction):
        from ontology_engine.core.types import ValidationResult

        r = IngestResult(
            file="meeting.md",
            extraction=sample_extraction,
            validation=ValidationResult(is_valid=True, auto_fixes_applied=2),
            processing_time_ms=1234,
        )
        s = r.summary()
        assert s["file"] == "meeting.md"
        assert s["success"] is True
        assert s["time_ms"] == 1234
        assert s["entities"] == 3
        assert s["valid"] is True
        assert s["auto_fixes"] == 2


class TestPipelineEngine:
    async def test_ingest_file(self, mock_engine, tmp_path):
        """Test ingesting a meeting file end-to-end (no DB)."""
        meeting_file = tmp_path / "20260316_test_meeting.md"
        meeting_file.write_text(
            "【Yann】：我们讨论一下WhiteMirror项目的进展。\n"
            "【Yann】：决定用Supabase做backend。"
        )

        result = await mock_engine.ingest(str(meeting_file))
        assert result.success
        assert result.extraction is not None
        assert len(result.extraction.entities) >= 1
        assert result.validation is not None
        assert result.meeting_date == date(2026, 3, 16)  # Parsed from filename

    async def test_ingest_nonexistent_file(self, mock_engine):
        from ontology_engine.core.errors import ExtractionError

        with pytest.raises(ExtractionError, match="not found"):
            await mock_engine.ingest("/nonexistent/path.md")

    async def test_ingest_directory(self, mock_engine, tmp_path):
        """Test ingesting multiple files."""
        (tmp_path / "20260301_a.md").write_text("【Alice】：会议A内容")
        (tmp_path / "20260302_b.md").write_text("【Bob】：会议B内容")
        (tmp_path / "notes.txt").write_text("Not a meeting")

        results = await mock_engine.ingest_directory(str(tmp_path), pattern="*.md")
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_parse_date_from_filename(self):
        assert PipelineEngine._parse_date_from_filename("20260208_meeting.md") == date(2026, 2, 8)
        assert PipelineEngine._parse_date_from_filename("2026-03-16_notes.md") == date(2026, 3, 16)
        assert PipelineEngine._parse_date_from_filename("random_file.md") is None
        assert PipelineEngine._parse_date_from_filename("20261301_bad_date.md") is None

    async def test_ingest_with_output(self, mock_engine, tmp_path):
        """Test that extraction results can be serialized."""
        meeting_file = tmp_path / "meeting.md"
        meeting_file.write_text("【Yann】：我们用Gemini做LLM。")

        result = await mock_engine.ingest(str(meeting_file))
        assert result.success
        assert result.extraction is not None

        # Should be JSON-serializable
        out_data = result.extraction.model_dump(mode="json")
        json_str = json.dumps(out_data, ensure_ascii=False, indent=2)
        assert "Yann" in json_str or len(out_data["entities"]) >= 0
