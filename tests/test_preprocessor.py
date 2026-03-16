"""Tests for pipeline/preprocessor.py."""

from __future__ import annotations

import json
from datetime import date

import pytest

from ontology_engine.core.config import OntologyConfig, PipelineConfig
from ontology_engine.pipeline.preprocessor import (
    FILLER_RE,
    SPEAKER_RE,
    MeetingPreprocessor,
    ProcessedMeeting,
    Segment,
)
from tests.conftest import MockLLMProvider


# =============================================================================
# Regex Tests
# =============================================================================


class TestFillerRegex:
    def test_removes_chinese_fillers(self):
        text = "嗯，我觉得，然后就是，我们应该做这个。"
        cleaned = FILLER_RE.sub("", text)
        assert "嗯" not in cleaned
        assert "然后就是" not in cleaned
        assert "我们应该做这个" in cleaned

    def test_removes_repetitive_fillers(self):
        text = "对对对，好好好，这个方案可以。"
        cleaned = FILLER_RE.sub("", text)
        assert "对对对" not in cleaned
        assert "好好好" not in cleaned
        assert "这个方案可以" in cleaned

    def test_preserves_normal_text(self):
        text = "我们需要在周五之前完成设计文档。"
        cleaned = FILLER_RE.sub("", text)
        assert cleaned == text


class TestSpeakerRegex:
    def test_bracket_format(self):
        m = SPEAKER_RE.search("【Alice】：你好")
        assert m is not None
        assert m.group(1) == "Alice"

    def test_colon_format(self):
        m = SPEAKER_RE.search("Bob：我同意")
        assert m is not None
        assert m.group(2) == "Bob"

    def test_english_colon(self):
        m = SPEAKER_RE.search("Charlie: I agree")
        assert m is not None
        assert m.group(2) == "Charlie"


# =============================================================================
# Preprocessor Tests
# =============================================================================


class TestMeetingPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        config = OntologyConfig(
            pipeline=PipelineConfig(
                remove_filler_words=True,
                segment_topics=False,  # Disable LLM calls for unit tests
                resolve_coreferences=False,
            )
        )
        llm = MockLLMProvider()
        return MeetingPreprocessor(llm, config)

    @pytest.fixture
    def preprocessor_with_llm(self):
        """Preprocessor with topic segmentation enabled and mock LLM responses."""
        config = OntologyConfig(
            pipeline=PipelineConfig(
                remove_filler_words=True,
                segment_topics=True,
                resolve_coreferences=True,
            )
        )
        responses = {
            "话题": {
                "topics": [
                    {"topic": "项目进度", "segment_ids": [0, 1]},
                    {"topic": "技术选型", "segment_ids": [2, 3]},
                ]
            },
            "代词消解": {
                "resolved": [
                    {"id": 0, "text": "Yann说Yann觉得这个方案不错"},
                ]
            },
        }
        llm = MockLLMProvider(responses)
        return MeetingPreprocessor(llm, config)

    async def test_clean_text(self, preprocessor):
        text = "嗯，我觉得，然后就是，这个项目很重要。"
        cleaned = preprocessor._clean_text(text)
        assert "嗯" not in cleaned
        assert "这个项目很重要" in cleaned

    async def test_clean_text_disabled(self):
        config = OntologyConfig(
            pipeline=PipelineConfig(remove_filler_words=False)
        )
        llm = MockLLMProvider()
        p = MeetingPreprocessor(llm, config)
        text = "嗯，我觉得这个好。"
        assert p._clean_text(text) == text

    async def test_extract_speakers_bracket(self, preprocessor):
        text = "【Alice】：你好世界。\n【Bob】：再见。"
        segments = preprocessor._extract_speakers(text)
        assert len(segments) == 2
        assert segments[0].speaker == "Alice"
        assert segments[1].speaker == "Bob"

    async def test_extract_speakers_no_labels(self, preprocessor):
        text = "This is a plain text meeting with no speaker labels."
        segments = preprocessor._extract_speakers(text)
        assert len(segments) == 1
        assert segments[0].speaker == ""
        assert segments[0].text == text

    async def test_detect_participants(self, preprocessor):
        segments = [
            Segment(text="Hi", speaker="Alice"),
            Segment(text="Hi", speaker="Bob"),
            Segment(text="Yes", speaker="Alice"),
        ]
        participants = preprocessor._detect_participants(segments)
        assert participants == ["Alice", "Bob"]

    async def test_full_process(self, preprocessor, sample_transcript):
        result = await preprocessor.process(
            sample_transcript,
            meeting_date=date(2026, 3, 16),
        )
        assert isinstance(result, ProcessedMeeting)
        assert result.meeting_date == date(2026, 3, 16)
        assert len(result.segments) > 0
        assert len(result.participants) > 0
        assert "Yann" in result.participants
        assert "Felix" in result.participants

    async def test_process_with_topic_segmentation(self, preprocessor_with_llm, sample_transcript):
        result = await preprocessor_with_llm.process(
            sample_transcript,
            meeting_date=date(2026, 3, 16),
        )
        assert isinstance(result, ProcessedMeeting)
        # Should have segments even with LLM processing
        assert len(result.segments) > 0
