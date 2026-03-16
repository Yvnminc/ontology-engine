"""Integration tests — real Gemini API + real meeting transcripts.

These tests are marked with @pytest.mark.integration and require:
- GEMINI_API_KEY environment variable
- Real meeting transcripts in the meeting data directory

Run with: pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from ontology_engine.core.config import LLMConfig, OntologyConfig, PipelineConfig
from ontology_engine.core.types import EntityType, ExtractionResult
from ontology_engine.llm.gemini import GeminiProvider
from ontology_engine.pipeline.engine import PipelineEngine
from ontology_engine.pipeline.extractor import StructuredExtractor
from ontology_engine.pipeline.preprocessor import MeetingPreprocessor
from ontology_engine.pipeline.validator import ExtractionValidator

MEETING_DIR = Path("/Users/yann/github/WhiteMirror_data/meeting")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# Small meeting file for quick tests
SMALL_MEETING = MEETING_DIR / "20260203_wn_meeting.txt"


def has_gemini() -> bool:
    return bool(GEMINI_KEY)


def has_meeting_data() -> bool:
    return MEETING_DIR.exists() and SMALL_MEETING.exists()


integration = pytest.mark.skipif(
    not (has_gemini() and has_meeting_data()),
    reason="Requires GEMINI_API_KEY and meeting data",
)


@pytest.fixture
def gemini_config() -> OntologyConfig:
    return OntologyConfig(
        llm=LLMConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key=GEMINI_KEY,
            fast_model="gemini-2.0-flash-lite",
        ),
        pipeline=PipelineConfig(
            remove_filler_words=True,
            segment_topics=False,  # Save API calls
            resolve_coreferences=False,  # Save API calls
            min_confidence=0.6,
        ),
    )


@pytest.fixture
def gemini_provider(gemini_config) -> GeminiProvider:
    return GeminiProvider(gemini_config.llm)


@integration
class TestGeminiLLM:
    """Test that Gemini API works end-to-end."""

    async def test_basic_generation(self, gemini_provider):
        """Verify Gemini API connectivity."""
        resp = await gemini_provider.generate(
            "Say 'hello' in JSON format: {\"greeting\": \"hello\"}",
            response_format="json",
        )
        assert resp.text
        data = resp.parse_json()
        assert "greeting" in data or "hello" in resp.text.lower()

    async def test_json_extraction(self, gemini_provider):
        """Verify structured JSON extraction works."""
        resp = await gemini_provider.generate_json(
            '从这段话中提取人名，输出JSON: {"names": [...]}。\n'
            '话："Yann说我们需要让Felix来负责运营。"'
        )
        assert "names" in resp
        names = resp["names"]
        assert any("Yann" in n for n in names)


@integration
class TestPreprocessorIntegration:
    async def test_preprocess_real_transcript(self, gemini_config, gemini_provider):
        """Test preprocessor on a real transcript."""
        preprocessor = MeetingPreprocessor(gemini_provider, gemini_config)
        raw_text = SMALL_MEETING.read_text(encoding="utf-8")

        result = await preprocessor.process(
            raw_text[:3000],  # Limit to first 3K chars for speed
            meeting_date=date(2026, 2, 3),
        )

        assert len(result.segments) >= 1
        assert result.cleaned_text
        assert len(result.cleaned_text) < len(raw_text[:3000])  # Fillers removed


@integration
class TestExtractorIntegration:
    async def test_extract_from_real_transcript(self, gemini_config, gemini_provider):
        """Test entity extraction on a real meeting."""
        preprocessor = MeetingPreprocessor(gemini_provider, gemini_config)
        extractor = StructuredExtractor(gemini_provider, gemini_config)

        raw_text = SMALL_MEETING.read_text(encoding="utf-8")
        processed = await preprocessor.process(
            raw_text[:3000],
            meeting_date=date(2026, 2, 3),
        )

        result = await extractor.extract(processed)

        # Should extract at least some entities from a real meeting
        assert len(result.entities) >= 1
        print(f"\n  Extracted {len(result.entities)} entities:")
        for e in result.entities:
            print(f"    [{e.entity_type.value}] {e.name} (conf={e.confidence:.2f})")

        print(f"  Extracted {len(result.links)} links")
        print(f"  Extracted {len(result.decisions)} decisions")
        print(f"  Extracted {len(result.action_items)} action items")


@integration
class TestFullPipelineIntegration:
    async def test_ingest_real_file(self, gemini_config):
        """Full pipeline test: file → preprocess → extract → validate."""
        engine = await PipelineEngine.create(gemini_config)
        try:
            result = await engine.ingest(
                str(SMALL_MEETING),
                meeting_date=date(2026, 2, 3),
            )

            assert result.success, f"Ingestion failed: {result.error}"
            assert result.extraction is not None
            assert result.validation is not None

            s = result.summary()
            print(f"\n  Full pipeline results:")
            print(f"    File: {s['file']}")
            print(f"    Time: {s['time_ms']}ms")
            print(f"    Entities: {s.get('entities', 0)}")
            print(f"    Links: {s.get('links', 0)}")
            print(f"    Decisions: {s.get('decisions', 0)}")
            print(f"    Action Items: {s.get('action_items', 0)}")
            print(f"    Valid: {s.get('valid')}")
            print(f"    Auto-fixes: {s.get('auto_fixes', 0)}")
            print(f"    Warnings: {s.get('warnings', 0)}")

            # Save results for inspection
            output_dir = Path("/Users/yann/github/ontology-engine/test_output")
            output_dir.mkdir(exist_ok=True)
            out_file = output_dir / "integration_result.json"
            out_data = result.extraction.model_dump(mode="json")
            out_file.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
            print(f"    Output saved to: {out_file}")

        finally:
            await engine.close()
