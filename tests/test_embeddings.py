"""Tests for the Embedding Generator."""

from __future__ import annotations

import pytest

from ontology_engine.fusion.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    def test_not_available_without_key(self):
        """Without OPENAI_API_KEY, available should be False."""
        gen = EmbeddingGenerator(api_key=None)
        # Clear env if set
        import os
        original = os.environ.pop("OPENAI_API_KEY", None)
        try:
            gen2 = EmbeddingGenerator()
            assert not gen2.available
        finally:
            if original:
                os.environ["OPENAI_API_KEY"] = original

    def test_available_with_key(self):
        gen = EmbeddingGenerator(api_key="test-key-123")
        assert gen.available

    def test_build_text_basic(self):
        gen = EmbeddingGenerator()
        text = gen.build_text("Yann Guo")
        assert text == "Yann Guo"

    def test_build_text_with_aliases(self):
        gen = EmbeddingGenerator()
        text = gen.build_text("Yann Guo", aliases=["郭博", "CEO"])
        assert "Yann Guo" in text
        assert "郭博" in text
        assert "CEO" in text

    def test_build_text_with_properties(self):
        gen = EmbeddingGenerator()
        text = gen.build_text(
            "Yann Guo",
            aliases=["郭博"],
            properties={"role": "CEO", "department": "Tech"},
        )
        assert "role: CEO" in text
        assert "department: Tech" in text

    def test_build_text_skips_meta_fields(self):
        gen = EmbeddingGenerator()
        text = gen.build_text(
            "Test",
            properties={
                "role": "Developer",
                "status": "active",       # should be skipped
                "created_at": "2026-01",  # should be skipped
                "id": "123",             # should be skipped
            },
        )
        assert "role: Developer" in text
        assert "status" not in text
        assert "created_at" not in text
        assert "id:" not in text  # careful: "id" could be in other words

    def test_build_text_empty_properties(self):
        gen = EmbeddingGenerator()
        text = gen.build_text("Test", properties={})
        assert text == "Test"
