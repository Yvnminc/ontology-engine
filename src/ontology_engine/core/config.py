"""Configuration for Ontology Engine."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "gemini"  # gemini | openai | anthropic
    model: str = "gemini-2.0-flash"  # Default: Gemini Flash (cost-effective)
    api_key: str = ""
    temperature: float = 0.1  # Low temperature for structured extraction
    max_tokens: int = 8192
    timeout_seconds: int = 60

    # Model tiers for different tasks
    fast_model: str = "gemini-2.0-flash-lite"  # Preprocessing, simple extraction
    default_model: str = "gemini-2.0-flash"  # Standard extraction & validation
    strong_model: str = "gemini-2.5-pro"  # Complex reasoning, conflict resolution


class DatabaseConfig(BaseModel):
    """PostgreSQL connection configuration."""

    url: str = ""  # postgresql://user:pass@host:port/db
    schema: str = "ontology"
    pool_min: int = 2
    pool_max: int = 10


class PipelineConfig(BaseModel):
    """Extraction pipeline configuration."""

    # Preprocessing
    remove_filler_words: bool = True
    resolve_coreferences: bool = True
    segment_topics: bool = True

    # Extraction
    min_confidence: float = 0.6  # Minimum confidence to keep an extraction
    enable_pass1_entities: bool = True
    enable_pass2_relations: bool = True
    enable_pass3_decisions: bool = True
    enable_pass4_actions: bool = True

    # Validation
    enable_factual_correction: bool = True
    enable_semantic_correction: bool = False  # Phase 2 — disabled for now
    enable_consistency_check: bool = True
    auto_fix_threshold: float = 0.9  # Auto-fix if confidence > this


class EntityAliasConfig(BaseModel):
    """Known entity aliases for disambiguation."""

    aliases: dict[str, list[str]] = Field(default_factory=dict)
    # Example: {"Yann": ["Yann哥", "验", "彦", "郭博", "Yanming"]}


class OntologyConfig(BaseModel):
    """Top-level configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    known_entities: EntityAliasConfig = Field(default_factory=EntityAliasConfig)

    # Paths
    meeting_dir: str = ""  # Directory containing meeting transcripts
    output_dir: str = ""  # Directory for extraction results / logs

    @classmethod
    def from_file(cls, path: str) -> OntologyConfig:
        """Load config from a TOML or JSON file."""
        import json
        from pathlib import Path

        p = Path(path)
        if p.suffix == ".json":
            return cls.model_validate_json(p.read_text())
        elif p.suffix == ".toml":
            import tomllib

            with open(p, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")
