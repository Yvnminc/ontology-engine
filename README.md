# Ontology Engine

> Structured knowledge extraction from meeting transcripts → PostgreSQL ontology graph.

An open-source framework that automatically transforms unstructured meeting transcripts into structured organizational knowledge graphs. Designed for multi-agent AI systems that need shared organizational memory.

## What It Does

```
Meeting Transcript (text)
    │
    ▼
┌─────────────────────────┐
│  Stage 1: Preprocessing │  Clean, segment, resolve pronouns
│  Stage 2: Extraction    │  4-Pass: Entities → Relations → Decisions → Actions
│  Stage 3: Validation    │  3-Layer: Factual → Semantic → Consistency
└─────────────────────────┘
    │
    ▼
PostgreSQL (JSONB + pgvector)
    │
    ▼
40+ AI Agents query shared knowledge
```

## Core Features

- **Meeting-to-Ontology Pipeline**: Multi-stage LLM pipeline for structured extraction
- **6 Core Object Types**: Person, Decision, ActionItem, Project, Risk, Deadline
- **13 Link Types**: Relationships between entities with temporal validity
- **3-Layer Auto-Correction**: Factual, semantic, and consistency validation
- **Provenance Tracking**: Every fact traced back to source transcript
- **Incremental Updates**: Each meeting enriches the existing knowledge graph
- **Version Control**: Full history of entity changes
- **SQL-Native**: PostgreSQL + JSONB — no SPARQL, no RDF, just SQL

## Tech Stack

- **Runtime**: Python 3.11+
- **Database**: PostgreSQL 15+ with pgvector, pg_trgm
- **LLM**: Pluggable — Gemini (default), OpenAI
- **Hosting**: Supabase-compatible

## Quick Start

### 1. Install

```bash
# Clone the repo
git clone https://github.com/Yvnminc/ontology-engine.git
cd ontology-engine

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

### 2. Set up your API key

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Verify the CLI works

```bash
ontology-engine --help
```

### 4. Process a meeting transcript (no database required)

```bash
# Ingest a single file, output results to JSON
ontology-engine ingest meeting_notes.md -o result.json

# Ingest all .md files in a directory
ontology-engine ingest-dir ./meetings/ --pattern "*.md"
```

### 5. (Optional) Connect to PostgreSQL for persistent storage

```bash
# Initialize the schema (requires PostgreSQL 15+ with pgvector, pg_trgm)
ontology-engine init --db-url "postgresql://user:pass@host:port/db"

# Ingest with storage
ontology-engine ingest meeting.md --db-url "postgresql://..."

# Query the graph
ontology-engine query "谁负责市场推广" --db-url "postgresql://..."

# View statistics
ontology-engine stats --db-url "postgresql://..."
```

### 6. Use as a Python library

```python
import asyncio
from ontology_engine.core.config import OntologyConfig, LLMConfig
from ontology_engine.pipeline.engine import PipelineEngine

async def main():
    config = OntologyConfig(
        llm=LLMConfig(
            provider="gemini",
            api_key="your-key",  # or set GEMINI_API_KEY env var
        )
    )
    engine = await PipelineEngine.create(config)
    result = await engine.ingest("meeting_notes.md")

    if result.success:
        print(f"Entities: {len(result.extraction.entities)}")
        print(f"Links: {len(result.extraction.links)}")
        print(f"Decisions: {len(result.extraction.decisions)}")
        print(f"Action Items: {len(result.extraction.action_items)}")

    await engine.close()

asyncio.run(main())
```

### Configuration

Create `config.json` or `config.toml`:

```json
{
  "llm": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "api_key": "your-key"
  },
  "database": {
    "url": "postgresql://localhost:5432/ontology"
  },
  "pipeline": {
    "min_confidence": 0.6,
    "segment_topics": true,
    "resolve_coreferences": true
  },
  "known_entities": {
    "aliases": {
      "Yann": ["Yann哥", "郭博", "验"]
    }
  }
}
```

Then use with: `ontology-engine -c config.json ingest meeting.md`

## Running Tests

```bash
# Unit tests (fast, no API key needed)
pytest tests/ --ignore=tests/test_integration.py -v

# Integration tests (requires GEMINI_API_KEY + meeting data)
pytest tests/test_integration.py -v -s
```

## Project Structure

```
ontology-engine/
├── src/ontology_engine/
│   ├── core/              # Core types, config, errors
│   │   ├── types.py           # 6 Entity + 13 Link type definitions
│   │   ├── config.py          # Pydantic configuration models
│   │   └── errors.py          # Custom exception hierarchy
│   ├── pipeline/          # Meeting-to-Ontology pipeline
│   │   ├── preprocessor.py    # Stage 1: Text cleaning, segmentation
│   │   ├── extractor.py       # Stage 2: 4-Pass structured extraction
│   │   ├── validator.py       # Stage 3: 3-Layer auto-correction
│   │   └── engine.py          # Pipeline orchestrator
│   ├── storage/           # PostgreSQL storage layer
│   │   ├── schema.py          # DDL (7 tables) & migration
│   │   └── repository.py      # Async CRUD operations
│   ├── llm/               # LLM provider abstraction
│   │   ├── base.py            # Abstract LLM interface
│   │   ├── gemini.py          # Google Gemini (default)
│   │   └── openai.py          # OpenAI
│   └── cli.py             # CLI entry point (click + rich)
├── tests/                 # 93 unit + 5 integration tests
├── pyproject.toml
└── README.md
```

## Known Limitations (Phase 1)

- **Semantic validation** (Layer 2) is disabled — contradiction detection is a Phase 2 feature
- **No embedding/vector search** — schema supports pgvector but it's not wired up yet
- **No incremental dedup** against DB — each ingest run is independent
- The `fast_model` tier (`gemini-2.0-flash-lite`) is used for preprocessing; extraction uses `gemini-2.5-flash`

## License

MIT

## Status

🟢 **Phase 1 MVP** — Core pipeline working: preprocessing → extraction → validation → storage
