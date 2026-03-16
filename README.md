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
- **LLM**: Pluggable (Gemini, OpenAI, Anthropic)
- **Hosting**: Supabase-compatible

## Quick Start

```bash
pip install ontology-engine

# Initialize database schema
ontology-engine init --db-url postgresql://...

# Process a meeting transcript
ontology-engine ingest meeting_notes.md

# Query the ontology
ontology-engine query "Who is responsible for LearningOS?"
```

## Project Structure

```
ontology-engine/
├── src/ontology_engine/
│   ├── core/              # Core types, config, errors
│   ├── pipeline/          # Meeting-to-Ontology pipeline
│   │   ├── preprocessor.py    # Stage 1: Text cleaning, segmentation
│   │   ├── extractor.py       # Stage 2: 4-Pass structured extraction
│   │   └── validator.py       # Stage 3: 3-Layer auto-correction
│   ├── storage/           # PostgreSQL storage layer
│   │   ├── schema.py          # DDL & migrations
│   │   ├── repository.py      # CRUD operations
│   │   └── query.py           # Query interface for agents
│   ├── llm/               # LLM provider abstraction
│   │   ├── base.py            # Abstract LLM interface
│   │   ├── gemini.py          # Google Gemini
│   │   └── openai.py          # OpenAI
│   └── cli.py             # CLI entry point
├── tests/
├── migrations/
├── pyproject.toml
└── README.md
```

## License

MIT

## Status

🚧 **Phase 1 MVP** — Under active development
