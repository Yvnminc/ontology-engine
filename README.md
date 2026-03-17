<p align="center">
  <h1 align="center">🧠 Ontology Engine</h1>
  <p align="center">
    <strong>Transform unstructured text into structured knowledge graphs — powered by LLMs, stored in PostgreSQL.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/ontology-engine/"><img src="https://img.shields.io/pypi/v/ontology-engine?color=blue" alt="PyPI"></a>
    <a href="https://pypi.org/project/ontology-engine/"><img src="https://img.shields.io/pypi/pyversions/ontology-engine" alt="Python"></a>
    <a href="https://github.com/Yvnminc/ontology-engine/actions"><img src="https://github.com/Yvnminc/ontology-engine/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
    <a href="https://github.com/Yvnminc/ontology-engine/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Yvnminc/ontology-engine" alt="License"></a>
  </p>
</p>

---

An open-source, schema-driven framework that extracts entities, relationships, decisions, and action items from meeting transcripts (or any text) into a PostgreSQL-backed knowledge graph. Designed for multi-agent AI systems that need shared organizational memory.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              Ontology Engine                  │
                    │                                              │
  Text Input ──────►  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
  (meetings,       │  │  BRONZE  │  │  SILVER  │  │   GOLD   │   │
   documents,      │  │          │  │          │  │          │   │
   reports)        │  │ Raw doc  ├──► Schema-  ├──► Entity   │   │
                    │  │ storage  │  │ driven   │  │ resoln + │   │
                    │  │ (append  │  │ extract  │  │ merge +  │   │
                    │  │  only)   │  │ + valid  │  │ embeds   │   │
                    │  └──────────┘  └──────────┘  └────┬─────┘   │
                    │                                    │         │
                    │  ┌──────────┐  ┌──────────────────┘         │
                    │  │  Agent   │  │                             │
                    │  │   SDK    ◄──┘  PostgreSQL + pgvector      │
                    │  │ + REST   │     (JSONB knowledge graph)    │
                    │  │   API    │                                │
                    │  └────┬─────┘                                │
                    └───────┼──────────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   AI Agents   │
                    │  query, write │
                    │  subscribe    │
                    └───────────────┘
```

**Bronze Layer** — Immutable, append-only raw document store with deduplication (SHA-256).

**Silver Layer** — Schema-driven LLM extraction pipeline: preprocessing → 4-pass extraction (entities, relations, decisions, actions) → 3-layer auto-correction.

**Gold Layer** — Entity resolution (Jaro-Winkler + alias matching + embeddings), deduplication, and computed aggregation. The canonical knowledge graph.

**Agent SDK** — Python async client + FastAPI REST API for AI agents to query, write, and subscribe to the knowledge graph.

## Quick Start

### Install

```bash
pip install ontology-engine[all]
```

### Extract knowledge from text (3 lines)

```python
import asyncio
from ontology_engine.pipeline.engine import PipelineEngine
from ontology_engine.core.config import OntologyConfig

async def main():
    engine = await PipelineEngine.create(OntologyConfig())
    result = await engine.ingest("meeting_notes.md")
    print(f"Found {len(result.extraction.entities)} entities, "
          f"{len(result.extraction.decisions)} decisions")

asyncio.run(main())
```

### Or use the CLI

```bash
# Set your LLM API key
export GEMINI_API_KEY="your-key"

# Process a single file
ontology-engine ingest meeting.md -o result.json

# Process a directory
ontology-engine ingest-dir ./meetings/ --pattern "*.md"

# Use a domain schema
ontology-engine ingest meeting.md --schema edtech
```

## Features

### 🟤 Bronze Layer — Raw Document Storage
- Append-only, immutable document store
- SHA-256 content deduplication
- Source tracking (type, URI, format, language)
- CLI: `ontology-engine bronze list`, `ontology-engine bronze show <id>`

### ⚪ Silver Layer — Schema-Driven Extraction
- **Preprocessing**: text cleaning, topic segmentation, coreference resolution
- **4-Pass extraction**: Entities → Relations → Decisions → Action Items
- **3-Layer validation**: factual correction, semantic checks, consistency enforcement
- **6 built-in entity types**: Person, Decision, ActionItem, Project, Risk, Deadline
- **13 link types**: participates_in, makes, assigned_to, relates_to, generates, supersedes, depends_on, mitigates, blocks, reports_to, collaborates_with, owns, deadline_for
- **Provenance tracking**: every fact traced to source text + extraction pass

### 🟡 Gold Layer — Entity Resolution & Embeddings
- **Entity resolution**: exact match, alias matching, Jaro-Winkler fuzzy matching
- **Configurable thresholds**: auto-merge (>0.95), review (0.80–0.95), distinct (<0.80)
- **Embedding generation**: OpenAI text-embedding-3-small for semantic search
- **Incremental builds**: only process new Silver entities
- **Full rebuild**: clear and recompute from Silver at any time

### 🤖 Agent SDK & REST API
- **Python async client** (`OntologyClient`): query, search, assert entities/links, subscribe to events
- **Agent registry**: agents register what they produce/consume
- **FastAPI REST API**: 12 endpoints for HTTP integration
- **Event system**: PostgreSQL LISTEN/NOTIFY for real-time change notifications

### 📋 Domain Schemas (YAML)
- Define custom entity types, link types, and properties per domain
- Built-in schemas: `default`, `edtech`, `finance`
- Extraction hints for LLM guidance
- Validation rules per schema

## Domain Schemas

Define domain-specific ontologies in YAML:

```yaml
domain: edtech
version: "1.0.0"
description: "Education technology domain"

entity_types:
  - name: Student
    description: "A student or learner"
    properties:
      - { name: student_id, type: string, required: true }
      - { name: grade_level, type: string }
      - { name: enrollment_status, type: enum, enum_values: ["active","graduated","withdrawn"] }
    extraction_hint: "学生、学员、同学等"

  - name: Course
    description: "A course or class"
    properties:
      - { name: course_code, type: string, required: true }
      - { name: credits, type: integer }

link_types:
  - { name: enrolled_in, source_types: [Student], target_types: [Course] }
  - { name: teaches, source_types: [Tutor], target_types: [Course] }

extraction:
  system_prompt: |
    You are an education-focused knowledge extraction system.
    Extract students, courses, knowledge units, and their relationships.
```

Use with CLI: `ontology-engine ingest meeting.md --schema edtech`

Manage schemas:
```bash
ontology-engine schema list          # List available schemas
ontology-engine schema show edtech   # Show schema details
ontology-engine schema validate my_schema.yaml
```

## Agent SDK

Build AI agents that read and write to the shared knowledge graph:

```python
from ontology_engine.sdk.client import OntologyClient

async with OntologyClient("postgresql://localhost/ontology") as client:
    # Register your agent
    await client.register_agent(
        id="meeting-bot",
        display_name="Meeting Bot",
        produces=["Decision", "ActionItem"],
        consumes=["Person", "Project"],
    )

    # Write: assert entities and links
    decision_id = await client.assert_entity(
        entity_type="Decision",
        name="Adopt microservices architecture",
        properties={"detail": "Migrate monolith to microservices by Q3", "rationale": "Scalability"},
        source="meeting:2024-01-15",
        confidence=0.95,
    )

    # Read: query and search
    decisions = await client.query("Decision", filters={"status": "active"})
    results = await client.search("microservices", limit=5)

    # Traverse the graph
    linked = await client.get_linked(decision_id, direction="outgoing", depth=2)

    # Schema introspection (for autonomous agents)
    schema = await client.describe("Decision")

    # Subscribe to changes (PostgreSQL LISTEN/NOTIFY)
    async def on_change(event):
        print(f"Entity changed: {event.entity_type} {event.entity_id}")

    await client.subscribe("Decision", on_change)
```

## API Reference

Start the API server:

```bash
# Via module
python -m ontology_engine.api.server --db-url "postgresql://..." --port 8000

# Or with uvicorn
uvicorn ontology_engine.api.server:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/query` | Query entities by type + filters |
| `GET` | `/api/v1/search?q=...` | Full-text search (trigram similarity) |
| `GET` | `/api/v1/entity/{id}` | Get entity by ID |
| `GET` | `/api/v1/linked/{id}` | Graph traversal from entity |
| `POST` | `/api/v1/ingest` | Ingest raw text (Bronze) |
| `POST` | `/api/v1/assert/entity` | Create/update entity |
| `POST` | `/api/v1/assert/link` | Create relationship |
| `GET` | `/api/v1/describe/{type}` | Schema introspection |
| `GET` | `/api/v1/agents` | List registered agents |
| `POST` | `/api/v1/agents/register` | Register an agent |

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | — |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) | — |
| `DATABASE_URL` | PostgreSQL connection URL | — |

### Config File

```json
{
  "llm": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "api_key": "your-key",
    "temperature": 0.1
  },
  "database": {
    "url": "postgresql://localhost:5432/ontology",
    "db_schema": "ontology"
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

Use with: `ontology-engine -c config.json ingest meeting.md`

### LLM Model Tiers

| Tier | Default Model | Used For |
|------|--------------|----------|
| Fast | `gemini-2.0-flash-lite` | Preprocessing, simple tasks |
| Default | `gemini-2.5-flash` | Extraction, validation |
| Strong | `gemini-2.5-pro` | Complex reasoning, conflict resolution |

Supports pluggable LLM providers: Gemini (default), OpenAI.

## Database Setup

```bash
# Initialize schema (creates tables, indexes, extensions)
ontology-engine init --db-url "postgresql://user:pass@localhost:5432/ontology"
```

Requires PostgreSQL 15+ with extensions:
- `pgvector` — vector similarity search
- `pg_trgm` — trigram text search

Or use Docker:
```bash
docker compose up -d
```

## Testing

```bash
# Unit tests (no API key or DB required, fast)
pytest tests/ --ignore=tests/integration/ -v

# Integration tests (requires GEMINI_API_KEY + PostgreSQL)
pytest tests/integration/ -v -s

# All tests
pytest -v

# With coverage
pytest --cov=ontology_engine --cov-report=term-missing
```

## Project Structure

```
ontology-engine/
├── src/ontology_engine/
│   ├── core/              # Types, config, errors, schema format
│   │   ├── types.py           # 6 Entity + 13 Link type definitions
│   │   ├── config.py          # Pydantic configuration models
│   │   ├── schema_format.py   # Domain schema Pydantic models
│   │   ├── schema_registry.py # Schema loader + registry
│   │   └── errors.py          # Custom exception hierarchy
│   ├── pipeline/          # Bronze→Silver extraction pipeline
│   │   ├── preprocessor.py    # Text cleaning, segmentation
│   │   ├── extractor.py       # 4-Pass LLM extraction
│   │   ├── validator.py       # 3-Layer auto-correction
│   │   └── engine.py          # Pipeline orchestrator
│   ├── fusion/            # Silver→Gold aggregation
│   │   ├── entity_resolver.py # Jaro-Winkler + alias entity resolution
│   │   ├── gold_builder.py    # Gold layer builder
│   │   └── embeddings.py      # OpenAI embedding generation
│   ├── storage/           # PostgreSQL persistence
│   │   ├── bronze.py          # Bronze document repository
│   │   ├── repository.py      # Silver entity/link CRUD
│   │   ├── gold_repository.py # Gold entity/link CRUD
│   │   └── schema.py          # DDL + migrations
│   ├── sdk/               # Agent SDK
│   │   ├── client.py          # OntologyClient (async)
│   │   └── registry.py        # Agent registration + discovery
│   ├── api/               # FastAPI REST API
│   │   ├── server.py          # App factory + lifespan
│   │   ├── routes.py          # 12 API endpoints
│   │   └── models.py          # Request/response schemas
│   ├── llm/               # LLM provider abstraction
│   │   ├── base.py            # Abstract interface
│   │   ├── gemini.py          # Google Gemini
│   │   └── openai.py          # OpenAI
│   ├── events/            # Event system
│   │   └── notifier.py        # PG LISTEN/NOTIFY pub/sub
│   └── cli.py             # CLI entry point (Click + Rich)
├── domain_schemas/        # Built-in YAML domain schemas
│   ├── default.yaml
│   ├── edtech.yaml
│   └── finance.yaml
├── tests/                 # 258 unit + 46 integration tests
├── docs/                  # Documentation
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── CHANGELOG.md
```

## Contributing

We welcome contributions! Here's how to get started:

```bash
# Clone and install in development mode
git clone https://github.com/Yvnminc/ontology-engine.git
cd ontology-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"

# Run linting
ruff check src/ tests/
ruff format src/ tests/

# Run type checking
mypy src/ontology_engine/

# Run tests
pytest -v
```

Please open an issue first to discuss significant changes.

## Roadmap

### ✅ Phase 1: Static Ontology (current — v0.1.0)
- Bronze → Silver → Gold full pipeline
- Schema-driven extraction with YAML domain schemas
- Entity resolution + embedding generation
- Agent SDK + FastAPI REST API
- PostgreSQL + pgvector storage
- CLI tools

### 🔜 Phase 2: Kinetic Ontology
- Real-time streaming ingestion
- Incremental Gold updates on every ingest
- Conflict detection and resolution
- Temporal queries (point-in-time knowledge graph)
- Webhook integrations

### 🔮 Phase 3: Dynamic Ontology
- Schema evolution and migration
- Cross-domain knowledge fusion
- Autonomous agent orchestration
- Graph neural network embeddings
- Multi-tenant deployment

## License

[MIT](LICENSE) © [Yann Guo](https://github.com/Yvnminc)
