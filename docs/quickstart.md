# Quick Start Guide

Get up and running with Ontology Engine in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- A Gemini API key (or OpenAI API key)
- PostgreSQL 15+ with pgvector and pg_trgm (optional — for persistent storage)

## Installation

### From PyPI

```bash
# Core only
pip install ontology-engine

# With all optional dependencies (recommended)
pip install ontology-engine[all]

# Specific extras
pip install ontology-engine[gemini]   # Google Gemini LLM
pip install ontology-engine[openai]   # OpenAI LLM + embeddings
pip install ontology-engine[api]      # FastAPI REST API
```

### From Source

```bash
git clone https://github.com/Yvnminc/ontology-engine.git
cd ontology-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

## Set Up Your API Key

```bash
# Google Gemini (default LLM provider)
export GEMINI_API_KEY="your-gemini-key"

# OpenAI (for embeddings in Gold layer)
export OPENAI_API_KEY="your-openai-key"  # optional
```

## Usage Option 1: CLI

### Process a Single File

```bash
# Extract knowledge from a meeting transcript, save to JSON
ontology-engine ingest meeting_notes.md -o result.json
```

Output:
```
✓ Processed: meeting_notes.md
  Time: 3420ms
  Entities: 12
  Links: 8
  Decisions: 3
  Action Items: 5
  Auto-fixes: 2
```

### Process a Directory

```bash
ontology-engine ingest-dir ./meetings/ --pattern "*.md"
```

### Use a Domain Schema

```bash
# Use the built-in edtech schema
ontology-engine ingest meeting.md --schema edtech

# List available schemas
ontology-engine schema list
```

### With PostgreSQL Storage

```bash
# Initialize database schema
ontology-engine init --db-url "postgresql://user:pass@localhost:5432/ontology"

# Ingest with persistent storage
ontology-engine ingest meeting.md --db-url "postgresql://user:pass@localhost:5432/ontology"

# Query the knowledge graph
ontology-engine query "市场推广" --db-url "postgresql://..."

# View statistics
ontology-engine stats --db-url "postgresql://..."
```

## Usage Option 2: Python Library

### Basic Extraction

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

        # Access individual entities
        for entity in result.extraction.entities:
            print(f"  [{entity.entity_type}] {entity.name} (confidence: {entity.confidence})")

    await engine.close()

asyncio.run(main())
```

### With Database Storage

```python
import asyncio
from ontology_engine.core.config import OntologyConfig
from ontology_engine.pipeline.engine import PipelineEngine

DB_URL = "postgresql://user:pass@localhost:5432/ontology"

async def main():
    config = OntologyConfig()
    engine = await PipelineEngine.create(config, db_url=DB_URL)

    # Process a meeting
    result = await engine.ingest("meeting.md")

    # Process all files in a directory
    results = await engine.ingest_directory("./meetings/", pattern="*.md")

    await engine.close()

asyncio.run(main())
```

### With Domain Schema

```python
import asyncio
from ontology_engine.core.config import OntologyConfig
from ontology_engine.core.schema_registry import DomainSchema
from ontology_engine.pipeline.engine import PipelineEngine

async def main():
    config = OntologyConfig()
    schema = DomainSchema.from_yaml("domain_schemas/edtech.yaml")

    engine = await PipelineEngine.create(config, domain_schema=schema)
    result = await engine.ingest("class_meeting.md")
    await engine.close()

asyncio.run(main())
```

## Usage Option 3: REST API

### Start the Server

```bash
python -m ontology_engine.api.server \
    --db-url "postgresql://user:pass@localhost:5432/ontology" \
    --port 8000
```

### Make Requests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Query entities
curl "http://localhost:8000/api/v1/query?entity_type=Decision&limit=10"

# Search
curl "http://localhost:8000/api/v1/search?q=microservices"

# Assert an entity
curl -X POST http://localhost:8000/api/v1/assert/entity \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "Decision",
    "name": "Adopt microservices",
    "properties": {"detail": "Migrate by Q3"},
    "source": "meeting:2024-01-15",
    "confidence": 0.95
  }'
```

Interactive docs: `http://localhost:8000/docs`

## Usage Option 4: Docker

```bash
# Start Ontology Engine + PostgreSQL + pgvector
docker compose up -d

# Initialize the database
docker compose exec ontology-engine ontology-engine init --db-url "$DATABASE_URL"

# Ingest a file
docker compose exec ontology-engine ontology-engine ingest /data/meeting.md
```

## What's Next?

- **[Domain Schemas](domain-schemas.md)** — Define custom ontologies for your domain
- **[Agent SDK](agent-sdk.md)** — Build AI agents that use the knowledge graph
- **[API Reference](api-reference.md)** — Full REST API documentation
- **[Architecture](architecture.md)** — Deep dive into the system design
