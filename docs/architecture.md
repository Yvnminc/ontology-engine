# Architecture Overview

Ontology Engine follows a **medallion architecture** (Bronze → Silver → Gold) to transform unstructured text into a structured, queryable knowledge graph.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Ontology Engine                                │
│                                                                         │
│  ┌─────────────┐    ┌─────────────────────┐    ┌────────────────────┐  │
│  │   BRONZE    │    │       SILVER        │    │       GOLD         │  │
│  │             │    │                     │    │                    │  │
│  │ Raw docs    │───►│ 1. Preprocess       │───►│ Entity Resolution  │  │
│  │ (immutable, │    │    - clean text     │    │ - exact match      │  │
│  │  append-    │    │    - segment topics │    │ - alias match      │  │
│  │  only)      │    │    - resolve coref  │    │ - fuzzy (J-W)      │  │
│  │             │    │                     │    │ - embeddings       │  │
│  │ SHA-256     │    │ 2. Extract (4-pass) │    │                    │  │
│  │ dedup       │    │    - entities       │    │ Gold Builder       │  │
│  │             │    │    - relations      │    │ - merge clusters   │  │
│  │ Metadata:   │    │    - decisions      │    │ - aggregate links  │  │
│  │ - source    │    │    - action items   │    │ - gen embeddings   │  │
│  │ - format    │    │                     │    │                    │  │
│  │ - language  │    │ 3. Validate (3-lyr) │    │ Outputs:           │  │
│  │             │    │    - factual fix    │    │ - gold_entities    │  │
│  │             │    │    - semantic chk   │    │ - gold_links       │  │
│  │             │    │    - consistency    │    │ - embeddings       │  │
│  └─────────────┘    └─────────────────────┘    └─────────┬──────────┘  │
│                                                           │            │
│  ┌──────────────────────────────────────────────────────┐ │            │
│  │                    Agent SDK                         │ │            │
│  │                                                      │ │            │
│  │  OntologyClient (async Python)                       ◄─┘            │
│  │  ├─ assert_entity / assert_link   (write)            │              │
│  │  ├─ query / search / get_linked   (read)             │              │
│  │  ├─ describe                      (introspect)       │              │
│  │  ├─ subscribe                     (events)           │              │
│  │  └─ register_agent               (discovery)         │              │
│  │                                                      │              │
│  │  FastAPI REST API (/api/v1/*)     (HTTP)             │              │
│  │  EventNotifier (PG LISTEN/NOTIFY) (pub/sub)          │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐              │
│  │               Domain Schema System                    │              │
│  │  YAML definitions → entity types, link types,         │              │
│  │  properties, extraction hints, validation rules       │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐              │
│  │                   PostgreSQL                          │              │
│  │  pgvector (embeddings) + pg_trgm (text search)        │              │
│  │  JSONB (flexible properties) + full SQL               │              │
│  └──────────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Layers in Detail

### Bronze Layer

The **Bronze layer** is the entry point. All raw documents are stored here exactly as received — no transformation, no extraction. This ensures full traceability.

**Key properties:**
- **Immutable** — documents are never modified after ingestion
- **Append-only** — new documents are added, never deleted
- **Deduplicated** — SHA-256 hash prevents duplicate storage
- **Metadata-rich** — source type, URI, format, language, ingestion timestamp

**Database table:** `bronze_documents`

```
bronze_documents
├── id (UUID)
├── source_type (text)     — "meeting_transcript", "document", "report"
├── source_uri (text)      — file path or URL
├── source_hash (text)     — SHA-256 of content (UNIQUE)
├── content (text)         — raw document text
├── content_format (text)  — "text", "markdown", "html"
├── language (text)        — "auto", "zh", "en"
├── metadata (jsonb)       — arbitrary key-value pairs
├── ingested_at (timestamptz)
└── ingested_by (text)
```

### Silver Layer

The **Silver layer** processes Bronze documents through a multi-stage LLM pipeline to produce structured entities, links, decisions, and action items.

#### Stage 1: Preprocessing
- Text cleaning and normalization
- Filler word removal
- Topic segmentation (split long transcripts into coherent sections)
- Coreference resolution (pronouns → entity names)

#### Stage 2: 4-Pass Extraction
Each pass runs a specialized LLM prompt:

| Pass | Extracts | Output Type |
|------|----------|-------------|
| Pass 1 | People, projects, risks, deadlines | `ExtractedEntity` |
| Pass 2 | Relationships between entities | `ExtractedLink` |
| Pass 3 | Decisions with context and rationale | `ExtractedDecision` |
| Pass 4 | Action items with owners and deadlines | `ExtractedActionItem` |

#### Stage 3: 3-Layer Validation
| Layer | Purpose | Example |
|-------|---------|---------|
| Factual | Field format validation | Date format, enum values, required fields |
| Semantic | Contradiction detection | (Phase 2 — disabled) |
| Consistency | Cross-entity validation | Link targets exist, no orphaned references |

**Database tables:** `ont_entities`, `ont_links`, `ont_provenance`

### Gold Layer

The **Gold layer** is a **computed view** — it can always be fully rebuilt from Silver. It provides the canonical, deduplicated knowledge graph.

#### Entity Resolution Pipeline

```
Silver entities → Group by type → Pairwise comparison → Union-Find clustering → Gold entities
```

**Comparison strategies (in order):**
1. **Exact name match** (case-insensitive) → similarity 1.0
2. **Alias match** (from config or entity aliases) → similarity 1.0
3. **Embedding cosine similarity** (if available)
4. **Jaro-Winkler fuzzy match**

**Merge decisions:**
- Similarity > 0.95 → **auto-merge** (no human review)
- 0.80 – 0.95 → **flag for review** (not auto-merged)
- < 0.80 → **treat as distinct**

**Database tables:** `gold_entities`, `gold_links`

### Agent SDK

The SDK provides a high-level async Python API for AI agents:

```python
async with OntologyClient("postgresql://...") as client:
    # Write
    eid = await client.assert_entity("Decision", {...}, source="meeting:2024-01-15")

    # Read
    results = await client.query("Decision", filters={"status": "active"})
    linked = await client.get_linked(eid, direction="outgoing", depth=2)

    # Subscribe
    await client.subscribe("Decision", my_callback)
```

### Domain Schema System

YAML files define custom ontologies per domain:

```yaml
domain: finance
entity_types:
  - name: Transaction
    properties:
      - { name: amount, type: number, required: true }
link_types:
  - { name: involves, source_types: [Transaction], target_types: [Account] }
```

The extraction pipeline dynamically adapts prompts, validation rules, and type definitions based on the active schema.

## Data Flow

```
1. User provides text (CLI, API, or SDK)
         │
         ▼
2. Bronze: Store raw document
         │
         ▼
3. Silver: Preprocess → Extract → Validate
         │
         ▼
4. Silver: Store entities, links, provenance in PostgreSQL
         │
         ▼
5. Gold: Resolve entities → Merge clusters → Aggregate links → Embed
         │
         ▼
6. Agents query Gold layer via SDK or REST API
```

## Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Async/await, type hints, LLM ecosystem |
| Database | PostgreSQL 15+ | JSONB flexibility, SQL power, extensions |
| Vectors | pgvector | Native PG extension, no separate vector DB |
| Text search | pg_trgm | Trigram similarity, fast fuzzy search |
| LLM | Gemini (default) | Cost-effective, structured output |
| Embeddings | OpenAI | text-embedding-3-small, 1536 dimensions |
| API | FastAPI | Async, auto-docs, Pydantic integration |
| CLI | Click + Rich | Composable commands, beautiful output |
| Config | Pydantic v2 | Type-safe, JSON/TOML support |
