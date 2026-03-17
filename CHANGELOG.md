# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-03-17

### 🎉 Initial Release — Phase 1: Static Ontology

First public release of Ontology Engine. Complete Bronze→Silver→Gold pipeline for transforming unstructured text into a structured knowledge graph.

### Added

#### Bronze Layer (M1)
- Immutable, append-only raw document storage
- SHA-256 content deduplication
- Source tracking with type, URI, format, and language metadata
- `BronzeRepository` async CRUD operations
- CLI commands: `bronze list`, `bronze show`

#### Silver Layer — Schema-Driven Extraction (M2)
- **Preprocessing pipeline**: text cleaning, filler word removal, topic segmentation, coreference resolution
- **4-pass LLM extraction**:
  - Pass 1: Entity extraction (Person, Decision, ActionItem, Project, Risk, Deadline)
  - Pass 2: Relationship extraction (13 link types with temporal validity)
  - Pass 3: Decision extraction (strategic/tactical/operational with rationale)
  - Pass 4: Action item extraction (with owners, due dates, priority)
- **3-layer auto-correction**:
  - Layer 1: Factual correction (field validation, format normalization)
  - Layer 2: Semantic validation (placeholder for Phase 2)
  - Layer 3: Consistency checking (cross-entity and cross-link validation)
- Full provenance tracking: every fact traced to source text + extraction pass
- `PipelineEngine` orchestrator with batch directory processing

#### Gold Layer — Entity Resolution & Embeddings (M3)
- **Entity resolution engine** with 4 strategies:
  - Exact name matching (case-insensitive)
  - Alias matching (via known_entities config + entity alias overlap)
  - Jaro-Winkler fuzzy string similarity
  - Embedding cosine similarity (when available)
- Configurable merge thresholds: auto-merge (>0.95), review (0.80–0.95)
- Union-Find clustering for multi-entity merges
- **Embedding generation** via OpenAI text-embedding-3-small (1536 dimensions)
- Incremental and full rebuild modes
- Gold entity/link aggregation with mention counting
- `GoldBuilder` pipeline with `GoldBuildResult` reporting

#### Agent SDK & FastAPI (M4)
- **`OntologyClient`** — async Python SDK for AI agents:
  - `assert_entity()` / `assert_link()` — upsert with automatic deduplication
  - `query()` / `search()` — type-filtered and trigram similarity search
  - `get_entity()` / `get_linked()` — graph traversal up to depth 5
  - `describe()` — schema introspection for autonomous agents
  - `register_agent()` / `list_agents()` — agent discovery
  - `subscribe()` — real-time event notifications
  - `ingest()` — raw text ingestion to Bronze
- **`AgentRegistry`** — PostgreSQL-backed agent registration and discovery
- **`EventNotifier`** — pub/sub via PostgreSQL LISTEN/NOTIFY
- **FastAPI REST API** — 12 endpoints under `/api/v1/`:
  - Health, query, search, entity CRUD, graph traversal
  - Agent registration and listing
  - Schema introspection
  - CORS enabled, Swagger UI at `/docs`

#### Domain Schema System (M5)
- YAML-based domain schema definitions
- `DomainSchema` loader with Pydantic validation
- `SchemaRegistry` singleton for multi-domain support
- Custom entity types with typed properties (string, integer, enum, date)
- Custom link types with source/target type constraints
- Extraction hints for LLM guidance
- Validation rules per schema
- Built-in schemas: `default`, `edtech`, `finance`
- CLI commands: `schema list`, `schema show`, `schema validate`

#### Infrastructure
- PostgreSQL 15+ with pgvector and pg_trgm extensions
- Pluggable LLM providers: Gemini (default), OpenAI
- 3-tier model configuration: fast, default, strong
- Pydantic v2 configuration with JSON/TOML file support
- Click + Rich CLI with subcommands
- 258 unit tests + 46 integration tests
- MIT license

[0.1.0]: https://github.com/Yvnminc/ontology-engine/releases/tag/v0.1.0
