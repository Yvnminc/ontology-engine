# API Reference

Ontology Engine provides a FastAPI-based REST API under `/api/v1/`. All endpoints return JSON.

## Starting the Server

```bash
# Via module entry point
python -m ontology_engine.api.server \
    --db-url "postgresql://user:pass@localhost:5432/ontology" \
    --host 0.0.0.0 \
    --port 8000 \
    --reload  # development mode

# Via uvicorn directly
DATABASE_URL="postgresql://..." uvicorn ontology_engine.api.server:app --reload

# Via Docker
docker compose up -d
```

Interactive Swagger UI is available at `http://localhost:8000/docs`.

## Endpoints

### Health

#### `GET /api/v1/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

### Query & Search

#### `GET /api/v1/query`

Query entities by type with optional filters.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entity_type` | string | ✅ | — | Entity type to query (e.g., `Decision`, `Person`) |
| `limit` | integer | — | 20 | Max results (1–200) |
| `offset` | integer | — | 0 | Pagination offset |
| `status` | string | — | — | Filter by status (`active`, `archived`) |
| `keyword` | string | — | — | Filter by keyword in properties |

**Example:**
```bash
curl "http://localhost:8000/api/v1/query?entity_type=Decision&status=active&limit=5"
```

**Response:**
```json
{
  "entities": [
    {
      "id": "ENT-a1b2c3d4",
      "entity_type": "Decision",
      "name": "Adopt microservices",
      "properties": {
        "detail": "Migrate to microservices by Q3",
        "decision_type": "strategic",
        "rationale": "Scalability"
      },
      "aliases": [],
      "confidence": 0.95,
      "version": 1,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "created_by": "meeting-bot"
    }
  ],
  "total": 1,
  "limit": 5,
  "offset": 0
}
```

#### `GET /api/v1/search`

Full-text search using PostgreSQL trigram similarity.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | ✅ | — | Search query |
| `limit` | integer | — | 10 | Max results (1–100) |
| `entity_type` | string | — | — | Filter by entity type |

**Example:**
```bash
curl "http://localhost:8000/api/v1/search?q=microservices&limit=5"
```

**Response:**
```json
{
  "results": [
    {
      "id": "ENT-a1b2c3d4",
      "entity_type": "Decision",
      "name": "Adopt microservices architecture",
      "properties": {"detail": "..."},
      "relevance": 0.85
    }
  ],
  "query": "microservices",
  "total": 1
}
```

---

### Entity CRUD

#### `GET /api/v1/entity/{entity_id}`

Fetch a single entity by ID.

**Example:**
```bash
curl "http://localhost:8000/api/v1/entity/ENT-a1b2c3d4"
```

**Response:**
```json
{
  "id": "ENT-a1b2c3d4",
  "entity_type": "Decision",
  "name": "Adopt microservices",
  "properties": {"detail": "..."},
  "aliases": [],
  "confidence": 0.95,
  "version": 1,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "created_by": "meeting-bot"
}
```

**Errors:**
- `404` — Entity not found

#### `POST /api/v1/assert/entity`

Create or update an entity. If an entity with the same type and name exists, properties are merged.

**Request body:**
```json
{
  "entity_type": "Decision",
  "name": "Adopt microservices",
  "properties": {
    "detail": "Migrate monolith to microservices by Q3",
    "rationale": "Scalability"
  },
  "source": "meeting:2024-01-15",
  "confidence": 0.95
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_type` | string | ✅ | Entity type |
| `name` | string | — | Entity name (auto-generated if omitted) |
| `properties` | object | ✅ | Entity properties |
| `source` | string | ✅ | Provenance source identifier |
| `confidence` | number | — | Confidence score (0.0–1.0, default 0.9) |

**Response:**
```json
{
  "id": "ENT-a1b2c3d4",
  "type": "entity",
  "action": "created"
}
```

#### `POST /api/v1/assert/link`

Create a relationship between two entities.

**Request body:**
```json
{
  "source_id": "ENT-a1b2c3d4",
  "link_type": "generates",
  "target_id": "ENT-e5f6g7h8",
  "properties": {
    "context": "Discussed during planning meeting"
  },
  "confidence": 0.9
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_id` | string | ✅ | Source entity ID |
| `link_type` | string | ✅ | Relationship type |
| `target_id` | string | ✅ | Target entity ID |
| `properties` | object | — | Link properties |
| `confidence` | number | — | Confidence score (0.0–1.0, default 0.9) |

**Response:**
```json
{
  "id": "LNK-x1y2z3",
  "type": "link",
  "action": "created"
}
```

---

### Graph Traversal

#### `GET /api/v1/linked/{entity_id}`

Traverse the knowledge graph from a given entity.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `link_type` | string | — | — | Filter by relationship type |
| `direction` | string | — | `both` | `outgoing`, `incoming`, or `both` |
| `depth` | integer | — | 1 | Traversal depth (1–5) |

**Example:**
```bash
curl "http://localhost:8000/api/v1/linked/ENT-a1b2c3d4?direction=outgoing&depth=2"
```

**Response:**
```json
{
  "entity_id": "ENT-a1b2c3d4",
  "linked": [
    {
      "id": "ENT-e5f6g7h8",
      "entity_type": "ActionItem",
      "name": "Create migration plan",
      "properties": {"task": "...", "priority": "high"},
      "link": {
        "link_id": "LNK-x1y2z3",
        "link_type": "generates",
        "direction": "outgoing",
        "depth": 1
      }
    }
  ],
  "total": 1
}
```

---

### Schema Introspection

#### `GET /api/v1/describe/{entity_type}`

Get the schema definition for an entity type, including properties, link types, and examples.

**Example:**
```bash
curl "http://localhost:8000/api/v1/describe/Decision"
```

**Response:**
```json
{
  "entity_type": "Decision",
  "description": "A decision made during a meeting",
  "properties": [
    {
      "name": "detail",
      "type": "string",
      "required": false,
      "description": "Detailed description of the decision",
      "enum_values": null
    },
    {
      "name": "decision_type",
      "type": "string",
      "required": false,
      "description": "",
      "enum_values": ["strategic", "tactical", "operational"]
    }
  ],
  "required_fields": ["summary"],
  "link_types": [
    {"link_type": "makes", "direction": "incoming"},
    {"link_type": "generates", "direction": "outgoing"},
    {"link_type": "relates_to", "direction": "outgoing"}
  ],
  "examples": [
    {
      "id": "ENT-abc123",
      "name": "Adopt microservices",
      "properties": {"detail": "..."}
    }
  ]
}
```

---

### Ingestion

#### `POST /api/v1/ingest`

Store raw text in the Bronze layer. This does **not** run the full extraction pipeline — use `PipelineEngine` for that.

**Request body:**
```json
{
  "text": "Meeting transcript content...",
  "source": "meeting:2024-01-15-standup",
  "source_type": "meeting_transcript"
}
```

**Response:**
```json
{
  "document_id": "DOC-abc123",
  "entities_created": 0,
  "links_created": 0,
  "entity_ids": [],
  "link_ids": [],
  "processing_time_ms": 12
}
```

---

### Agent Management

#### `GET /api/v1/agents`

List registered agents.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | `active` | Filter by status |
| `produces` | string | — | Filter by entity type produced |
| `consumes` | string | — | Filter by entity type consumed |

**Example:**
```bash
curl "http://localhost:8000/api/v1/agents?produces=Decision"
```

**Response:**
```json
{
  "agents": [
    {
      "id": "meeting-bot",
      "display_name": "Meeting Bot",
      "description": "Extracts knowledge from meeting transcripts",
      "produces": ["Decision", "ActionItem"],
      "consumes": ["Person", "Project"],
      "capabilities": ["extraction"],
      "version": "1.0.0",
      "status": "active",
      "metadata": {},
      "registered_at": "2024-01-15T10:00:00Z",
      "last_seen_at": "2024-01-15T12:00:00Z"
    }
  ],
  "total": 1
}
```

#### `POST /api/v1/agents/register`

Register a new agent (or update existing).

**Request body:**
```json
{
  "id": "meeting-bot",
  "display_name": "Meeting Bot",
  "description": "Extracts knowledge from meeting transcripts",
  "produces": ["Decision", "ActionItem"],
  "consumes": ["Person", "Project"],
  "capabilities": ["extraction", "summarization"],
  "version": "1.0.0",
  "metadata": {
    "model": "gemini-2.5-flash"
  }
}
```

**Response:** Same as agent object in list response.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error description"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (invalid parameters) |
| 404 | Entity not found |
| 422 | Validation error (invalid request body) |
| 503 | Service unavailable (database not connected) |

## CORS

CORS is enabled for all origins by default. Configure in `create_app()` if you need to restrict access.

## Authentication

Phase 1 does not include authentication. The API is intended for internal/trusted network use. Authentication will be added in Phase 2.
