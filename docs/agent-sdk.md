# Agent SDK Guide

The Agent SDK provides a high-level async Python API for AI agents to interact with the Ontology Engine knowledge graph. Agents can query, write, discover each other, and subscribe to real-time changes.

## Overview

The SDK consists of three core components:

| Component | Purpose |
|-----------|---------|
| `OntologyClient` | Primary API — read/write entities, links, search, graph traversal |
| `AgentRegistry` | Agent registration and discovery |
| `EventNotifier` | Real-time event pub/sub via PostgreSQL LISTEN/NOTIFY |

## Quick Start

```python
from ontology_engine.sdk.client import OntologyClient

async with OntologyClient("postgresql://user:pass@localhost/ontology") as client:
    # Register this agent
    await client.register_agent(
        id="my-agent",
        display_name="My Agent",
        produces=["Decision"],
        consumes=["Person", "Project"],
    )

    # Write an entity
    eid = await client.assert_entity(
        entity_type="Decision",
        name="Adopt microservices",
        properties={"detail": "Migrate to microservices by Q3"},
        source="meeting:2024-01-15",
    )

    # Query
    decisions = await client.query("Decision")
    print(f"Found {len(decisions)} decisions")
```

## OntologyClient API

### Connection Lifecycle

```python
# Option 1: Context manager (recommended)
async with OntologyClient("postgresql://...") as client:
    # client is connected and ready
    pass
# client is automatically closed

# Option 2: Manual connect/close
client = OntologyClient("postgresql://...")
await client.connect()
# ... use client ...
await client.close()
```

### Writing Data

#### `assert_entity()`

Create or update an entity. If an entity with the same type and name exists, its properties are merged (new properties overwrite old ones).

```python
entity_id = await client.assert_entity(
    entity_type="Decision",         # Required: entity type
    name="Adopt microservices",     # Optional: entity name (auto-generated if omitted)
    properties={                     # Required: entity properties
        "detail": "Migrate monolith to microservices by Q3",
        "rationale": "Improve scalability and deployment speed",
        "decision_type": "strategic",
    },
    source="meeting:2024-01-15",    # Required: provenance source
    confidence=0.95,                 # Optional: confidence score (0.0–1.0)
)
print(f"Entity ID: {entity_id}")    # e.g., "ENT-a1b2c3d4"
```

#### `assert_link()`

Create a relationship between two entities. Idempotent — if the same link already exists, it updates properties.

```python
link_id = await client.assert_link(
    source_id=person_id,            # Required: source entity ID
    link_type="makes",              # Required: relationship type
    target_id=decision_id,          # Required: target entity ID
    properties={                     # Optional: link properties
        "context": "During Q1 planning meeting",
    },
    confidence=0.9,                  # Optional: confidence score
)
```

#### `ingest()`

Store raw text in the Bronze layer. For full pipeline extraction, use `PipelineEngine` instead.

```python
result = await client.ingest(
    text="Meeting transcript content...",
    source="meeting:2024-01-15",
    source_type="meeting_transcript",
)
print(f"Document ID: {result.document_id}")
```

### Reading Data

#### `query()`

Query entities by type with optional filters.

```python
# All active decisions
decisions = await client.query("Decision")

# Filtered query
active_decisions = await client.query(
    "Decision",
    filters={"status": "active"},
    limit=10,
    offset=0,
)

# Each result is a dict
for d in active_decisions:
    print(f"{d['name']}: {d['properties'].get('detail', '')}")
```

#### `search()`

Full-text search using PostgreSQL trigram similarity. Searches across entity names and key properties.

```python
results = await client.search(
    "microservices",
    limit=5,
    entity_type="Decision",  # Optional: filter by type
)

for r in results:
    print(f"[{r['entity_type']}] {r['name']} (relevance: {r['relevance']:.2f})")
```

#### `get_entity()`

Fetch a single entity by ID.

```python
entity = await client.get_entity("ENT-a1b2c3d4")
if entity:
    print(f"{entity['entity_type']}: {entity['name']}")
    print(f"Properties: {entity['properties']}")
```

#### `get_linked()`

Graph traversal — find entities connected to a given entity.

```python
# All entities linked to this decision (outgoing + incoming)
linked = await client.get_linked(decision_id)

# Only outgoing links, specific type
actions = await client.get_linked(
    decision_id,
    link_type="generates",    # Optional: filter by link type
    direction="outgoing",     # "outgoing", "incoming", or "both"
    depth=2,                  # How many hops (1–5)
)

for entity in actions:
    link_info = entity["_link"]
    print(f"  [{link_info['link_type']}] → {entity['name']} (depth: {link_info['depth']})")
```

#### `describe()`

Schema introspection — useful for autonomous agents that need to understand what entity types exist and their schemas.

```python
schema = await client.describe("Decision")

print(f"Type: {schema.entity_type}")
print(f"Description: {schema.description}")
print(f"Required fields: {schema.required_fields}")

for prop in schema.properties:
    print(f"  {prop['name']}: {prop['type']} (required: {prop['required']})")

for lt in schema.link_types:
    print(f"  Link: {lt['link_type']} ({lt['direction']})")

for ex in schema.examples:
    print(f"  Example: {ex['name']}")
```

### Agent Registration

#### `register_agent()`

Register your agent so other agents can discover it.

```python
agent_info = await client.register_agent(
    id="meeting-bot",                       # Unique identifier
    display_name="Meeting Summary Bot",     # Human-readable name
    description="Extracts knowledge from meeting transcripts",
    produces=["Decision", "ActionItem"],    # Entity types this agent creates
    consumes=["Person", "Project"],         # Entity types this agent reads
    capabilities=["extraction", "summarization"],
    version="1.2.0",
    metadata={"model": "gemini-2.5-flash"},
)
```

#### `list_agents()`

Discover other registered agents.

```python
# All active agents
agents = await client.list_agents()

# Agents that produce Decisions
decision_producers = await client.list_agents(produces="Decision")

# Agents that consume Person entities
person_consumers = await client.list_agents(consumes="Person")

for agent in agents:
    print(f"{agent.display_name} (v{agent.version})")
    print(f"  Produces: {agent.produces}")
    print(f"  Consumes: {agent.consumes}")
```

### Event Subscription

Subscribe to real-time entity change notifications using PostgreSQL LISTEN/NOTIFY.

```python
async def on_decision_changed(event):
    """Called when any Decision entity is created or updated."""
    print(f"Event: {event.event_type}")
    print(f"Entity: {event.entity_type} ({event.entity_id})")
    print(f"Source: {event.source_agent}")
    print(f"Payload: {event.payload}")

# Subscribe to Decision changes
await client.subscribe("Decision", on_decision_changed)
```

**Event types:**
- `entity.created` — new entity created
- `entity.updated` — existing entity updated
- `link.created` — new link created
- `gold.fused` — Gold layer updated
- `conflict.detected` — conflicting data detected

## Building a Complete Agent

Here's a full example of an agent that monitors meetings and creates follow-up tasks:

```python
import asyncio
from ontology_engine.sdk.client import OntologyClient

DB_URL = "postgresql://user:pass@localhost/ontology"

async def follow_up_agent():
    async with OntologyClient(DB_URL) as client:
        # 1. Register
        await client.register_agent(
            id="follow-up-agent",
            display_name="Follow-Up Agent",
            description="Creates follow-up tasks from decisions",
            produces=["ActionItem"],
            consumes=["Decision"],
        )

        # 2. Find recent decisions without follow-ups
        decisions = await client.query("Decision", filters={"status": "active"})

        for decision in decisions:
            # Check if this decision already has action items
            linked = await client.get_linked(
                decision["id"],
                link_type="generates",
                direction="outgoing",
            )

            if not linked:
                # 3. Create a follow-up action item
                action_id = await client.assert_entity(
                    entity_type="ActionItem",
                    properties={
                        "task": f"Follow up on: {decision['name']}",
                        "priority": "medium",
                        "status": "pending",
                    },
                    source=f"agent:follow-up-agent",
                )

                # 4. Link it to the decision
                await client.assert_link(
                    source_id=decision["id"],
                    link_type="generates",
                    target_id=action_id,
                )

                print(f"Created follow-up for: {decision['name']}")

asyncio.run(follow_up_agent())
```

## Using with FastAPI REST API

If you prefer HTTP over the Python SDK, start the REST API server:

```bash
python -m ontology_engine.api.server --db-url "postgresql://..." --port 8000
```

Then make HTTP requests from any language:

```bash
# Register agent
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-agent",
    "display_name": "My Agent",
    "produces": ["Decision"],
    "consumes": ["Person"]
  }'

# Assert entity
curl -X POST http://localhost:8000/api/v1/assert/entity \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "Decision",
    "name": "Adopt microservices",
    "properties": {"detail": "Migrate by Q3"},
    "source": "meeting:2024-01-15",
    "confidence": 0.95
  }'

# Search
curl "http://localhost:8000/api/v1/search?q=microservices&limit=5"

# Graph traversal
curl "http://localhost:8000/api/v1/linked/ENT-abc123?direction=outgoing&depth=2"
```

## Error Handling

```python
from ontology_engine.core.errors import StorageError

try:
    entity = await client.get_entity("nonexistent-id")
except StorageError as e:
    print(f"Storage error: {e}")
```

The SDK raises `StorageError` for database-related errors. Connection errors from `asyncpg` may also be raised if the database is unreachable.
