# Domain Schemas

Domain schemas define custom ontologies in YAML. They tell the extraction pipeline what entity types, link types, and properties to look for in your specific domain.

## Overview

Out of the box, Ontology Engine provides 6 built-in entity types (Person, Decision, ActionItem, Project, Risk, Deadline) and 13 link types. Domain schemas let you extend or replace these with types specific to your domain — education, finance, healthcare, etc.

## Built-in Schemas

| Schema | Domain | Entity Types | Link Types |
|--------|--------|-------------|------------|
| `default` | General (meetings) | Person, Decision, ActionItem, Project, Risk, Deadline | 13 standard types |
| `edtech` | Education technology | Student, Course, KnowledgeUnit, Tutor, Decision, ActionItem | enrolled_in, teaches, covers, mastered, etc. |
| `finance` | Financial analysis | (custom financial entities and relationships) | (custom financial link types) |

## Schema File Format

```yaml
# Required: domain identifier and version
domain: my_domain
version: "1.0.0"
description: "Description of this domain schema"

# Entity types to extract
entity_types:
  - name: EntityTypeName        # Required, PascalCase
    description: "What this entity represents"
    properties:
      - name: property_name     # Required
        type: string            # string | integer | number | boolean | date | enum
        required: true          # Optional, default false
        description: "What this property means"
        default: "value"        # Optional default value
        enum_values: [...]      # Required if type is enum
    extraction_hint: "Natural language hints for the LLM to identify this entity"

# Relationship types between entities
link_types:
  - name: link_type_name        # Required, snake_case
    source_types: [TypeA]       # Which entity types can be the source
    target_types: [TypeB]       # Which entity types can be the target
    description: "What this relationship means"

# Optional: customize the LLM extraction prompt
extraction:
  system_prompt: |
    Custom system prompt for the LLM extractor.
    Guides the LLM on what to look for and how to structure output.

# Optional: validation rules
validation_rules:
  - field: confidence
    rule: min_confidence
    params: { min_value: 0.6 }
    message: "Entity confidence must be >= 0.6"
```

## Creating a Custom Schema

### Step 1: Define Your Entity Types

Think about the key objects in your domain. Each entity type should have:
- A clear **name** (PascalCase)
- A human-readable **description**
- **Properties** that capture the important attributes
- An **extraction_hint** (in the language of your transcripts) to help the LLM identify these entities

```yaml
entity_types:
  - name: Patient
    description: "A patient in the healthcare system"
    properties:
      - { name: patient_id, type: string, required: true, description: "Medical record number" }
      - { name: age, type: integer, description: "Patient age" }
      - { name: diagnosis, type: string, description: "Primary diagnosis" }
      - { name: severity, type: enum, enum_values: ["critical","serious","stable","minor"], default: "stable" }
    extraction_hint: "患者、病人、病患等"
```

### Step 2: Define Link Types

Identify the relationships between your entities:

```yaml
link_types:
  - name: treated_by
    source_types: [Patient]
    target_types: [Doctor]
    description: "Patient is treated by a doctor"

  - name: prescribed
    source_types: [Doctor]
    target_types: [Medication]
    description: "Doctor prescribed medication"

  - name: diagnosed_with
    source_types: [Patient]
    target_types: [Condition]
    description: "Patient diagnosed with a condition"
```

### Step 3: Add Extraction Hints

Extraction hints are especially valuable for multilingual extraction. They tell the LLM what words/phrases in the source text correspond to each entity type:

```yaml
entity_types:
  - name: Doctor
    extraction_hint: "医生、大夫、主治、主任等"

  - name: Medication
    extraction_hint: "药物、药品、处方药、用药等"
```

### Step 4: Customize Extraction Prompt (Optional)

Override the default system prompt for domain-specific extraction:

```yaml
extraction:
  system_prompt: |
    You are a healthcare knowledge extraction system.
    Extract patients, doctors, medications, conditions, and their relationships.
    Pay special attention to dosages, treatment plans, and follow-up schedules.
    Output must be valid JSON.
```

### Step 5: Add Validation Rules (Optional)

```yaml
validation_rules:
  - field: confidence
    rule: min_confidence
    params: { min_value: 0.7 }
    message: "Healthcare entities require confidence >= 0.7"
```

## Complete Example: Healthcare Domain

```yaml
domain: healthcare
version: "1.0.0"
description: "Healthcare domain — patients, doctors, treatments"

entity_types:
  - name: Patient
    description: "A patient receiving medical care"
    properties:
      - { name: patient_id, type: string, required: true }
      - { name: age, type: integer }
      - { name: gender, type: enum, enum_values: ["male","female","other"] }
      - { name: blood_type, type: string }
    extraction_hint: "患者、病人、病患等"

  - name: Doctor
    description: "A medical professional"
    properties:
      - { name: specialty, type: string, required: true }
      - { name: department, type: string }
      - { name: title, type: enum, enum_values: ["attending","resident","fellow","intern"] }
    extraction_hint: "医生、大夫、主治、主任等"

  - name: Medication
    description: "A prescribed medication"
    properties:
      - { name: generic_name, type: string, required: true }
      - { name: dosage, type: string }
      - { name: frequency, type: string }
    extraction_hint: "药物、药品、处方等"

  - name: Condition
    description: "A medical condition or diagnosis"
    properties:
      - { name: icd_code, type: string }
      - { name: severity, type: enum, enum_values: ["critical","serious","moderate","mild"] }
    extraction_hint: "疾病、诊断、病症等"

  - name: Decision
    description: "A clinical decision"
    properties:
      - { name: detail, type: string }
      - { name: rationale, type: string }

  - name: ActionItem
    description: "A follow-up action"
    properties:
      - { name: priority, type: enum, enum_values: ["urgent","high","medium","low"] }
      - { name: due_date, type: date }

link_types:
  - { name: treated_by, source_types: [Patient], target_types: [Doctor] }
  - { name: prescribed, source_types: [Doctor], target_types: [Medication] }
  - { name: diagnosed_with, source_types: [Patient], target_types: [Condition] }
  - { name: treats, source_types: [Medication], target_types: [Condition] }
  - { name: refers_to, source_types: [Doctor], target_types: [Doctor] }
  - { name: makes, source_types: [Doctor], target_types: [Decision] }
  - { name: assigned_to, source_types: [ActionItem], target_types: [Doctor] }

extraction:
  system_prompt: |
    You are a healthcare knowledge extraction system.
    Extract patients, doctors, medications, conditions, and their relationships
    from clinical meeting transcripts and medical notes.
    Pay attention to treatment plans, medication changes, and follow-up actions.
    Output must be valid JSON.

validation_rules:
  - field: confidence
    rule: min_confidence
    params: { min_value: 0.7 }
    message: "Healthcare entities require confidence >= 0.7"
```

## Using Schemas

### CLI

```bash
# Use a built-in schema
ontology-engine ingest meeting.md --schema edtech

# Use a custom schema file
ontology-engine ingest meeting.md --schema ./my_schemas/healthcare.yaml

# List available schemas
ontology-engine schema list

# Show schema details
ontology-engine schema show edtech

# Validate a schema file
ontology-engine schema validate my_schema.yaml
```

### Python

```python
from ontology_engine.core.schema_registry import DomainSchema, SchemaRegistry

# Load a schema
schema = DomainSchema.from_yaml("domain_schemas/edtech.yaml")

# Use with pipeline
engine = await PipelineEngine.create(config, domain_schema=schema)
result = await engine.ingest("meeting.md")

# Use the registry for multi-domain support
registry = SchemaRegistry()
registry.register_from_yaml("domain_schemas/edtech.yaml")
registry.register_from_yaml("domain_schemas/finance.yaml")
registry.set_active("edtech")

active = registry.get_active()
print(active.entity_type_names())  # ['Student', 'Course', ...]
```

### Schema Introspection

```python
# Check what types a schema defines
schema = DomainSchema.from_yaml("edtech.yaml")

print(f"Domain: {schema.domain} v{schema.version}")
print(f"Entity types: {schema.entity_type_names()}")
print(f"Link types: {schema.link_type_names()}")

# Get details about a specific type
student_type = schema.get_entity_type("Student")
if student_type:
    for prop in student_type.properties:
        print(f"  {prop.name}: {prop.type} (required: {prop.required})")
```

## Property Types

| Type | Python Equivalent | Example |
|------|-------------------|---------|
| `string` | `str` | `"John Doe"` |
| `integer` | `int` | `42` |
| `number` | `float` | `3.14` |
| `boolean` | `bool` | `true` |
| `date` | `date` | `"2024-01-15"` |
| `enum` | `str` (constrained) | `"active"` (from enum_values list) |

## Best Practices

1. **Keep schemas focused** — one schema per domain, don't try to cover everything
2. **Use extraction hints** — especially for non-English text, they significantly improve extraction accuracy
3. **Include Decision and ActionItem** — these are useful in almost every domain
4. **Set realistic confidence thresholds** — 0.6 is a good default, raise to 0.7+ for critical domains
5. **Version your schemas** — use semantic versioning so you can evolve them over time
6. **Test with real data** — validate your schema against actual transcripts to tune entity types and hints
