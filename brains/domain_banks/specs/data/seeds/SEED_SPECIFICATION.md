# Domain Bank Seed Specification

**Version:** 1.0.0
**Date:** 2025-11-15
**Status:** Active

## Overview

This document defines the specification for domain bank seed files. Seeds provide foundational knowledge that the Maven system uses for deterministic reasoning, planning, coding, and decision-making.

## Purpose

Domain bank seeds serve to:

1. **Initialize** domain banks with canonical knowledge
2. **Ensure** deterministic behavior across system operations
3. **Provide** authoritative sources for specialist brains
4. **Maintain** consistency in system behavior and outputs

## Seed File Format

All seed files use **JSONL** (JSON Lines) format:
- One JSON object per line
- Each line is a complete, valid JSON object
- No commas between lines
- UTF-8 encoding

## Entry Schema

Each seed entry MUST contain the following required fields:

### Required Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `id` | string | Unique identifier within bank | Format: `{bank}:{kind}:{slug}` |
| `bank` | string | Target domain bank | Must be valid bank name |
| `kind` | string | Entry type/category | Must be valid kind |
| `content` | object | Knowledge payload | Must contain title & description |
| `tier` | string | Memory tier | Always "ltm" for seeds |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `confidence` | number | Confidence level (0.0-1.0) | 1.0 |
| `source` | string | Entry source | "seed" |
| `deterministic` | boolean | Is deterministic knowledge | true |

### Content Object

The `content` field MUST be an object with:

**Required:**
- `title` (string, 3-200 chars): Short name/title
- `description` (string, min 10 chars): Detailed explanation

**Optional:**
- `tags` (array of strings): Categorization tags
- `examples` (array of strings): Usage examples
- `related_ids` (array of strings): Related entry IDs
- `metadata` (object): Additional metadata

## Allowed Banks

Seeds can target these domain banks:

1. **science** - Scientific laws, concepts, definitions
2. **technology** - Technology concepts and systems
3. **language_arts** - Grammar, syntax, linguistic rules
4. **working_theories** - Causal patterns and theories
5. **personal** - System identity and self-knowledge
6. **governance_rules** - System governance and policies
7. **coding_patterns** - Code patterns and templates
8. **planning_patterns** - Task decomposition patterns
9. **creative_templates** - Creative generation templates
10. **environment_rules** - Environmental constraints
11. **conflict_resolution_patterns** - Conflict resolution strategies

## Allowed Kinds

Entry `kind` must be one of:

- **law** - Fundamental laws (e.g., physical laws)
- **concept** - Abstract concepts
- **definition** - Precise definitions
- **principle** - Guiding principles
- **rule** - Explicit rules
- **pattern** - Repeatable patterns
- **template** - Reusable templates
- **constraint** - Hard constraints
- **theory** - Theoretical frameworks
- **fact** - Factual statements
- **guideline** - Best practice guidelines
- **strategy** - Strategic approaches
- **heuristic** - Decision-making heuristics

## ID Format

Entry IDs MUST follow this format:

```
{bank}:{kind}:{slug}
```

**Rules:**
- Lowercase only
- Underscores allowed
- No spaces or special characters
- Must be unique within the bank
- Slug should be descriptive and stable

**Examples:**
```
science:law:thermodynamics_2
coding_patterns:error_handling:try_except_basic
planning_patterns:strategy:divide_and_conquer
governance_rules:rule:no_randomness
```

## Validation Rules

### Schema Validation

All entries MUST:
1. Match the JSON schema in `seed_schema.json`
2. Have all required fields present
3. Have valid types for all fields
4. Have valid enum values for bank and kind

### Uniqueness Validation

Within each bank:
1. Entry IDs MUST be unique
2. No duplicate IDs across seed files for same bank

### Consistency Validation

1. Entry `id` prefix MUST match `bank` field
2. Entry `id` kind component SHOULD match `kind` field
3. Entry `tier` MUST be "ltm"
4. Entry `source` SHOULD be "seed"

### Content Validation

1. `content.title` must be 3-200 characters
2. `content.description` must be at least 10 characters
3. Tags must be lowercase alphanumeric + underscores
4. Related IDs must follow ID format rules

## Example Entries

### Science Entry

```json
{
  "id": "science:law:thermodynamics_2",
  "bank": "science",
  "kind": "law",
  "content": {
    "title": "Second Law of Thermodynamics",
    "description": "The total entropy of an isolated system can never decrease over time. Entropy can only increase or remain constant in ideal reversible processes.",
    "tags": ["thermodynamics", "entropy", "physics"],
    "related_ids": ["science:law:thermodynamics_1"]
  },
  "tier": "ltm",
  "confidence": 1.0,
  "source": "seed",
  "deterministic": true
}
```

### Coding Pattern Entry

```json
{
  "id": "coding_patterns:pattern:service_api_structure",
  "bank": "coding_patterns",
  "kind": "pattern",
  "content": {
    "title": "Service API Structure Pattern",
    "description": "Standard structure for brain service APIs. Accept msg dict with 'op' and 'payload', return dict with 'ok' and 'payload' or 'error'.",
    "examples": [
      "def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:\n    op = msg.get('op')\n    payload = msg.get('payload', {})\n    if op == 'OPERATION':\n        return {'ok': True, 'payload': result}\n    return {'ok': False, 'error': 'unknown_op'}"
    ],
    "tags": ["api", "structure", "deterministic", "brain_service"]
  },
  "tier": "ltm",
  "confidence": 1.0,
  "source": "seed",
  "deterministic": true
}
```

### Governance Rule Entry

```json
{
  "id": "governance_rules:rule:no_randomness",
  "bank": "governance_rules",
  "kind": "rule",
  "content": {
    "title": "No Randomness Rule",
    "description": "All system operations must be deterministic. No use of random number generation, no time-based variation in logic paths. Same inputs must always produce same outputs.",
    "tags": ["determinism", "governance", "mandatory"],
    "related_ids": ["governance_rules:rule:python_only"]
  },
  "tier": "ltm",
  "confidence": 1.0,
  "source": "seed",
  "deterministic": true
}
```

## File Organization

Seeds are organized as follows:

```
brains/domain_banks/specs/data/seeds/
├── seed_registry.json              # Central registry
├── seed_schema.json                # JSON schema
├── SEED_SPECIFICATION.md           # This document
├── science_seeds.jsonl
├── technology_seeds.jsonl
├── language_seeds.jsonl
├── working_theories_seeds.jsonl
├── personal_system_seeds.jsonl
├── governance_rules_seeds.jsonl
├── coding_patterns_seeds.jsonl
├── planning_patterns_seeds.jsonl
├── creative_templates_seeds.jsonl
├── environment_rules_seeds.jsonl
└── conflict_resolution_patterns_seeds.jsonl
```

## Seeding Process

1. **Load** all seed files
2. **Validate** schema compliance
3. **Validate** uniqueness constraints
4. **Validate** consistency rules
5. **Apply** to runtime domain banks (if not validate-only mode)
6. **Report** results (counts, errors, warnings)

## Determinism Guarantees

Seeds must support deterministic system behavior:

1. **No timestamps** (except for metadata/tracking)
2. **No random values**
3. **Stable IDs** that don't change between runs
4. **Idempotent application** - same seeds produce same output
5. **Order-independent** within a bank

## Error Handling

Validation failures MUST hard-fail with clear error messages:

- **Schema violations**: Report field, expected type, actual value
- **Duplicate IDs**: Report bank, ID, file locations
- **Invalid references**: Report ID, invalid related_id
- **Malformed JSONL**: Report file, line number, parse error

## Versioning

- Seed schema version: Semantic versioning (MAJOR.MINOR.PATCH)
- Breaking changes: Increment MAJOR
- New optional fields: Increment MINOR
- Bug fixes/clarifications: Increment PATCH

Current version: **1.0.0**

---

**Document Status:** Complete
**Next Review:** Phase 7 completion
