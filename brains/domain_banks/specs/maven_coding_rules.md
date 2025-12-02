# Maven Coding Rules

## Core Principles

### 1. Deterministic Over Stochastic
**Rule**: Always prefer rule-based, deterministic logic over LLM generation when possible.

**Example**:
```python
# CORRECT
if "what do i like" in text.lower():
    preferences = get_all_preferences(user_id)
    return build_preference_summary(preferences)

# WRONG
if "what do i like" in text.lower():
    prompt = f"Summarize user preferences: {preferences}"
    return llm.generate(prompt)
```

### 2. Memory as Source of Truth
**Rule**: Always retrieve from memory before generating new content.

**Example**:
```python
# CORRECT
name = episodic_last_declared_identity(ctx["recent_queries"])
if not name:
    name = wm_get("user_identity")
if name:
    return f"You are {name}"

# WRONG
return llm.generate("Tell me the user's name")
```

### 3. Intent-Driven Routing
**Rule**: Parse intent first, then route to appropriate handler.

**Example**:
```python
# CORRECT
intent = stage3_language.get("intent")
if intent == "preference_query":
    return handle_preference_query(ctx)
elif intent == "identity_query":
    return handle_identity_query(ctx)

# WRONG
text = ctx["text"]
response = llm.generate(f"Answer this: {text}")
```

### 4. Verdict Before Storage
**Rule**: Never store content without a reasoning verdict.

**Example**:
```python
# CORRECT
verdict = reasoning.evaluate_fact(proposed_fact, evidence)
if verdict["verdict"] == "TRUE":
    store_to_bank("factual", fact, confidence=verdict["confidence"])

# WRONG
store_to_bank("factual", user_input)  # No validation!
```

## Error Handling

### Fail Gracefully
**Rule**: Catch exceptions at operation boundaries, never let errors propagate to user.

```python
# CORRECT
try:
    preferences = get_all_preferences(user_id)
    if preferences:
        return summarize_preferences(preferences)
    else:
        return "You haven't told me your preferences yet."
except Exception as e:
    log_error("preference_retrieval_failed", e)
    return "I'm having trouble accessing your preferences right now."

# WRONG
preferences = get_all_preferences(user_id)  # May crash
return summarize_preferences(preferences)   # May crash on None
```

### Log, Don't Crash
**Rule**: Use try-except blocks around all external operations.

```python
# CORRECT
try:
    result = reasoning_brain.service_api({"op": "EVALUATE_FACT", "payload": data})
    verdict = result.get("payload", {}).get("verdict", "UNKNOWN")
except Exception as e:
    verdict = "UNKNOWN"
    log_error("reasoning_call_failed", e)

# WRONG
result = reasoning_brain.service_api({"op": "EVALUATE_FACT", "payload": data})
verdict = result["payload"]["verdict"]  # May crash if missing keys
```

## Stage Discipline

### Follow the Pipeline
**Rule**: Never skip stages or call out of order.

**Pipeline Order**:
1. Stage 1: Sensorium (normalization)
2. Stage 2: Planner (for commands)
3. Stage 3: Language (intent parsing)
4. Stage 2R: Memory retrieval
5. Stage 4: Pattern recognition
6. Stage 5: Affect
7. Stage 6: Candidate generation
8. Stage 8: Reasoning/validation
9. Stage 8b: Governance
10. Stage 9: Storage
11. Stage 10: Finalization

**Example**:
```python
# CORRECT
ctx["stage_3_language"] = language.parse(text)
ctx["stage_2R_memory"] = retrieve_evidence(ctx)
ctx["stage_8_validation"] = reasoning.evaluate(ctx)
if ctx["stage_8_validation"]["verdict"] == "TRUE":
    store_to_memory(ctx)

# WRONG
store_to_memory(text)  # Skipped parsing, retrieval, validation!
```

### Context Flows Forward
**Rule**: Each stage reads from `ctx`, writes to `ctx["stage_N_..."]`.

```python
# CORRECT
def stage6_generate(ctx: Dict) -> Dict:
    stage3 = ctx.get("stage_3_language", {})
    intent = stage3.get("intent")
    # Generate candidate
    ctx["stage_6_candidates"] = {"candidates": [candidate]}
    return ctx

# WRONG
def stage6_generate(text: str) -> str:
    return generate_answer(text)  # Loses context!
```

## Naming Conventions

### Functions
- **Private helpers**: `_helper_function()`
- **Public API**: `service_api(msg)`
- **Stage handlers**: `stage6_generate(ctx)`

### Variables
- **Context dictionary**: `ctx`
- **User input**: `text` or `query`
- **Confidence**: `conf` (float 0.0-1.0)
- **Verdict**: `verdict` (string: TRUE/FALSE/THEORY/etc.)

### Constants
- **ALL_CAPS**: `GREETINGS = {"hi", "hello", "hey"}`
- **Patterns**: `identity_triggers = ["who are you", ...]`

## Memory Storage Rules

### JSONL Format
**Rule**: All memory banks use line-delimited JSON.

```python
# CORRECT
record = {
    "content": "User likes cats",
    "confidence": 0.9,
    "tags": ["preference", "animals"],
    "user_id": "josh",
    "timestamp": int(time.time())
}
with open(bank_path, "a") as f:
    f.write(json.dumps(record) + "\n")

# WRONG
records = [rec1, rec2, rec3]
with open(bank_path, "w") as f:
    json.dump(records, f)  # Not JSONL, loses atomicity
```

### Tagging Discipline
**Rule**: Always tag records with searchable metadata.

```python
# CORRECT
tags = ["preference", domain, "user_declared"]

# WRONG
tags = []  # No way to find this later!
```

### Duplicate Prevention
**Rule**: Check for exact matches before storing.

```python
# CORRECT
existing = search_bank(proposed_content)
if not existing or not exact_match(existing, proposed_content):
    store_to_bank(proposed_content)

# WRONG
store_to_bank(proposed_content)  # May create duplicates
```

## Intent Detection Patterns

### Pattern Matching Style
**Rule**: Use lowercase normalization and substring matching.

```python
# CORRECT
lower_text = text.lower().strip().rstrip("?!.,")
if "what do i like" in lower_text:
    return handle_preference_query()

# WRONG
if text == "What do I like?":  # Too strict, misses variations
    return handle_preference_query()
```

### Regex for Precise Matching
**Rule**: Use regex for structural patterns, not semantic matching.

```python
# CORRECT - structural pattern
if re.search(r"\bwho\s+am\s+i\b", lower_text):
    return handle_identity_query()

# WRONG - semantic matching better done with substring
if re.search(r".*prefer.*", lower_text):  # Too broad
    return handle_preference_query()
```

## Verdict Assignment Rules

### Match Intent to Verdict
**Rule**: Different intents require different verdicts.

| Intent | Verdict | Reason |
|--------|---------|--------|
| `preference_query` | `PREFERENCE` or `SKIP_STORAGE` | User query, not a fact |
| `identity_query` | `SKIP_STORAGE` | Meta-query, don't store |
| `relationship_query` | `TRUE` or `SKIP_STORAGE` | Depends on if updating fact |
| `math_compute` | `TRUE` | Deterministic, validated |
| Regular question | `TRUE/FALSE/UNKNOWN` | Based on evidence |
| Regular statement | `TRUE/THEORY` | Based on confidence |

### Set Mode with Verdict
**Rule**: Mode field explains why verdict was assigned.

```python
# CORRECT
ctx["stage_8_validation"] = {
    "verdict": "SKIP_STORAGE",
    "mode": "PREFERENCE_QUERY",
    "confidence": 0.9,
    # ...
}

# WRONG
ctx["stage_8_validation"] = {
    "verdict": "SKIP_STORAGE"  # Missing mode context
}
```

## LLM Interaction Rules

### Gate LLM Access
**Rule**: Always check gates before calling LLM.

```python
# CORRECT
if not _passed_memory(ctx):
    return {"error": "memory_gate_failed"}
if not _governance_permit_generate(ctx):
    return {"error": "governance_blocked"}
# Now safe to call LLM
response = llm.generate(prompt)

# WRONG
response = llm.generate(prompt)  # Ungated!
```

### Confidence Calibration
**Rule**: LLM answers get lower confidence than deterministic answers.

```python
# CORRECT
if deterministic_answer:
    confidence = 1.0
elif memory_based_answer:
    confidence = 0.9
elif llm_answer:
    confidence = 0.7  # Lower for generated content

# WRONG
confidence = 0.95  # Always high, regardless of source
```

## Testing and Validation

### Unit Tests for Each Intent
**Rule**: Every new intent must have test cases.

```python
# Example test cases
test_cases = [
    ("what do i like", "preference_query"),
    ("what animals do i like", "preference_query", {"domain": "animals"}),
    ("tell me who you are", "identity_query"),
    ("who am i", "identity_query"),
]
```

### Regression Tests
**Rule**: Behavioral contracts must pass before merging.

```json
{
  "identity_suite": [
    {"input": "who are you", "expected_pattern": "I'm Maven"},
    {"input": "describe yourself", "expected_pattern": "living system"}
  ],
  "preference_suite": [
    {"input": "what do i like", "expected_pattern": "Based on what you've told me"}
  ]
}
```

## Performance Rules

### Lazy Loading
**Rule**: Don't load resources until needed.

```python
# CORRECT
def _brain_module(name):
    # Load on first use
    if name not in _brain_cache:
        _brain_cache[name] = import_module(f"brains.{name}")
    return _brain_cache[name]

# WRONG
# Load all brains at startup
language = import_module("brains.language")
reasoning = import_module("brains.reasoning")
# ...
```

### Parallel Retrieval
**Rule**: Fan out memory searches when possible.

```python
# CORRECT
results = parallel_search(banks=["factual", "personal", "science"])

# WRONG
results = []
for bank in ["factual", "personal", "science"]:
    results.extend(search_bank(bank))  # Sequential
```

## Security and Privacy

### User Isolation
**Rule**: Always filter by `user_id` when retrieving memories.

```python
# CORRECT
preferences = get_all_preferences(user_id=ctx["user_id"])

# WRONG
preferences = get_all_preferences()  # Leaks across users!
```

### Governance Compliance
**Rule**: All storage must pass governance checks.

```python
# CORRECT
gov_result = governance.enforce(action="STORE", content=text)
if gov_result["allowed"]:
    store_to_bank(text)

# WRONG
store_to_bank(text)  # Bypasses governance!
```

## Code Organization

### Keep Functions Small
**Rule**: Functions should do one thing, < 50 lines.

```python
# CORRECT
def handle_preference_query(ctx):
    user_id = ctx["user_id"]
    domain = ctx["stage_3_language"].get("preference_domain")
    preferences = list_preferences(user_id, domain)
    return build_preference_answer(preferences, domain)

# WRONG
def handle_preference_query(ctx):
    # 200 lines of inline logic mixing retrieval, formatting, validation...
```

### Extract Shared Logic
**Rule**: If code appears twice, extract to helper.

```python
# CORRECT
def _build_self_description(identity_hit, creation_hit):
    # Shared logic for all identity intents
    if identity_hit and not creation_hit:
        return IDENTITY_TEXT
    # ...

# WRONG
# Copy-paste same identity text in 5 different places
```

## Documentation

### Docstrings for Public Functions
**Rule**: Every `service_api` and public helper needs a docstring.

```python
# CORRECT
def list_preferences(user_id: str, domain: str | None = None) -> list:
    """
    List stored preferences for a user, optionally filtered by domain.

    Args:
        user_id: The user identifier
        domain: Optional domain/category filter

    Returns:
        List of preference records matching criteria
    """
    # ...

# WRONG
def list_preferences(user_id, domain=None):
    # No docstring, unclear what it does
```

### Inline Comments for Complex Logic
**Rule**: Explain *why*, not *what*.

```python
# CORRECT
# Priority reordering: Maven-specific facts should appear first when
# the query is about system identity. This prevents confusion when
# personal facts about someone named "Maven" might otherwise rank higher.
if any(p in query for p in identity_triggers):
    results = maven_hits + other_hits

# WRONG
# Sort results
results = maven_hits + other_hits  # Doesn't explain why
```

## Prohibited Patterns

### ❌ Never Use These

1. **Globals for state**: Use `ctx` dictionary
2. **Hardcoded paths**: Use `Path(__file__).parent` or config
3. **Uncaught exceptions**: Always wrap in try-except
4. **LLM for facts**: Use memory retrieval first
5. **Storage without validation**: Always get verdict first
6. **Cross-user leakage**: Always filter by user_id
7. **Blocking governance**: All actions must pass policy checks

## Extension Checklist

When adding new functionality:

- [ ] Intent pattern added to language_brain.py
- [ ] Stage 6 handler added to memory_librarian.py
- [ ] Appropriate verdict set in stage_8_validation
- [ ] Storage skip configured if needed (Stage 9)
- [ ] Test cases added to behavior contracts
- [ ] Documentation updated in maven_design.md
- [ ] Code follows all rules in this document
