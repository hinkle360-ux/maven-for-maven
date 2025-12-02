# Teacher Integration Patterns

## Overview

Maven provides **three valid patterns** for cognitive brains to learn from the Teacher (LLM). All patterns achieve the same goal: check memory first, call Teacher if needed, validate results, and store learned patterns.

Choose the pattern that best fits your brain's architecture and coding style.

---

## Pattern 1: TeacherHelper Class (Recommended for Most Brains)

**Status**: ✅ Proven and used by 30/30 cognitive brains
**Best for**: Standard cognitive brains that need internal pattern learning
**Complexity**: Low - highest level of abstraction

### How It Works

The `TeacherHelper` class encapsulates the entire learning loop:
1. Check brain's own memory for existing patterns
2. Call Teacher only if no high-confidence pattern found
3. Validate results with TruthClassifier
4. Store patterns to brain's own memory (if `store_internal=True`)
5. Store facts to domain brains (if `store_domain=True`)

### Example Implementation

```python
# In your brain's service file (e.g., planner_brain.py)

from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
from memory_system.brain_memory import BrainMemory

# Initialize at module level
_teacher_helper = TeacherHelper("planner")
_memory = BrainMemory("planner")

def plan_task(task_description: str, parameters: dict) -> dict:
    """Plan a task using learned planning patterns."""

    # Try existing patterns first
    existing_plans = _memory.retrieve(
        query=f"planning pattern: {task_description[:50]}",
        limit=3
    )

    for plan in existing_plans:
        if plan.get("confidence", 0) >= 0.7:
            return {"plan": plan.get("content"), "source": "memory"}

    # No good pattern found - ask Teacher
    teacher_result = _teacher_helper.maybe_call_teacher(
        question=f"How should I plan this task: {task_description}?",
        context={
            "task": task_description,
            "parameters": parameters,
            "complexity": "high" if len(task_description) > 100 else "low"
        },
        check_memory_first=True  # Double-check memory inside helper
    )

    if teacher_result and teacher_result.get("answer"):
        patterns_stored = teacher_result.get("patterns_stored", 0)
        print(f"[PLANNER] Learned {patterns_stored} new planning patterns")
        return {
            "plan": teacher_result["answer"],
            "source": "teacher",
            "patterns_learned": patterns_stored
        }

    # Fallback if Teacher fails
    return {"plan": "default_plan", "source": "fallback"}
```

### Configuration

Each brain needs a contract in `brains/teacher_contracts.py`:

```python
"planner": {
    "operation": "TEACH",
    "prompt_mode": "planning_patterns",
    "store_internal": True,   # Store patterns to own memory
    "store_domain": False,    # Don't store to domain brains
    "enabled": True,
    "description": "Learns task decomposition patterns"
}
```

### Pros & Cons

**Pros:**
- ✅ Highest level abstraction - just call one method
- ✅ Automatic memory checking and storage
- ✅ TruthClassifier integration built-in
- ✅ Domain brain routing handled automatically
- ✅ Proven across 30 brains

**Cons:**
- ❌ Less control over the exact storage logic
- ❌ Requires contract configuration
- ❌ Slightly more opaque (logic inside helper class)

---

## Pattern 2: teach_for_brain() Direct Function (New)

**Status**: ✅ Added in this PR
**Best for**: Brains that want full control over memory and storage
**Complexity**: Medium - mid-level abstraction

### How It Works

Call `teach_for_brain()` directly and handle storage yourself:
1. You check memory (your own logic)
2. Call `teach_for_brain(brain_name, situation)`
3. You validate the results (your own logic)
4. You store patterns/facts (your own logic)

### Example Implementation

```python
# In your brain's service file (e.g., autonomy_brain.py)

from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from memory_system.brain_memory import BrainMemory
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

_memory = BrainMemory("autonomy")

def prioritize_tasks(tasks: list) -> list:
    """Prioritize tasks using learned autonomy strategies."""

    # 1. Check own memory first
    task_signature = "-".join([t.get("type", "unknown") for t in tasks[:3]])

    existing = _memory.retrieve(
        query={"kind": "teacher_pattern", "task_signature": task_signature},
        limit=1
    )

    if existing and existing[0].get("confidence", 0) >= 0.7:
        print("[AUTONOMY] Using learned prioritization pattern")
        return existing[0].get("content", [])

    # 2. No pattern found - call Teacher directly
    print("[AUTONOMY] No pattern found, calling Teacher...")

    result = teach_for_brain(
        brain_name="autonomy",
        situation={
            "question": f"How should I prioritize these {len(tasks)} tasks?",
            "tasks": [t.get("description", "") for t in tasks[:5]],
            "task_types": [t.get("type", "unknown") for t in tasks],
            "urgency": "high" if any(t.get("urgent") for t in tasks) else "normal"
        }
    )

    # 3. Validate result
    if result["verdict"] == "ERROR":
        print(f"[AUTONOMY] Teacher error: {result.get('error')}")
        return []

    if result["verdict"] == "NO_ANSWER":
        print("[AUTONOMY] Teacher had no answer")
        return []

    # 4. Validate patterns with TruthClassifier
    patterns = result.get("patterns", [])
    validated_patterns = []

    for pattern in patterns:
        classification = TruthClassifier.classify(
            content=pattern.get("pattern", ""),
            confidence=result.get("confidence", 0.7),
            evidence=None
        )

        if classification["type"] != "RANDOM" and classification["allow_memory_write"]:
            validated_patterns.append(pattern)

    # 5. Store validated patterns to own memory
    if validated_patterns:
        _memory.store(
            content=validated_patterns,
            metadata={
                "kind": "teacher_pattern",
                "task_signature": task_signature,
                "source": "llm_teacher",
                "confidence": result.get("confidence", 0.7),
                "pattern_count": len(validated_patterns)
            }
        )
        print(f"[AUTONOMY] Stored {len(validated_patterns)} validated patterns")

    return validated_patterns
```

### Pros & Cons

**Pros:**
- ✅ Full control over memory checking logic
- ✅ Full control over validation logic
- ✅ Full control over storage logic
- ✅ No contract configuration needed (TEACHER_MODES already defined)
- ✅ Simpler to understand (direct function call)

**Cons:**
- ❌ More code to write per brain
- ❌ You must implement validation yourself
- ❌ You must implement storage yourself
- ❌ More room for inconsistencies between brains

---

## Pattern 3: _maybe_learn_from_teacher() Method (Hybrid)

**Status**: ✅ Can be added to any brain
**Best for**: Brains that want encapsulation + control
**Complexity**: Medium-High - combines both approaches

### How It Works

Create a method in your brain that encapsulates the learning loop using `teach_for_brain()`:
1. Method checks brain's own memory
2. Calls `teach_for_brain()` if needed
3. Validates and stores results
4. Returns learned patterns

### Example Implementation

```python
# In your brain's service file (e.g., belief_tracker_brain.py)

from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from memory_system.brain_memory import BrainMemory
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

class BeliefTracker:
    def __init__(self):
        self.name = "belief_tracker"
        self.memory = BrainMemory(self.name)

    def _build_situation_key(self, context: dict) -> str:
        """Build a unique key for this type of situation."""
        belief_type = context.get("belief_type", "unknown")
        update_type = context.get("update_type", "unknown")
        return f"{belief_type}:{update_type}"

    def _is_valid_teacher_result(self, result: dict) -> bool:
        """Validate Teacher result with governance checks."""
        if result.get("verdict") == "ERROR":
            return False

        if result.get("confidence", 0.0) < 0.5:
            return False

        patterns = result.get("patterns", [])
        if not patterns:
            return False

        # Validate with TruthClassifier
        for pattern in patterns:
            classification = TruthClassifier.classify(
                content=pattern.get("pattern", ""),
                confidence=result.get("confidence", 0.7),
                evidence=None
            )

            if classification["type"] == "RANDOM" or not classification["allow_memory_write"]:
                return False

        return True

    def _maybe_learn_from_teacher(self, context: dict) -> dict | None:
        """
        Use Teacher as a helper: only when we don't already have a pattern
        for this kind of situation. Store any learned patterns in our own
        BrainMemory for next time.

        Args:
            context: Dict with belief update context

        Returns:
            Learned patterns dict or None
        """
        # 0. Decide what key represents the "situation type"
        situation_key = self._build_situation_key(context)

        # 1. Check own memory for an existing pattern
        existing = self.memory.retrieve(
            query={"kind": "teacher_pattern", "situation_key": situation_key},
            limit=1
        )

        if existing and existing[0].get("confidence", 0) >= 0.7:
            print(f"[{self.name.upper()}] Using existing pattern from memory")
            return existing[0].get("content")

        # 2. Call Teacher via teach_for_brain
        print(f"[{self.name.upper()}] No pattern found, calling Teacher...")

        teacher_result = teach_for_brain(
            brain_name=self.name,
            situation={
                "question": f"How should I handle belief update: {context.get('belief')}?",
                "belief_type": context.get("belief_type"),
                "update_type": context.get("update_type"),
                "evidence": context.get("evidence", ""),
                "current_confidence": context.get("confidence", 0.5)
            }
        )

        # 3. Truth / governance checks
        if not self._is_valid_teacher_result(teacher_result):
            print(f"[{self.name.upper()}] Teacher result failed validation")
            return None

        # 4. Store pattern in this brain's own memory
        patterns = teacher_result.get("patterns", [])

        self.memory.store(
            content=patterns,
            metadata={
                "kind": "teacher_pattern",
                "situation_key": situation_key,
                "source": "teacher",
                "confidence": teacher_result.get("confidence", 0.7),
                "pattern_count": len(patterns)
            }
        )

        print(f"[{self.name.upper()}] Stored {len(patterns)} new belief patterns")

        return patterns

    def update_belief(self, belief: str, evidence: dict) -> dict:
        """Update a belief using learned patterns."""

        # Use the learning method
        patterns = self._maybe_learn_from_teacher({
            "belief": belief,
            "belief_type": evidence.get("type", "factual"),
            "update_type": "strengthen" if evidence.get("supports") else "weaken",
            "evidence": str(evidence),
            "confidence": evidence.get("confidence", 0.5)
        })

        if patterns:
            return {"updated": True, "patterns_used": len(patterns)}
        else:
            return {"updated": False, "fallback": True}


# Module-level instance
_tracker = BeliefTracker()

def service_api(msg: dict) -> dict:
    """Service API using the belief tracker."""
    op = msg.get("op", "").upper()
    payload = msg.get("payload", {})

    if op == "UPDATE_BELIEF":
        result = _tracker.update_belief(
            belief=payload.get("belief", ""),
            evidence=payload.get("evidence", {})
        )
        return {"ok": True, "payload": result}

    return {"ok": False, "error": {"code": "UNSUPPORTED_OP", "message": op}}
```

### Pros & Cons

**Pros:**
- ✅ Encapsulated in a clean method
- ✅ Full control over validation and storage
- ✅ Easy to test and debug
- ✅ Consistent pattern across brains that use it
- ✅ Self-documenting code

**Cons:**
- ❌ More code per brain
- ❌ Requires implementing several helper methods
- ❌ Can lead to code duplication across brains

---

## Pattern Comparison

| Feature | TeacherHelper | teach_for_brain() | _maybe_learn_from_teacher() |
|---------|---------------|-------------------|----------------------------|
| **Abstraction Level** | Highest | Medium | Medium-High |
| **Code to Write** | Minimal | Medium | Most |
| **Control** | Least | Most | Full |
| **Configuration** | Contract required | TEACHER_MODES only | TEACHER_MODES only |
| **Validation** | Automatic | Manual | Manual |
| **Storage** | Automatic | Manual | Manual |
| **Current Usage** | 30/30 brains | 0 brains (new) | 0 brains (pattern) |
| **Best For** | Standard brains | Control freaks | OOP enthusiasts |

---

## Which Pattern Should You Use?

### Use TeacherHelper if:
- ✅ You want the simplest integration
- ✅ You're okay with automatic storage to memory
- ✅ You want TruthClassifier validation built-in
- ✅ Your brain follows the standard pattern (internal skills)

### Use teach_for_brain() if:
- ✅ You need full control over memory checking
- ✅ You need custom validation logic
- ✅ You want to avoid contract configuration
- ✅ You prefer functional style over OOP

### Use _maybe_learn_from_teacher() if:
- ✅ You want encapsulation within your brain class
- ✅ You need custom logic but want it organized
- ✅ You prefer method-based APIs
- ✅ You want to follow the exact pattern from the spec

---

## Memory Architecture (All Patterns)

Regardless of which pattern you use, all brains follow the same memory rules:

### Cognitive Brains (Internal Skills)
```python
# Store patterns to OWN memory only
metadata = {
    "kind": "teacher_pattern",      # or "learned_pattern"
    "source": "llm_teacher",         # or "teacher"
    "situation_key": "...",          # Unique situation identifier
    "confidence": 0.7,               # Teacher confidence
}

# Stored in: maven2_fix/memory/brains/{brain_name}/...
```

### Reasoning Brain (World Facts)
```python
# Store facts to DOMAIN brains, not own memory
# Facts go through TruthClassifier first
# Routed to: science, history, personal, etc.

metadata = {
    "kind": "learned_fact",
    "source": "llm_teacher_via_reasoning",
    "truth_type": "CERTAIN",
    "fact_type": "world_fact",
}

# Stored in: maven2_fix/memory/brains/{domain_brain_name}/...
```

### Memory Librarian (Routing Rules)
```python
# Store routing rules to OWN memory
metadata = {
    "kind": "routing_rule",
    "question": "...",
    "banks": [{"bank": "science", "weight": 0.9}],
    "aliases": ["..."],
}

# Stored in: maven2_fix/memory/brains/memory_librarian/...
```

---

## TEACHER_MODES Reference

All brains are pre-configured in `teacher_brain.py:TEACHER_MODES`:

```python
TEACHER_MODES = {
    # Core
    "reasoning": "world_question",
    "memory_librarian": "routing_learning",
    "teacher": None,

    # Cognitive brains
    "planner": "planning_patterns",
    "autonomy": "autonomy_strategies",
    "attention": "attention_alloc_rules",
    "learning": "learning_strategies",
    "belief_tracker": "belief_update_patterns",
    "context_management": "context_decay_strategies",
    # ... (30 total)
}
```

Each mode corresponds to a specific prompt template in `_build_mode_specific_prompt()`.

---

## Best Practices (All Patterns)

### 1. Check Memory First
Always check your brain's own memory before calling Teacher. This:
- Avoids redundant LLM calls
- Saves tokens/cost
- Respects budget governance
- Uses learned patterns

### 2. Validate Results
Always validate Teacher results before storing:
- Check `verdict` is not "ERROR"
- Check `confidence` is above threshold (usually 0.5-0.7)
- Use TruthClassifier for governance
- Filter out "RANDOM" classifications

### 3. Store with Good Metadata
Store patterns with rich metadata for future retrieval:
- `kind`: "teacher_pattern" or "learned_pattern"
- `source`: "llm_teacher" or "teacher"
- `situation_key`: Unique identifier for pattern type
- `confidence`: Teacher's confidence score
- Other context-specific keys

### 4. Log Learning Events
Log when you learn from Teacher:
```python
print(f"[{BRAIN_NAME}] Learned {count} new patterns from Teacher")
print(f"[{BRAIN_NAME}] Using learned pattern from memory")
```

### 5. Respect Budget Limits
Don't spam Teacher:
- Check memory thoroughly first
- Limit to 1 Teacher call per request
- Use confidence thresholds to avoid re-learning

---

## Example: Converting Between Patterns

### TeacherHelper → teach_for_brain()

**Before (TeacherHelper):**
```python
result = _teacher_helper.maybe_call_teacher(
    question="How should I plan this?",
    context={"task": task},
    check_memory_first=True
)
```

**After (teach_for_brain):**
```python
# You handle memory checking
existing = _memory.retrieve(query=..., limit=1)
if not existing:
    result = teach_for_brain(
        brain_name="planner",
        situation={"question": "How should I plan this?", "task": task}
    )
    # You handle storage
    _memory.store(content=result["patterns"], metadata={...})
```

### teach_for_brain() → _maybe_learn_from_teacher()

**Before (direct call):**
```python
result = teach_for_brain("planner", {"question": "..."})
# Manual validation
# Manual storage
```

**After (method wrapper):**
```python
def _maybe_learn_from_teacher(self, context):
    # Check memory
    # Call teach_for_brain
    result = teach_for_brain(self.name, context)
    # Validate
    # Store
    return patterns

# Usage
patterns = self._maybe_learn_from_teacher(context)
```

---

## Testing Recommendations

### Unit Tests
Test each pattern independently:

```python
def test_teacher_helper_pattern():
    """Test TeacherHelper integration."""
    helper = TeacherHelper("test_brain")
    result = helper.maybe_call_teacher(
        question="test",
        context={},
        check_memory_first=False  # Skip memory for test
    )
    assert "verdict" in result

def test_teach_for_brain_pattern():
    """Test direct teach_for_brain call."""
    result = teach_for_brain("planner", "test question")
    assert result["verdict"] in ["LEARNED", "NO_ANSWER", "ERROR"]

def test_maybe_learn_pattern():
    """Test _maybe_learn_from_teacher method."""
    brain = MyBrain()
    patterns = brain._maybe_learn_from_teacher({"test": "context"})
    assert patterns is None or isinstance(patterns, (list, dict))
```

### Integration Tests
Test the full learning loop:

```python
def test_learning_loop():
    """Test complete learning loop."""
    brain = MyBrain()

    # First call - should learn from Teacher
    result1 = brain.process_task("new task type")
    assert result1["source"] == "teacher"

    # Second call - should use memory
    result2 = brain.process_task("new task type")
    assert result2["source"] == "memory"
```

---

## Troubleshooting

### "No teaching mode defined for brain"
- Add your brain to `TEACHER_MODES` in `teacher_brain.py`
- Or create a contract in `teacher_contracts.py` (for TeacherHelper)

### "LLM service not available"
- Check that `brains/tools/llm_service.py` is configured
- This is expected in test environments without LLM setup

### Patterns not being stored
- Check that validation is passing (confidence >= threshold)
- Check that TruthClassifier is not marking as "RANDOM"
- Verify `allow_memory_write` is True

### Patterns not being retrieved
- Check that `situation_key` or query matches stored metadata
- Check memory tiers - try `tiers=["stm", "mtm", "ltm"]`
- Check confidence threshold in retrieval

---

## Summary

All three patterns are valid and supported:

1. **TeacherHelper** - Easiest, proven, recommended for most brains
2. **teach_for_brain()** - Direct function, full control, new option
3. **_maybe_learn_from_teacher()** - Method wrapper, encapsulated, hybrid approach

Choose based on your brain's needs and your coding style preferences. All achieve the same learning loop goal.
