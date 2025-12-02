# Teacher Integration Examples

This directory contains comprehensive examples showing how to integrate Teacher (LLM learning) into Maven cognitive brains.

## Overview

Maven provides **three patterns** for Teacher integration. These examples demonstrate all three patterns with concrete, runnable code.

See [`../TEACHER_INTEGRATION_PATTERNS.md`](../TEACHER_INTEGRATION_PATTERNS.md) for detailed documentation.

---

## Files in This Directory

### Pattern Implementations

1. **`simple_teach_for_brain_usage.py`** - Pattern 2 (Direct Function)
   - Simplest approach: call `teach_for_brain()` directly
   - 7 complete examples from minimal to advanced
   - Shows validation, storage, error handling
   - Best for: Control freaks, functional style

2. **`learning_brain_with_maybe_learn.py`** - Pattern 3 (Method Wrapper)
   - Complete Learning brain implementation
   - Uses `_maybe_learn_from_teacher()` method
   - Shows meta-learning strategy learning
   - Best for: OOP enthusiasts, encapsulation lovers

3. **`attention_brain_with_maybe_learn.py`** - Pattern 3 (Method Wrapper)
   - Complete Attention brain implementation
   - Shows attention allocation rule learning
   - Includes focus management and shift decisions
   - Best for: Understanding complex learning loops

4. **`belief_tracker_with_maybe_learn.py`** - Pattern 3 (Method Wrapper)
   - Complete Belief Tracker brain implementation
   - Shows belief update pattern learning
   - Includes confidence adjustment logic
   - Best for: Learning about belief management

### Pattern 1 (TeacherHelper)

For Pattern 1 examples using the TeacherHelper class, see the actual cognitive brain implementations:

- `maven2_fix/brains/cognitive/planner/service/planner_brain.py`
- `maven2_fix/brains/cognitive/coder/service/coder_brain.py`
- `maven2_fix/brains/cognitive/affect_priority/service/affect_priority_brain.py`
- (30+ brains use this pattern)

---

## Quick Start

### Pattern 2: Direct Function Call

```python
from brains.cognitive.teacher.service.teacher_brain import teach_for_brain

# Call Teacher directly
result = teach_for_brain(
    brain_name="planner",
    situation="How should I break down a web scraping task?"
)

if result['verdict'] == 'LEARNED':
    for pattern in result['patterns']:
        print(pattern['pattern'])
```

**Run the examples:**
```bash
python maven2_fix/docs/examples/simple_teach_for_brain_usage.py
```

### Pattern 3: _maybe_learn_from_teacher() Method

```python
class MyBrain:
    def _maybe_learn_from_teacher(self, context):
        # 1. Check memory
        # 2. Call teach_for_brain if needed
        # 3. Validate with TruthClassifier
        # 4. Store to own memory
        # 5. Return patterns
        pass

    def my_main_function(self, inputs):
        patterns = self._maybe_learn_from_teacher(context)
        # Use patterns to process inputs
```

**Run the examples:**
```bash
python maven2_fix/docs/examples/learning_brain_with_maybe_learn.py
python maven2_fix/docs/examples/attention_brain_with_maybe_learn.py
python maven2_fix/docs/examples/belief_tracker_with_maybe_learn.py
```

### Pattern 1: TeacherHelper Class

```python
from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

_helper = TeacherHelper("my_brain")

result = _helper.maybe_call_teacher(
    question="How should I do X?",
    context={"key": "value"},
    check_memory_first=True  # Handles memory and storage automatically
)
```

---

## What You'll Learn

### From `simple_teach_for_brain_usage.py`

- ✅ Minimal usage (Example 1)
- ✅ Rich context passing (Example 2)
- ✅ Manual memory storage (Example 3)
- ✅ Validation with TruthClassifier (Example 4)
- ✅ Error handling (Example 5)
- ✅ Using different brain types (Example 6)
- ✅ Immediate use without storage (Example 7)

### From `learning_brain_with_maybe_learn.py`

- ✅ Complete `_maybe_learn_from_teacher()` implementation
- ✅ Situation key building
- ✅ Teacher result validation
- ✅ Budget management (limiting Teacher calls)
- ✅ Memory tier usage (STM, MTM, LTM)
- ✅ Meta-learning strategy application

### From `attention_brain_with_maybe_learn.py`

- ✅ Attention allocation rule learning
- ✅ Focus management patterns
- ✅ Multi-input handling
- ✅ Urgency-based decision making
- ✅ Pattern application to real decisions

### From `belief_tracker_with_maybe_learn.py`

- ✅ Belief confidence updating
- ✅ Evidence strength handling
- ✅ Contradictory evidence processing
- ✅ Belief revision patterns
- ✅ Confidence delta calculation

---

## Running the Examples

### Prerequisites

These examples work in two modes:

**1. Test Mode (No LLM)**
- Examples run but return ERROR verdicts
- Demonstrates structure and error handling
- No LLM configuration needed

**2. Production Mode (With LLM)**
- Examples actually learn from Teacher
- Requires LLM service configured in `brains/tools/llm_service.py`
- Returns LEARNED verdicts with real patterns

### Run All Examples

```bash
# Run each example individually
cd /home/user/maven

python maven2_fix/docs/examples/simple_teach_for_brain_usage.py
python maven2_fix/docs/examples/learning_brain_with_maybe_learn.py
python maven2_fix/docs/examples/attention_brain_with_maybe_learn.py
python maven2_fix/docs/examples/belief_tracker_with_maybe_learn.py
```

### Expected Output (Test Mode)

```
=== Example 1: Minimal Usage ===
Verdict: ERROR
Confidence: 0.0
Patterns learned: 0

=== Example 2: Rich Context ===
Verdict: ERROR

=== Example 3: Manual Storage ===
No pattern found, calling Teacher...
Verdict: ERROR
...
```

### Expected Output (Production Mode)

```
=== Example 1: Minimal Usage ===
Verdict: LEARNED
Confidence: 0.7
Patterns learned: 3
  Pattern 1: Break task into data extraction, parsing, and storage...
  Pattern 2: Identify target website structure and selectors...
  Pattern 3: Handle rate limiting and error cases...
...
```

---

## Pattern Comparison

| Feature | Pattern 1<br>(TeacherHelper) | Pattern 2<br>(teach_for_brain) | Pattern 3<br>(_maybe_learn) |
|---------|------------------------------|--------------------------------|----------------------------|
| **Example File** | *(see actual brains)* | `simple_teach_for_brain_usage.py` | `*_with_maybe_learn.py` |
| **Lines of Code** | ~10 lines | ~30 lines | ~100 lines |
| **Control Level** | Low (automatic) | High (manual) | High (manual) |
| **Memory Handling** | Automatic | Manual | Manual |
| **Validation** | Automatic | Manual | Manual |
| **Storage** | Automatic | Manual | Manual |
| **Best For** | Quick integration | Full control | OOP encapsulation |

---

## Next Steps

### To Use Pattern 2 in Your Brain

1. Import `teach_for_brain` from teacher_brain.py
2. Call it when you need Teacher's help
3. Validate the results yourself
4. Store patterns to memory yourself

See `simple_teach_for_brain_usage.py` for examples.

### To Use Pattern 3 in Your Brain

1. Copy the `_maybe_learn_from_teacher()` method from any example
2. Customize `_build_situation_key()` for your brain's situations
3. Customize `_is_valid_teacher_result()` for your validation rules
4. Call `_maybe_learn_from_teacher()` from your main functions

See `learning_brain_with_maybe_learn.py` for a complete template.

### To Use Pattern 1 in Your Brain

1. Import `TeacherHelper` from teacher_helper.py
2. Initialize: `_helper = TeacherHelper("your_brain_name")`
3. Call: `result = _helper.maybe_call_teacher(question, context)`
4. Use: `answer = result.get("answer")`

See actual cognitive brains for 30+ real examples.

---

## Architecture Notes

### Memory Storage

All patterns store to the same location:
- **Cognitive brains**: Store patterns to own memory under `maven2_fix/memory/brains/{brain_name}/`
- **Reasoning brain**: Stores facts to domain brains (science, history, personal, etc.)
- **Memory Librarian**: Stores routing rules to own memory

### Metadata Standards

All patterns should store with this metadata:

```python
metadata = {
    "kind": "teacher_pattern",      # or "learned_pattern"
    "source": "teacher",             # or "llm_teacher"
    "situation_key": "...",          # Unique situation identifier
    "confidence": 0.7,               # Teacher confidence
    # Brain-specific keys...
}
```

### Governance

All patterns should validate using TruthClassifier:

```python
from brains.cognitive.reasoning.truth_classifier import TruthClassifier

classification = TruthClassifier.classify(
    content=pattern_text,
    confidence=result['confidence'],
    evidence=None
)

if classification['type'] != 'RANDOM' and classification['allow_memory_write']:
    # Store pattern
```

---

## Troubleshooting

### "No teaching mode defined for brain"
Add your brain to `TEACHER_MODES` in teacher_brain.py:
```python
TEACHER_MODES = {
    "your_brain": "your_mode_name",
    # ...
}
```

### Examples return ERROR verdict
This is expected in test mode without LLM configured. The examples still demonstrate structure and error handling correctly.

### Patterns not being retrieved from memory
Check that:
- `situation_key` in storage matches query
- `confidence` threshold is not too high
- Memory tiers include the tier you stored to

### Want to see real learning in action?
Configure LLM service in `brains/tools/llm_service.py` and re-run the examples.

---

## Additional Resources

- **Full Documentation**: [`../TEACHER_INTEGRATION_PATTERNS.md`](../TEACHER_INTEGRATION_PATTERNS.md)
- **Teacher Brain Source**: `maven2_fix/brains/cognitive/teacher/service/teacher_brain.py`
- **TeacherHelper Source**: `maven2_fix/brains/cognitive/teacher/service/teacher_helper.py`
- **Contracts**: `maven2_fix/brains/teacher_contracts.py`
- **Tests**: `maven2_fix/tests/test_teach_for_brain.py`

---

## Contributing

To add a new example:

1. Create a new file: `{brain_name}_with_{pattern}.py`
2. Follow the structure from existing examples
3. Include docstrings explaining the pattern
4. Add runnable `if __name__ == "__main__"` section
5. Update this README

---

## License

These examples are part of the Maven project and follow the same license.
