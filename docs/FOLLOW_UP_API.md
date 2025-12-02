# Follow-Up API for Cognitive Brains

## Overview

All cognitive brains must implement the unified follow-up framework to enable Maven to properly handle conversation continuations like "tell me more", "what else", and "continue".

This document describes the three required signals every brain must implement.

## Required Signals

Every cognitive brain MUST use these three signals from `brains.cognitive.continuation_helpers`:

### 1. Follow-up Detection: `is_continuation()`

**Purpose**: Detect if the user's input is a continuation of previous conversation.

**Usage**:
```python
from brains.cognitive.continuation_helpers import is_continuation

def process_query(query: str, context: Dict[str, Any]):
    if is_continuation(query, context):
        # Handle as follow-up
        pass
    else:
        # Handle as new query
        pass
```

**Detection patterns**:
- Explicit phrases: "tell me more", "what else", "continue", "keep going"
- Sensorium classification: `norm_type == "follow_up_question"`
- Short pronouns: "that", "it", "this" in queries ≤ 3 words

### 2. History Access: `get_conversation_context()`

**Purpose**: Retrieve the last topic and conversation state.

**Usage**:
```python
from brains.cognitive.continuation_helpers import get_conversation_context

def process_query(query: str):
    conv_context = get_conversation_context()
    last_topic = conv_context.get("last_topic")
    last_question = conv_context.get("last_user_question")

    # Use context to inform processing
    if last_topic:
        enhanced_query = f"{query} about {last_topic}"
```

**Returned fields**:
- `last_topic`: Most recent conversation topic
- `last_user_question`: Previous user query
- `last_answer_subject`: Subject of last response
- `thread_entities`: Entities mentioned in conversation
- `conversation_depth`: Number of turns

### 3. Routing Hints: `create_routing_hint()`

**Purpose**: Provide standardized routing suggestions to the Integrator.

**Usage**:
```python
from brains.cognitive.continuation_helpers import create_routing_hint

def generate_routing_hint(is_continuation: bool):
    if is_continuation:
        return create_routing_hint(
            brain_name="reasoning",
            action="expand_previous_reasoning",
            confidence=0.9,
            context_tags=["follow_up", "factual"],
            metadata={"requires_topic": True}
        )
```

## Integration Checklist

For each cognitive brain, ensure:

- [ ] Import continuation_helpers at top of module
- [ ] Call `is_continuation()` when processing queries
- [ ] Call `get_conversation_context()` for topic awareness
- [ ] Use enhanced queries for follow-ups (append topic context)
- [ ] Generate routing hints with `create_routing_hint()`
- [ ] Store current topic via `system_history.SET_TOPIC` after answering

## Example: Full Integration

```python
from brains.cognitive.continuation_helpers import (
    is_continuation,
    get_conversation_context,
    create_routing_hint,
    enhance_query_with_context
)

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    op = msg.get("op", "").upper()
    payload = msg.get("payload", {})

    if op == "PROCESS_QUERY":
        query = payload.get("query", "")
        context = payload.get("context", {})

        # 1. Detect continuation
        is_cont = is_continuation(query, context)

        # 2. Get conversation history
        conv_context = get_conversation_context()

        # 3. Enhance query if continuation
        if is_cont:
            enhanced_query = enhance_query_with_context(query, conv_context)
        else:
            enhanced_query = query

        # 4. Process with enhanced query
        result = process_internal(enhanced_query, context)

        # 5. Generate routing hint
        routing_hint = create_routing_hint(
            brain_name="my_brain",
            action="expand_previous" if is_cont else "new_query",
            confidence=0.9,
            context_tags=["follow_up"] if is_cont else []
        )

        # 6. Store topic for next turn
        if result.get("topic"):
            from brains.cognitive.system_history.service.system_history_brain import service_api as history_api
            history_api({
                "op": "SET_TOPIC",
                "payload": {"topic": result["topic"]}
            })

        return {
            "ok": True,
            "payload": {
                "result": result,
                "routing_hint": routing_hint,
                "is_continuation": is_cont
            }
        }
```

## Brains Status

### ✅ Implemented
- **Sensorium**: Detects follow_up_question classification
- **System_History**: Stores/retrieves last topic via SET_TOPIC/GET_LAST_TOPIC
- **Reasoning**: Uses enhanced queries for continuations
- **Integrator**: Routes follow_up_question to REASONING

### 🔄 Needs Integration
- **Language**: Add continuation awareness
- **Pattern_Recognition**: Detect continuation patterns
- **Memory_Librarian**: Track conversation threads
- **Affect_Priority**: Adjust tone for continuations
- **Planner**: Expand previous plans on continuation
- **Motivation**: Maintain topic interest across turns

## Benefits

When all brains use this API:

1. **Consistent behavior**: "tell me more" works the same across all brains
2. **Teacher learning**: Teacher can learn optimal continuation handling patterns
3. **Context preservation**: Topics aren't lost between turns
4. **Better routing**: Integrator makes smarter decisions with routing hints
5. **User experience**: Conversations feel natural and continuous

## Testing

Test your brain's integration:

```python
# Test continuation detection
assert is_continuation("tell me more")
assert is_continuation("what else?")
assert not is_continuation("what is a lion?")

# Test context retrieval
context = get_conversation_context()
assert "last_topic" in context
assert "last_user_question" in context

# Test routing hint generation
hint = create_routing_hint("my_brain", "expand", 0.8)
assert hint["brain"] == "my_brain"
assert hint["routing_hint"] == "expand"
assert 0 <= hint["confidence"] <= 1
```

## Migration Guide

1. Add import: `from brains.cognitive.continuation_helpers import ...`
2. Update query processing to check `is_continuation()`
3. Call `get_conversation_context()` at start of processing
4. Use enhanced query for Teacher/memory lookups
5. Generate routing hints in responses
6. Set topic after successful answers

See `brains/cognitive/reasoning/service/reasoning_brain.py` for reference implementation.
