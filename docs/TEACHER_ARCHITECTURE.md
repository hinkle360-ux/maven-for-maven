# Teacher Architecture

## Overview

The Teacher module provides external LLM (Language Model) assistance to Maven's cognitive brains.
However, Teacher is designed as an **advisor, not an oracle**. Teacher never directly writes
authoritative facts to memory - it only returns structured proposals that brains can evaluate
and decide whether to accept.

## Core Principles

### 1. Teacher as Proposal Engine, Not Oracle

- Teacher returns **TeacherProposal** objects containing hypotheses and suggestions
- All proposals have `status="proposal"` and `source="teacher"`
- Brains evaluate proposals against their own beliefs before storing
- Facts from Teacher start as `status="tentative"`, requiring confirmation

### 2. Memory-First Architecture

The memory-first pattern is ALWAYS enforced:

1. Check Brain's own BrainMemory (STM/MTM/LTM)
2. Check relevant domain banks
3. Check strategy tables and Q->A cache
4. ONLY if memory fails -> call LLM as teacher
5. Store lessons/facts -> next time answer from memory

### 3. Forbidden Teacher Topics

Teacher MUST NOT answer questions about:

- **Self-identity**: "Who are you?", "What is Maven?" -> Route to self_model
- **Capabilities**: "Can you search the web?" -> Route to capability_snapshot
- **Conversation history**: "What did we discuss?" -> Route to system_history
- **User memory**: "What do you remember about me?" -> Route to personal bank
- **Explain previous**: "Why did you answer that way?" -> Route to EXPLAIN_LAST

## TeacherProposal Schema

```python
@dataclass
class TeacherProposal:
    candidate_questions: List[CandidateQuestion]  # Questions to clarify
    hypotheses: List[Hypothesis]                  # Proposed facts
    strategy_suggestions: List[StrategySuggestion]  # Suggested approaches
    answer: Optional[str]                         # Direct answer text
    verdict: str                                  # PROPOSAL, NO_ANSWER, BLOCKED
    original_question: Optional[str]

@dataclass
class Hypothesis:
    statement: str
    confidence: float  # 0.0-1.0
    kind: HypothesisKind  # FACTUAL, PROCEDURAL, SELF_DESCRIPTION, CAPABILITY, etc.
    source: str = "teacher"  # Always "teacher"
    status: str = "proposal"  # Never "committed" from Teacher

    def is_safe_to_evaluate(self) -> bool:
        """Self-description and capability hypotheses are NOT safe."""
        return self.kind not in (SELF_DESCRIPTION, CAPABILITY)
```

## Feature Flags

In `config/features.json`:

```json
{
    "teacher_learning": true,        // Enable Teacher-assisted learning
    "teacher_direct_fact_write": false,  // MUST be false - no direct writes
    "teacher_proposal_mode": true,   // Enable proposal-only mode
    "capability_startup_scan": true  // Enable capability verification at startup
}
```

## Responsibility Split

| Component | Responsibility |
|-----------|---------------|
| **Teacher** | Proposes hypotheses and patterns (NEVER commits facts) |
| **Brains** | Evaluate proposals, decide what to store, promote tentative->committed |
| **Governance/Policy** | Enforce storage rules, block forbidden topics |
| **self_model** | Answer identity and capability questions |
| **capability_snapshot** | Truthful answers about what Maven can do |

## Flow Diagram

```
User Question
     |
     v
+----------------+
|  Reasoning     |---> Is self-identity? ---> self_model
|    Brain       |---> Is capability?    ---> capability_snapshot
+----------------+---> Is explain?       ---> EXPLAIN_LAST
     |
     | (Not forbidden)
     v
+----------------+
| Memory Check   |---> Found? --> Return from memory (no LLM call)
+----------------+
     |
     | (Memory miss)
     v
+----------------+
| TeacherHelper  |---> Calls Teacher with guards
+----------------+
     |
     v
+----------------+
|    Teacher     |---> Returns TeacherProposal
+----------------+     (status=proposal, source=teacher)
     |
     v
+----------------+
|    Brain       |---> Evaluates proposal
+----------------+---> Stores as tentative if compatible
                 ---> Promotes to committed after confirmation
```

## Capability Registry

The capability registry provides truthful answers about what Maven can do.

### Startup Scan

At system startup, `run_capability_startup_scan()`:
- Checks web_client availability
- Checks LLM service status
- Verifies filesystem access
- Tests git tool availability
- Checks Python sandbox
- Tests browser runtime

### Capability Snapshot

`get_capability_snapshot()` returns:
- `web_search_enabled`: Can Maven search the web?
- `code_execution_enabled`: Can Maven run code?
- `filesystem_scope`: What files can Maven access?
- `can_control_programs`: ALWAYS FALSE
- `autonomous_tools`: ALWAYS FALSE

### Answering Capability Questions

`answer_capability_question()` handles:
- "Can you search the web for me?"
- "Can you run code for me?"
- "Can you control other programs on my computer?"
- "Can you read or change files on my system?"
- "Can you use tools or the internet without me asking you?"

All answers come from the capability snapshot, NEVER from Teacher.

## Usage Example

```python
from brains.cognitive.teacher.service.teacher_helper import TeacherHelper

helper = TeacherHelper("reasoning")

# TeacherHelper enforces memory-first and forbidden topic guards
result = helper.maybe_call_teacher(
    question="What is the capital of France?",
    context={}
)

if result:
    if result["source"] == "local_memory":
        # Answered from memory - no LLM call made
        pass
    elif result["source"] == "teacher":
        # Learned from LLM
        # Facts stored as tentative (status="tentative")
        pass
    elif result["source"] == "llm_disabled":
        # Question was blocked (self-identity, capability, etc.)
        # Route to appropriate handler
        pass
```

## Testing

The test suite in `tests/test_capabilities_and_boundaries.py` verifies:

1. Self-identity questions are blocked by Teacher
2. Capability questions use capability_snapshot
3. Teacher returns proposals (not direct facts)
4. Feature flags are correctly configured
5. Startup scan works correctly

Run tests with:
```bash
pytest tests/test_capabilities_and_boundaries.py -v
```

## Migration from Legacy Mode

If migrating from legacy (direct fact write) mode:

1. Set `teacher_proposal_mode: true` in features.json (default)
2. Set `teacher_direct_fact_write: false` (default)
3. Existing facts will continue to work
4. New facts from Teacher will be stored as `status="tentative"`
5. Brains can promote tentative facts after confirmation

To enable legacy mode (NOT RECOMMENDED):
```json
{
    "teacher_proposal_mode": false,
    "teacher_direct_fact_write": true
}
```
