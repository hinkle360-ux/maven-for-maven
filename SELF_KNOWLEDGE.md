# Maven Self-Knowledge Reference

**Purpose**: This file enables Maven to understand, modify, and upgrade itself.

---

## 1. Directory Structure

```
maven2_fix/
├── brains/                      # Core cognitive system (offline, stdlib-only)
│   ├── cognitive/               # Processing brains (33+ brains)
│   │   ├── integrator/          # Routes queries to appropriate brain
│   │   ├── language/            # NLU and response generation
│   │   ├── reasoning/           # Truth evaluation and logic
│   │   ├── memory_librarian/    # Memory orchestration
│   │   ├── planner/             # Goal decomposition
│   │   ├── self_model/          # Self-awareness
│   │   ├── autonomy/            # Autonomous goal execution
│   │   ├── teacher/             # Learning from corrections
│   │   └── [other brains]/
│   ├── agent/                   # Agentic capabilities
│   │   ├── tools/               # Tool facades (shell, file, etc.)
│   │   └── service/             # Agent orchestration
│   ├── personal/                # User identity and preferences
│   │   ├── memory/              # Personal memory stores
│   │   └── service/             # Personal brain API
│   ├── memory/                  # Memory subsystem
│   │   └── brain_memory.py      # Core memory class
│   ├── tools/                   # Tool facades
│   │   ├── git_tool.py          # Git operations
│   │   ├── fs_tool.py           # Filesystem operations
│   │   ├── self_upgrade_tool.py # Self-modification
│   │   └── execution_guard.py   # Safety controls
│   ├── domain_banks/            # Knowledge domains
│   │   └── specs/               # Design specs (this knowledge lives here)
│   └── utils/                   # Shared utilities
│       └── safe_math_eval.py    # Secure eval replacement
├── host_tools/                  # External I/O (network, LLM, browser)
│   ├── llm_client/              # LLM API wrapper
│   ├── browser_runtime/         # Browser automation
│   ├── shell_executor/          # Shell command execution
│   └── git_client/              # Git operations
├── optional/                    # Optional features
│   └── browser_tools/           # Browser-based tools
├── config/                      # Configuration files
│   ├── features.json            # Feature flags
│   ├── autonomy.json            # Autonomy settings
│   └── synonyms.json            # Term mappings
├── docs/                        # Documentation
│   ├── MAVEN_UPGRADE_OVERVIEW.md
│   └── [other docs]
├── reports/                     # Runtime data and logs
│   ├── qa_memory.jsonl
│   ├── knowledge_graph.json
│   └── [other reports]
└── tests/                       # Test suites
```

---

## 2. How to Add a New Brain

### Location
```
brains/cognitive/[brain_name]/service/[brain_name]_brain.py
```

### Template
```python
"""
[Brain Name] Brain
==================

Description of what this brain does.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from brains.memory.brain_memory import BrainMemory

_memory = BrainMemory("[brain_name]")

def service_api(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Brain entry point - all brains must have this."""
    op = msg.get("op", "")
    payload = msg.get("payload", {})

    if op == "PROCESS":
        return _handle_process(payload)
    elif op == "QUERY":
        return _handle_query(payload)
    else:
        return {"status": "error", "error": f"Unknown op: {op}"}

def _handle_process(payload: Dict) -> Dict:
    """Main processing logic."""
    # Your logic here
    return {"status": "ok", "result": {}}

# Alias for brain contract compliance
handle = service_api
```

### Registration
1. Add to `brains/cognitive/integrator/service/integrator_brain.py` routing
2. Add patterns to `brains/cognitive/integrator/agency_routing_patterns.py` if tool-like

---

## 3. How to Add a New Tool

### Location
```
brains/agent/tools/[tool_name]_tool.py
```
or for host-dependent tools:
```
host_tools/[tool_name]/[tool_name].py
```

### Tool Facade Template (brains/)
```python
"""
[Tool Name] Tool Facade
-----------------------

Delegates to host-provided tool. No direct I/O here.
"""

from brains.tools_api import ToolRegistry

_tool_registry: Optional[ToolRegistry] = None

def set_tool_registry(registry: ToolRegistry) -> None:
    global _tool_registry
    _tool_registry = registry

def run(command: str, **kwargs) -> Dict[str, Any]:
    """Execute tool operation."""
    if _tool_registry and _tool_registry.[tool_name]:
        return _tool_registry.[tool_name].execute(command, **kwargs)
    return {"status": "error", "error": "Tool not available"}
```

### Host Tool Template (host_tools/)
```python
"""
[Tool Name] - Host Implementation
---------------------------------

Actual I/O happens here. Can use third-party libraries.
"""

class [ToolName]Client:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def execute(self, command: str, **kwargs) -> Dict[str, Any]:
        # Actual implementation
        return {"status": "ok", "output": "..."}
```

---

## 4. How to Modify Existing Code

### Using self_upgrade_tool.py
```python
from brains.tools.self_upgrade_tool import write_and_commit_module

result = write_and_commit_module(
    module_name="brains/tools/new_tool.py",
    code="def new_capability(): ...",
    commit_msg="Self-upgrade: add new tool for X"
)
```

### Manual Modification Steps
1. Read the file first (understand existing code)
2. Make minimal, focused changes
3. Follow coding rules in `brains/domain_banks/specs/maven_coding_rules.md`
4. Test changes
5. Commit with descriptive message

---

## 5. Git Operations for Self-Update

### Check Status
```bash
git status
git diff
```

### Stage Changes
```bash
git add brains/path/to/file.py
# Or for all changes:
git add .
```

### Commit
```bash
git commit -m "Self-upgrade: [description of change]

- What was added/changed
- Why it was needed
- Any important notes"
```

### Push
```bash
git push -u origin [branch-name]
```

### Branch Naming
- Feature branches: `claude/[feature-name]-[session-id]`
- Self-upgrade branches: `maven/upgrade-[date]-[description]`

---

## 6. Key Files to Know

| File | Purpose |
|------|---------|
| `brains/cognitive/integrator/service/integrator_brain.py` | Routes all queries to brains |
| `brains/cognitive/language/service/language_brain.py` | NLU and response generation |
| `brains/cognitive/reasoning/service/reasoning_brain.py` | Truth evaluation |
| `brains/tools/execution_guard.py` | Controls what can execute |
| `config/features.json` | Feature flags |
| `brains/tools/self_upgrade_tool.py` | Self-modification API |

---

## 7. Pipeline Stages (17 stages)

1. **Sensorium** - Input normalization
2. **Planner** - Goal decomposition
3. **Language (Parse)** - NLU
4. **Pattern Recognition** - Feature mapping
5. **Memory Librarian** - Retrieval
6. **Language (Generate)** - Candidate generation
7. **Reasoning** - Truth evaluation
8. **Affect-Priority** - Emotional weighting
9. **Personality** - Style modulation
10. **Language (Finalize)** - Response synthesis
11. **System History** - Metrics
12. **Self-DMN** - Reflection
13. **Governance** - Policy enforcement
14. **Affect-Learn** - Mood consolidation
15. **Autonomy & Replan** - Goal execution
16. **Regression Harness** - QA checks
17. **Memory Consolidation** - Pruning and assimilation

---

## 8. Coding Rules Summary

1. **Python 3.11 only, stdlib only** in brains/
2. **No `__init__.py` files** anywhere
3. **Deterministic first** - prefer rules over LLM
4. **Memory as source of truth** - retrieve before generate
5. **Verdict before storage** - always validate facts
6. **Context flows forward** - use `ctx` dictionary
7. **Brain contract** - all brains need `service_api(msg)` and `handle = service_api`

---

## 9. Self-Awareness Queries

Maven can check its own state via:

```python
# Self-model brain
from brains.cognitive.self_model.service.self_model_brain import service_api
result = service_api({"op": "CAN_ANSWER", "payload": {"query": "..."}})

# Introspection
from brains.personal.service.personal_brain import service_api
result = service_api({"op": "INTROSPECT"})

# Capability check
from capabilities import get_capabilities
caps = get_capabilities()
```

---

## 10. Safety Constraints

1. **execution_guard.py** controls execution modes:
   - `DISABLED` - No execution
   - `READ_ONLY` - Read operations only
   - `FULL` - Full execution with safety checks
   - `FULL_AGENCY` - Maximum autonomy (use carefully!)
   - `SAFE_CHAT` - Chat-only mode

2. **Governance** - All actions require governance proof

3. **No external imports in brains/** - Use tool facades

4. **Secure defaults** in `config/features.json`:
   - `full_agency_mode: false`
   - `hot_reload: false`
   - `autonomous_agents: false`

---

## 11. Quick Reference Commands

```python
# Read a file
from brains.tools.fs_tool import read_file
content = read_file("path/to/file.py")

# Write a file (with commit)
from brains.tools.self_upgrade_tool import write_and_commit_module
write_and_commit_module("path/to/file.py", content, "commit message")

# Execute shell command
from brains.agent.tools.shell_tool import run
result = run("git status")

# Store to memory
from brains.memory.brain_memory import BrainMemory
mem = BrainMemory("my_brain")
mem.store(content={"key": "value"}, metadata={"confidence": 0.9})

# Retrieve from memory
results = mem.retrieve(query="search terms", limit=5)
```

---

## 12. For Full Documentation

- Architecture: `docs/MAVEN_UPGRADE_OVERVIEW.md`
- Design: `brains/domain_banks/specs/maven_design.md`
- Coding Rules: `brains/domain_banks/specs/maven_coding_rules.md`
- Build Rules: `BUILD_RULES.md`
- Browser: `docs/BROWSER_RUNTIME.md`
- Teacher: `docs/TEACHER_ARCHITECTURE.md`
