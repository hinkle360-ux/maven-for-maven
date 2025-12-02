# Maven Build Rules

## Directory Layout

- **brains/** – Maven core (offline, stdlib-only)
- **host_tools/** – External I/O (HTTP, LLM, shell, git, browser)
- **optional/** – Extra features (browser runtime, UI), not required for core

## Requirements Files

- **requirements_core.txt** – Must remain empty (or only true stdlib backports)
- **requirements_host_tools.txt** – All third-party dependencies
- **requirements_dev.txt** – Dev-only tooling

## __init__.py Rule

- **No __init__.py anywhere in the repo**
- If any appear, they must be deleted and imports rewired to module-based imports
- Applies to: brains/, host_tools/, optional/, tests/, everything

## Brain Contract

- All brain modules under `brains/**/service/*.py` must have:
  - `def handle(context) -> dict` – canonical entry point
  - `service_api = handle` – alias at bottom of file
- Helper modules in service/ directories that are not brains:
  - Should NOT be forced into brain shape
  - Should be clearly documented as helpers

### Helper Allowlist (exempt from brain contract)

These files live in `service/` directories but are support modules, not brains:

- `brains/agent/service/blackboard.py`
- `brains/personal/service/identity_user_store.py`
- `brains/cognitive/memory_librarian/service/librarian_memory.py`
- `brains/cognitive/reasoning/service/dynamic_confidence.py`
- `brains/cognitive/self_model/service/self_introspection.py`
- `brains/cognitive/teacher/service/teacher_helper.py`
- `brains/cognitive/teacher/service/prompt_templates.py`

Brain contract scans should skip these paths.

## Forbidden Imports in brains/

- No third-party/network/browser/LLM libraries:
  - requests, httpx, openai, fastapi, uvicorn, playwright, aiohttp, websockets, selenium
- No dangerous stdlib for core:
  - subprocess, socket, os.system, urllib.request, urllib.parse, urllib.error, http.client

## Memory API Invariants

- `BrainMemory.store()` always writes to STM only
- No tier parameter exposed to external callers
- Spill logic handles overflow (STM → MTM → LTM → Archive)

## Governance Rules

- Governance modules cannot modify memory directly
- Governance modules cannot override truth classification
- Only Reasoning brain can classify truth types via `truth_classifier.py`

## Tool Injection Pattern

- Host runtime creates tools via `host_tools.factory.create_host_tools()`
- Injects via `brains.tool_injection.inject_tools(registry)`
- Brains access tools via tool registry, never import host_tools directly
