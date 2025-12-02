# Optional Runtime Integrations

This directory contains optional runtime components that extend Maven's
capabilities but are NOT part of the core Maven brain system.

## Why This Separation?

Per Maven specification:
- Core brains must be stdlib-only and offline
- No external dependencies (requests, httpx, openai, playwright, etc.)
- Browser/HTTP work is provided by the host via abstract interfaces

## Contents

### browser_runtime/
The Playwright-based browser automation server. Runs as a separate service.
- Dependencies: playwright, fastapi, uvicorn, pydantic, python-dotenv
- Start with: `python -m optional.browser_runtime`

### maven_browser_client/
HTTP client for communicating with the browser runtime.
- Dependencies: httpx
- Used by browser_tools to execute browser plans

### browser_tools/
Maven tool implementations for browser automation.
- browser_tool.py: Main tool for executing browser plans
- intent_resolver.py: Converts natural language to browser plans
- plan_validator.py: Validates browser plans before execution
- pattern_store.py: Stores learned browsing patterns
- task_executor.py: Task execution sandbox
- reflection.py: Post-task reflection and learning

### browser_brain/
Cognitive brain wrapper for browser capabilities.
- Integrates browser automation into Maven's cognitive stack
- NOT imported by core brains

## Usage

These components should be imported only by host applications that:
1. Have installed the required external dependencies
2. Are running the browser runtime service
3. Explicitly enable browser capabilities

Core Maven brains must NEVER import from this directory.
