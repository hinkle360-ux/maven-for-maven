# Host Tools - External I/O Implementations

This directory contains the concrete implementations of tool interfaces
that perform actual I/O operations. These tools are used by the host
runtime and should **NEVER** be imported by core brains.

## Architecture

```
brains/tools_api.py     <-- Abstract interfaces (stdlib-only)
       ^
       |  (implements)
       |
host_tools/             <-- Concrete implementations (I/O)
    web_client/         <-- HTTP operations
    llm_client/         <-- LLM API calls
    shell_executor/     <-- Subprocess execution
    git_client/         <-- Git operations
```

## Usage

The host runtime creates a `ToolRegistry` with concrete implementations
and injects it into the brain context:

```python
from host_tools.factory import create_host_tools

# Create all available tools
tools = create_host_tools(
    enable_web=True,
    enable_llm=True,
    enable_shell=True,
    enable_git=True,
    enable_sandbox=True,
    root_dir="/path/to/maven"
)

# Inject into brain context
context.tools = tools

# Or pass to brain directly
brain.handle(context, tools=tools)
```

## Tool Implementations

### web_client/
HTTP operations for web search and page fetching.
- `HostWebSearchTool`: Web search via DuckDuckGo HTML API
- `HostWebFetchTool`: Page fetching with text extraction

### llm_client/
LLM completion operations.
- `HostLLMTool`: Ollama-based LLM with pattern learning

### shell_executor/
Shell command and Python sandbox execution.
- `HostShellTool`: Shell command execution with policy enforcement
- `HostPythonSandboxTool`: Isolated Python code execution

### git_client/
Git repository operations.
- `HostGitTool`: Full git operations (status, commit, push, etc.)

## Key Principles

1. **Brains never import from host_tools** - All tool access goes through
   the abstract interfaces in `brains/tools_api.py`

2. **Host injects tools** - The runtime wires concrete implementations
   into the brain context before invoking any brain

3. **Graceful degradation** - Brains should handle missing tools:
   ```python
   if tools.web_search:
       result = tools.web_search.search("query")
   else:
       # Handle offline mode
       pass
   ```

4. **Testability** - Use `create_null_tools()` from `brains.tools_api`
   for unit testing brains without I/O
