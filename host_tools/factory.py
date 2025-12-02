"""
Host Tools Factory
==================

Factory function for creating a ToolRegistry with all available host tool
implementations. This is the main entry point for the host runtime.

This package contains concrete implementations of the tool interfaces
defined in brains.tools_api. These implementations perform actual I/O
operations (HTTP, subprocess, etc.) and should NEVER be imported by
core brains.

The host runtime imports these implementations and injects them into
the brain context via the ToolRegistry.

Package structure:
    host_tools/
        web_client/     - HTTP client for web search and page fetching
        llm_client/     - LLM service client (Ollama, OpenAI, etc.)
        shell_executor/ - Shell command execution
        git_client/     - Git operations via subprocess

Dependencies:
    These modules may use external packages such as:
    - urllib/http.client (stdlib, but performs network I/O)
    - subprocess (stdlib, but executes external processes)
    - requests, httpx (optional, for advanced HTTP)
    - openai (optional, for OpenAI API)

Core brains must NEVER import from this package.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brains.tools_api import ToolRegistry


def create_host_tools(
    enable_web: bool = True,
    enable_llm: bool = True,
    enable_shell: bool = True,
    enable_git: bool = True,
    enable_sandbox: bool = True,
    root_dir: str | None = None
) -> "ToolRegistry":
    """
    Create a ToolRegistry with all available host tool implementations.

    This is the main entry point for the host runtime to create tools
    before invoking Maven brains.

    Args:
        enable_web: Enable web search/fetch tools
        enable_llm: Enable LLM completion tool
        enable_shell: Enable shell execution tool
        enable_git: Enable git operations tool
        enable_sandbox: Enable Python sandbox tool
        root_dir: Root directory for confined operations

    Returns:
        ToolRegistry with concrete tool implementations
    """
    from brains.tools_api import ToolRegistry

    registry = ToolRegistry()

    if enable_web:
        try:
            from host_tools.web_client.client import HostWebSearchTool, HostWebFetchTool
            registry.web_search = HostWebSearchTool()
            registry.web_fetch = HostWebFetchTool()
        except Exception as e:
            print(f"[HOST_TOOLS] Failed to load web tools: {e}")

    if enable_llm:
        try:
            from host_tools.llm_client.client import HostLLMTool
            registry.llm = HostLLMTool()
        except Exception as e:
            print(f"[HOST_TOOLS] Failed to load LLM tool: {e}")

    if enable_shell:
        try:
            from host_tools.shell_executor.executor import HostShellTool
            registry.shell = HostShellTool(root_dir=root_dir)
        except Exception as e:
            print(f"[HOST_TOOLS] Failed to load shell tool: {e}")

    if enable_git:
        try:
            from host_tools.git_client.client import HostGitTool
            registry.git = HostGitTool(root_dir=root_dir)
        except Exception as e:
            print(f"[HOST_TOOLS] Failed to load git tool: {e}")

    if enable_sandbox:
        try:
            from host_tools.shell_executor.executor import HostPythonSandboxTool
            registry.python_sandbox = HostPythonSandboxTool(root_dir=root_dir)
        except Exception as e:
            print(f"[HOST_TOOLS] Failed to load sandbox tool: {e}")

    return registry
