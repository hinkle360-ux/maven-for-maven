"""
Abstract Tool Interfaces for Maven Brains
==========================================

This module defines the pure Python (stdlib-only) interfaces that brains
use to access external capabilities. Brains NEVER import concrete implementations
or make direct I/O calls. Instead, the host runtime injects concrete tool
implementations that satisfy these interfaces into the context.

Usage in brains:
    # Access tools via context (injected by host)
    result = context.tools.web_search.search("query")

    # Or via explicit tool parameter
    def handle(context, tools: ToolRegistry):
        result = tools.llm.complete("prompt")

The host is responsible for:
1. Instantiating concrete tool implementations
2. Injecting them into the context/tool registry before invoking brains
3. Handling all actual I/O (HTTP, subprocess, etc.)

This keeps brains pure, testable, and stdlib-only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


# =============================================================================
# Web/HTTP Tool Interfaces
# =============================================================================

@dataclass
class WebSearchResult:
    """Result from a web search operation."""
    text: str
    url: Optional[str]
    confidence: float
    raw_results: List[Dict[str, Any]] = field(default_factory=list)


class WebSearchTool(Protocol):
    """Interface for web search operations."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        language: str = "en",
        timeout: int = 30
    ) -> WebSearchResult:
        """
        Perform a web search and return results.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            language: Language code for results
            timeout: Timeout in seconds

        Returns:
            WebSearchResult with text content and metadata
        """
        ...


@dataclass
class WebFetchResult:
    """Result from fetching a web page."""
    text: str
    url: str
    status_code: int
    success: bool
    error: Optional[str] = None


class WebFetchTool(Protocol):
    """Interface for fetching web pages."""

    def fetch(
        self,
        url: str,
        *,
        timeout: int = 30,
        max_chars: int = 8000
    ) -> WebFetchResult:
        """
        Fetch a web page and extract text content.

        Args:
            url: URL to fetch
            timeout: Timeout in seconds
            max_chars: Maximum characters to return

        Returns:
            WebFetchResult with extracted text
        """
        ...


# =============================================================================
# LLM Tool Interface
# =============================================================================

@dataclass
class LLMCompletionResult:
    """Result from an LLM completion call."""
    ok: bool
    text: str
    source: str  # e.g., "ollama", "openai", "learned_template"
    llm_used: bool
    confidence: float = 1.0
    error: Optional[str] = None
    error_type: Optional[str] = None


class LLMTool(Protocol):
    """Interface for LLM completion operations."""

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMCompletionResult:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            context: Optional context dict for template substitution

        Returns:
            LLMCompletionResult with the generated text
        """
        ...

    @property
    def enabled(self) -> bool:
        """Whether the LLM service is enabled."""
        ...


# =============================================================================
# Shell/Process Execution Tool Interface
# =============================================================================

@dataclass
class ShellResult:
    """Result from a shell command execution."""
    status: str  # "completed", "error", "timeout", "denied"
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


class ShellTool(Protocol):
    """Interface for shell command execution."""

    def run(
        self,
        cmd: str,
        *,
        cwd: Optional[str] = None,
        timeout: int = 120,
        check_policy: bool = True
    ) -> ShellResult:
        """
        Execute a shell command.

        Args:
            cmd: Command to execute
            cwd: Working directory
            timeout: Timeout in seconds
            check_policy: Whether to check against deny patterns

        Returns:
            ShellResult with output and status
        """
        ...


# =============================================================================
# Git Tool Interface
# =============================================================================

@dataclass
class GitStatusResult:
    """Result from git status."""
    branch: Optional[str]
    ahead: int
    behind: int
    staged: List[str]
    modified: List[str]
    untracked: List[str]
    deleted: List[str]
    is_clean: bool


@dataclass
class GitLogEntry:
    """A single git log entry."""
    hash: str
    author_name: str
    author_email: str
    timestamp: int
    message: str


@dataclass
class GitResult:
    """Generic result from a git operation."""
    ok: bool
    output: str
    error: Optional[str] = None
    returncode: int = 0


class GitTool(Protocol):
    """Interface for git operations."""

    def status(self, *, short: bool = True, porcelain: bool = False) -> str:
        """Get git status output."""
        ...

    def status_detailed(self) -> GitStatusResult:
        """Get detailed git status as structured data."""
        ...

    def diff(
        self,
        *,
        cached: bool = False,
        file_path: Optional[str] = None,
        commit: Optional[str] = None
    ) -> str:
        """Get git diff output."""
        ...

    def log(
        self,
        *,
        max_count: int = 10,
        oneline: bool = False
    ) -> List[GitLogEntry]:
        """Get git log entries."""
        ...

    def add(self, paths: List[str]) -> GitResult:
        """Stage files."""
        ...

    def commit(self, message: str, *, allow_empty: bool = False) -> GitResult:
        """Create a commit."""
        ...

    def current_branch(self) -> str:
        """Get current branch name."""
        ...

    def is_clean(self) -> bool:
        """Check if working tree is clean."""
        ...


# =============================================================================
# Python Sandbox Tool Interface
# =============================================================================

@dataclass
class SandboxResult:
    """Result from sandbox code execution."""
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    error: Optional[str] = None
    timed_out: bool = False


class PythonSandboxTool(Protocol):
    """Interface for sandboxed Python execution."""

    def execute(
        self,
        code: str,
        *,
        timeout_ms: int = 3000,
        cwd: Optional[str] = None
    ) -> SandboxResult:
        """
        Execute Python code in an isolated subprocess.

        Args:
            code: Python code to execute
            timeout_ms: Timeout in milliseconds
            cwd: Working directory

        Returns:
            SandboxResult with output and status
        """
        ...


# =============================================================================
# Tool Registry
# =============================================================================

@dataclass
class ToolRegistry:
    """
    Registry of available tools, injected into brains by the host.

    Usage in brains:
        if tools.web_search:
            result = tools.web_search.search("query")
        else:
            # Graceful degradation when tool unavailable
            result = None
    """
    web_search: Optional[WebSearchTool] = None
    web_fetch: Optional[WebFetchTool] = None
    llm: Optional[LLMTool] = None
    shell: Optional[ShellTool] = None
    git: Optional[GitTool] = None
    python_sandbox: Optional[PythonSandboxTool] = None
    # Browser-based tools (ChatGPT, Grok, etc.) - Dict[name, callable]
    browser_tools: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        # Check browser_tools dict first
        if name in getattr(self, 'browser_tools', {}):
            return self.browser_tools[name]
        return getattr(self, name, None)

    def available(self) -> List[str]:
        """List available (non-None) tools."""
        tools = [
            name for name in ["web_search", "web_fetch", "llm", "shell", "git", "python_sandbox"]
            if getattr(self, name, None) is not None
        ]
        # Add browser tools
        tools.extend(self.browser_tools.keys())
        return tools

    def register_browser_tool(self, name: str, tool: Any) -> None:
        """Register a browser-based tool (ChatGPT, Grok, etc.)."""
        self.browser_tools[name] = tool


# =============================================================================
# Null/Stub Tool Implementations for Testing
# =============================================================================

class NullWebSearchTool:
    """Null implementation that returns empty results."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        language: str = "en",
        timeout: int = 30
    ) -> WebSearchResult:
        return WebSearchResult(text="", url=None, confidence=0.0)


class NullWebFetchTool:
    """Null implementation that returns empty results."""

    def fetch(
        self,
        url: str,
        *,
        timeout: int = 30,
        max_chars: int = 8000
    ) -> WebFetchResult:
        return WebFetchResult(
            text="",
            url=url,
            status_code=0,
            success=False,
            error="Web fetch not available"
        )


class NullLLMTool:
    """Null implementation that returns empty results."""

    @property
    def enabled(self) -> bool:
        return False

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 500,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMCompletionResult:
        return LLMCompletionResult(
            ok=False,
            text="",
            source="null",
            llm_used=False,
            error="LLM not available"
        )


class NullShellTool:
    """Null implementation that denies all commands."""

    def run(
        self,
        cmd: str,
        *,
        cwd: Optional[str] = None,
        timeout: int = 120,
        check_policy: bool = True
    ) -> ShellResult:
        return ShellResult(
            status="denied",
            error="Shell execution not available"
        )


class NullGitTool:
    """Null implementation that returns empty results."""

    def status(self, *, short: bool = True, porcelain: bool = False) -> str:
        return ""

    def status_detailed(self) -> GitStatusResult:
        return GitStatusResult(
            branch=None,
            ahead=0,
            behind=0,
            staged=[],
            modified=[],
            untracked=[],
            deleted=[],
            is_clean=True
        )

    def diff(
        self,
        *,
        cached: bool = False,
        file_path: Optional[str] = None,
        commit: Optional[str] = None
    ) -> str:
        return ""

    def log(
        self,
        *,
        max_count: int = 10,
        oneline: bool = False
    ) -> List[GitLogEntry]:
        return []

    def add(self, paths: List[str]) -> GitResult:
        return GitResult(ok=False, output="", error="Git not available")

    def commit(self, message: str, *, allow_empty: bool = False) -> GitResult:
        return GitResult(ok=False, output="", error="Git not available")

    def current_branch(self) -> str:
        return ""

    def is_clean(self) -> bool:
        return True


class NullPythonSandboxTool:
    """Null implementation that denies execution."""

    def execute(
        self,
        code: str,
        *,
        timeout_ms: int = 3000,
        cwd: Optional[str] = None
    ) -> SandboxResult:
        return SandboxResult(
            ok=False,
            stdout="",
            stderr="",
            returncode=-1,
            error="Python sandbox not available"
        )


def create_null_tools() -> ToolRegistry:
    """Create a ToolRegistry with all null implementations."""
    return ToolRegistry(
        web_search=NullWebSearchTool(),
        web_fetch=NullWebFetchTool(),
        llm=NullLLMTool(),
        shell=NullShellTool(),
        git=NullGitTool(),
        python_sandbox=NullPythonSandboxTool()
    )
