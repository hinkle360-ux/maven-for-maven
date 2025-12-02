"""
Web Client Facade
=================

This module provides a facade for web search operations that delegates
to the host-provided web tools. It maintains backward compatibility
with existing code while ensuring no direct network I/O occurs in brains.

IMPORTANT: This module should not perform direct HTTP operations.
All web operations are delegated to the tool registry.

For direct HTTP access, use host_tools.web_client.client directly
from the host runtime (not from brains).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Import the abstract tool interface (stdlib-only)
from brains.tools_api import (
    WebSearchTool,
    WebSearchResult,
    NullWebSearchTool,
    ToolRegistry,
)


# Global tool registry - set by host runtime
_tool_registry: Optional[ToolRegistry] = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry (called by host runtime)."""
    global _tool_registry
    _tool_registry = registry


def get_web_search_tool() -> WebSearchTool:
    """Get the web search tool from the registry."""
    if _tool_registry and _tool_registry.web_search:
        return _tool_registry.web_search
    return NullWebSearchTool()


def _web_enabled() -> bool:
    """Check if web research is enabled via environment."""
    try:
        env_override = os.getenv("MAVEN_ENABLE_WEB_RESEARCH")
        if env_override is not None:
            return env_override.strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        pass

    # Try to load from config
    try:
        from api.utils import CFG
        legacy_cfg = bool((CFG.get("web_research") or {}).get("enabled", False))
        return bool(CFG.get("ENABLE_WEB_RESEARCH", True) or legacy_cfg)
    except Exception:
        return True  # Default to enabled


def search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Performs web search via the host-provided tool.

    This function delegates to the tool registry instead of performing
    direct HTTP operations. The host runtime must inject the web search
    tool before this function can return meaningful results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Dict with 'text', 'url', and 'confidence' keys
    """
    if not _web_enabled():
        return {"text": "", "url": None, "confidence": 0.0}

    print(f"[RESEARCH_WEB] Searching web for '{query}' (max_results={max_results})")

    tool = get_web_search_tool()
    result = tool.search(query, max_results=max_results)

    return {
        "text": result.text,
        "url": result.url,
        "confidence": result.confidence
    }

