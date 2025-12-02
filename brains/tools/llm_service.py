"""
LLM Service Facade
==================

This module provides a facade for LLM operations that delegates to the
host-provided LLM tool. It maintains backward compatibility with existing
code while ensuring no direct network I/O occurs in brains.

IMPORTANT: This module should not perform direct HTTP operations.
All LLM operations are delegated to the tool registry.

For direct LLM access, use host_tools.llm_client.client directly
from the host runtime (not from brains).
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from brains.maven_paths import get_brains_path, get_maven_root
from brains.tools_api import (
    LLMTool,
    LLMCompletionResult,
    NullLLMTool,
    ToolRegistry,
)


# Global tool registry - set by host runtime
_tool_registry: Optional[ToolRegistry] = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry (called by host runtime)."""
    global _tool_registry
    _tool_registry = registry


def get_llm_tool() -> LLMTool:
    """Get the LLM tool from the registry."""
    if _tool_registry and _tool_registry.llm:
        return _tool_registry.llm
    return NullLLMTool()


class OllamaLLMService:
    """LLM service facade that delegates to the host-provided LLM tool.

    This class maintains the same interface as the original OllamaLLMService
    but delegates all HTTP operations to the tool registry. Pattern learning
    state is still maintained locally using file I/O (which is stdlib-only).
    """

    def __init__(self) -> None:
        self.config: dict = self._load_config()
        self.enabled: bool = bool(self.config.get("enabled", True))
        self.patterns_dir = get_brains_path("personal", "memory", "learned_patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.pattern_stats = self.patterns_dir / "pattern_stats.json"

    def _load_config(self) -> dict:
        """Load LLM configuration from config/llm.json if present."""
        cfg_path = get_maven_root() / "config" / "llm.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or {}
                return data
            except Exception:
                pass
        return {
            "enabled": True,
            "provider": "ollama",
            "ollama_url": "http://localhost:11434",
            "model": "llama3.2",
            "learning": {
                "enabled": True,
                "min_interactions_to_learn": 10,
                "similarity_threshold": 0.8,
                "min_pattern_occurrences": 3,
            },
        }

    def call(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        context: dict | None = None,
    ) -> dict:
        """Call the LLM via the host-provided tool.

        This method delegates to the tool registry instead of performing
        direct HTTP operations. The host runtime must inject the LLM tool
        before this function can make actual LLM calls.
        """
        if not self.enabled:
            return {"ok": False, "error": "LLM disabled"}

        tool = get_llm_tool()
        result = tool.complete(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            context=context
        )

        return {
            "ok": result.ok,
            "text": result.text,
            "source": result.source,
            "llm_used": result.llm_used,
            "confidence": result.confidence,
            "error": result.error,
            "error_type": result.error_type,
        }

    def learn_patterns(self) -> None:
        """Trigger pattern learning on the host tool if available."""
        tool = get_llm_tool()
        if hasattr(tool, "learn_patterns"):
            tool.learn_patterns()

    def get_learning_stats(self) -> dict:
        """Return current learning statistics."""
        if self.pattern_stats.exists():
            try:
                with open(self.pattern_stats, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
            except Exception:
                pass
        return {"templates": 0, "interactions": 0}


# Instantiate a global service for convenience
llm_service = OllamaLLMService()