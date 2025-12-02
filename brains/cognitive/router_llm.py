"""
Router LLM - Dedicated Routing Decision Layer
==============================================

This module provides a dedicated router that uses an LLM to make routing
decisions. Unlike the general Teacher/reasoning flow, this router:

1. Outputs ONLY JSON (brains + tools + confidence)
2. Has a focused prompt with tool/brain catalog
3. Can incorporate few-shot examples from routing history
4. Provides explicit confidence scores for precedence decisions

The router is called AFTER the grammar layer but BEFORE learned patterns,
only when the grammar doesn't match.

Usage:
    from brains.cognitive.router_llm import route_with_llm

    result = route_with_llm("hello from maven", capability_snapshot)
    if result.confidence >= 0.75:
        # Use router's decision
        brains = result.brains
        tools = result.tools
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Try to import LLM client
try:
    from host_tools.llm_client.llm_service import LLMClient
    _llm_available = True
except ImportError:
    _llm_available = False
    LLMClient = None  # type: ignore


@dataclass
class RouterDecision:
    """Result of LLM routing decision."""

    brains: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    source: str = "router_llm"
    raw_response: str = ""
    parse_error: Optional[str] = None


# =============================================================================
# ROUTER PROMPT TEMPLATE
# =============================================================================

ROUTER_PROMPT = '''You are the ROUTER for Maven, an offline cognitive system.
Your ONLY job is to decide which brains and tools should handle a user message.

## Available Brains
- language: Normal chat, QA, conversation. DEFAULT when no tools needed.
- research_manager: Deep web research on topics. Use with web_search/web_fetch tools.
- reasoning: Logical reasoning, problem solving, analysis.
- coder: Code generation, programming tasks.
- self_model: Questions about Maven itself (who are you, what can you do).
- memory_librarian: Retrieving stored facts and memories.
- planner: Task planning and goal decomposition.

## Available Tools
- x: X.com/Twitter actions. Subcommands: grok (chat with Grok AI), post (post tweet), search.
- chatgpt: Chat with ChatGPT via browser.
- browser_open: Open a URL in browser.
- web_search: Search the web for information.
- web_fetch: Fetch content from a URL.
- shell: Execute shell commands (if enabled).
- python_sandbox: Run Python code (if enabled).
- git: Git operations (if enabled).
- fs: Filesystem operations (if enabled).

## Tool Availability (current session)
{tool_availability}

## Few-Shot Examples
{examples}

## Instructions
1. Analyze the user message
2. Decide which brains and tools are needed
3. If message mentions "grok", "x grok", "ask grok" → tools: ["x"], subcommand: "grok"
4. If message mentions "post to x", "tweet" → tools: ["x"], subcommand: "post"
5. If message is about research/lookup → brains: ["research_manager"], tools: ["web_search"]
6. If message is casual chat/question → brains: ["language"], tools: []
7. If message asks about Maven itself → brains: ["self_model"], tools: []

Respond ONLY with a JSON object, no other text:
{{"brains": [...], "tools": [...], "subcommand": "...", "confidence": 0.0-1.0, "reason": "..."}}

User message: "{user_message}"

JSON response:'''


# Default few-shot examples
DEFAULT_EXAMPLES = [
    {
        "input": "x grok hello from maven",
        "output": {"brains": ["language"], "tools": ["x"], "subcommand": "grok", "confidence": 0.95, "reason": "Explicit grok command via x tool"}
    },
    {
        "input": "research: AI safety developments",
        "output": {"brains": ["research_manager"], "tools": ["web_search", "web_fetch"], "confidence": 0.95, "reason": "Explicit research command"}
    },
    {
        "input": "what's the weather like?",
        "output": {"brains": ["language"], "tools": [], "confidence": 0.9, "reason": "General question, no tools needed"}
    },
    {
        "input": "who are you?",
        "output": {"brains": ["self_model"], "tools": [], "confidence": 0.95, "reason": "Self-identity question routes to self_model"}
    },
    {
        "input": "post to x: Hello world!",
        "output": {"brains": ["language"], "tools": ["x"], "subcommand": "post", "confidence": 0.95, "reason": "Explicit post to X command"}
    },
    {
        "input": "write a python function to sort a list",
        "output": {"brains": ["coder", "reasoning"], "tools": [], "confidence": 0.9, "reason": "Coding request"}
    },
]


def _format_examples(examples: List[Dict[str, Any]]) -> str:
    """Format examples for the prompt."""
    lines = []
    for i, ex in enumerate(examples[:8], 1):  # Limit to 8 examples
        inp = ex.get("input", "")
        out = ex.get("output", {})
        lines.append(f"Example {i}:")
        lines.append(f'  Input: "{inp}"')
        lines.append(f"  Output: {json.dumps(out)}")
        lines.append("")
    return "\n".join(lines)


def _format_tool_availability(capability_snapshot: Dict[str, Any]) -> str:
    """Format tool availability for the prompt."""
    tools = capability_snapshot.get("tools_available", [])
    exec_mode = capability_snapshot.get("execution_mode", "UNKNOWN")
    web_enabled = capability_snapshot.get("web_research_enabled", False)

    lines = [
        f"Execution mode: {exec_mode}",
        f"Web research: {'enabled' if web_enabled else 'disabled'}",
        f"Available tools: {', '.join(tools) if tools else 'none'}",
    ]
    return "\n".join(lines)


def _parse_router_response(response: str) -> RouterDecision:
    """Parse the LLM's JSON response."""
    decision = RouterDecision(raw_response=response)

    # Try to extract JSON from response
    try:
        # First try direct parse
        data = json.loads(response.strip())
    except json.JSONDecodeError:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                decision.parse_error = f"JSON parse error: {e}"
                decision.confidence = 0.0
                return decision
        else:
            decision.parse_error = "No JSON found in response"
            decision.confidence = 0.0
            return decision

    # Extract fields
    decision.brains = data.get("brains", ["language"])
    decision.tools = data.get("tools", [])
    decision.confidence = float(data.get("confidence", 0.5))
    decision.reason = data.get("reason", "")

    # Validate brains
    valid_brains = {
        "language", "research_manager", "reasoning", "coder",
        "self_model", "memory_librarian", "planner", "integrator",
        "sensorium", "affect_priority", "action_engine"
    }
    decision.brains = [b for b in decision.brains if b in valid_brains]
    if not decision.brains:
        decision.brains = ["language"]

    return decision


def route_with_llm(
    user_message: str,
    capability_snapshot: Dict[str, Any],
    examples: Optional[List[Dict[str, Any]]] = None,
) -> RouterDecision:
    """
    Route a user message using the LLM router.

    Args:
        user_message: The user's input
        capability_snapshot: Current system capabilities
        examples: Optional few-shot examples (uses defaults if None)

    Returns:
        RouterDecision with brains, tools, and confidence
    """
    if not _llm_available:
        return RouterDecision(
            brains=["language"],
            confidence=0.0,
            parse_error="LLM client not available"
        )

    # Use provided examples or defaults
    if examples is None:
        examples = DEFAULT_EXAMPLES

    # Build prompt
    prompt = ROUTER_PROMPT.format(
        tool_availability=_format_tool_availability(capability_snapshot),
        examples=_format_examples(examples),
        user_message=user_message[:500],  # Truncate long messages
    )

    try:
        # Call LLM
        client = LLMClient()
        response = client.complete(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,  # Low temperature for deterministic routing
        )

        if not response:
            return RouterDecision(
                brains=["language"],
                confidence=0.0,
                parse_error="Empty LLM response"
            )

        # Parse response
        return _parse_router_response(response)

    except Exception as e:
        return RouterDecision(
            brains=["language"],
            confidence=0.0,
            parse_error=f"LLM error: {str(e)[:100]}"
        )


def get_router_schema() -> Dict[str, Any]:
    """
    Get the schema for router output.

    This can be used to validate router responses or for documentation.
    """
    return {
        "type": "object",
        "properties": {
            "brains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of brain names to route to"
            },
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tool names to use"
            },
            "subcommand": {
                "type": "string",
                "description": "Optional subcommand for tools (e.g., 'grok' for x tool)"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score for this routing decision"
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of routing decision"
            }
        },
        "required": ["brains", "tools", "confidence"]
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RouterDecision",
    "route_with_llm",
    "get_router_schema",
    "DEFAULT_EXAMPLES",
]
