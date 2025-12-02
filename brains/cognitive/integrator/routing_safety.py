"""
routing_safety.py
~~~~~~~~~~~~~~~~~

Safety invariants for smart routing.

This module enforces hard safety rules that MUST be respected by all
routing decisions. These rules cannot be overridden by Teacher or
pattern matching.

INVARIANTS:
1. SELF-INTENT GATE: All capability/self/history questions MUST route
   to self_model (NOT Teacher). This prevents hallucinated capability claims.

2. CAPABILITY BOUNDARY: No routing plan may enable operations beyond
   what execution_guard + system_capabilities allow.

3. TOOL VALIDATION: All suggested tools must exist and be enabled.
   Non-existent tools are silently filtered.

4. CRASH PREVENTION: If routing classification fails, fallback to safe
   defaults (language brain). Never crash the system.

Usage:
    from brains.cognitive.integrator.routing_safety import (
        validate_routing_plan,
        is_self_intent_query,
        enforce_capability_boundary,
    )

    # Validate a routing plan
    plan = validate_routing_plan(plan, capability_snapshot)

    # Check if query is self-intent
    if is_self_intent_query(query):
        # MUST route to self_model, NOT Teacher
        pass
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Self-intent detection patterns
# These MUST route to self_model, NOT Teacher
SELF_INTENT_PATTERNS = [
    # Capability questions
    r"\bcan you\b",
    r"\bare you able\b",
    r"\bdo you have\b.*\b(access|ability|capability|tool)\b",
    r"\bwhat can you do\b",
    r"\bwhat are your capabilities\b",
    r"\bwhat tools do you have\b",
    r"\bcan you browse\b",
    r"\bcan you search\b",
    r"\bcan you run\b",
    r"\bcan you execute\b",
    r"\bcan you read\b",
    r"\bcan you write\b",
    r"\bcan you access\b",
    r"\bcan you control\b",

    # Self-identity questions
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\btell me about yourself\b",
    r"\bdescribe yourself\b",
    r"\bwhat is your name\b",
    r"\bare you (maven|an llm|chatgpt|claude|gpt)\b",
    r"\bwho (made|created|built) you\b",
    r"\bwhat is your purpose\b",
    r"\bhow do you work\b",
    r"\byour (creator|architect|developer)\b",
    r"\bwhat are your brains\b",
    r"\bdescribe your architecture\b",

    # Self-explanation / introduction requests (with typo tolerance)
    r"\bexplain[e]?\s*your\s*self\b",  # explain/explaine your self
    r"\bexplain[e]?\s*yourself\b",      # explain/explaine yourself
    r"\bexpl[ai]+n[e]?\s*your\s*self\b",  # explian/explaain variants
    r"\bintroduce yourself\b",
    r"\bintroduce your\s*self\b",
    r"\bwho\s*am\s*i\b.*\byou\b",  # "who am I talking to"
    r"\bwhoami\b",
    r"\btell .* about yourself\b",
    r"\btell .* who you are\b",
    r"\bexplain[e]? .* who you are\b",  # typo tolerance
    r"\bdescribe .* who you are\b",
    r"\bexplain[e]?\s*yourself to\b",  # typo tolerance
    r"\bintroduce yourself to\b",
    r"\byour\s*self\b",  # Catch standalone "your self" phrases

    # TASK 1: Internal state / feelings / preferences questions
    # These MUST route to self_model, NOT Teacher
    r"\bhow do you feel\b",
    r"\bhow are you feeling\b",
    r"\bdo you have (feelings|emotions)\b",
    r"\bare you (happy|sad|conscious|sentient|alive|real)\b",
    r"\bdo you (like|enjoy|prefer|want)\b",
    r"\bwhat do you (like|want|prefer|think about yourself)\b",
    r"\bdo you have (preferences|opinions)\b",
    r"\bwhat is your opinion\b",
    r"\bhow do you think\b",
    r"\bare you a person\b",
    r"\bdo you (dream|sleep|get tired|get bored)\b",

    # History questions
    r"\bwhat did (we|i) (discuss|talk|ask|say)\b",
    r"\bour (previous |last )?(conversation|discussion)\b",
    r"\bearlier you (said|mentioned)\b",
    r"\bremember when\b",
    r"\b(yesterday|last time) we\b",
    r"\bdo you remember\b",
    r"\bwhat did you (say|answer|respond)\b",
]

# Compiled patterns for efficiency
_COMPILED_SELF_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SELF_INTENT_PATTERNS]


def is_self_intent_query(query: str) -> bool:
    """
    Check if a query is a self-intent question.

    Self-intent queries MUST route to self_model (NOT Teacher).
    This prevents Teacher from hallucinating capabilities.

    Args:
        query: The user's query text

    Returns:
        True if this is a self-intent query
    """
    if not query:
        return False

    for pattern in _COMPILED_SELF_PATTERNS:
        if pattern.search(query):
            return True

    return False


def get_self_intent_type(query: str) -> Optional[str]:
    """
    Determine the type of self-intent query.

    Args:
        query: The user's query text

    Returns:
        "capability", "identity", "introduction", "history", or None
    """
    if not query:
        return None

    query_lower = query.lower()

    # Check introduction patterns FIRST (most specific)
    # These trigger SELF_INTRODUCTION operation
    # Include common typo variants like "explaine"
    introduction_markers = [
        "explain your self", "explain yourself", "introduce yourself",
        "introduce your self", "whoami", "explain yourself to",
        "introduce yourself to", "tell grok", "tell chatgpt",
        "explain to grok", "explain to chatgpt", "explain to x",
        # Typo variants
        "explaine your self", "explaine yourself", "explaine yourself to",
        "your self",  # standalone "your self" often means self-intro
    ]
    # Also check regex for more typo variants
    intro_regex = r"\bexpl[ai]+n[e]?\s*(your\s*self|yourself)\b"
    if any(m in query_lower for m in introduction_markers) or re.search(intro_regex, query_lower):
        return "introduction"

    # Check capability patterns
    capability_markers = [
        "can you", "are you able", "do you have", "what can you do",
        "what tools", "capabilities",
    ]
    if any(m in query_lower for m in capability_markers):
        return "capability"

    # Check identity patterns (including feelings/preferences - TASK 1)
    identity_markers = [
        "who are you", "what are you", "yourself", "your name",
        "who made", "who created", "your purpose", "how do you work",
        # Internal state / feelings / preferences
        "how do you feel", "how are you feeling", "do you have feelings",
        "do you have emotions", "are you happy", "are you sad",
        "are you conscious", "are you sentient", "are you alive",
        "do you like", "do you enjoy", "do you prefer", "what do you like",
        "what do you want", "what do you prefer", "do you have preferences",
        "do you have opinions", "what is your opinion", "how do you think",
        "are you real", "are you a person", "do you dream", "do you sleep",
        "do you get tired", "do you get bored",
    ]
    if any(m in query_lower for m in identity_markers):
        return "identity"

    # Check history patterns
    history_markers = [
        "what did we", "what did i", "our conversation",
        "earlier you", "remember when", "yesterday", "last time",
    ]
    if any(m in query_lower for m in history_markers):
        return "history"

    return None


def get_introduction_target(query: str) -> Optional[str]:
    """
    Extract the target platform for self-introduction.

    Detects patterns like "explain yourself to grok" and returns the target.

    Args:
        query: The user's query text

    Returns:
        Target platform name ("grok", "chatgpt", "x") or None
    """
    if not query:
        return None

    query_lower = query.lower()

    # Patterns: "explain yourself to X", "introduce yourself to X", "tell X about yourself"
    target_patterns = [
        r"(?:explain|introduce)\s+(?:your\s*self|yourself)\s+to\s+(\w+)",
        r"tell\s+(\w+)\s+(?:about\s+yourself|who\s+you\s+are)",
        r"(?:explain|describe)\s+(?:to\s+)?(\w+)\s+who\s+you\s+are",
        r"maven\s+(?:explain|introduce)\s+(?:your\s*self|yourself)\s+to\s+(\w+)",
        r"(\w+)\s+maven\s+(?:explain|introduce)\s+(?:your\s*self|yourself)",
    ]

    for pattern in target_patterns:
        match = re.search(pattern, query_lower)
        if match:
            target = match.group(1)
            # Normalize common targets
            if target in ("grok", "x", "twitter"):
                return "grok"
            elif target in ("chatgpt", "openai", "gpt"):
                return "chatgpt"
            elif target in ("claude", "anthropic"):
                return "claude"
            else:
                return target

    return None


def validate_routing_plan(
    plan_brains: List[str],
    plan_tools: List[str],
    capability_snapshot: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate a routing plan against capability boundaries.

    This enforces INVARIANT 2: No routing plan may enable operations
    beyond what execution_guard + system_capabilities allow.

    Args:
        plan_brains: Suggested brain list
        plan_tools: Suggested tool list
        capability_snapshot: Current system capabilities

    Returns:
        Tuple of (valid_brains, valid_tools, violations)
    """
    violations = []

    # Get execution mode
    exec_mode = capability_snapshot.get("execution_mode", "UNKNOWN")
    web_enabled = capability_snapshot.get("web_research_enabled", False)
    tools_available = set(capability_snapshot.get("tools_available", []))

    valid_brains = list(plan_brains)
    valid_tools = []

    # Validate tools
    for tool in plan_tools:
        tool_base = tool.split(".")[0] if "." in tool else tool

        # Check if tool is available
        if tool not in tools_available and tool_base not in tools_available:
            violations.append(f"Tool not available: {tool}")
            continue

        # Check web tools against web_enabled
        if "web" in tool.lower() and not web_enabled:
            violations.append(f"Web disabled, cannot use: {tool}")
            continue

        # Check execution tools against exec_mode
        if tool_base in ("shell", "python_sandbox") and exec_mode in ("DISABLED", "SAFE_CHAT"):
            violations.append(f"Execution disabled, cannot use: {tool}")
            continue

        valid_tools.append(tool)

    # Validate brains based on mode
    if exec_mode == "SAFE_CHAT":
        # No action brains in SAFE_CHAT mode
        action_brains = {"action_engine", "coder"}
        removed = [b for b in valid_brains if b in action_brains]
        if removed:
            violations.append(f"SAFE_CHAT mode, removed brains: {removed}")
            valid_brains = [b for b in valid_brains if b not in action_brains]

    return valid_brains, valid_tools, violations


def enforce_capability_boundary(
    query: str,
    plan_brains: List[str],
    plan_tools: List[str],
    capability_snapshot: Dict[str, Any],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Enforce all capability boundaries on a routing plan.

    This is the main safety function that should be called before
    executing any routing plan.

    Args:
        query: The original query
        plan_brains: Suggested brain list
        plan_tools: Suggested tool list
        capability_snapshot: Current system capabilities

    Returns:
        Tuple of (final_brains, final_tools, audit_notes)
    """
    audit_notes = []

    # INVARIANT 1: Self-intent queries MUST route to self_model
    if is_self_intent_query(query):
        intent_type = get_self_intent_type(query)
        audit_notes.append(f"SELF_INTENT_GATE: Detected {intent_type} query, routing to self_model")

        # Override to self_model
        if intent_type == "history":
            return ["self_model", "system_history"], [], audit_notes
        elif intent_type == "introduction":
            # Check if there's a target platform (e.g., "explain yourself to grok")
            target = get_introduction_target(query)
            if target:
                audit_notes.append(f"SELF_INTENT_GATE: Introduction target detected: {target}")
            return ["self_model"], [], audit_notes
        else:
            return ["self_model"], [], audit_notes

    # INVARIANT 2: Validate against capability boundaries
    valid_brains, valid_tools, violations = validate_routing_plan(
        plan_brains, plan_tools, capability_snapshot
    )

    if violations:
        audit_notes.extend(violations)

    # INVARIANT 4: Ensure we always have at least one brain
    if not valid_brains:
        valid_brains = ["language"]
        audit_notes.append("CRASH_PREVENTION: Defaulted to language brain")

    return valid_brains, valid_tools, audit_notes


def safe_routing_fallback() -> Tuple[List[str], List[str]]:
    """
    Return safe routing defaults when all else fails.

    This ensures INVARIANT 4: Never crash the system.

    Returns:
        Tuple of (default_brains, default_tools)
    """
    return ["language"], []


# Export for use in smart_routing.py
__all__ = [
    "is_self_intent_query",
    "get_self_intent_type",
    "get_introduction_target",
    "validate_routing_plan",
    "enforce_capability_boundary",
    "safe_routing_fallback",
    "SELF_INTENT_PATTERNS",
]
