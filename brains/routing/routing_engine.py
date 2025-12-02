"""
Routing Engine - Unified Entry Point for All Routing Decisions
==============================================================

This module is the single entry point for routing user messages to brains/tools.
It orchestrates the layered routing system with strict precedence:

ROUTING PRECEDENCE (in order):
1. GRAMMAR (Priority -1) - Deterministic regex patterns for explicit commands
   - "x grok hello" → x tool with grok subcommand
   - "research: AI" → research_manager brain
   - ALWAYS respected, cannot be overridden

2. SELF-INTENT GATE (Priority 0) - Self/capability questions
   - "who are you" → self_model brain
   - "can you browse" → self_model brain
   - Prevents Teacher from hallucinating capabilities

3. ROUTER LLM (Priority 0.5) - JSON-only routing LLM
   - Called when grammar doesn't match
   - Returns confidence score
   - Only used if confidence >= threshold

4. LEARNED PATTERNS (Priority 1) - Pattern store / routing_learning
   - Historical patterns from routing_examples
   - Lowest priority, easily overridden

5. SAFE DEFAULT (Fallback) - language brain
   - Used when all else fails
   - Never crashes the system

Usage:
    from brains.routing.routing_engine import build_routing_plan

    decision = build_routing_plan(
        query="x grok hello from maven",
        capability_snapshot=snapshot,
    )

    # decision.brains = ["language"]
    # decision.tools = ["x"]
    # decision.subcommand = "grok"
    # decision.confidence = 1.0
    # decision.source = "grammar:x_grok"
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from brains.routing.router_schema import (
    RoutingDecision,
    ParsedCommand,
    LLMRouterResult,
    RoutingContext,
    RoutingExample,
    VALID_BRAINS,
)

# Text normalizer for handling typos
try:
    from brains.routing.normalizer import normalize, NormalizationResult
    _normalizer_available = True
except ImportError:
    _normalizer_available = False
    normalize = None  # type: ignore


# =============================================================================
# GRAMMAR PATTERNS (Priority -1)
# =============================================================================

# Pattern: x grok <message> or x: grok <message>
PATTERN_X_GROK = re.compile(
    r"^x\s*[:\s]+grok\s+(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: grok <message> (standalone grok command)
PATTERN_GROK_STANDALONE = re.compile(
    r"^grok\s+(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: x post <message> or post to x: <message>
PATTERN_X_POST = re.compile(
    r"^(?:x\s+post|post\s+(?:to\s+)?x)\s*[:\s]*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: research: <topic> or research: "<topic>" web:N
PATTERN_RESEARCH = re.compile(
    r'^research\s*:\s*(?:"([^"]+)"|(.+?))(?:\s+web\s*:\s*(\d+))?$',
    re.IGNORECASE | re.DOTALL
)

# Pattern: browser_open: <url> or open <url>
PATTERN_BROWSER_OPEN = re.compile(
    r"^(?:browser_open|open)\s*:\s*(https?://\S+)$",
    re.IGNORECASE
)

# Pattern: search: <query> or web search: <query>
PATTERN_WEB_SEARCH = re.compile(
    r"^(?:web\s+)?search\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: use <tool> tool: <args>
PATTERN_EXPLICIT_TOOL = re.compile(
    r"^use\s+(\w+)\s+tool\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: use grok: <message> or use grok <message> (direct grok command)
# This is a shorthand for "x grok: <message>"
PATTERN_USE_GROK = re.compile(
    r"^use\s+grok\s*[:\s]+(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: use chatgpt: <message> or use chatgpt <message>
PATTERN_USE_CHATGPT = re.compile(
    r"^use\s+chatgpt\s*[:\s]+(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: chatgpt <message> (standalone chatgpt command)
PATTERN_CHATGPT = re.compile(
    r"^chatgpt\s+(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: x <subcommand> <args> (generic x command)
PATTERN_X_GENERIC = re.compile(
    r"^x\s*[:\s]+(\w+)\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: shell: <command> or run: <command>
PATTERN_SHELL = re.compile(
    r"^(?:shell|run)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: python: <code>
PATTERN_PYTHON = re.compile(
    r"^python\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: code: <request>
PATTERN_CODE = re.compile(
    r"^code\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: maven <action> (direct maven commands)
PATTERN_MAVEN_COMMAND = re.compile(
    r"^maven\s+(explain|introduce|whoami|scan|status)\s*(.*)$",
    re.IGNORECASE
)


# =============================================================================
# SELF-INTENT PATTERNS (Priority 0)
# =============================================================================

SELF_INTENT_PATTERNS = [
    # Capability questions
    r"\bcan you\b",
    r"\bare you able\b",
    r"\bdo you have\b.*\b(access|ability|capability|tool)\b",
    r"\bwhat can you do\b",
    r"\bwhat are your capabilities\b",
    r"\bwhat tools do you have\b",

    # Self-identity questions
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\btell me about yourself\b",
    r"\bdescribe yourself\b",
    r"\bwhat is your name\b",
    r"\bare you (maven|an llm|chatgpt|claude|gpt)\b",
    r"\bwho (made|created|built) you\b",

    # Self-introduction requests (with common typo variants)
    r"\bexplain[e]?\s*your\s*self\b",  # explain/explaine your self
    r"\bexplain[e]?\s*yourself\b",      # explain/explaine yourself
    r"\bexpl[ai]+n[e]?\s*your\s*self\b",  # explian/explaain variants
    r"\bintroduce yourself\b",
    r"\bwhoami\b",
    r"\byour\s*self\b",  # Catch standalone "your self" phrases

    # Internal state questions
    r"\bhow do you feel\b",
    r"\bdo you have (feelings|emotions)\b",
    r"\bare you (happy|sad|conscious|sentient)\b",
    r"\bdo you (like|enjoy|prefer)\b",

    # History questions
    r"\bwhat did (we|i) (discuss|talk|ask)\b",
    r"\bdo you remember\b",
]

_COMPILED_SELF_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SELF_INTENT_PATTERNS]


# =============================================================================
# GRAMMAR LAYER (Priority -1)
# =============================================================================

def _parse_grammar(query: str) -> Optional[ParsedCommand]:
    """
    Parse user input against the hard grammar.

    This is Priority -1 - matches here ALWAYS take precedence.

    Args:
        query: Raw user input text

    Returns:
        ParsedCommand if matched, None otherwise
    """
    if not query:
        return None

    # Normalize: strip, collapse whitespace
    text = query.strip()
    text_normalized = re.sub(r'\s+', ' ', text)

    # Try patterns in order (most specific first)

    # 1. x grok <message>
    match = PATTERN_X_GROK.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["x"],
            brains=["language"],
            subcommand="grok",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="x_grok",
            metadata={"platform": "x", "action": "grok_chat"}
        )

    # 2. grok <message> (standalone)
    match = PATTERN_GROK_STANDALONE.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["x"],
            brains=["language"],
            subcommand="grok",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="grok_standalone",
            metadata={"platform": "x", "action": "grok_chat"}
        )

    # 2.5. use grok: <message> (shorthand for x grok)
    match = PATTERN_USE_GROK.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["x"],
            brains=["language"],
            subcommand="grok",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="use_grok",
            metadata={"platform": "x", "action": "grok_chat"}
        )

    # 2.6. use chatgpt: <message>
    match = PATTERN_USE_CHATGPT.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["chatgpt"],
            brains=["language"],
            subcommand="chat",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="use_chatgpt",
            metadata={"platform": "chatgpt", "action": "chat"}
        )

    # 3. x post / post to x
    match = PATTERN_X_POST.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["x"],
            brains=["language"],
            subcommand="post",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="x_post",
            metadata={"platform": "x", "action": "post"}
        )

    # 4. research: <topic>
    match = PATTERN_RESEARCH.match(text_normalized)
    if match:
        topic = match.group(1) or match.group(2)
        web_limit = int(match.group(3)) if match.group(3) else 10
        return ParsedCommand(
            intent="research",
            tools=["web_search", "web_fetch"],
            brains=["research_manager", "reasoning"],
            args=topic.strip() if topic else "",
            raw_input=text,
            matched_pattern="research",
            metadata={"web_limit": web_limit, "mode": "deep_research"}
        )

    # 5. browser_open: <url>
    match = PATTERN_BROWSER_OPEN.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_open",
            tools=["browser_open"],
            brains=["language"],
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="browser_open",
            metadata={"url": match.group(1).strip()}
        )

    # 6. search: <query>
    match = PATTERN_WEB_SEARCH.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="web_search",
            tools=["web_search"],
            brains=["research_manager"],
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="web_search",
            metadata={"query": match.group(1).strip()}
        )

    # 7. use <tool> tool: <args>
    match = PATTERN_EXPLICIT_TOOL.match(text_normalized)
    if match:
        tool_name = match.group(1).lower()
        return ParsedCommand(
            intent="explicit_tool",
            tools=[tool_name],
            brains=["language"],
            args=match.group(2).strip(),
            raw_input=text,
            matched_pattern="explicit_tool",
            metadata={"explicit_tool": tool_name}
        )

    # 8. chatgpt <message>
    match = PATTERN_CHATGPT.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="browser_tool",
            tools=["chatgpt"],
            brains=["language"],
            subcommand="chat",
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="chatgpt",
            metadata={"platform": "chatgpt", "action": "chat"}
        )

    # 9. x <subcommand> <args> (generic)
    match = PATTERN_X_GENERIC.match(text_normalized)
    if match:
        subcommand = match.group(1).lower()
        args = match.group(2).strip() if match.group(2) else ""
        return ParsedCommand(
            intent="browser_tool",
            tools=["x"],
            brains=["language"],
            subcommand=subcommand,
            args=args,
            raw_input=text,
            matched_pattern="x_generic",
            metadata={"platform": "x", "action": subcommand}
        )

    # 10. shell: <command>
    match = PATTERN_SHELL.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="shell_execution",
            tools=["shell"],
            brains=["coder"],
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="shell",
            metadata={"command": match.group(1).strip()}
        )

    # 11. python: <code>
    match = PATTERN_PYTHON.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="python_execution",
            tools=["python_sandbox"],
            brains=["coder"],
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="python",
            metadata={"code": match.group(1).strip()}
        )

    # 12. code: <request>
    match = PATTERN_CODE.match(text_normalized)
    if match:
        return ParsedCommand(
            intent="coding",
            tools=[],
            brains=["coder", "reasoning"],
            args=match.group(1).strip(),
            raw_input=text,
            matched_pattern="code",
            metadata={"request": match.group(1).strip()}
        )

    # 13. maven <action>
    match = PATTERN_MAVEN_COMMAND.match(text_normalized)
    if match:
        action = match.group(1).lower()
        extra = match.group(2).strip() if match.group(2) else ""

        if action in ("explain", "introduce", "whoami"):
            return ParsedCommand(
                intent="self_introduction",
                tools=[],
                brains=["self_model"],
                subcommand="introduction",
                args=extra,
                raw_input=text,
                matched_pattern="maven_command",
                metadata={"action": action}
            )
        elif action == "scan":
            return ParsedCommand(
                intent="self_scan",
                tools=[],
                brains=["self_model"],
                subcommand="scan",
                args=extra or "self",
                raw_input=text,
                matched_pattern="maven_command",
                metadata={"action": "scan", "target": extra or "self"}
            )
        elif action == "status":
            return ParsedCommand(
                intent="status",
                tools=[],
                brains=["self_model"],
                subcommand="status",
                args="",
                raw_input=text,
                matched_pattern="maven_command",
                metadata={"action": "status"}
            )

    # No grammar match
    return None


# =============================================================================
# SELF-INTENT GATE (Priority 0)
# =============================================================================

def _is_self_intent(query: str) -> bool:
    """Check if query is a self-intent question."""
    if not query:
        return False
    for pattern in _COMPILED_SELF_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _get_self_intent_type(query: str) -> Optional[str]:
    """Get the type of self-intent (capability, identity, introduction, history)."""
    if not query:
        return None

    query_lower = query.lower()

    # Introduction patterns (most specific, with typo tolerance)
    intro_markers = [
        "explain your self", "explain yourself", "introduce yourself",
        "explaine your self", "explaine yourself",  # common typos
        "your self",  # standalone "your self" is usually about self
        "whoami", "tell grok", "tell chatgpt",
    ]
    # Also check regex for typo variants
    intro_regex = r"\bexpl[ai]+n[e]?\s*(your\s*self|yourself)\b"
    if any(m in query_lower for m in intro_markers) or re.search(intro_regex, query_lower):
        return "introduction"

    # Capability patterns
    cap_markers = ["can you", "are you able", "what can you do", "what tools"]
    if any(m in query_lower for m in cap_markers):
        return "capability"

    # Identity patterns
    id_markers = [
        "who are you", "what are you", "yourself", "your name",
        "who made", "who created", "how do you feel",
    ]
    if any(m in query_lower for m in id_markers):
        return "identity"

    # History patterns
    hist_markers = ["what did we", "what did i", "do you remember"]
    if any(m in query_lower for m in hist_markers):
        return "history"

    return None


def _get_introduction_target(query: str) -> Optional[str]:
    """Extract target platform for self-introduction (grok, chatgpt, etc.)."""
    if not query:
        return None

    query_lower = query.lower()

    target_patterns = [
        r"(?:explain|introduce)\s+(?:your\s*self|yourself)\s+to\s+(\w+)",
        r"tell\s+(\w+)\s+(?:about\s+yourself|who\s+you\s+are)",
    ]

    for pattern in target_patterns:
        match = re.search(pattern, query_lower)
        if match:
            target = match.group(1)
            if target in ("grok", "x", "twitter"):
                return "grok"
            elif target in ("chatgpt", "openai", "gpt"):
                return "chatgpt"
            return target

    return None


def _handle_self_intent(ctx: RoutingContext) -> RoutingDecision:
    """Handle self-intent queries by routing to self_model."""
    intent_type = _get_self_intent_type(ctx.query)
    target = _get_introduction_target(ctx.query)

    decision = RoutingDecision(
        brains=["self_model"],
        tools=[],
        intent=f"self_{intent_type}" if intent_type else "self_query",
        confidence=1.0,
        source="self_intent_gate",
        bypass_teacher=True,
    )

    decision.add_audit(f"SELF_INTENT_GATE: Detected {intent_type} query")

    if intent_type == "history":
        decision.brains = ["self_model", "system_history"]
    elif intent_type == "introduction" and target:
        # Introduction with target = generate intro AND send to target platform
        decision.metadata["introduction_target"] = target
        decision.metadata["requires_tool_send"] = True
        decision.add_audit(f"Introduction target: {target}")

        # Add browser tool for sending to target
        if target in ("grok", "x"):
            decision.tools = ["x"]
            decision.subcommand = "grok"
            decision.metadata["platform"] = "x"
            decision.metadata["action"] = "grok_chat"
        elif target in ("chatgpt", "gpt", "openai"):
            decision.tools = ["chatgpt"]
            decision.subcommand = "chat"
            decision.metadata["platform"] = "chatgpt"
            decision.metadata["action"] = "chat"

    return decision


# =============================================================================
# CAPABILITY VALIDATION
# =============================================================================

def _validate_against_capabilities(
    decision: RoutingDecision,
    capability_snapshot: Dict[str, Any],
) -> RoutingDecision:
    """Validate and filter decision against system capabilities."""
    exec_mode = capability_snapshot.get("execution_mode", "UNKNOWN")
    web_enabled = capability_snapshot.get("web_research_enabled", False)
    tools_available = set(capability_snapshot.get("tools_available", []))

    valid_tools = []
    for tool in decision.tools:
        tool_base = tool.split(".")[0] if "." in tool else tool

        # Check availability
        if tool not in tools_available and tool_base not in tools_available:
            decision.add_violation(f"Tool not available: {tool}")
            continue

        # Check web tools
        if "web" in tool.lower() and not web_enabled:
            decision.add_violation(f"Web disabled: {tool}")
            continue

        # Check execution tools
        if tool_base in ("shell", "python_sandbox") and exec_mode in ("DISABLED", "SAFE_CHAT"):
            decision.add_violation(f"Execution disabled: {tool}")
            continue

        valid_tools.append(tool)

    decision.tools = valid_tools

    # Validate brains
    valid_brains = [b for b in decision.brains if b in VALID_BRAINS]
    if not valid_brains:
        valid_brains = ["language"]
        decision.add_audit("CRASH_PREVENTION: Defaulted to language brain")
    decision.brains = valid_brains

    return decision


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def build_routing_plan(
    query: str,
    capability_snapshot: Optional[Dict[str, Any]] = None,
    last_tool_name: Optional[str] = None,
    last_intent: Optional[str] = None,
    llm_router_enabled: bool = True,
    llm_confidence_threshold: float = 0.75,
) -> RoutingDecision:
    """
    Build a routing plan for a user query.

    This is the main entry point for all routing decisions.
    It follows strict precedence:
        Normalize > Grammar > Self-Intent > LLM Router > Learned > Default

    Args:
        query: The user's input text
        capability_snapshot: Current system capabilities
        last_tool_name: Last tool used (for continuation)
        last_intent: Last intent detected (for continuation)
        llm_router_enabled: Whether to use LLM router
        llm_confidence_threshold: Minimum confidence for LLM router

    Returns:
        RoutingDecision with brains, tools, and metadata
    """
    if capability_snapshot is None:
        capability_snapshot = {}

    # =========================================================================
    # STEP 0: NORMALIZE INPUT (Typo correction BEFORE routing)
    # =========================================================================
    # This fixes common typos so grammar and self-intent patterns work correctly.
    # e.g., "explaine your self" → "explain yourself"
    # e.g., "x gork helo" → "x grok hello"
    original_query = query
    normalized_query = query

    if _normalizer_available and query:
        try:
            norm_result = normalize(query)
            if norm_result.was_modified:
                normalized_query = norm_result.normalized
                print(f"[ROUTING] Normalized: '{query}' → '{normalized_query}'")
        except Exception as e:
            print(f"[ROUTING] Normalization failed: {e}, using original")

    # Build routing context with NORMALIZED query
    ctx = RoutingContext(
        query=normalized_query,  # Use normalized for routing
        capability_snapshot=capability_snapshot,
        last_tool_name=last_tool_name,
        last_intent=last_intent,
    )
    # Store original for logging/display
    ctx.metadata = {"original_query": original_query}

    # =========================================================================
    # PRIORITY -1: GRAMMAR LAYER (HARD FLOOR)
    # =========================================================================
    # Grammar runs on NORMALIZED text
    parsed = _parse_grammar(normalized_query)
    if parsed:
        ctx.grammar_matched = True
        ctx.grammar_result = parsed
        decision = parsed.to_routing_decision()
        decision.metadata["original_query"] = original_query
        decision.metadata["normalized_query"] = normalized_query
        decision = _validate_against_capabilities(decision, capability_snapshot)
        ctx.final_decision = decision
        return decision

    # =========================================================================
    # PRIORITY 0: SELF-INTENT GATE
    # =========================================================================
    # Self-intent runs on NORMALIZED text
    if _is_self_intent(normalized_query):
        ctx.self_intent_detected = True
        ctx.self_intent_type = _get_self_intent_type(normalized_query)
        ctx.introduction_target = _get_introduction_target(normalized_query)
        decision = _handle_self_intent(ctx)
        decision.metadata["original_query"] = original_query
        decision.metadata["normalized_query"] = normalized_query
        decision = _validate_against_capabilities(decision, capability_snapshot)
        ctx.final_decision = decision
        return decision

    # =========================================================================
    # PRIORITY 0.5: LLM ROUTER (Optional)
    # =========================================================================
    if llm_router_enabled:
        try:
            from brains.cognitive.router_llm import route_with_llm
            llm_result = route_with_llm(query, capability_snapshot)

            if llm_result.confidence >= llm_confidence_threshold:
                ctx.llm_result = LLMRouterResult(
                    brains=llm_result.brains,
                    tools=llm_result.tools,
                    confidence=llm_result.confidence,
                    reason=llm_result.reason,
                    raw_response=llm_result.raw_response,
                )
                decision = ctx.llm_result.to_routing_decision()
                decision = _validate_against_capabilities(decision, capability_snapshot)
                ctx.final_decision = decision
                return decision

        except ImportError:
            pass  # LLM router not available
        except Exception:
            pass  # LLM router failed, continue to next layer

    # =========================================================================
    # PRIORITY 1: LEARNED PATTERNS (routing_learning / pattern_store)
    # =========================================================================
    try:
        from brains.cognitive.pattern_store import get_best_match
        learned = get_best_match(query)
        if learned and learned.get("confidence", 0) >= 0.7:
            ctx.learned_result = learned
            decision = RoutingDecision(
                brains=learned.get("brains", ["language"]),
                tools=learned.get("tools", []),
                confidence=learned.get("confidence", 0.7),
                source="learned_pattern",
                audit_notes=[f"Learned pattern match: {learned.get('pattern', 'unknown')}"],
            )
            decision = _validate_against_capabilities(decision, capability_snapshot)
            ctx.final_decision = decision
            return decision
    except ImportError:
        pass  # Pattern store not available
    except Exception:
        pass  # Pattern lookup failed

    # =========================================================================
    # FALLBACK: SAFE DEFAULT
    # =========================================================================
    decision = RoutingDecision(
        brains=["language"],
        tools=[],
        confidence=0.3,
        source="safe_default",
        audit_notes=["FALLBACK: No patterns matched, using language brain"],
    )
    decision = _validate_against_capabilities(decision, capability_snapshot)
    ctx.final_decision = decision
    return decision


def is_explicit_command(query: str) -> bool:
    """Quick check if query is an explicit grammar command."""
    return _parse_grammar(query) is not None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "build_routing_plan",
    "is_explicit_command",
    "RoutingDecision",
    "ParsedCommand",
    "RoutingContext",
]
