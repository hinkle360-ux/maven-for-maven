"""
Command Pre-Parser - Hard Grammar Layer
========================================

This module provides a deterministic pre-parser that routes explicit tool
commands BEFORE they reach sensorium/integrator. This is the "hard floor"
that guarantees correct routing when learned patterns are weak.

ROUTING PRECEDENCE:
1. Hard Grammar (this module) - Explicit tool syntax, never overridden
2. Router LLM - Dedicated routing decisions with confidence
3. Learned Patterns - routing_learning / pattern_store
4. Safe Default - language brain with clarification

GRAMMAR PATTERNS:
- `x grok <message>` → browser tool x, subcommand grok
- `x: grok <message>` → same as above
- `grok <message>` → browser tool x, subcommand grok
- `research: <topic>` → research_manager brain
- `research: "<topic>" web:N` → research with N web requests
- `browser_open: <url>` → browser_open tool
- `use <tool> tool: <args>` → explicit tool call
- `post to x: <message>` → x_post tool
- `search: <query>` → web_search tool

When this layer matches, it:
1. Bypasses sensorium + reasoning entirely
2. Builds a routing plan directly
3. Logs as "gold" routing for learning

Usage:
    from brains.cognitive.command_pre_parser import parse_command, is_explicit_command

    result = parse_command("x grok hello from maven")
    if result:
        # result.intent = "browser_tool"
        # result.tools = ["x"]
        # result.subcommand = "grok"
        # result.args = "hello from maven"
        # result.bypass_integrator = True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedCommand:
    """Result of command pre-parsing."""

    intent: str  # e.g., "browser_tool", "research", "web_search", "explicit_tool"
    tools: List[str] = field(default_factory=list)  # Tools to invoke
    brains: List[str] = field(default_factory=list)  # Brains to route to
    subcommand: Optional[str] = None  # e.g., "grok", "post", "search"
    args: str = ""  # Remaining arguments
    raw_input: str = ""  # Original input
    matched_pattern: str = ""  # Which pattern matched
    bypass_integrator: bool = True  # Skip integrator routing?
    confidence: float = 1.0  # Always 1.0 for grammar matches
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GRAMMAR PATTERNS - Order matters! More specific patterns first.
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

# Pattern: pc: <command> or pc <command> (PC control tool)
PATTERN_PC_CONTROL = re.compile(
    r"^pc\s*[:\s]+(.*)$",
    re.IGNORECASE | re.DOTALL
)

# Pattern: human: <command> or human <command> (desktop control tool)
PATTERN_HUMAN_CONTROL = re.compile(
    r"^human\s*[:\s]+(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_command(text: str) -> Optional[ParsedCommand]:
    """
    Parse user input against the hard grammar.

    Args:
        text: Raw user input text

    Returns:
        ParsedCommand if matched, None if no grammar match
    """
    if not text:
        return None

    # Normalize: strip, collapse whitespace
    text = text.strip()
    text_normalized = re.sub(r'\s+', ' ', text)

    # Try each pattern in order (most specific first)

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

    # 14. pc: <command> (PC control - system monitoring, cleanup, security)
    match = PATTERN_PC_CONTROL.match(text_normalized)
    if match:
        pc_command = match.group(1).strip() if match.group(1) else "status"
        return ParsedCommand(
            intent="browser_tool",
            tools=["pc"],
            brains=["language"],
            subcommand=pc_command.split()[0] if pc_command else "status",
            args=pc_command,
            raw_input=text,
            matched_pattern="pc_control",
            metadata={"tool": "pc", "command": pc_command}
        )

    # 15. human: <command> (desktop control - mouse, keyboard, screenshots)
    match = PATTERN_HUMAN_CONTROL.match(text_normalized)
    if match:
        human_command = match.group(1).strip() if match.group(1) else ""
        return ParsedCommand(
            intent="browser_tool",
            tools=["human"],
            brains=["language"],
            subcommand=human_command.split()[0] if human_command else "help",
            args=human_command,
            raw_input=text,
            matched_pattern="human_control",
            metadata={"tool": "human", "command": human_command}
        )

    # No grammar match
    return None


def is_explicit_command(text: str) -> bool:
    """
    Quick check if text is an explicit command that should bypass integrator.

    Args:
        text: User input text

    Returns:
        True if this is an explicit tool command
    """
    return parse_command(text) is not None


def get_routing_plan(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Convert a ParsedCommand to a routing plan for integrator.

    Args:
        parsed: The parsed command result

    Returns:
        Routing plan dictionary compatible with integrator
    """
    return {
        "brains": parsed.brains,
        "tools": parsed.tools,
        "source": f"grammar:{parsed.matched_pattern}",
        "confidence": parsed.confidence,
        "bypass_learned": True,  # Don't let learned patterns override
        "bypass_llm_router": True,  # Don't need LLM router
        "intent": parsed.intent,
        "subcommand": parsed.subcommand,
        "args": parsed.args,
        "metadata": parsed.metadata,
    }


def log_gold_routing(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Create a gold routing example for learning.

    These examples can be used to train the routing patterns and
    provide few-shot examples to the router LLM.

    Args:
        parsed: The parsed command

    Returns:
        Gold routing example for storage
    """
    return {
        "input": parsed.raw_input,
        "normalized": re.sub(r'\s+', ' ', parsed.raw_input.strip()),
        "intent": parsed.intent,
        "tools": parsed.tools,
        "brains": parsed.brains,
        "subcommand": parsed.subcommand,
        "pattern": parsed.matched_pattern,
        "confidence": 1.0,
        "source": "grammar",
        "is_gold": True,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ParsedCommand",
    "parse_command",
    "is_explicit_command",
    "get_routing_plan",
    "log_gold_routing",
]
