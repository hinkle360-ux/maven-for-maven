"""
Router Schema - Unified Data Classes for Routing
=================================================

This module defines all data classes used by the routing system.
Having a single schema file ensures consistency across all routing components.

Classes:
    - RoutingDecision: The final routing decision returned by routing_engine
    - ParsedCommand: Result of grammar parsing (command_grammar.py)
    - LLMRouterResult: Result from the LLM router (router_llm.py)
    - RoutingExample: A stored routing example for learning
    - RoutingContext: Context passed through the routing pipeline

Usage:
    from brains.routing.router_schema import RoutingDecision, ParsedCommand

    decision = RoutingDecision(
        brains=["language"],
        tools=[],
        confidence=0.95,
        source="grammar",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RoutingDecision:
    """
    Final routing decision from the routing engine.

    This is the main output of build_routing_plan() and contains all
    information needed to route a user message.
    """

    # Core routing
    brains: List[str] = field(default_factory=lambda: ["language"])
    tools: List[str] = field(default_factory=list)

    # Tool details
    subcommand: Optional[str] = None
    args: str = ""

    # Decision metadata
    intent: str = ""
    confidence: float = 0.5
    source: str = "unknown"  # grammar, router_llm, learned, fallback
    matched_pattern: str = ""

    # Control flags
    bypass_teacher: bool = False
    bypass_integrator: bool = False
    is_continuation: bool = False

    # Audit trail
    audit_notes: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "brains": self.brains,
            "tools": self.tools,
            "subcommand": self.subcommand,
            "args": self.args,
            "intent": self.intent,
            "confidence": self.confidence,
            "source": self.source,
            "matched_pattern": self.matched_pattern,
            "bypass_teacher": self.bypass_teacher,
            "bypass_integrator": self.bypass_integrator,
            "is_continuation": self.is_continuation,
            "audit_notes": self.audit_notes,
            "violations": self.violations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def add_audit(self, note: str) -> None:
        """Add an audit note."""
        self.audit_notes.append(note)

    def add_violation(self, violation: str) -> None:
        """Add a violation note."""
        self.violations.append(violation)


@dataclass
class ParsedCommand:
    """
    Result of command grammar parsing.

    This is produced by the hard grammar layer (command_grammar.py) for
    explicit tool commands.
    """

    intent: str  # e.g., "browser_tool", "research", "web_search", "explicit_tool"
    tools: List[str] = field(default_factory=list)
    brains: List[str] = field(default_factory=list)
    subcommand: Optional[str] = None
    args: str = ""
    raw_input: str = ""
    matched_pattern: str = ""
    bypass_integrator: bool = True
    confidence: float = 1.0  # Always 1.0 for grammar matches
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_routing_decision(self) -> RoutingDecision:
        """Convert to a RoutingDecision."""
        return RoutingDecision(
            brains=self.brains,
            tools=self.tools,
            subcommand=self.subcommand,
            args=self.args,
            intent=self.intent,
            confidence=self.confidence,
            source=f"grammar:{self.matched_pattern}",
            matched_pattern=self.matched_pattern,
            bypass_teacher=True,
            bypass_integrator=self.bypass_integrator,
            audit_notes=[f"Grammar match: {self.matched_pattern}"],
            metadata=self.metadata,
        )


@dataclass
class LLMRouterResult:
    """
    Result from the LLM router.

    The LLM router outputs JSON decisions with confidence scores.
    """

    brains: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    subcommand: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    source: str = "router_llm"
    raw_response: str = ""
    parse_error: Optional[str] = None

    def to_routing_decision(self) -> RoutingDecision:
        """Convert to a RoutingDecision."""
        return RoutingDecision(
            brains=self.brains if self.brains else ["language"],
            tools=self.tools,
            subcommand=self.subcommand,
            confidence=self.confidence,
            source="router_llm",
            bypass_teacher=self.confidence >= 0.8,
            audit_notes=[f"LLM router: {self.reason}"] if self.reason else [],
            metadata={"raw_response": self.raw_response} if self.raw_response else {},
        )


@dataclass
class RoutingExample:
    """
    A stored routing example for learning.

    These examples are used for few-shot prompting and pattern learning.
    """

    input_text: str
    normalized_text: str = ""
    intent: str = ""
    tools: List[str] = field(default_factory=list)
    brains: List[str] = field(default_factory=list)
    subcommand: Optional[str] = None
    verdict: str = "ok"  # ok, wrong_tool, wrong_brain, overkill, underkill
    source: str = "unknown"  # grammar, router_llm, learned, feedback
    confidence: float = 1.0
    timestamp: str = ""
    signature: str = ""
    is_gold: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import re

        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.normalized_text:
            self.normalized_text = re.sub(r"\s+", " ", self.input_text.strip().lower())
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute a routing signature for pattern matching."""
        parts = []
        if self.intent:
            parts.append(f"intent:{self.intent}")
        if self.tools:
            parts.append(f"tools:{','.join(sorted(self.tools))}")
        words = self.normalized_text.split()[:5]
        if words:
            parts.append(f"words:{','.join(words)}")
        return "|".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "input_text": self.input_text,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "tools": self.tools,
            "brains": self.brains,
            "subcommand": self.subcommand,
            "verdict": self.verdict,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "is_gold": self.is_gold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingExample":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RoutingContext:
    """
    Context passed through the routing pipeline.

    This accumulates state as the message passes through routing layers.
    """

    # Original input
    query: str
    normalized_query: str = ""

    # Capability snapshot
    capability_snapshot: Dict[str, Any] = field(default_factory=dict)

    # State tracking for continuation
    last_tool_name: Optional[str] = None
    last_intent: Optional[str] = None
    last_brains: List[str] = field(default_factory=list)

    # Layer results
    grammar_result: Optional[ParsedCommand] = None
    llm_result: Optional[LLMRouterResult] = None
    learned_result: Optional[Dict[str, Any]] = None

    # Processing flags
    grammar_matched: bool = False
    self_intent_detected: bool = False
    self_intent_type: Optional[str] = None
    introduction_target: Optional[str] = None

    # Final decision
    final_decision: Optional[RoutingDecision] = None

    def __post_init__(self):
        import re

        if not self.normalized_query:
            self.normalized_query = re.sub(r"\s+", " ", self.query.strip().lower())


@dataclass
class RouterToolChoice:
    """A tool choice from the router."""

    name: str
    subcommand: Optional[str] = None
    args: str = ""
    confidence: float = 1.0


@dataclass
class RouterBrainChoice:
    """A brain choice from the router."""

    name: str
    priority: int = 0  # Lower = higher priority
    reason: str = ""


# =============================================================================
# VALID BRAIN AND TOOL NAMES
# =============================================================================

VALID_BRAINS = {
    "language",
    "research_manager",
    "reasoning",
    "coder",
    "self_model",
    "memory_librarian",
    "planner",
    "integrator",
    "sensorium",
    "affect_priority",
    "action_engine",
    "system_history",
}

VALID_TOOLS = {
    "x",
    "chatgpt",
    "browser_open",
    "web_search",
    "web_fetch",
    "shell",
    "python_sandbox",
    "git",
    "fs",
    "grok",
}


def validate_brain_name(name: str) -> bool:
    """Check if a brain name is valid."""
    return name in VALID_BRAINS


def validate_tool_name(name: str) -> bool:
    """Check if a tool name is valid."""
    return name in VALID_TOOLS


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RoutingDecision",
    "ParsedCommand",
    "LLMRouterResult",
    "RoutingExample",
    "RoutingContext",
    "RouterToolChoice",
    "RouterBrainChoice",
    "VALID_BRAINS",
    "VALID_TOOLS",
    "validate_brain_name",
    "validate_tool_name",
]
