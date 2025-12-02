"""
routing_intent.py
~~~~~~~~~~~~~~~~~

Data structures for LLM-assisted smart routing.

This module defines the core types used for routing classification:
- RoutingIntent: The classified intent of a user message
- RoutingSuggestion: Teacher's routing recommendation
- RoutingPlan: Final validated routing plan after capability checks

The design goal is to make routing feel like a smart chat assistant:
- Robust to phrasing: "help me debug this", "why is this breaking", "this code is weird"
  should all route to the same path.
- Dynamic: adjusts based on context, prior turns, capabilities.
- LLM helps teach routing, but never lies about capabilities or internal state.

This module is self-contained with no LLM dependencies so that:
1. Types can be used anywhere without circular imports
2. Future non-LLM classifiers can use the same types
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class PrimaryIntent(str, Enum):
    """
    Primary intent categories for routing.

    These represent the top-level classification of what the user wants.
    """
    # Chat/conversation intents
    CHAT_ANSWER = "chat_answer"          # General conversational response
    SMALL_TALK = "small_talk"            # Casual chat, greetings

    # Task intents
    CODE_TASK = "code_task"              # Write, debug, or modify code
    RESEARCH_QUESTION = "research_question"  # Research/investigation needed
    TOOL_REQUEST = "tool_request"        # Direct tool execution request
    WEB_SEARCH = "web_search"            # Web search for current info (games, news, etc.)

    # Self-knowledge intents (MUST route to self_model, NOT Teacher)
    CAPABILITY_QUESTION = "capability_question"  # "can you X?" questions
    SELF_QUESTION = "self_question"      # "who are you?" identity questions
    HISTORY_QUESTION = "history_question"  # "what did we discuss?" history

    # Follow-up intents
    TASK_FOLLOWUP = "task_followup"      # Continuation of previous task
    CLARIFICATION = "clarification"      # "I meant X" clarifications

    # Meta intents
    META_INSTRUCTION = "meta_instruction"  # Instructions about behavior
    FEEDBACK = "feedback"                # User feedback on responses

    # Fallback
    UNKNOWN = "unknown"                  # Could not classify


class Urgency(str, Enum):
    """Urgency level for routing decisions."""
    LOW = "low"           # Can take time, exploration allowed
    MEDIUM = "medium"     # Normal priority
    HIGH = "high"         # Needs quick response
    CRITICAL = "critical"  # Safety-critical, minimal processing


class Complexity(str, Enum):
    """Complexity level for routing decisions."""
    SIMPLE = "simple"     # Single step, direct answer
    MODERATE = "moderate"  # Multiple steps, some reasoning
    COMPLEX = "complex"   # Multi-stage, requires planning
    VERY_COMPLEX = "very_complex"  # Major task, full orchestration


@dataclass
class RoutingIntent:
    """
    Classified intent of a user message.

    This is the output of classify_intent() and represents what we believe
    the user wants, independent of how we'll fulfill that want.

    Attributes:
        primary_intent: Main intent category
        secondary_tags: Additional context tags (e.g., ["browser", "python", "project:maven"])
        urgency: How urgent is this request?
        complexity: How complex is this request?
        confidence: How confident are we in this classification (0.0-1.0)?
        raw_signals: Debug info about what signals led to this classification
    """
    primary_intent: PrimaryIntent
    secondary_tags: List[str] = field(default_factory=list)
    urgency: Urgency = Urgency.MEDIUM
    complexity: Complexity = Complexity.MODERATE
    confidence: float = 0.5
    raw_signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "primary_intent": self.primary_intent.value,
            "secondary_tags": self.secondary_tags,
            "urgency": self.urgency.value,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "raw_signals": self.raw_signals,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RoutingIntent:
        """Deserialize from dictionary."""
        return RoutingIntent(
            primary_intent=PrimaryIntent(d.get("primary_intent", "unknown")),
            secondary_tags=d.get("secondary_tags", []),
            urgency=Urgency(d.get("urgency", "medium")),
            complexity=Complexity(d.get("complexity", "moderate")),
            confidence=float(d.get("confidence", 0.5)),
            raw_signals=d.get("raw_signals", {}),
        )

    def is_self_intent(self) -> bool:
        """Check if this intent should route to self_model (NOT Teacher)."""
        return self.primary_intent in (
            PrimaryIntent.CAPABILITY_QUESTION,
            PrimaryIntent.SELF_QUESTION,
            PrimaryIntent.HISTORY_QUESTION,
        )


@dataclass
class RoutingSuggestion:
    """
    Teacher's routing recommendation.

    This is what the Teacher LLM suggests for routing, but it has NOT been
    validated against actual capabilities yet.

    IMPORTANT: Teacher suggestions MUST be validated before use:
    1. Check suggested brains exist in brain_roles
    2. Check suggested tools exist and are enabled
    3. Intersect with system_capabilities

    Attributes:
        recommended_brains: Brain names Teacher suggests (e.g., ["reasoning", "coder"])
        recommended_tools: Tool paths Teacher suggests (e.g., ["web_client.search"])
        notes: Teacher's explanation (for debugging/training only)
        confidence: Teacher's confidence in this suggestion
        raw_response: Original Teacher response for logging
    """
    recommended_brains: List[str] = field(default_factory=list)
    recommended_tools: List[str] = field(default_factory=list)
    notes: str = ""
    confidence: float = 0.5
    raw_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "recommended_brains": self.recommended_brains,
            "recommended_tools": self.recommended_tools,
            "notes": self.notes,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RoutingSuggestion:
        """Deserialize from dictionary."""
        return RoutingSuggestion(
            recommended_brains=d.get("recommended_brains", []),
            recommended_tools=d.get("recommended_tools", []),
            notes=d.get("notes", ""),
            confidence=float(d.get("confidence", 0.5)),
            raw_response=d.get("raw_response", {}),
        )


@dataclass
class RoutingPlan:
    """
    Final validated routing plan.

    This is the output of compute_routing_plan() and represents the ACTUAL
    routing that will be executed. All suggestions have been validated
    against real capabilities.

    Attributes:
        final_brains: Validated brain names to activate
        final_tools: Validated tool paths to use
        intent: The classified intent that led to this plan
        suggestion_source: Where the suggestion came from
        reasons: Audit trail of why each decision was made
        validation_notes: What was filtered out and why
    """
    final_brains: List[str] = field(default_factory=list)
    final_tools: List[str] = field(default_factory=list)
    intent: Optional[RoutingIntent] = None
    suggestion_source: str = "fallback"  # "teacher", "pattern_match", "rl_routing", "fallback"
    reasons: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "final_brains": self.final_brains,
            "final_tools": self.final_tools,
            "intent": self.intent.to_dict() if self.intent else None,
            "suggestion_source": self.suggestion_source,
            "reasons": self.reasons,
            "validation_notes": self.validation_notes,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> RoutingPlan:
        """Deserialize from dictionary."""
        intent_data = d.get("intent")
        return RoutingPlan(
            final_brains=d.get("final_brains", []),
            final_tools=d.get("final_tools", []),
            intent=RoutingIntent.from_dict(intent_data) if intent_data else None,
            suggestion_source=d.get("suggestion_source", "fallback"),
            reasons=d.get("reasons", []),
            validation_notes=d.get("validation_notes", []),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Intent Classification Patterns
# =============================================================================

# These patterns are used for fast local classification before calling Teacher.
# They are gradually learned/updated by the distillation job.

SELF_INTENT_PATTERNS = {
    # Capability questions - MUST route to system_capabilities, NOT Teacher
    "capability_question": [
        "can you",
        "are you able to",
        "do you have",
        "what can you do",
        "what are your capabilities",
        "what tools do you have",
        "can you browse",
        "can you search",
        "can you run",
        "can you execute",
        "can you read",
        "can you write",
        "can you access",
        "can you control",
        "do you have access",
        "do you have internet",
    ],
    # Self-identity questions - MUST route to self_model, NOT Teacher
    "self_question": [
        "who are you",
        "what are you",
        "tell me about yourself",
        "describe yourself",
        "what is your name",
        "are you maven",
        "are you an llm",
        "are you chatgpt",
        "are you claude",
        "who made you",
        "who created you",
        "what is your purpose",
        "how do you work",
        # TASK 1: Internal state / feelings / preferences questions
        # These MUST route to self_model, NOT Teacher
        "how do you feel",
        "how are you feeling",
        "do you have feelings",
        "do you have emotions",
        "are you happy",
        "are you sad",
        "are you conscious",
        "are you sentient",
        "are you alive",
        "do you like",
        "do you enjoy",
        "do you prefer",
        "what do you think about yourself",
        "what do you like",
        "what do you want",
        "what do you prefer",
        "do you have preferences",
        "do you have opinions",
        "what is your opinion",
        "how do you think",
        "are you real",
        "are you a person",
        "do you dream",
        "do you sleep",
        "do you get tired",
        "do you get bored",
    ],
    # History questions - MUST route to system_history, NOT Teacher
    "history_question": [
        "what did we discuss",
        "what did i ask",
        "what did you say",
        "our conversation",
        "previous question",
        "earlier you said",
        "remember when",
        "last time we",
        "yesterday we",
        "do you remember",
    ],
}

CODE_TASK_PATTERNS = [
    "write a function",
    "write code",
    "debug this",
    "fix this",
    "help me code",
    "implement",
    "refactor",
    "optimize this code",
    "what's wrong with this code",
    "why is this breaking",
    "this code is weird",
    "code review",
    "generate code",
    "create a script",
    "write a class",
    "write a method",
    "extend this function",
]

RESEARCH_PATTERNS = [
    "research",
    "find information",
    "look up",
    "investigate",
    "search for",
    "what is the latest",
    "tell me about",
    "explain",
    "what does",
    "how does",
    "why does",
]

# Web search patterns - for queries requiring current/live web information
WEB_SEARCH_PATTERNS = [
    "google search",
    "search google",
    "google for",
    "bing search",
    "search bing",
    "duckduckgo",
    "search the web",
    "web search",
    "search online",
    "look up online",
    "find online",
    "search internet",
    "what new games",
    "latest games",
    "upcoming games",
    "new games coming out",
    "games releasing",
    "latest news",
    "current news",
    "news about",
    "what's happening",
    "recent updates",
    "new releases",
]

TOOL_REQUEST_PATTERNS = [
    "run",
    "execute",
    "git status",
    "git commit",
    "scan",
    "list files",
    "read file",
    "write file",
    "open url",
    "browse to",
]


def _pattern_match_score(text: str, patterns: List[str]) -> float:
    """
    Calculate how well text matches a list of patterns.

    Returns a score between 0.0 and 1.0.
    """
    text_lower = text.lower()
    matches = sum(1 for p in patterns if p in text_lower)
    if matches == 0:
        return 0.0
    # Normalize by pattern count with diminishing returns
    return min(1.0, matches * 0.3)


def classify_intent_local(
    user_message: str,
    context: Optional[Dict[str, Any]] = None
) -> RoutingIntent:
    """
    Local (non-LLM) intent classification using pattern matching.

    This provides a fast first-pass classification that can be used:
    1. To determine if Teacher classification is needed
    2. As a fallback if Teacher is unavailable
    3. To validate Teacher suggestions

    Args:
        user_message: The user's message text
        context: Optional context with conversation history, etc.

    Returns:
        RoutingIntent with classification result
    """
    context = context or {}
    msg_lower = user_message.lower().strip()
    raw_signals: Dict[str, Any] = {}

    # Check for self-intent patterns first (highest priority)
    for intent_type, patterns in SELF_INTENT_PATTERNS.items():
        score = _pattern_match_score(msg_lower, patterns)
        raw_signals[f"{intent_type}_score"] = score
        if score > 0.3:
            intent_map = {
                "capability_question": PrimaryIntent.CAPABILITY_QUESTION,
                "self_question": PrimaryIntent.SELF_QUESTION,
                "history_question": PrimaryIntent.HISTORY_QUESTION,
            }
            return RoutingIntent(
                primary_intent=intent_map[intent_type],
                secondary_tags=["self_intent"],
                urgency=Urgency.MEDIUM,
                complexity=Complexity.SIMPLE,
                confidence=min(0.9, score + 0.5),
                raw_signals=raw_signals,
            )

    # Check for code task patterns
    code_score = _pattern_match_score(msg_lower, CODE_TASK_PATTERNS)
    raw_signals["code_task_score"] = code_score
    if code_score > 0.3:
        # Determine complexity based on message length and keywords
        if len(user_message) > 200 or "refactor" in msg_lower or "optimize" in msg_lower:
            complexity = Complexity.COMPLEX
        elif "debug" in msg_lower or "fix" in msg_lower:
            complexity = Complexity.MODERATE
        else:
            complexity = Complexity.MODERATE

        return RoutingIntent(
            primary_intent=PrimaryIntent.CODE_TASK,
            secondary_tags=_extract_code_tags(msg_lower),
            urgency=Urgency.MEDIUM,
            complexity=complexity,
            confidence=min(0.85, code_score + 0.4),
            raw_signals=raw_signals,
        )

    # Check for tool request patterns
    tool_score = _pattern_match_score(msg_lower, TOOL_REQUEST_PATTERNS)
    raw_signals["tool_request_score"] = tool_score
    if tool_score > 0.3:
        return RoutingIntent(
            primary_intent=PrimaryIntent.TOOL_REQUEST,
            secondary_tags=_extract_tool_tags(msg_lower),
            urgency=Urgency.MEDIUM,
            complexity=Complexity.SIMPLE,
            confidence=min(0.85, tool_score + 0.4),
            raw_signals=raw_signals,
        )

    # Check for web search patterns (before research for higher priority)
    web_search_score = _pattern_match_score(msg_lower, WEB_SEARCH_PATTERNS)
    raw_signals["web_search_score"] = web_search_score
    if web_search_score > 0.25:
        return RoutingIntent(
            primary_intent=PrimaryIntent.WEB_SEARCH,
            secondary_tags=["web_search", "browser"],
            urgency=Urgency.MEDIUM,
            complexity=Complexity.SIMPLE,
            confidence=min(0.9, web_search_score + 0.45),
            raw_signals=raw_signals,
        )

    # Check for research patterns
    research_score = _pattern_match_score(msg_lower, RESEARCH_PATTERNS)
    raw_signals["research_score"] = research_score
    if research_score > 0.2:
        return RoutingIntent(
            primary_intent=PrimaryIntent.RESEARCH_QUESTION,
            secondary_tags=[],
            urgency=Urgency.LOW,
            complexity=Complexity.MODERATE,
            confidence=min(0.7, research_score + 0.3),
            raw_signals=raw_signals,
        )

    # Check for follow-up patterns
    if _is_followup(msg_lower, context):
        raw_signals["is_followup"] = True
        return RoutingIntent(
            primary_intent=PrimaryIntent.TASK_FOLLOWUP,
            secondary_tags=["continuation"],
            urgency=Urgency.MEDIUM,
            complexity=Complexity.SIMPLE,
            confidence=0.7,
            raw_signals=raw_signals,
        )

    # Check for small talk
    if _is_small_talk(msg_lower):
        raw_signals["is_small_talk"] = True
        return RoutingIntent(
            primary_intent=PrimaryIntent.SMALL_TALK,
            secondary_tags=[],
            urgency=Urgency.LOW,
            complexity=Complexity.SIMPLE,
            confidence=0.8,
            raw_signals=raw_signals,
        )

    # Default to chat answer with low confidence
    return RoutingIntent(
        primary_intent=PrimaryIntent.CHAT_ANSWER,
        secondary_tags=[],
        urgency=Urgency.MEDIUM,
        complexity=Complexity.MODERATE,
        confidence=0.4,
        raw_signals=raw_signals,
    )


def _extract_code_tags(text: str) -> List[str]:
    """Extract code-related tags from text."""
    tags = []
    language_keywords = {
        "python": ["python", "py", "def ", "import "],
        "javascript": ["javascript", "js", "node", "npm"],
        "typescript": ["typescript", "ts"],
        "rust": ["rust", "cargo"],
        "go": ["golang", "go "],
    }
    for lang, keywords in language_keywords.items():
        if any(kw in text for kw in keywords):
            tags.append(lang)

    if "debug" in text or "fix" in text or "error" in text:
        tags.append("debugging")
    if "refactor" in text:
        tags.append("refactoring")
    if "test" in text:
        tags.append("testing")

    return tags


def _extract_tool_tags(text: str) -> List[str]:
    """Extract tool-related tags from text."""
    tags = []
    if "git" in text:
        tags.append("git")
    if "file" in text or "read" in text or "write" in text:
        tags.append("filesystem")
    if "url" in text or "browse" in text or "web" in text:
        tags.append("browser")
    if "run" in text or "execute" in text:
        tags.append("execution")
    return tags


def _is_followup(text: str, context: Dict[str, Any]) -> bool:
    """Check if message is a follow-up to previous conversation."""
    followup_indicators = [
        "that", "it", "this", "those", "these", "them",
        "more", "continue", "go on", "tell me more",
        "what about", "and", "also", "another",
    ]
    # Short message with pronoun reference
    if len(text.split()) <= 5:
        if any(ind in text for ind in followup_indicators[:6]):
            return True
    # Explicit continuation phrases
    if any(ind in text for ind in followup_indicators[6:]):
        return True
    return False


def _is_small_talk(text: str) -> bool:
    """Check if message is small talk."""
    small_talk_patterns = [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "what's up", "thanks",
        "thank you", "bye", "goodbye", "see you",
    ]
    return any(p in text for p in small_talk_patterns)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PrimaryIntent",
    "Urgency",
    "Complexity",
    "RoutingIntent",
    "RoutingSuggestion",
    "RoutingPlan",
    "classify_intent_local",
    "SELF_INTENT_PATTERNS",
]
