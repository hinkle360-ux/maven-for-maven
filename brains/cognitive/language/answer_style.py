"""
Answer Style Module
===================

Controls the detail level and length of responses based on question complexity,
user context, and conversation state.

This module provides intelligent answer-length determination that replaces
the hard-coded "concise 2-3 sentences" behavior with dynamic length selection.

DetailLevel determines how much detail to include:
- SHORT: 1-2 sentences for simple facts/calculations
- MEDIUM: 3-6 sentences for general "what is X" questions
- LONG: Multiple paragraphs for explanations, follow-ups asking for more detail
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional
import re


class DetailLevel(Enum):
    """Response detail level classification."""
    SHORT = "short"    # 1-2 sentences: simple facts, calculations, yes/no
    MEDIUM = "medium"  # 3-6 sentences: general questions, definitions
    LONG = "long"      # Multiple paragraphs: explanations, follow-ups, complex topics


# Patterns that indicate user wants more detail (triggers LONG)
# Comprehensive list covering all "tell me more" variations
EXPANSION_PATTERNS = [
    r"\btell me more\b",
    r"\bmore about\b",
    r"\bgo deeper\b",
    r"\bdive deeper\b",
    r"\bmore detail\b",
    r"\bmore details\b",
    r"\bfurther detail\b",
    r"\belaborate\b",
    r"\bexpand on\b",
    r"\bcan you expand\b",
    r"\bexplain further\b",
    r"\bexplain more\b",
    r"\bexplain in detail\b",
    r"\bin detail\b",
    r"\bin more detail\b",
    r"\bwhat else\b",
    r"\banything else\b",
    r"\banything more\b",
    r"\bis there more\b",
    r"\bsomething more\b",
    r"\bkeep going\b",
    r"\bkeep talking\b",
    r"\bcontinue\b",
    r"\bgo on\b",
    r"\bdon't stop\b",
    r"\bmore info\b",
    r"\bmore information\b",
    r"\bgive me more\b",
    r"\bi want more\b",
    r"\bwant more\b",
    r"\bfull explanation\b",
    r"\bcomprehensive\b",
    r"\bthorough\b",
    r"\bwalk me through\b",
    r"\bstep by step\b",
    r"\bin depth\b",
    r"\bdeeper explanation\b",
    r"\bdescribe more\b",
    r"\bsay more\b",
    r"\bwhat more\b",
]

# Patterns that indicate complex questions (triggers LONG)
COMPLEX_QUESTION_PATTERNS = [
    r"^why\s+",          # Why questions need explanation
    r"^how does\s+",     # Mechanism questions
    r"^how do\s+",       # Process questions
    r"^explain\s+",      # Explicit explanation request
    r"^describe\s+",     # Description request
    r"^compare\s+",      # Comparison (needs multiple points)
    r"^what are the differences?\s+",
    r"^analyze\s+",
    r"^discuss\s+",
]

# Patterns for simple factual questions (triggers SHORT)
SIMPLE_FACT_PATTERNS = [
    r"^\d+\s*[\+\-\*\/]\s*\d+",      # Math: 2+2, 10*5
    r"^what is the capital of\s+",   # Capital questions
    r"^what year\s+",                # Year questions
    r"^when was\s+.{1,20}\s+(born|founded|created|invented|discovered)\b",
    r"^who is the\s+.{1,30}$",       # Short "who is the X" questions
    r"^how old is\s+",               # Age questions
    r"^how tall is\s+",              # Height questions
    r"^how many\s+.{1,30}\s+in\s+",  # Unit conversions
    r"^convert\s+\d+",               # Conversions
    r"^what is\s+\d+",               # What is 5+3, etc.
    r"^(yes|no)\s*\?",               # Yes/no questions
    r"^is\s+.{1,20}\s+(true|false|correct|right)\?*$",
    r"^(true|false)\s*:",            # True/false questions
]


def _matches_any_pattern(text: str, patterns: list) -> bool:
    """Check if text matches any of the given regex patterns."""
    text_lower = text.lower().strip()
    for pattern in patterns:
        try:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        except Exception:
            continue
    return False


def _is_short_query(text: str) -> bool:
    """Check if query is short enough to likely be a simple question."""
    words = text.split()
    return len(words) <= 6


def _is_follow_up_expansion(text: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if this is a follow-up asking for more detail on a previous topic.

    Args:
        text: The user's query
        context: Conversation context with last_topic, is_continuation, etc.

    Returns:
        True if this appears to be a follow-up asking for expansion
    """
    if not text:
        return False

    text_lower = text.lower().strip()

    # Direct expansion patterns
    if _matches_any_pattern(text, EXPANSION_PATTERNS):
        return True

    # Check context for continuation signals
    if context:
        # If marked as continuation with expansion intent
        continuation_intent = context.get("continuation_intent", "")
        if continuation_intent == "expansion":
            return True

        # If this is a continuation and the query is short (likely "tell me more" type)
        is_continuation = context.get("is_continuation", False)
        if is_continuation and len(text_lower.split()) <= 5:
            return True

    return False


def infer_detail_level(
    question: str,
    context: Optional[Dict[str, Any]] = None
) -> DetailLevel:
    """
    Infer the appropriate detail level for answering a question.

    This replaces hard-coded "concise 2-3 sentences" with intelligent selection
    based on:
    - Question complexity (simple fact vs explanation)
    - User signals (explicit "tell me more", "explain", etc.)
    - Conversation state (follow-ups on previous topics)

    Args:
        question: The user's question text
        context: Optional context dict containing:
            - is_continuation: bool - whether this follows a previous question
            - continuation_intent: str - "expansion", "clarification", "related"
            - last_topic: str - topic from previous turn
            - last_answer_source: str - where the last answer came from
            - user_verbosity_preference: str - user's preferred verbosity

    Returns:
        DetailLevel.SHORT, DetailLevel.MEDIUM, or DetailLevel.LONG

    Examples:
        >>> infer_detail_level("2+2")
        DetailLevel.SHORT

        >>> infer_detail_level("what is music")
        DetailLevel.MEDIUM

        >>> infer_detail_level("tell me more about music")
        DetailLevel.LONG

        >>> infer_detail_level("why does the sun rise in the east")
        DetailLevel.LONG
    """
    if not question:
        return DetailLevel.MEDIUM

    question_clean = question.strip()
    question_lower = question_clean.lower()

    # RULE 1: Check for explicit expansion/detail requests -> LONG
    if _is_follow_up_expansion(question, context):
        return DetailLevel.LONG

    # RULE 2: Check for complex question patterns -> LONG
    if _matches_any_pattern(question, COMPLEX_QUESTION_PATTERNS):
        return DetailLevel.LONG

    # RULE 3: Check for simple fact patterns -> SHORT
    if _matches_any_pattern(question, SIMPLE_FACT_PATTERNS):
        return DetailLevel.SHORT

    # RULE 4: Very short questions with simple structure -> SHORT
    # e.g., "capital of France", "2+2", "what is 5*3"
    words = question_clean.split()
    if len(words) <= 4:
        # Check if it's a math expression
        if re.match(r"^\d+\s*[\+\-\*\/\^]\s*\d+", question_clean):
            return DetailLevel.SHORT
        # Very short "what is X" where X is 1-2 words
        if question_lower.startswith(("what is ", "what's ")) and len(words) <= 4:
            return DetailLevel.MEDIUM  # Medium for definitions

    # RULE 5: Check context for previous web search follow-up
    if context:
        last_source = context.get("last_answer_source", "")
        is_continuation = context.get("is_continuation", False)

        # Follow-up to a web search result -> give more detail
        if is_continuation and last_source == "web_search":
            return DetailLevel.LONG

        # User has a verbosity preference
        user_pref = context.get("user_verbosity_preference", "").lower()
        if user_pref in ("verbose", "detailed", "long", "comprehensive"):
            return DetailLevel.LONG
        elif user_pref in ("brief", "short", "terse", "concise"):
            return DetailLevel.SHORT

    # RULE 6: Default to MEDIUM for general questions
    return DetailLevel.MEDIUM


def get_length_instruction(detail_level: DetailLevel) -> str:
    """
    Get the prompt instruction for the given detail level.

    Args:
        detail_level: The determined DetailLevel

    Returns:
        A string to include in the prompt that guides response length
    """
    if detail_level == DetailLevel.SHORT:
        return "Answer briefly in 1-2 sentences. Be direct and factual."

    elif detail_level == DetailLevel.MEDIUM:
        return "Provide a clear, helpful answer in a short paragraph (3-6 sentences)."

    elif detail_level == DetailLevel.LONG:
        return (
            "Provide a comprehensive, detailed explanation. "
            "Use multiple paragraphs if needed. Include examples, context, "
            "and different aspects of the topic. Structure with bullet points "
            "or sections if appropriate."
        )

    # Fallback
    return "Provide a clear, helpful answer."


def get_web_search_synthesis_instruction(
    detail_level: DetailLevel,
    is_followup: bool = False
) -> str:
    """
    Get synthesis instructions for web search results based on detail level.

    Args:
        detail_level: The determined DetailLevel
        is_followup: Whether this is a follow-up to a previous search

    Returns:
        Instruction string for synthesizing web search results
    """
    base_rules = [
        "Synthesize the information into a clear, helpful answer",
        "Do NOT make up information beyond what's in the results",
        "If the results don't fully answer the question, acknowledge the limitations",
    ]

    if detail_level == DetailLevel.SHORT:
        length_rule = "Be concise: answer in 1-2 sentences"
    elif detail_level == DetailLevel.MEDIUM:
        length_rule = "Provide a clear answer in a short paragraph (3-6 sentences)"
    else:  # LONG
        if is_followup:
            length_rule = (
                "This is a follow-up 'tell me more' request. Expand SIGNIFICANTLY "
                "beyond the previous 1-2 sentence summary.\n\n"
                "REQUIREMENTS:\n"
                "- You MUST answer in at least 3-4 sentences, ideally 2+ paragraphs\n"
                "- You MUST cover MULTIPLE aspects of the topic: history/origin, "
                "types or categories, specific examples, and real-world relevance\n"
                "- Do NOT just rephrase the previous short answer - provide NEW information\n"
                "- Use bullet points or numbered lists to organize different aspects\n"
                "- Include concrete examples, statistics, or facts from the search results"
            )
        else:
            length_rule = (
                "Provide a comprehensive, detailed explanation. "
                "Cover multiple aspects of the topic. Use multiple paragraphs "
                "and structure with bullet points or sections if appropriate."
            )

    base_rules.insert(1, length_rule)

    return "\n".join(f"- {rule}" for rule in base_rules)
