"""
Continuation Helpers
====================

Shared utilities for detecting and handling conversation continuations
across all cognitive brains.

Every cognitive brain must use these helpers to implement the three
required signals:
1. Follow-up detection (is_continuation)
2. History access (get_conversation_context)
3. Routing hints (create_routing_hint)

This ensures consistent continuation handling and enables Teacher learning.
"""

from __future__ import annotations
from typing import Dict, Any, Optional


# Follow-up patterns that indicate continuation
# Comprehensive list covering all variations of "tell me more" type requests
FOLLOW_UP_PATTERNS = [
    # Core expansion patterns
    "tell me more", "more about", "what about", "can you expand",
    "explain further", "more details", "continue", "go on",
    "elaborate", "what else", "anything else", "more on",
    "explain that", "about that", "more info", "keep going",
    "and?", "also?", "besides?", "additionally",
    "what about that", "say more", "more please",
    # Additional expansion variations
    "go deeper", "dive deeper", "can you expand on that",
    "expand on that", "tell me more about that",
    "more information", "more detail", "further detail",
    "give me more", "i want more", "want more",
    "deeper explanation", "in more detail",
    "explain more", "describe more", "say more about",
    "keep talking", "don't stop", "what more",
    "is there more", "anything more", "something more",
]


def is_continuation(text: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Detect if input text is a follow-up/continuation of previous conversation.

    Args:
        text: The user's input text
        context: Optional context dict with norm_type or other hints

    Returns:
        True if this appears to be a continuation, False otherwise

    Examples:
        >>> is_continuation("tell me more")
        True
        >>> is_continuation("what is a lion?")
        False
    """
    if not text:
        return False

    text_lower = text.lower().strip()

    # Check explicit follow-up patterns
    if any(pattern in text_lower for pattern in FOLLOW_UP_PATTERNS):
        return True

    # Check context hints (from sensorium classification)
    if context:
        norm_type = context.get("norm_type", "")
        if norm_type == "follow_up_question":
            return True

        classification = context.get("classification", "")
        if classification == "follow_up_question":
            return True

    # Very short queries that reference "that", "it", "this" are likely continuations
    if len(text_lower.split()) <= 3:
        pronoun_refs = ["that", "it", "this", "those", "these", "them"]
        if any(ref in text_lower for ref in pronoun_refs):
            return True

    return False


def get_conversation_context() -> Dict[str, Any]:
    """
    Retrieve conversation history from system_history and conversation state.

    Returns:
        Dictionary with:
        - last_topic: Most recent conversation topic
        - last_user_question: Previous user query
        - last_answer_subject: Subject of last response
        - thread_entities: Entities mentioned in conversation
        - conversation_depth: Number of turns in conversation
        - last_web_search: Last web search data (if any)
        - last_answer_source: Source of last answer (e.g., "web_search", "reasoning")

    This enables brains to attach follow-ups to previous context.
    """
    context = {
        "last_topic": None,
        "last_user_question": None,
        "last_answer_subject": None,
        "thread_entities": [],
        "conversation_depth": 0,
        "last_web_search": None,
        "last_answer_source": None,
    }

    # Try to get from system_history
    try:
        from brains.cognitive.system_history.service.system_history_brain import service_api
        result = service_api({"op": "GET_LAST_TOPIC", "payload": {}})
        if result.get("ok"):
            payload = result.get("payload", {})
            context["last_topic"] = payload.get("last_topic")
    except Exception:
        pass

    # Try to get from memory_librarian conversation state
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import _CONVERSATION_STATE
        context["last_user_question"] = _CONVERSATION_STATE.get("last_query", "")
        context["last_answer_subject"] = _CONVERSATION_STATE.get("last_response", "")
        context["thread_entities"] = _CONVERSATION_STATE.get("thread_entities", [])
        context["conversation_depth"] = _CONVERSATION_STATE.get("conversation_depth", 0)

        # Fallback to last_topic from conversation state if system_history didn't have it
        if not context["last_topic"]:
            context["last_topic"] = _CONVERSATION_STATE.get("last_topic", "")
    except Exception:
        pass

    # Try to get last web search state
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import get_last_web_search
        last_web = get_last_web_search()
        if last_web:
            context["last_web_search"] = last_web
            context["last_answer_source"] = "web_search"
    except Exception:
        pass

    return context


def get_last_web_search_state() -> Optional[Dict[str, Any]]:
    """
    Get the most recent web search results for follow-up handling.

    Returns:
        Dict with query, results, engine, answer, sources, or None if no search stored
    """
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import get_last_web_search
        return get_last_web_search()
    except Exception:
        return None


def followup_refers_to_last_search(query: str) -> bool:
    """
    Check if a follow-up query refers to the last web search topic.

    Args:
        query: The user's follow-up query

    Returns:
        True if the follow-up appears to reference the last web search topic
    """
    try:
        from brains.cognitive.memory_librarian.service.memory_librarian import followup_refers_to_web_search
        return followup_refers_to_web_search(query)
    except Exception:
        return False


def create_routing_hint(
    brain_name: str,
    action: str,
    confidence: float,
    context_tags: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized routing hint for the Integrator.

    Args:
        brain_name: Name of the brain providing the hint
        action: Suggested routing action (e.g., "expand_previous_reasoning")
        confidence: Confidence in this routing (0.0 to 1.0)
        context_tags: Optional list of context tags
        metadata: Optional additional metadata

    Returns:
        Standardized routing hint dictionary

    Example:
        >>> create_routing_hint("reasoning", "expand_previous_reasoning", 0.9,
        ...                     context_tags=["follow_up", "factual"])
        {
            "brain": "reasoning",
            "routing_hint": "expand_previous_reasoning",
            "confidence": 0.9,
            "context_tags": ["follow_up", "factual"],
            "metadata": {}
        }
    """
    return {
        "brain": brain_name,
        "routing_hint": action,
        "confidence": max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
        "context_tags": context_tags or [],
        "metadata": metadata or {}
    }


def enhance_query_with_context(query: str, context: Dict[str, Any]) -> str:
    """
    Enhance a follow-up query with conversation context.

    For continuation queries, append the last topic to improve retrieval
    and reasoning.

    Args:
        query: Original user query
        context: Conversation context from get_conversation_context()

    Returns:
        Enhanced query string

    Example:
        >>> enhance_query_with_context("tell me more",
        ...                           {"last_topic": "lion"})
        "tell me more about lion"
    """
    if not query or not context:
        return query

    last_topic = context.get("last_topic", "")

    # If this is a bare follow-up and we have a topic, attach it
    if is_continuation(query) and last_topic:
        # Avoid duplication
        if last_topic.lower() not in query.lower():
            return f"{query} about {last_topic}"

    return query


def should_use_continuation_routing(context: Dict[str, Any]) -> bool:
    """
    Determine if continuation-specific routing should be used.

    Args:
        context: Context dict with classification and history

    Returns:
        True if continuation routing is appropriate

    This helps brains decide whether to use expansion vs. fresh analysis.
    """
    # Check if classified as continuation
    if context.get("is_continuation"):
        return True

    # Check if there's a recent topic to continue from
    if context.get("last_topic") and context.get("conversation_depth", 0) > 0:
        return True

    return False


def extract_continuation_intent(text: str) -> str:
    """
    Extract the type of continuation being requested.

    Returns one of:
    - "expansion": User wants more details/elaboration
    - "clarification": User wants simpler explanation
    - "related": User wants related information
    - "unknown": Cannot determine intent

    This helps route to appropriate brain for handling.
    """
    text_lower = text.lower().strip()

    # Expansion requests
    expansion_markers = [
        "more", "expand", "elaborate", "continue", "further",
        "deeper", "detail", "tell me more"
    ]
    if any(marker in text_lower for marker in expansion_markers):
        return "expansion"

    # Clarification requests
    clarification_markers = [
        "explain", "clarify", "simpler", "don't understand",
        "what does that mean", "confused"
    ]
    if any(marker in text_lower for marker in clarification_markers):
        return "clarification"

    # Related information requests
    related_markers = [
        "what about", "how about", "what else", "also",
        "related", "similar", "besides"
    ]
    if any(marker in text_lower for marker in related_markers):
        return "related"

    return "unknown"
