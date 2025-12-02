"""
Context Management
===================

This module implements simple context management utilities for the
Maven cognitive architecture.  Context refers to the mutable
pipeline state that accumulates information across stages and turns.
Without management, context can grow indefinitely and bias future
decisions.  The functions here implement basic temporal decay and
reconstruction heuristics.

Features include:

* ``apply_decay`` – reduce the influence of old numeric fields by
  multiplying them by a decay factor.
* ``reconstruct_context`` – merge a list of prior context snapshots
  into a fresh context, preferring newer values and combining lists.

These helpers are intentionally lightweight.  They can be extended
in future releases to support more sophisticated decay functions
(e.g. exponential, contextual) or to persist and retrieve context
across sessions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Iterable, Optional, Union

# Pattern store for unified learning across all brains
from brains.cognitive.pattern_store import (
    get_pattern_store,
    Pattern,
    verdict_to_reward
)
from brains.cognitive.context_management.initial_patterns import (
    initialize_context_management_patterns
)

# Teacher integration for learning context tracking and management strategies
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("context_management")
except Exception as e:
    print(f"[CONTEXT_MANAGEMENT] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Initialize pattern store
_pattern_store = get_pattern_store()

# Load initial patterns if not already present
try:
    initialize_context_management_patterns()
except Exception as e:
    print(f"[CONTEXT_MANAGEMENT] Failed to initialize patterns: {e}")

# Track which pattern was used for the current decay
_current_pattern: Optional[Pattern] = None

# =============================================================================
# PHASE 2: ACTION REQUEST TRACKING
# =============================================================================
# Track the last actionable request so follow-ups like "do it please" can
# execute the previous request.

from dataclasses import dataclass, field
from datetime import datetime
import re

@dataclass
class ActionRequest:
    """Represents a pending actionable request that can be executed on confirmation."""
    request_type: str  # e.g., "write_story", "search", "explain", "create"
    original_text: str  # The original user request text
    payload: Dict[str, Any]  # Structured data extracted from the request
    timestamp: datetime = field(default_factory=datetime.now)
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_type": self.request_type,
            "original_text": self.original_text,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "executed": self.executed,
        }


# Module-level storage for the last open action request
_last_open_action_request: Optional[ActionRequest] = None


# Patterns for detecting confirmation/acknowledgment messages
CONFIRMATION_PATTERNS = [
    r"^do\s+it\s*,?\s*please\.?$",
    r"^do\s+it\.?$",
    r"^yes\s*,?\s*please\.?$",
    r"^go\s+ahead\.?$",
    r"^ok\s*,?\s*do\s+it\.?$",
    r"^please\s+do\.?$",
    r"^please\s+do\s+it\.?$",
    r"^yes\s*,?\s*do\s+it\.?$",
    r"^yeah\s*,?\s*do\s+it\.?$",
    r"^sure\s*,?\s*go\s+ahead\.?$",
    r"^alright\s*,?\s*do\s+it\.?$",
    r"^ok(ay)?\s*,?\s*please\.?$",
    r"^proceed\.?$",
    r"^execute\.?$",
    r"^make\s+it\s+happen\.?$",
    r"^do\s+that\.?$",
]


# Patterns for detecting actionable requests (not just questions)
# Format: (pattern, request_type, topic_group_index)
# topic_group_index: which capture group contains the topic (0 = no specific topic group)
ACTION_REQUEST_PATTERNS = [
    # Write/Create patterns with topic extraction
    # "write a story about birds" -> topic = "birds"
    (r"\b(can you |could you )?(write|create|make|generate|compose)\s+(?:me\s+)?(?:a\s+)?(story|poem|essay|script|article|song|code|function|program)\s+(?:about|on|regarding|for)\s+(.+)", "write_content"),
    (r"\b(can you |could you )?(write|create|make|generate|compose)\s+(?:me\s+)?(?:a\s+)?(story|poem|essay|script|article|song)\b", "write_content"),
    (r"\b(write|create|make|generate|compose)\s+(?:me\s+)?(?:a\s+)?(.+)", "write_content"),
    # Search patterns
    (r"\b(can you |could you )?(search|look up|find|research)\s+(?:for\s+)?(.+)", "search"),
    (r"\b(can you |could you )?(search|look up|find|research)\b", "search"),
    # Explain patterns
    (r"\b(can you |could you )?(explain|describe|tell me about|elaborate on)\s+(.+)", "explain"),
    (r"\b(can you |could you )?(explain|describe|tell me about|elaborate on)\b", "explain"),
    # Calculate/Compute patterns
    (r"\b(can you |could you )?(calculate|compute|figure out|determine)\s+(.+)", "calculate"),
    (r"\b(can you |could you )?(calculate|compute|figure out|determine)\b", "calculate"),
    # Run/Execute patterns
    (r"\b(can you |could you )?(run|execute|perform|do)\b.*\b(command|script|code|program)\b", "execute"),
    # Summarize patterns
    (r"\b(can you |could you )?(summarize|sum up|give me a summary)\s+(?:of\s+)?(.+)", "summarize"),
    (r"\b(can you |could you )?(summarize|sum up|give me a summary)\b", "summarize"),
    # Translate patterns
    (r"\b(can you |could you )?(translate)\s+(.+)", "translate"),
    (r"\b(can you |could you )?(translate)\b", "translate"),
]


def is_confirmation_message(text: str) -> bool:
    """
    Check if a message is a short acknowledgment/confirmation like "do it please".

    Args:
        text: The user message text

    Returns:
        True if the message matches a confirmation pattern
    """
    if not text:
        return False

    text_clean = text.strip().lower()

    # Only short messages can be confirmations
    if len(text_clean) > 50:
        return False

    for pattern in CONFIRMATION_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True

    return False


def extract_action_request(text: str) -> Optional[ActionRequest]:
    """
    Extract an actionable request from user text.

    If the text matches a pattern for an actionable request (like "can you write
    me a story about birds"), extract the request type and payload.

    Args:
        text: The user message text

    Returns:
        ActionRequest if the text is an actionable request, None otherwise
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    for pattern, request_type in ACTION_REQUEST_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            # Extract the subject/topic from the request
            # For write_content requests, try to extract:
            # 1. The content type (story, poem, etc.)
            # 2. The topic (what it's about)
            groups = match.groups()
            content_type = ""
            topic = ""

            if request_type == "write_content":
                # Look for content type (story, poem, etc.) and topic
                for i, g in enumerate(groups):
                    if g and g.strip():
                        g_clean = g.strip()
                        # Check if this is a content type
                        if g_clean in ["story", "poem", "essay", "script", "article", "song", "code", "function", "program"]:
                            content_type = g_clean
                        # The last non-empty, non-content-type group is likely the topic
                        elif g_clean not in ["can you ", "could you ", "can you", "could you"]:
                            topic = g_clean

                # If we didn't find a topic in groups, try to extract it from the full text
                if not topic:
                    # Try to extract topic after "about", "on", "regarding", "for"
                    topic_match = re.search(r"\b(?:about|on|regarding|for)\s+(.+?)(?:\?|$)", text_lower)
                    if topic_match:
                        topic = topic_match.group(1).strip()

            else:
                # For other request types, use the last non-empty group as subject
                for g in reversed(groups):
                    if g and g.strip() and g.strip() not in ["can you ", "could you "]:
                        topic = g.strip()
                        break

            # Create the action request with structured payload
            payload = {
                "subject": topic or content_type,
                "matched_pattern": pattern,
                "match_groups": [g for g in groups if g],
            }

            # Add specific fields for write_content requests
            if request_type == "write_content":
                payload["intent"] = "write_story" if content_type == "story" else f"write_{content_type}" if content_type else "write_content"
                payload["content_type"] = content_type
                payload["topic"] = topic
                payload["arguments"] = {"topic": topic} if topic else {}

            print(f"[CONTEXT_MANAGER] Extracted action request: type={request_type}, content_type={content_type}, topic={topic}")

            return ActionRequest(
                request_type=request_type,
                original_text=text,
                payload=payload
            )

    return None


def store_action_request(request: ActionRequest) -> None:
    """
    Store an actionable request for potential follow-up execution.

    Args:
        request: The ActionRequest to store
    """
    global _last_open_action_request
    _last_open_action_request = request
    print(f"[CONTEXT_MANAGER] Stored action request: type={request.request_type}, text='{request.original_text[:50]}...'")


def get_last_action_request() -> Optional[ActionRequest]:
    """
    Get the last stored action request.

    Returns:
        The last ActionRequest, or None if none stored
    """
    return _last_open_action_request


def clear_action_request() -> None:
    """Clear the stored action request."""
    global _last_open_action_request
    _last_open_action_request = None


def mark_action_executed() -> None:
    """Mark the current action request as executed."""
    global _last_open_action_request
    if _last_open_action_request:
        _last_open_action_request.executed = True


def get_follow_up_context() -> Optional[Dict[str, Any]]:
    """
    Get context for a follow-up execution.

    If there's a pending action request and the user sends a confirmation,
    this returns the context needed to execute that action.

    Returns:
        Dict with action context if available, None otherwise
    """
    if not _last_open_action_request:
        return None

    if _last_open_action_request.executed:
        return None

    return {
        "is_follow_up": True,
        "original_request": _last_open_action_request.original_text,
        "request_type": _last_open_action_request.request_type,
        "payload": _last_open_action_request.payload,
        "timestamp": _last_open_action_request.timestamp.isoformat(),
    }


def _ensure_pattern_object(pattern: Any, default_signature: str = "global:default") -> Optional[Pattern]:
    """
    Ensure that a pattern is a proper Pattern object.

    Handles the bug where _current_pattern can become a string or dict
    instead of a Pattern object.

    Args:
        pattern: The pattern to normalize
        default_signature: Default signature if conversion is needed

    Returns:
        A proper Pattern object, or None if pattern is None/invalid
    """
    if pattern is None:
        return None

    if isinstance(pattern, Pattern):
        return pattern

    if isinstance(pattern, str):
        # Pattern is a plain string - convert to Pattern object
        print(f"[CONTEXT_MANAGEMENT] WARNING: Converting string to Pattern: {pattern[:50]}")
        return Pattern(
            brain="context_management",
            signature=pattern if pattern else default_signature,
            action={},
            score=0.5
        )

    if isinstance(pattern, dict):
        # Pattern is a dict - convert to Pattern object
        print(f"[CONTEXT_MANAGEMENT] WARNING: Converting dict to Pattern")
        return Pattern(
            brain="context_management",
            signature=str(pattern.get("signature", pattern.get("text", default_signature))),
            action=pattern.get("action", {}) if isinstance(pattern.get("action"), dict) else {},
            score=float(pattern.get("score", 0.5))
        )

    # Unknown type - log and return default
    print(f"[CONTEXT_MANAGEMENT] WARNING: Unknown pattern type: {type(pattern).__name__}")
    return Pattern(
        brain="context_management",
        signature=default_signature,
        action={},
        score=0.5
    )


def _classify_decay_type(ctx: Dict[str, Any]) -> str:
    """Classify context to determine which decay pattern to use."""
    if not isinstance(ctx, dict) or len(ctx) < 2:
        return "global:default_decay"

    # Check for specific content types
    if any(k in ctx for k in ["answer", "response", "output"]):
        return "global:answer_decay"
    if any(k in ctx for k in ["affect", "emotion", "sentiment"]):
        return "global:affect_decay"

    return "global:default_decay"


def apply_decay(
    ctx: Dict[str, Any],
    decay: float = 0.9,
    session_context: Optional[Dict[str, Any]] = None,
    _is_recursive: bool = False
) -> Dict[str, Any]:
    """
    Apply temporal decay to numeric fields in a context dictionary using learned patterns.

    Numeric values (ints or floats) are multiplied by the provided
    decay factor, reducing their magnitude over time.  Nested
    dictionaries are decayed recursively.  Non‑numeric fields are
    preserved unchanged.

    Args:
        ctx: The context dictionary to decay.
        decay: A multiplicative factor between 0 and 1.  Values
            closer to zero cause faster forgetting.
        session_context: Optional context about the session type (for pattern matching)
        _is_recursive: Internal flag to avoid pattern lookup on recursive calls

    Returns:
        A new dictionary with decayed values; the original context is
        not modified.
    """
    global _current_pattern

    if not isinstance(ctx, dict):
        return {}

    # Only do pattern lookup at the top level, not in recursive calls
    effective_decay = decay

    if not _is_recursive:
        global _current_pattern

        # Classify the decay type
        decay_type = _classify_decay_type(ctx)

        # Try to find a learned pattern
        pattern = _pattern_store.get_best_pattern(
            brain="context_management",
            signature=decay_type,
            score_threshold=0.0  # Accept any pattern above 0
        )

        # Track which pattern we're using for later updates
        # Use _ensure_pattern_object to handle any type safely
        _current_pattern = _ensure_pattern_object(pattern, decay_type)

        # Determine effective decay factor
        if _current_pattern and hasattr(_current_pattern, 'score') and _current_pattern.score > -0.5:
            action = _current_pattern.action if hasattr(_current_pattern, 'action') else {}
            # Defensive check: ensure action is a dict before calling .get()
            if isinstance(action, dict):
                learned_decay = action.get("decay_factor", decay)
                effective_decay = learned_decay
                print(f"[CONTEXT_MANAGEMENT] Using learned decay: {effective_decay:.3f} for {decay_type}")
            else:
                print(f"[CONTEXT_MANAGEMENT] Warning: pattern.action is not a dict (type={type(action).__name__}), using default decay")
                effective_decay = decay
        else:
            print(f"[CONTEXT_MANAGEMENT] Using default decay: {effective_decay:.3f}")

    # Apply decay recursively
    decayed: Dict[str, Any] = {}
    for key, value in ctx.items():
        if isinstance(value, dict):
            # Pass _is_recursive=True to avoid pattern lookup in nested dicts
            decayed[key] = apply_decay(value, effective_decay, session_context, _is_recursive=True)
        elif isinstance(value, (int, float)):
            decayed[key] = value * effective_decay
        else:
            decayed[key] = value

    return decayed


def update_from_verdict(verdict: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Update the current pattern's score based on SELF_REVIEW/Teacher verdict.

    For CONTEXT_MANAGEMENT, we also consider:
    - context_loss: caused incoherent answer → negative reward
    - too_much_context: kept too much, repetitive → negative reward

    Args:
        verdict: One of 'ok', 'minor_issue', 'major_issue'
        metadata: Optional dict with 'context_loss', 'too_much_context' flags
    """
    global _current_pattern

    if not _current_pattern:
        print("[CONTEXT_MANAGEMENT] No current pattern to update")
        return

    # Normalize _current_pattern to ensure it's a proper Pattern object
    _current_pattern = _ensure_pattern_object(_current_pattern, "global:default")

    if not _current_pattern:
        print("[CONTEXT_MANAGEMENT] Could not normalize current pattern, skipping update")
        return

    # Start with base reward from verdict
    reward = verdict_to_reward(verdict)

    # Adjust based on context-specific metadata
    if metadata and isinstance(metadata, dict):
        if metadata.get("context_loss"):
            reward -= 0.5  # Context loss is bad
        if metadata.get("too_much_context"):
            reward -= 0.3  # Too much context is wasteful

    # Clamp reward to [-1, 1]
    reward = max(-1.0, min(1.0, reward))

    # Update pattern score
    _pattern_store.update_pattern_score(
        pattern=_current_pattern,
        reward=reward,
        alpha=0.85  # Learning rate: 0.85 = slower, more stable
    )

    print(f"[CONTEXT_MANAGEMENT] Updated pattern {_current_pattern.signature} "
          f"based on verdict={verdict} (reward={reward:+.1f})")


def reconstruct_context(history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Reconstruct a context from a sequence of prior contexts.

    When resuming a long session, it may be necessary to combine
    multiple context snapshots (e.g. from previous turns) into a
    single current context.  This function iterates through the
    provided contexts in order, merging their keys.  Later contexts
    override earlier ones for scalar values.  For lists, values are
    concatenated.  Nested dictionaries are merged recursively.

    Args:
        history: An iterable of context dictionaries ordered from
            oldest to newest.
    Returns:
        A single context dictionary representing the merged result.
    """
    merged: Dict[str, Any] = {}
    for ctx in history:
        if not isinstance(ctx, dict):
            continue
        for key, value in ctx.items():
            if key not in merged:
                merged[key] = value
            else:
                existing = merged[key]
                # If both are dicts, merge recursively
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged[key] = reconstruct_context([existing, value])
                # If both are lists, append new items
                elif isinstance(existing, list) and isinstance(value, list):
                    merged[key] = existing + value
                # Otherwise, prefer the newer value
                else:
                    merged[key] = value
    return merged


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context Management service API.

    Supported operations:
    - APPLY_DECAY: Apply temporal decay to context
    - RECONSTRUCT: Reconstruct context from history
    - HEALTH: Health check

    Args:
        msg: Request with 'op' and optional 'payload'

    Returns:
        Response dict with 'ok' and 'payload' or 'error'
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload") or {}

    if op == "APPLY_DECAY":
        try:
            ctx = payload.get("context", {})
            decay = payload.get("decay", 0.9)
            decayed = apply_decay(ctx, decay)
            return {
                "ok": True,
                "payload": {
                    "context": decayed
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "DECAY_FAILED",
                    "message": str(e)
                }
            }

    if op == "RECONSTRUCT":
        try:
            history = payload.get("history", [])
            merged = reconstruct_context(history)
            return {
                "ok": True,
                "payload": {
                    "context": merged
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "RECONSTRUCT_FAILED",
                    "message": str(e)
                }
            }

    if op == "HEALTH":
        num_patterns = len(_pattern_store.get_patterns_by_brain("context_management"))
        return {
            "ok": True,
            "payload": {
                "status": "operational",
                "patterns_loaded": num_patterns
            }
        }

    if op == "UPDATE_FROM_VERDICT":
        try:
            verdict = str(payload.get("verdict", "ok"))
            metadata = payload.get("metadata")
            update_from_verdict(verdict, metadata)
            return {
                "ok": True,
                "payload": {
                    "updated": True
                }
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "UPDATE_FAILED",
                    "message": str(e)
                }
            }

    # PHASE 2: Action request tracking operations
    if op == "CHECK_CONFIRMATION":
        # Check if the message is a confirmation like "do it please"
        text = str(payload.get("text", ""))
        is_confirmation = is_confirmation_message(text)
        follow_up_context = None
        if is_confirmation:
            follow_up_context = get_follow_up_context()
        return {
            "ok": True,
            "payload": {
                "is_confirmation": is_confirmation,
                "has_pending_action": follow_up_context is not None,
                "follow_up_context": follow_up_context,
            }
        }

    if op == "EXTRACT_ACTION_REQUEST":
        # Extract an actionable request from user text
        text = str(payload.get("text", ""))
        action_request = extract_action_request(text)
        if action_request:
            # Store it for potential follow-up
            store_action_request(action_request)
            return {
                "ok": True,
                "payload": {
                    "is_action_request": True,
                    "action_request": action_request.to_dict(),
                }
            }
        else:
            return {
                "ok": True,
                "payload": {
                    "is_action_request": False,
                    "action_request": None,
                }
            }

    if op == "GET_PENDING_ACTION":
        # Get the last pending action request
        action_request = get_last_action_request()
        if action_request:
            return {
                "ok": True,
                "payload": {
                    "has_pending_action": True,
                    "action_request": action_request.to_dict(),
                }
            }
        else:
            return {
                "ok": True,
                "payload": {
                    "has_pending_action": False,
                    "action_request": None,
                }
            }

    if op == "MARK_ACTION_EXECUTED":
        # Mark the current action as executed
        mark_action_executed()
        return {
            "ok": True,
            "payload": {
                "marked": True
            }
        }

    if op == "CLEAR_ACTION_REQUEST":
        # Clear the stored action request
        clear_action_request()
        return {
            "ok": True,
            "payload": {
                "cleared": True
            }
        }

    return {
        "ok": False,
        "error": {
            "code": "UNSUPPORTED_OP",
            "message": op
        }
    }


# Standard service contract: handle is the entry point
service_api = handle