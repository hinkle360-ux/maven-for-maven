
from __future__ import annotations
import json, time, re
from pathlib import Path
from typing import Dict, Any, Optional

# Root directories
HERE = Path(__file__).resolve().parent
BRAIN_ROOT = HERE.parent

# Pattern store for unified learning across all brains
from brains.cognitive.pattern_store import (
    get_pattern_store,
    Pattern,
    verdict_to_reward
)
from brains.cognitive.affect_priority.initial_patterns import (
    initialize_affect_priority_patterns
)

# Teacher integration for learning emotional response and priority patterns
try:
    from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
    _teacher_helper = TeacherHelper("affect_priority")
except Exception as e:
    print(f"[AFFECT_PRIORITY] Teacher helper not available: {e}")
    _teacher_helper = None  # type: ignore

# Continuation helpers for follow-up and conversation context tracking
try:
    from brains.cognitive.continuation_helpers import (
        is_continuation,
        get_conversation_context,
        create_routing_hint,
        extract_continuation_intent,
        enhance_query_with_context
    )
except Exception as e:
    print(f"[AFFECT_PRIORITY] Continuation helpers not available: {e}")
    # Provide fallback stubs
    is_continuation = lambda text, context=None: False  # type: ignore
    get_conversation_context = lambda: {}  # type: ignore
    create_routing_hint = lambda *args, **kwargs: {}  # type: ignore
    extract_continuation_intent = lambda text: "unknown"  # type: ignore
    enhance_query_with_context = lambda query, context: query  # type: ignore

# Initialize pattern store
_pattern_store = get_pattern_store()

# Load initial patterns if not already present
try:
    initialize_affect_priority_patterns()
except Exception as e:
    print(f"[AFFECT_PRIORITY] Failed to initialize patterns: {e}")

# Track which pattern was used for the current computation
_current_pattern: Optional[Pattern] = None


def _classify_situation(text: str) -> str:
    """
    Classify text into a situation type for pattern matching.

    Returns one of:
    - high_stress+direct_question
    - high_stress+self_blame
    - small_talk
    - task_request
    - direct_question
    - urgent
    - positive
    - negative
    - neutral+research_command
    - default
    """
    t = (text or "")
    lower = t.lower()
    exclamations = t.count("!")

    # Check for stress markers
    stress_words = ["urgent", "asap", "immediately", "worried", "stress", "afraid", "fear"]
    is_high_stress = exclamations >= 2 or any(word in lower for word in stress_words)

    # Check for self-blame
    self_blame_words = ["my fault", "i failed", "i'm sorry", "i apologize", "i messed up"]
    has_self_blame = any(phrase in lower for phrase in self_blame_words)

    # High stress combinations (highest priority)
    if is_high_stress and has_self_blame:
        return "high_stress+self_blame"
    if is_high_stress and "?" in t:
        return "high_stress+direct_question"

    # Urgent (no stress markers, but explicit urgency)
    if any(word in lower for word in ["urgent", "asap", "immediately"]):
        return "urgent"

    # Small talk
    greetings = ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]
    if any(word in lower for word in greetings):
        return "small_talk"

    # Task requests
    request_words = ["please", "can you", "could you", "would you", "help me"]
    if any(phrase in lower for phrase in request_words):
        return "task_request"

    # Research commands
    if any(word in lower for word in ["research", "investigate", "study", "analyze"]):
        return "neutral+research_command"

    # Direct questions
    if "?" in t:
        return "direct_question"

    # Sentiment-based classification
    positive_words = ["happy", "joy", "love", "good", "great", "awesome", "excited", "excellent"]
    negative_words = ["worried", "sad", "terrible", "bad", "upset", "anxious", "hate", "angry"]

    pos_count = sum(1 for word in positive_words if word in lower)
    neg_count = sum(1 for word in negative_words if word in lower)

    if pos_count > neg_count and pos_count > 0:
        return "positive"
    if neg_count > pos_count and neg_count > 0:
        return "negative"

    return "default"

def _compute_affect(text: str) -> Dict[str, Any]:
    """
    Compute affect response using learned patterns from the pattern store.

    Enhanced with continuation detection to avoid re-escalating affect
    on follow-up questions.  Continuations maintain or reduce the previous
    affect level rather than treating them as new urgent concerns.

    Returns configuration for tone, length, and safety checks based on
    the situation type classified from the text.
    """
    global _current_pattern

    t = (text or "")
    if not t:
        situation = "default"
    else:
        # -----------------------------------------------------------------
        # Continuation detection and affect adjustment
        #
        # Detect if this is a follow-up to avoid re-escalating urgency.
        # Follow-ups like "tell me more" or "what else" should maintain
        # or reduce the previous affect, not trigger a new high-stress
        # response.
        try:
            conv_context = get_conversation_context()
            is_cont = is_continuation(text)
            continuation_intent = extract_continuation_intent(text) if is_cont else "unknown"
        except Exception:
            conv_context = {}
            is_cont = False
            continuation_intent = "unknown"

        # Classify the base situation
        situation = _classify_situation(t)

        # For continuations, adjust situation to avoid re-escalation
        if is_cont:
            # Clarifications should be lower priority than new concerns
            if continuation_intent == "clarification":
                if situation in ["high_stress+direct_question", "high_stress+self_blame", "urgent"]:
                    situation = "direct_question"  # Downgrade from high stress
                    print(f"[AFFECT_PRIORITY] Continuation (clarification) detected, downgrading from high stress")
            # Expansions maintain current level but don't escalate
            elif continuation_intent == "expansion":
                if situation in ["high_stress+direct_question", "high_stress+self_blame", "urgent"]:
                    situation = "direct_question"  # Maintain but don't re-escalate
                    print(f"[AFFECT_PRIORITY] Continuation (expansion) detected, maintaining level")
            # Related topics are treated as new but gentle
            elif continuation_intent == "related":
                # Keep the situation but note it's a continuation
                print(f"[AFFECT_PRIORITY] Continuation (related topic) detected")

    print(f"[AFFECT_PRIORITY] Situation: {situation}")

    # Retrieve best matching pattern from store
    pattern = _pattern_store.get_best_pattern(
        brain="affect_priority",
        signature=situation,
        score_threshold=0.0  # Accept any pattern above 0
    )

    # Track which pattern we're using for later updates
    _current_pattern = pattern

    if pattern and pattern.score > -0.5:  # Use pattern if score is reasonable
        action = pattern.action
        print(f"[AFFECT_PRIORITY] Using learned pattern: {situation} -> {action.get('tone')}")

        # Return the pattern's action as the affect configuration
        return {
            "tone": action.get("tone", "neutral"),
            "max_length": action.get("max_length", "normal"),
            "extra_teacher_checks": action.get("extra_teacher_checks", 0),
            "force_self_review_mode": action.get("force_self_review_mode", "quick"),
            "situation_type": situation
        }
    else:
        # No good pattern found, use safe defaults
        print(f"[AFFECT_PRIORITY] No good pattern for {situation}, using safe defaults")
        return {
            "tone": "neutral",
            "max_length": "normal",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick",
            "situation_type": situation
        }


def update_from_verdict(verdict: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Update the current pattern's score based on SELF_REVIEW/Teacher verdict.

    This is the learning mechanism: after each interaction, we get feedback
    and use it to adjust pattern scores.

    Args:
        verdict: One of 'ok', 'minor_issue', 'major_issue'
        metadata: Optional metadata (not used by AFFECT_PRIORITY, but kept for API consistency)
    """
    global _current_pattern

    if not _current_pattern:
        print("[AFFECT_PRIORITY] No current pattern to update")
        return

    # Convert verdict to reward signal
    reward = verdict_to_reward(verdict)

    # Update pattern score
    _pattern_store.update_pattern_score(
        pattern=_current_pattern,
        reward=reward,
        alpha=0.85  # Learning rate: 0.85 = slower, more stable
    )

    print(f"[AFFECT_PRIORITY] Updated pattern {_current_pattern.signature} "
          f"based on verdict={verdict} (reward={reward:+.1f})")

def handle(msg):
    """
    Central entry point for the affect priority brain.
    Supports affect scoring using learned patterns from the pattern store.
    """
    from api.utils import generate_mid, success_response, error_response  # type: ignore
    op = (msg or {}).get("op", " ").upper()
    mid = msg.get("mid") or generate_mid()
    payload = msg.get("payload") or {}

    # HEALTH operation
    if op == "HEALTH":
        num_patterns = len(_pattern_store.get_patterns_by_brain("affect_priority"))
        return success_response(op, mid, {
            "status": "operational",
            "patterns_loaded": num_patterns
        })

    # SCORE operation: evaluate a text and return affect configuration
    if op == "SCORE":
        try:
            text = str(payload.get("text", ""))
        except Exception:
            text = ""
        config = _compute_affect(text)
        return success_response(op, mid, config)

    # UPDATE_FROM_VERDICT operation: learn from feedback
    if op == "UPDATE_FROM_VERDICT":
        try:
            verdict = str(payload.get("verdict", "ok"))
            update_from_verdict(verdict)
            return success_response(op, mid, {"updated": True})
        except Exception as e:
            return error_response(op, mid, "ERROR", f"Failed to update: {e}")

    # LEARN_FROM_RUN operation: learn from execution data
    if op == "LEARN_FROM_RUN":
        try:
            run_data = payload.get("run_data", {})
            if not isinstance(run_data, dict):
                return error_response(op, mid, "INVALID_INPUT", "run_data must be a dict")

            # Extract affect-relevant data from run
            verdict = run_data.get("verdict", "ok")
            situation_type = run_data.get("situation_type", "default")

            # Learn from the run by updating patterns
            update_from_verdict(verdict)

            print(f"[AFFECT_PRIORITY] Learned from run: situation={situation_type}, verdict={verdict}")

            return success_response(op, mid, {"success": True})
        except Exception as e:
            return error_response(op, mid, "ERROR", f"Failed to learn from run: {e}")

    # Unsupported operations
    return error_response(op, mid, "UNSUPPORTED_OP", op)

def bid_for_attention(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bid for attention based on affect priority needs.

    The affect priority brain bids higher when:
    - High stress or urgent situations are detected
    - Emotional content requires tonal adjustment
    - Safety checks are needed

    For continuations, adjusts priority to avoid re-escalation.
    """
    try:
        # Get language analysis results
        lang_info = ctx.get("stage_3_language", {}) or {}
        is_cont = lang_info.get("is_continuation", False)
        continuation_intent = lang_info.get("continuation_intent", "unknown")
        conv_context = lang_info.get("conversation_context", {})

        # Check if we have affect analysis data
        affect_info = ctx.get("affect_priority", {}) or {}
        situation = affect_info.get("situation_type", "default")
        tone = affect_info.get("tone", "neutral")

        # High priority for high-stress situations
        if situation in ["high_stress+direct_question", "high_stress+self_blame", "urgent"]:
            # For continuations, don't re-escalate
            if is_cont and continuation_intent in ["clarification", "expansion"]:
                routing_hint = create_routing_hint(
                    brain_name="affect_priority",
                    action="maintain_affect",
                    confidence=0.60,
                    context_tags=["continuation", "maintain_level", continuation_intent],
                    metadata={
                        "last_topic": conv_context.get("last_topic", ""),
                        "continuation_type": continuation_intent,
                        "situation": situation
                    }
                )
                return {
                    "brain_name": "affect_priority",
                    "priority": 0.60,
                    "reason": "continuation_maintain_affect",
                    "evidence": {"routing_hint": routing_hint, "is_continuation": is_cont},
                }
            else:
                # New high-stress situation
                routing_hint = create_routing_hint(
                    brain_name="affect_priority",
                    action="high_priority_response",
                    confidence=0.90,
                    context_tags=["high_stress", "urgent", situation],
                    metadata={"situation": situation}
                )
                return {
                    "brain_name": "affect_priority",
                    "priority": 0.90,
                    "reason": "high_stress_detected",
                    "evidence": {"routing_hint": routing_hint, "situation": situation},
                }

        # Medium priority for emotional content (positive/negative)
        if situation in ["positive", "negative"]:
            routing_hint = create_routing_hint(
                brain_name="affect_priority",
                action="emotional_response",
                confidence=0.55,
                context_tags=["emotional", situation],
                metadata={"is_continuation": is_cont, "tone": tone}
            )
            return {
                "brain_name": "affect_priority",
                "priority": 0.55,
                "reason": "emotional_content",
                "evidence": {"routing_hint": routing_hint, "situation": situation},
            }

        # Low priority for small talk
        if situation == "small_talk":
            routing_hint = create_routing_hint(
                brain_name="affect_priority",
                action="social",
                confidence=0.30,
                context_tags=["social", "low_priority"],
                metadata={}
            )
            return {
                "brain_name": "affect_priority",
                "priority": 0.30,
                "reason": "small_talk",
                "evidence": {"routing_hint": routing_hint},
            }

        # Default low priority
        routing_hint = create_routing_hint(
            brain_name="affect_priority",
            action="default",
            confidence=0.15,
            context_tags=["default"],
            metadata={"is_continuation": is_cont}
        )
        return {
            "brain_name": "affect_priority",
            "priority": 0.15,
            "reason": "default",
            "evidence": {"routing_hint": routing_hint},
        }
    except Exception:
        # On error, return safe default
        return {
            "brain_name": "affect_priority",
            "priority": 0.15,
            "reason": "default",
            "evidence": {},
        }

# Standard service contract: handle is the entry point
service_api = handle
