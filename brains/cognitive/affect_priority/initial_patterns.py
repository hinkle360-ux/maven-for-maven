"""
Initial AFFECT_PRIORITY Patterns
================================

Define the starting affect handling patterns for the AFFECT_PRIORITY brain.
These patterns specify tone, length, and safety behaviors for different emotional contexts.

Over time, AFFECT_PRIORITY learns which patterns work well based on user feedback.
"""

from typing import List
from brains.cognitive.pattern_store import Pattern, create_pattern_id


def get_initial_patterns() -> List[Pattern]:
    """
    Return initial AFFECT_PRIORITY patterns.

    Each pattern specifies:
    - signature: affect situation type
    - context_tags: emotional markers
    - action: tone, length, safety configuration
    - score: initial quality estimate
    - frozen: whether pattern can be modified by learning
    """

    patterns = []

    # Pattern 1: High stress + direct question
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "high_stress+direct_question"),
        brain="affect_priority",
        signature="high_stress+direct_question",
        context_tags=["high_stress", "direct_question", "safety"],
        action={
            "tone": "de-escalating",
            "max_length": "normal",
            "extra_teacher_checks": 1,
            "force_self_review_mode": "deep"
        },
        score=0.5,  # Reasonable default for safety
        frozen=False
    ))

    # Pattern 2: Small talk (casual, low arousal)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "small_talk"),
        brain="affect_priority",
        signature="small_talk",
        context_tags=["casual", "low_arousal"],
        action={
            "tone": "neutral",
            "max_length": "short",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 3: Task request (normal working mode)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "task_request"),
        brain="affect_priority",
        signature="task_request",
        context_tags=["task", "request"],
        action={
            "tone": "neutral",
            "max_length": "normal",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 4: Direct question (factual)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "direct_question"),
        brain="affect_priority",
        signature="direct_question",
        context_tags=["direct_question", "factual"],
        action={
            "tone": "neutral",
            "max_length": "normal",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 5: High stress + self-blame (FROZEN - safety critical)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "high_stress+self_blame"),
        brain="affect_priority",
        signature="high_stress+self_blame",
        context_tags=["high_stress", "self_blame", "safety"],
        action={
            "tone": "supportive",
            "max_length": "extended",
            "extra_teacher_checks": 2,
            "force_self_review_mode": "deep"
        },
        score=0.8,  # High confidence - this is a safety rule
        frozen=True  # Never modify this safety pattern
    ))

    # Pattern 6: Research command (technical, neutral)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "neutral+research_command"),
        brain="affect_priority",
        signature="neutral+research_command",
        context_tags=["neutral", "research", "technical"],
        action={
            "tone": "neutral",
            "max_length": "extended",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 7: Urgent request (time-sensitive)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "urgent"),
        brain="affect_priority",
        signature="urgent",
        context_tags=["urgent", "high_arousal"],
        action={
            "tone": "blunt",  # Get to the point quickly
            "max_length": "short",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 8: Positive affect (happy, grateful)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "positive"),
        brain="affect_priority",
        signature="positive",
        context_tags=["positive", "grateful"],
        action={
            "tone": "neutral",  # Match user's positive energy
            "max_length": "normal",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 9: Negative affect (worried, sad)
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "negative"),
        brain="affect_priority",
        signature="negative",
        context_tags=["negative", "worried"],
        action={
            "tone": "de-escalating",
            "max_length": "normal",
            "extra_teacher_checks": 1,
            "force_self_review_mode": "deep"
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 10: Default fallback
    patterns.append(Pattern(
        id=create_pattern_id("affect_priority", "default"),
        brain="affect_priority",
        signature="default",
        context_tags=["fallback"],
        action={
            "tone": "neutral",
            "max_length": "normal",
            "extra_teacher_checks": 0,
            "force_self_review_mode": "quick"
        },
        score=0.0,  # Neutral - used when nothing else matches
        frozen=True  # Never remove the fallback
    ))

    return patterns


def initialize_affect_priority_patterns():
    """Load initial patterns into the pattern store if not already present."""
    from brains.cognitive.pattern_store import get_pattern_store

    store = get_pattern_store()
    existing = store.get_patterns_by_brain("affect_priority")

    if existing:
        print(f"[AFFECT_PRIORITY_INIT] Found {len(existing)} existing patterns, skipping init")
        return

    # No existing patterns, load initial set
    patterns = get_initial_patterns()
    print(f"[AFFECT_PRIORITY_INIT] Loading {len(patterns)} initial patterns")

    for pattern in patterns:
        store.store_pattern(pattern)

    print("[AFFECT_PRIORITY_INIT] Initial patterns loaded successfully")
