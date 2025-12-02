"""
Initial CONTEXT_MANAGEMENT Patterns
====================================

Define the starting context management patterns for the CONTEXT_MANAGEMENT brain.
These patterns specify how aggressively to forget/summarize context.

Over time, CONTEXT_MANAGEMENT learns optimal configurations based on coherence and efficiency.
"""

from typing import List
from brains.cognitive.pattern_store import Pattern, create_pattern_id


def get_initial_patterns() -> List[Pattern]:
    """
    Return initial CONTEXT_MANAGEMENT patterns.

    Each pattern specifies:
    - signature: context situation type
    - context_tags: session characteristics
    - action: decay_factor, max_turns_in_context, summary_trigger_threshold
    - score: initial quality estimate
    - frozen: whether pattern can be modified by learning
    """

    patterns = []

    # Pattern 1: Long session (preserve more context)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "context:long_session"),
        brain="context_management",
        signature="context:long_session",
        context_tags=["long_session", "extended"],
        action={
            "decay_factor": 0.95,  # Slow decay
            "max_turns_in_context": 20,
            "summary_trigger_threshold": 15
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 2: Short session (faster decay)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "context:short_session"),
        brain="context_management",
        signature="context:short_session",
        context_tags=["short_session", "quick"],
        action={
            "decay_factor": 0.90,  # Moderate decay
            "max_turns_in_context": 10,
            "summary_trigger_threshold": 8
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 3: Default decay (general use)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "global:default_decay"),
        brain="context_management",
        signature="global:default_decay",
        context_tags=["default", "general"],
        action={
            "decay_factor": 0.92,
            "max_turns_in_context": 15,
            "summary_trigger_threshold": 12
        },
        score=0.5,  # Good default
        frozen=False
    ))

    # Pattern 4: Answer decay (after providing an answer)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "global:answer_decay"),
        brain="context_management",
        signature="global:answer_decay",
        context_tags=["answer", "response"],
        action={
            "decay_factor": 0.88,  # Faster decay after answering
            "max_turns_in_context": 12,
            "summary_trigger_threshold": 10
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 5: Affect decay (emotional context fades faster)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "global:affect_decay"),
        brain="context_management",
        signature="global:affect_decay",
        context_tags=["affect", "emotional"],
        action={
            "decay_factor": 0.85,  # Fast decay for emotional content
            "max_turns_in_context": 8,
            "summary_trigger_threshold": 6
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 6: Research session (preserve lots of context)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "context:research_session"),
        brain="context_management",
        signature="context:research_session",
        context_tags=["research", "deep"],
        action={
            "decay_factor": 0.98,  # Very slow decay
            "max_turns_in_context": 30,
            "summary_trigger_threshold": 25
        },
        score=0.5,
        frozen=False
    ))

    # Pattern 7: Coding session (keep recent code context)
    patterns.append(Pattern(
        id=create_pattern_id("context_management", "context:coding_session"),
        brain="context_management",
        signature="context:coding_session",
        context_tags=["coding", "implementation"],
        action={
            "decay_factor": 0.93,
            "max_turns_in_context": 18,
            "summary_trigger_threshold": 15
        },
        score=0.4,
        frozen=False
    ))

    return patterns


def initialize_context_management_patterns():
    """Load initial patterns into the pattern store if not already present."""
    from brains.cognitive.pattern_store import get_pattern_store

    store = get_pattern_store()
    existing = store.get_patterns_by_brain("context_management")

    if existing:
        print(f"[CONTEXT_MANAGEMENT_INIT] Found {len(existing)} existing patterns, skipping init")
        return

    # No existing patterns, load initial set
    patterns = get_initial_patterns()
    print(f"[CONTEXT_MANAGEMENT_INIT] Loading {len(patterns)} initial patterns")

    for pattern in patterns:
        store.store_pattern(pattern)

    print("[CONTEXT_MANAGEMENT_INIT] Initial patterns loaded successfully")
