"""
Initial INTEGRATOR Patterns
===========================

Define the starting routing patterns for the INTEGRATOR brain.
These patterns specify which pipeline/brain combo to run for different input types.

Over time, INTEGRATOR learns which patterns work well and adjusts scores based on
SELF_REVIEW/Teacher feedback.
"""

from typing import List
from brains.cognitive.pattern_store import Pattern, create_pattern_id


def get_initial_patterns() -> List[Pattern]:
    """
    Return initial INTEGRATOR routing patterns.

    Each pattern specifies:
    - signature: type of user input
    - context_tags: additional context markers
    - action: pipeline configuration (which brains to run, in what order)
    - score: initial quality estimate (0.0 = neutral)
    - frozen: whether pattern can be modified by learning
    """

    patterns = []

    # Pattern 1: Direct factual questions
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "direct_question:what_is"),
        brain="integrator",
        signature="direct_question:what_is",
        context_tags=["direct_question", "factual"],
        action={
            "pipeline": [
                "SENSORIUM",
                "AFFECT_PRIORITY",
                "INTEGRATOR",
                "REASONING",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": []
        },
        score=0.3,  # Reasonable default
        frozen=False
    ))

    # Pattern 2: How-to questions (may need research)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "direct_question:how_to"),
        brain="integrator",
        signature="direct_question:how_to",
        context_tags=["direct_question", "instructional"],
        action={
            "pipeline": [
                "SENSORIUM",
                "AFFECT_PRIORITY",
                "INTEGRATOR",
                "RESEARCH_MANAGER",
                "REASONING",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": []
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 3: Self-diagnostics (bypass normal pipeline)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "self_diag:diagnose_web"),
        brain="integrator",
        signature="self_diag:diagnose_web",
        context_tags=["self_diag", "meta"],
        action={
            "pipeline": [
                "SELF_MODEL"  # Go straight to self-model for introspection
            ],
            "force_teacher": False,
            "skip_brains": ["SENSORIUM", "REASONING"]  # Skip normal processing
        },
        score=0.5,  # Higher - we know this works well
        frozen=False
    ))

    # Pattern 4: Tool calls (action-oriented)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "tool_call:pc_scan"),
        brain="integrator",
        signature="tool_call:pc_scan",
        context_tags=["tool_call", "action"],
        action={
            "pipeline": [
                "SENSORIUM",
                "ACTION_ENGINE",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": ["REASONING"]  # Direct action, no deep reasoning
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 5: Casual statements (minimal processing)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "casual_statement"),
        brain="integrator",
        signature="casual_statement",
        context_tags=["casual", "social"],
        action={
            "pipeline": [
                "SENSORIUM",
                "AFFECT_PRIORITY",
                "LANGUAGE"
            ],
            "force_teacher": False,
            "skip_brains": ["RESEARCH_MANAGER", "REASONING"]  # Keep it light
        },
        score=0.2,
        frozen=False
    ))

    # Pattern 6: Deep research commands
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "research_command:deep"),
        brain="integrator",
        signature="research_command:deep",
        context_tags=["research", "explicit"],
        action={
            "pipeline": [
                "SENSORIUM",
                "AFFECT_PRIORITY",
                "RESEARCH_MANAGER",
                "REASONING",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": []
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 7: Contradiction resolution (FROZEN - safety critical)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "contradiction_detected"),
        brain="integrator",
        signature="contradiction_detected",
        context_tags=["contradiction", "safety"],
        action={
            "pipeline": [
                "REASONING",  # Always use reasoning for contradictions
                "TEACHER",    # Always consult teacher
                "SELF_REVIEW"
            ],
            "force_teacher": True,
            "skip_brains": []
        },
        score=0.8,  # High confidence - this is a safety rule
        frozen=True  # Never modify this pattern
    ))

    # Pattern 8: Unanswered questions (FROZEN - safety critical)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "unanswered_question"),
        brain="integrator",
        signature="unanswered_question",
        context_tags=["unanswered", "safety"],
        action={
            "pipeline": [
                "LANGUAGE",   # Language brain handles Q&A
                "REASONING",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": []
        },
        score=0.7,
        frozen=True  # Never modify this pattern
    ))

    # Pattern 9: Multi-brain conflicts
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "multi_brain_conflict"),
        brain="integrator",
        signature="multi_brain_conflict",
        context_tags=["conflict", "complex"],
        action={
            "pipeline": [
                "SENSORIUM",
                "INTEGRATOR",  # Recursive integration
                "REASONING",
                "TEACHER",
                "SELF_REVIEW"
            ],
            "force_teacher": True,  # Get teacher input for conflicts
            "skip_brains": []
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 10: Follow-up questions (context-aware expansion)
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "follow_up_question"),
        brain="integrator",
        signature="follow_up_question",
        context_tags=["follow_up", "context_dependent"],
        action={
            "pipeline": [
                "SYSTEM_HISTORY",  # Get last topic first
                "REASONING",       # Reasoning handles context + expansion
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": ["SENSORIUM"]  # Skip normalization, already done
        },
        score=0.5,  # Good default - follow-ups are common
        frozen=False
    ))

    # Pattern 11: Default fallback
    patterns.append(Pattern(
        id=create_pattern_id("integrator", "default"),
        brain="integrator",
        signature="default",
        context_tags=["fallback"],
        action={
            "pipeline": [
                "SENSORIUM",
                "AFFECT_PRIORITY",
                "INTEGRATOR",
                "REASONING",
                "SELF_REVIEW"
            ],
            "force_teacher": False,
            "skip_brains": []
        },
        score=0.0,  # Neutral - used when nothing else matches
        frozen=True  # Never remove the fallback
    ))

    return patterns


def initialize_integrator_patterns():
    """Load initial patterns into the pattern store if not already present."""
    from brains.cognitive.pattern_store import get_pattern_store

    store = get_pattern_store()
    existing = store.get_patterns_by_brain("integrator")

    if existing:
        print(f"[INTEGRATOR_INIT] Found {len(existing)} existing patterns, skipping init")
        return

    # No existing patterns, load initial set
    patterns = get_initial_patterns()
    print(f"[INTEGRATOR_INIT] Loading {len(patterns)} initial patterns")

    for pattern in patterns:
        store.store_pattern(pattern)

    print("[INTEGRATOR_INIT] Initial patterns loaded successfully")
