"""
Initial RESEARCH_MANAGER Patterns
==================================

Define the starting research configuration patterns for the RESEARCH_MANAGER brain.
These patterns specify depth, web access, and time budgets for different research topics.

Over time, RESEARCH_MANAGER learns optimal configurations based on quality and efficiency.
"""

from typing import List
from brains.cognitive.pattern_store import Pattern, create_pattern_id


def get_initial_patterns() -> List[Pattern]:
    """
    Return initial RESEARCH_MANAGER patterns.

    Each pattern specifies:
    - signature: topic or research type
    - context_tags: research characteristics
    - action: depth, web_enabled, time_budget_seconds, max_web_requests
    - score: initial quality estimate
    - frozen: whether pattern can be modified by learning
    """

    patterns = []

    # Pattern 1: Deep research with web (explicit request)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "explicit:deep+web"),
        brain="research_manager",
        signature="explicit:deep+web",
        context_tags=["explicit", "deep", "web"],
        action={
            "depth": 4,
            "web_enabled": True,
            "time_budget_seconds": 180,
            "max_web_requests": 5
        },
        score=0.5,  # Good default for explicit requests
        frozen=False
    ))

    # Pattern 2: Quick research (no web)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "explicit:quick"),
        brain="research_manager",
        signature="explicit:quick",
        context_tags=["explicit", "quick"],
        action={
            "depth": 1,
            "web_enabled": False,
            "time_budget_seconds": 30,
            "max_web_requests": 0
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 3: System self-diagnostics (internal only)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:system_self_diagnostics"),
        brain="research_manager",
        signature="topic:system_self_diagnostics",
        context_tags=["self_diag", "internal"],
        action={
            "depth": 2,
            "web_enabled": False,  # No web for internal diagnostics
            "time_budget_seconds": 60,
            "max_web_requests": 0
        },
        score=0.6,  # We know this works well
        frozen=False
    ))

    # Pattern 4: Scientific topics (may need web)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:science"),
        brain="research_manager",
        signature="topic:science",
        context_tags=["science", "factual"],
        action={
            "depth": 3,
            "web_enabled": False,  # Start without web, learn if needed
            "time_budget_seconds": 60,
            "max_web_requests": 0
        },
        score=0.3,
        frozen=False
    ))

    # Pattern 5: Current events (needs web)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:current_events"),
        brain="research_manager",
        signature="topic:current_events",
        context_tags=["current", "news", "events"],
        action={
            "depth": 2,
            "web_enabled": True,  # Current events always need web
            "time_budget_seconds": 90,
            "max_web_requests": 3
        },
        score=0.5,
        frozen=False
    ))

    # Pattern 6: Technical documentation
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:technical_docs"),
        brain="research_manager",
        signature="topic:technical_docs",
        context_tags=["technical", "documentation"],
        action={
            "depth": 2,
            "web_enabled": False,
            "time_budget_seconds": 45,
            "max_web_requests": 0
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 7: Trivial questions (minimal research)
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:trivial"),
        brain="research_manager",
        signature="topic:trivial",
        context_tags=["trivial", "simple"],
        action={
            "depth": 1,
            "web_enabled": False,
            "time_budget_seconds": 15,
            "max_web_requests": 0
        },
        score=0.5,  # We're confident about keeping it minimal
        frozen=False
    ))

    # Pattern 8: Code/implementation topics
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "topic:code"),
        brain="research_manager",
        signature="topic:code",
        context_tags=["code", "implementation"],
        action={
            "depth": 2,
            "web_enabled": False,
            "time_budget_seconds": 45,
            "max_web_requests": 0
        },
        score=0.4,
        frozen=False
    ))

    # Pattern 9: Default fallback
    patterns.append(Pattern(
        id=create_pattern_id("research_manager", "default"),
        brain="research_manager",
        signature="default",
        context_tags=["fallback"],
        action={
            "depth": 2,
            "web_enabled": False,
            "time_budget_seconds": 60,
            "max_web_requests": 0
        },
        score=0.0,  # Neutral - used when nothing else matches
        frozen=True  # Never remove the fallback
    ))

    return patterns


def initialize_research_manager_patterns():
    """Load initial patterns into the pattern store if not already present."""
    from brains.cognitive.pattern_store import get_pattern_store

    store = get_pattern_store()
    existing = store.get_patterns_by_brain("research_manager")

    if existing:
        print(f"[RESEARCH_MANAGER_INIT] Found {len(existing)} existing patterns, skipping init")
        return

    # No existing patterns, load initial set
    patterns = get_initial_patterns()
    print(f"[RESEARCH_MANAGER_INIT] Loading {len(patterns)} initial patterns")

    for pattern in patterns:
        store.store_pattern(pattern)

    print("[RESEARCH_MANAGER_INIT] Initial patterns loaded successfully")
