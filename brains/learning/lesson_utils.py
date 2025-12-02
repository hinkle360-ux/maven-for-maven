"""
Lesson Utilities
================

Central helpers for creating and storing lesson records. These are pure data
utilities with NO network calls or LLM tool imports.

CONCEPT KEY NORMALIZATION
=========================
The `canonical_concept_key()` function provides consistent normalization of
questions to a "concept key" that allows matching semantically equivalent
questions like "what are birds" and "are birds" to the same key "birds".

This is critical for the memory-first architecture:
- When storing facts/lessons, we store the concept_key in metadata
- When looking up facts/lessons, we compute the concept_key and match on it
- This ensures "learn once, reuse forever" for the same concept

Lesson Schema:
    brain: str              - Brain that generated the lesson (e.g. "reasoning")
    topic: str              - Short topic identifier (e.g. "logical_choice")
    input_signature: Dict   - Signature describing the input:
        - problem_type: str
        - domain: Optional[str]
        - risk_level: Optional[str]
        - ... additional context fields
    llm_prompt: str         - The prompt sent to the LLM
    llm_response: str       - The response received (possibly truncated)
    distilled_rule: str     - Extracted reasoning steps or rule
    examples: List[Dict]    - List of {"input": ..., "output": ...} examples
    confidence: float       - Confidence score (0.0 - 1.0)
    mode: str              - Learning mode when generated ("training", "shadow", "production")
    status: str            - Lifecycle status:
        - "new": Just created, not yet reviewed
        - "trusted": Reviewed and approved
        - "provisional": Tentatively approved
        - "rejected": Marked as incorrect
        - "integrated": Successfully used to update strategies
    timestamp: str         - ISO 8601 timestamp

Usage:
    from brains.learning.lesson_utils import create_lesson_record, store_lesson

    lesson = create_lesson_record(
        brain="reasoning",
        topic="logical_choice",
        input_signature={"problem_type": "comparison", "domain": "science"},
        llm_prompt="Explain which is larger...",
        llm_response="Jupiter is larger because...",
        distilled_rule="Compare by physical dimensions for size questions",
        examples=[{"input": "Is Mars bigger than Earth?", "output": "No, Earth is larger"}],
        confidence=0.85,
        mode="training",
        status="new"
    )

    store_lesson("reasoning", lesson, brain_memory_instance)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # Only import for type hints; BrainMemory is injected at runtime
    from brains.memory.brain_memory import BrainMemory


# Common filler words to strip when extracting concept keys
_FILLER_WORDS = frozenset([
    "what", "is", "are", "the", "a", "an", "does", "do", "can", "could",
    "would", "should", "how", "why", "when", "where", "who", "which",
    "tell", "me", "about", "explain", "describe", "define", "please",
    "of", "for", "to", "in", "on", "at", "by", "with", "from", "as",
    "be", "been", "being", "have", "has", "had", "having",
])

# =============================================================================
# PLANNING CONCEPT KEY MAPPINGS (Step 1 Enhancement)
# =============================================================================
# These canonical keys allow planning-related queries to be normalized,
# so "plan my week", "help me schedule", "organize my tasks" all map to
# the same concept key for memory-first lookups.

_PLANNING_CONCEPT_MAPPINGS: Dict[str, List[str]] = {
    # Weekly/daily planning
    "planning_week": [
        "plan my week", "plan the week", "weekly planning", "week plan",
        "plan this week", "schedule my week", "weekly schedule",
        "organize my week", "week ahead", "plan for the week",
        "help me plan my week", "plan out my week",
    ],
    "planning_day": [
        "plan my day", "plan the day", "daily planning", "day plan",
        "plan today", "schedule my day", "daily schedule",
        "organize my day", "today's plan", "plan for today",
        "help me plan my day", "what should i do today",
    ],
    # Task organization
    "task_organization": [
        "organize my tasks", "organize tasks", "task organization",
        "prioritize tasks", "prioritize my tasks", "task list",
        "to do list", "todo list", "manage tasks", "task management",
        "sort my tasks", "order my tasks", "rank my tasks",
    ],
    # Scheduling
    "scheduling": [
        "help me schedule", "schedule this", "scheduling help",
        "make a schedule", "create schedule", "build schedule",
        "time management", "manage my time", "allocate time",
        "when should i", "find time for", "fit this in",
    ],
    # Goal setting
    "goal_setting": [
        "set goals", "goal setting", "my goals", "define goals",
        "establish goals", "create goals", "plan goals",
        "what are my goals", "help with goals", "goal planning",
    ],
    # Project planning
    "project_planning": [
        "plan project", "project planning", "project plan",
        "plan this project", "organize project", "project breakdown",
        "project steps", "project phases", "plan the project",
    ],
    # Habit tracking
    "habit_tracking": [
        "track habits", "habit tracking", "my habits",
        "build habits", "habit building", "daily habits",
        "routine", "daily routine", "morning routine",
    ],
    # Decision making
    "decision_making": [
        "help me decide", "make a decision", "decision making",
        "should i", "which should i", "what should i choose",
        "compare options", "weigh options", "pros and cons",
    ],
}

# Common punctuation to strip
_PUNCTUATION = "?!.,;:'\"()[]{}*"


def planning_concept_key(question: str) -> Optional[str]:
    """
    Check if a question maps to a canonical planning concept key.

    This function enables the Planner brain to recognize that queries like
    "plan my week", "help me schedule my week", and "organize my weekly tasks"
    all refer to the same underlying concept: `planning_week`.

    This is critical for:
    - Memory-first lookups: Same concept key = same stored strategy
    - Personal learning: Maven learns YOUR planning style, not generic methods
    - Pattern recognition: Similar queries benefit from learned patterns

    Args:
        question: The user's question or query string

    Returns:
        A canonical planning concept key (e.g., "planning_week") if found,
        or None if the question doesn't match any planning patterns.

    Examples:
        >>> planning_concept_key("plan my week")
        'planning_week'
        >>> planning_concept_key("help me schedule my day")
        'planning_day'
        >>> planning_concept_key("organize my tasks")
        'task_organization'
        >>> planning_concept_key("what is the capital of France?")
        None
    """
    if not question:
        return None

    # Normalize: lowercase, strip whitespace
    q = question.lower().strip()

    # Remove punctuation
    for char in _PUNCTUATION:
        q = q.replace(char, " ")

    # Normalize whitespace
    q = " ".join(q.split())

    # Check each canonical key's patterns
    for concept_key, patterns in _PLANNING_CONCEPT_MAPPINGS.items():
        for pattern in patterns:
            # Check for exact match or if pattern appears as substring
            if pattern in q or q in pattern:
                return concept_key

            # Check for high word overlap (fuzzy match)
            pattern_words = set(pattern.split())
            q_words = set(q.split())
            if pattern_words and q_words:
                overlap = len(pattern_words & q_words)
                # If 2+ significant words match, consider it a match
                if overlap >= 2 and overlap >= len(pattern_words) * 0.5:
                    return concept_key

    return None


def canonical_concept_key(question: str) -> str:
    """
    Extract a canonical concept key from a question.

    This normalizes questions so that semantically equivalent queries
    like "what are birds", "are birds", "tell me about birds", and "birds"
    all produce the same concept key: "birds".

    The algorithm:
    1. Lowercase and strip whitespace
    2. Remove punctuation
    3. Remove common filler/question words
    4. Collapse whitespace
    5. Return the remaining content as the concept key

    Args:
        question: The user's question or query string

    Returns:
        A normalized concept key string (e.g., "birds" from "what are birds?")
        Returns the original (normalized) question if no simplification possible.

    Examples:
        >>> canonical_concept_key("what are birds")
        'birds'
        >>> canonical_concept_key("are birds")
        'birds'
        >>> canonical_concept_key("tell me about birds")
        'birds'
        >>> canonical_concept_key("What is the capital of France?")
        'capital france'
        >>> canonical_concept_key("why do cats purr")
        'cats purr'
    """
    if not question:
        return ""

    # 1. Lowercase and strip
    q = question.lower().strip()

    # 2. Remove punctuation
    for char in _PUNCTUATION:
        q = q.replace(char, " ")

    # 3. Split into words and filter out filler words
    words = q.split()
    significant_words = [w for w in words if w not in _FILLER_WORDS]

    # 4. If we removed everything, fall back to last 2-3 words of original
    if not significant_words:
        # Take last few words as they're often the subject
        words = q.split()
        significant_words = words[-3:] if len(words) >= 3 else words

    # 5. Join and collapse whitespace
    key = " ".join(significant_words).strip()

    # Final fallback: if still empty, use original normalized question
    if not key:
        key = q.strip()

    return key


def create_lesson_record(
    brain: str,
    topic: str,
    input_signature: Dict[str, Any],
    llm_prompt: str,
    llm_response: str,
    distilled_rule: str,
    examples: List[Dict[str, Any]],
    confidence: float,
    mode: str,
    status: str = "new",
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a lesson record with the canonical schema.

    Args:
        brain: Name of the brain that generated this lesson
        topic: Short topic identifier (e.g. problem type)
        input_signature: Dict describing the input characteristics
        llm_prompt: The prompt sent to the LLM
        llm_response: The LLM response (may be truncated)
        distilled_rule: Extracted reasoning rule or steps
        examples: List of example dicts with "input" and "output" keys
        confidence: Confidence score (0.0 - 1.0)
        mode: Learning mode ("training", "shadow", "offline")
        status: Lifecycle status ("new", "trusted", "provisional", "rejected", "integrated")
        timestamp: ISO timestamp (auto-generated if None)

    Returns:
        Dict containing all lesson fields
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "brain": brain,
        "topic": topic,
        "input_signature": input_signature,
        "llm_prompt": llm_prompt,
        "llm_response": llm_response,
        "distilled_rule": distilled_rule,
        "examples": examples,
        "confidence": confidence,
        "mode": mode,
        "status": status,
        "timestamp": timestamp
    }


def store_lesson(
    brain_name: str,
    lesson_record: Dict[str, Any],
    brain_memory: "BrainMemory",
    original_question: Optional[str] = None
) -> bool:
    """
    Store a lesson record to the brain's memory.

    Uses the BrainMemory.store() API with metadata tagging the record
    as a lesson for later retrieval and filtering.

    IMPORTANT: Pass original_question to enable concept-key based lookups.
    This allows questions like "what are birds" and "are birds" to match
    the same stored lesson via the normalized concept_key.

    Args:
        brain_name: Name of the brain (should match lesson_record["brain"])
        lesson_record: The lesson dict from create_lesson_record()
        brain_memory: BrainMemory instance to store to
        original_question: The original question that triggered this lesson.
                          Used to compute concept_key for later lookups.

    Returns:
        True if stored successfully, False otherwise
    """
    try:
        # Build metadata with concept_key for lookups
        metadata = {
            "type": "lesson",
            "brain": brain_name,
            "topic": lesson_record.get("topic", "unknown"),
            "status": lesson_record.get("status", "new"),
            "mode": lesson_record.get("mode", "unknown"),
            "confidence": lesson_record.get("confidence", 0.0)
        }

        # Add original_question and concept_key for memory-first lookups
        if original_question:
            metadata["original_question"] = original_question
            metadata["concept_key"] = canonical_concept_key(original_question)
            print(f"[LESSON_UTILS] Storing lesson with concept_key='{metadata['concept_key']}' for question='{original_question[:50]}...'")

        brain_memory.store(
            content=lesson_record,
            metadata=metadata
        )
        return True
    except Exception as e:
        print(f"[LESSON_UTILS] Failed to store lesson for {brain_name}: {e}")
        return False


def retrieve_lessons(
    brain_name: str,
    brain_memory: "BrainMemory",
    status_filter: Optional[List[str]] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieve lesson records from brain memory.

    Args:
        brain_name: Name of the brain to retrieve lessons for
        brain_memory: BrainMemory instance to query
        status_filter: If provided, only return lessons with these statuses
        limit: Maximum number of lessons to retrieve

    Returns:
        List of lesson records matching the criteria
    """
    try:
        # Use a broad query to get lessons
        results = brain_memory.retrieve(query="lesson", limit=limit)

        lessons = []
        for rec in results:
            # Check if this is a lesson record
            metadata = rec.get("metadata", {})
            if metadata.get("type") != "lesson":
                continue
            if metadata.get("brain") != brain_name:
                continue

            content = rec.get("content")
            if not isinstance(content, dict):
                continue

            # Apply status filter if provided
            if status_filter:
                lesson_status = content.get("status", "unknown")
                if lesson_status not in status_filter:
                    continue

            lessons.append(content)

        return lessons

    except Exception as e:
        print(f"[LESSON_UTILS] Failed to retrieve lessons for {brain_name}: {e}")
        return []


def update_lesson_status(
    brain_name: str,
    lesson_record: Dict[str, Any],
    new_status: str,
    brain_memory: "BrainMemory"
) -> bool:
    """
    Update the status of a lesson record.

    This stores a new version of the lesson with updated status.
    The old version remains in memory (immutable append-only pattern).

    Args:
        brain_name: Name of the brain
        lesson_record: The lesson to update
        new_status: New status value
        brain_memory: BrainMemory instance

    Returns:
        True if update succeeded, False otherwise
    """
    try:
        # Create updated lesson with new status and timestamp
        updated_lesson = lesson_record.copy()
        updated_lesson["status"] = new_status
        updated_lesson["timestamp"] = datetime.now(timezone.utc).isoformat()

        return store_lesson(brain_name, updated_lesson, brain_memory)

    except Exception as e:
        print(f"[LESSON_UTILS] Failed to update lesson status for {brain_name}: {e}")
        return False


# Export public API
__all__ = [
    "canonical_concept_key",
    "planning_concept_key",
    "create_lesson_record",
    "store_lesson",
    "retrieve_lessons",
    "update_lesson_status",
]
