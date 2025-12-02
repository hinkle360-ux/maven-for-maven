"""
Learning Mode Enum
==================

Defines the three learning modes that control behavior across the cognitive stack.

CORE PRINCIPLE: Memory-First, LLM-as-Teacher
--------------------------------------------
The system operates with "brains operate from memory/context first, then LLM
as teacher if needed, never the other way around."

The LLM (Ollama locally, or remote APIs later) is always available but is used
as a TEACHER, not the primary source of answers. LearningMode is a BEHAVIORAL
switch, not a kill switch for the LLM.

Memory-First Order of Operations:
1. Check BrainMemory (STM/MTM/LTM)
2. Check domain-specific banks
3. Check strategy tables and cached answers
4. ONLY if memory fails -> call LLM as teacher

Mode Definitions:
-----------------
- TRAINING (default): Normal operation with learning enabled.
  Behavior: Memory-first lookup -> if miss -> call Ollama-as-teacher ->
  store lesson & facts -> next time answer from memory without re-asking.
  Use this mode for all normal operations.

- OFFLINE: Evaluation-only mode, no new learning.
  Behavior: Use strategies and memory only. Do NOT call the LLM for new
  lessons. Do NOT write lessons/facts to memory. Existing strategies and
  cached knowledge still work. This is how we test "true offline cognition"
  with no new learning, not a complete LLM kill switch.

- SHADOW: Teacher commentary mode for side-by-side evaluation.
  Behavior: LLM can run in the background to critique answers, but its
  outputs NEVER change memory or influence the final user answer. Used
  for comparing LLM vs memory-based answers without contamination.

Usage:
    from brains.learning.learning_mode import LearningMode

    # In context dict (TRAINING is the default everywhere)
    context["learning_mode"] = LearningMode.TRAINING

    # In brain handle functions
    def handle(context: Dict):
        mode = context.get("learning_mode", LearningMode.TRAINING)

        # 1. Always try memory/strategies first
        strategy = select_strategy(problem_type, domain)
        if strategy and strategy["confidence"] > 0.7:
            return apply_strategy(strategy, context)  # No LLM call

        # 2. Only if no strategy, check mode
        if mode == LearningMode.TRAINING:
            # Call LLM, create lesson, store facts
            result = call_teacher_and_learn(context)
        elif mode == LearningMode.OFFLINE:
            # No learning - use fallback heuristics only
            result = apply_fallback_heuristics(context)
        elif mode == LearningMode.SHADOW:
            # Run LLM in background for comparison only
            result = apply_fallback_heuristics(context)
            shadow_result = call_teacher_for_comparison(context)

Note: This enum should be the ONLY source of learning mode values.
Do not use ad-hoc strings like "training" or "offline" elsewhere.
"""

from __future__ import annotations

from enum import Enum


class LearningMode(str, Enum):
    """
    Learning mode controlling cognitive behavior and lesson generation.

    Inherits from str to allow easy serialization and comparison.

    IMPORTANT: This is a BEHAVIORAL switch, not an LLM kill switch.
    All modes operate memory-first. The LLM (Ollama) is always available
    as a teacher, but is only called when memory/strategies fail.
    """

    TRAINING = "training"
    """
    Normal operation with learning enabled (DEFAULT).

    Behavior: Memory-first -> if miss -> call LLM-as-teacher -> store lessons/facts.
    Next time the same question is asked, answer from memory without re-asking.
    Use this for all normal operations.
    """

    OFFLINE = "offline"
    """
    Evaluation-only mode, no new learning.

    Behavior: Use strategies and memory only. Do NOT call LLM for new lessons.
    Do NOT write lessons/facts. This tests "true offline cognition" - existing
    learned strategies still work, but no new learning occurs.
    """

    SHADOW = "shadow"
    """
    Teacher commentary mode for side-by-side evaluation.

    Behavior: LLM can run in background to critique answers, but its outputs
    NEVER change memory or influence final user answer. Used for comparing
    LLM vs memory-based answers without contaminating the learning process.
    """


# Export public API
__all__ = ["LearningMode"]
