"""
Example: Learning Brain with _maybe_learn_from_teacher() Pattern
==================================================================

This example shows how to implement the _maybe_learn_from_teacher()
pattern in the Learning (meta-learning) brain using teach_for_brain()
directly instead of TeacherHelper.

This is an EXAMPLE - the actual learning brain uses TeacherHelper.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from brains.memory.brain_memory import BrainMemory
from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from brains.cognitive.reasoning.truth_classifier import TruthClassifier


class LearningBrainExample:
    """
    Meta-learning brain that learns learning strategies.

    Uses the _maybe_learn_from_teacher() pattern to:
    1. Check own memory for learned strategies
    2. Call Teacher if no strategy found
    3. Validate with TruthClassifier
    4. Store validated strategies
    """

    def __init__(self):
        self.name = "learning"
        self.memory = BrainMemory(self.name)

        # Track Teacher calls to avoid spam
        self.teacher_calls_this_session = 0
        self.max_teacher_calls = 10

    def _build_situation_key(self, context: Dict[str, Any]) -> str:
        """
        Build a unique key for this type of learning situation.

        Args:
            context: Learning context dict

        Returns:
            Situation key string
        """
        learning_type = context.get("learning_type", "unknown")
        content_type = context.get("content_type", "unknown")
        difficulty = context.get("difficulty", "unknown")

        return f"{learning_type}:{content_type}:{difficulty}"

    def _is_valid_teacher_result(self, teacher_result: Dict[str, Any]) -> bool:
        """
        Truth / governance checks using existing TruthClassifier.

        Args:
            teacher_result: Result from teach_for_brain()

        Returns:
            True if result is valid and should be stored
        """
        # Check verdict
        if teacher_result.get("verdict") == "ERROR":
            print(f"[{self.name.upper()}] Teacher returned ERROR")
            return False

        # Check confidence
        confidence = teacher_result.get("confidence", 0.0)
        if confidence < 0.6:  # Meta-learning needs higher confidence
            print(f"[{self.name.upper()}] Confidence too low: {confidence}")
            return False

        # Check patterns exist
        patterns = teacher_result.get("patterns", [])
        if not patterns:
            print(f"[{self.name.upper()}] No patterns in Teacher result")
            return False

        # Validate each pattern with TruthClassifier
        for pattern in patterns:
            pattern_text = pattern.get("pattern", "")

            classification = TruthClassifier.classify(
                content=pattern_text,
                confidence=confidence,
                evidence=None
            )

            # Reject if marked as RANDOM or not allowed to write
            if classification["type"] == "RANDOM":
                print(f"[{self.name.upper()}] Pattern rejected as RANDOM")
                return False

            if not classification.get("allow_memory_write", True):
                print(f"[{self.name.upper()}] Pattern not allowed to write to memory")
                return False

        return True

    def _maybe_learn_from_teacher(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Use Teacher as a helper: only when we don't already have a pattern
        for this kind of situation. Store any learned patterns in our own
        BrainMemory for next time.

        This implements the standardized learning loop:
        1. Check own memory for existing pattern
        2. Call Teacher if no pattern found
        3. Validate result with governance checks
        4. Store validated patterns
        5. Return patterns for immediate use

        Args:
            context: Dict with learning situation context
                Required keys:
                - learning_type: "concept", "skill", "pattern", etc.
                - content_type: "factual", "procedural", "meta", etc.
                - difficulty: "easy", "medium", "hard"
                - content: The actual content to learn

        Returns:
            Dict with learned strategies, or None if learning failed
        """
        # 0. Decide what key represents the "situation type"
        situation_key = self._build_situation_key(context)

        # 1. Check own memory for an existing pattern
        existing = self.memory.retrieve(
            query={
                "kind": "teacher_pattern",
                "situation_key": situation_key
            },
            limit=1
        )

        if existing and existing[0].get("confidence", 0) >= 0.7:
            print(f"[{self.name.upper()}] Using existing learning strategy from memory")
            return {
                "strategies": existing[0].get("content"),
                "source": "memory",
                "confidence": existing[0].get("confidence")
            }

        # 2. Check budget - don't spam Teacher
        if self.teacher_calls_this_session >= self.max_teacher_calls:
            print(f"[{self.name.upper()}] Teacher call limit reached for this session")
            return None

        # 3. Call Teacher via teach_for_brain
        print(f"[{self.name.upper()}] No strategy found, calling Teacher...")

        teacher_result = teach_for_brain(
            brain_name=self.name,
            situation={
                "question": f"What's the best strategy to learn {context.get('learning_type')} content?",
                "learning_type": context.get("learning_type"),
                "content_type": context.get("content_type"),
                "difficulty": context.get("difficulty"),
                "content_preview": str(context.get("content", ""))[:200]
            }
        )

        self.teacher_calls_this_session += 1

        # 4. Truth / governance checks
        if not self._is_valid_teacher_result(teacher_result):
            print(f"[{self.name.upper()}] Teacher result failed validation")
            return None

        # 5. Store pattern in this brain's own memory
        patterns = teacher_result.get("patterns", [])

        self.memory.store(
            content=patterns,
            metadata={
                "kind": "teacher_pattern",
                "situation_key": situation_key,
                "source": "teacher",
                "confidence": teacher_result.get("confidence", 0.7),
                "pattern_count": len(patterns),
                "learning_type": context.get("learning_type"),
                "content_type": context.get("content_type")
            }
        )

        print(f"[{self.name.upper()}] Stored {len(patterns)} new learning strategies")

        return {
            "strategies": patterns,
            "source": "teacher",
            "confidence": teacher_result.get("confidence"),
            "pattern_count": len(patterns)
        }

    def suggest_learning_strategy(
        self,
        content: Any,
        learning_type: str = "concept",
        content_type: str = "factual",
        difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """
        Suggest the best learning strategy for given content.

        This is the main API that uses _maybe_learn_from_teacher internally.

        Args:
            content: The content to learn
            learning_type: Type of learning ("concept", "skill", "pattern", etc.)
            content_type: Content category ("factual", "procedural", "meta", etc.)
            difficulty: Difficulty level ("easy", "medium", "hard")

        Returns:
            Dict with suggested strategies and metadata
        """
        # Use the learning helper
        result = self._maybe_learn_from_teacher({
            "learning_type": learning_type,
            "content_type": content_type,
            "difficulty": difficulty,
            "content": content
        })

        if result:
            return {
                "ok": True,
                "strategies": result["strategies"],
                "source": result["source"],
                "confidence": result.get("confidence", 0.7)
            }
        else:
            # Fallback to generic strategy
            return {
                "ok": True,
                "strategies": [
                    {"pattern": "Break content into smaller chunks"},
                    {"pattern": "Practice with examples"},
                    {"pattern": "Review and reinforce"}
                ],
                "source": "fallback",
                "confidence": 0.5
            }


# Example usage
if __name__ == "__main__":
    brain = LearningBrainExample()

    # First call - will learn from Teacher
    result1 = brain.suggest_learning_strategy(
        content="How to implement quicksort algorithm",
        learning_type="skill",
        content_type="procedural",
        difficulty="medium"
    )

    print(f"\nFirst call result: {result1['source']}")
    print(f"Strategies count: {len(result1['strategies'])}")

    # Second call with same type - will use memory
    result2 = brain.suggest_learning_strategy(
        content="How to implement merge sort algorithm",
        learning_type="skill",  # Same type
        content_type="procedural",  # Same type
        difficulty="medium"  # Same difficulty
    )

    print(f"\nSecond call result: {result2['source']}")
    print("Should be 'memory' if first call succeeded!")
