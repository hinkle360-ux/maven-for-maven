"""
Example: Belief Tracker Brain with _maybe_learn_from_teacher() Pattern
========================================================================

This example shows how to implement the _maybe_learn_from_teacher()
pattern in the Belief Tracker brain using teach_for_brain() directly.

This is an EXAMPLE - shows the pattern implementation approach.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from brains.memory.brain_memory import BrainMemory
from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from brains.cognitive.reasoning.truth_classifier import TruthClassifier


class BeliefTrackerExample:
    """
    Belief tracking brain that learns belief update patterns.

    Uses _maybe_learn_from_teacher() to learn:
    - How to update belief confidence based on new evidence
    - When to revise beliefs vs. strengthen them
    - How to handle conflicting evidence
    """

    def __init__(self):
        self.name = "belief_tracker"
        self.memory = BrainMemory(self.name)

        # Track beliefs (simplified)
        self.beliefs = {}

    def _build_situation_key(self, context: Dict[str, Any]) -> str:
        """Build unique key for this belief update situation."""
        belief_type = context.get("belief_type", "unknown")
        update_type = context.get("update_type", "unknown")
        evidence_strength = context.get("evidence_strength", "medium")

        return f"{belief_type}:{update_type}:{evidence_strength}"

    def _is_valid_teacher_result(self, teacher_result: Dict[str, Any]) -> bool:
        """Validate Teacher result with governance checks."""
        if teacher_result.get("verdict") == "ERROR":
            return False

        # For belief updates, require higher confidence
        if teacher_result.get("confidence", 0.0) < 0.6:
            return False

        patterns = teacher_result.get("patterns", [])
        if not patterns:
            return False

        # Validate with TruthClassifier
        for pattern in patterns:
            classification = TruthClassifier.classify(
                content=pattern.get("pattern", ""),
                confidence=teacher_result.get("confidence", 0.7),
                evidence=None
            )

            if classification["type"] == "RANDOM":
                return False

            if not classification.get("allow_memory_write", True):
                return False

        return True

    def _maybe_learn_from_teacher(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Learn belief update patterns from Teacher.

        Args:
            context: Dict with belief update context
                - belief: str (the belief statement)
                - belief_type: str ("factual", "opinion", "hypothesis")
                - current_confidence: float (current belief confidence)
                - evidence: str (new evidence)
                - evidence_strength: str ("weak", "medium", "strong")
                - update_type: str ("strengthen", "weaken", "revise", "contradict")

        Returns:
            Dict with learned patterns, or None
        """
        # 0. Build situation key
        situation_key = self._build_situation_key(context)

        # 1. Check own memory for existing pattern
        existing = self.memory.retrieve(
            query={
                "kind": "teacher_pattern",
                "situation_key": situation_key
            },
            limit=1
        )

        if existing and existing[0].get("confidence", 0) >= 0.7:
            print(f"[{self.name.upper()}] Using learned belief update pattern from memory")
            return {
                "patterns": existing[0].get("content"),
                "source": "memory",
                "confidence": existing[0].get("confidence")
            }

        # 2. Call Teacher
        print(f"[{self.name.upper()}] No belief pattern found, calling Teacher...")

        teacher_result = teach_for_brain(
            brain_name=self.name,
            situation={
                "question": f"How should I update belief confidence: {context.get('belief', '')[:100]}?",
                "belief": context.get("belief"),
                "belief_type": context.get("belief_type"),
                "current_confidence": context.get("current_confidence"),
                "evidence": context.get("evidence", "")[:200],
                "evidence_strength": context.get("evidence_strength"),
                "update_type": context.get("update_type")
            }
        )

        # 3. Validate
        if not self._is_valid_teacher_result(teacher_result):
            print(f"[{self.name.upper()}] Teacher result failed validation")
            return None

        # 4. Store patterns
        patterns = teacher_result.get("patterns", [])

        self.memory.store(
            content=patterns,
            metadata={
                "kind": "teacher_pattern",
                "situation_key": situation_key,
                "source": "teacher",
                "confidence": teacher_result.get("confidence", 0.7),
                "pattern_count": len(patterns),
                "belief_type": context.get("belief_type"),
                "update_type": context.get("update_type")
            }
        )

        print(f"[{self.name.upper()}] Stored {len(patterns)} belief update patterns")

        return {
            "patterns": patterns,
            "source": "teacher",
            "confidence": teacher_result.get("confidence"),
            "pattern_count": len(patterns)
        }

    def update_belief(
        self,
        belief: str,
        evidence: str,
        evidence_strength: str = "medium"
    ) -> Dict[str, Any]:
        """
        Update a belief based on new evidence.

        Uses learned patterns to determine how to adjust belief confidence.

        Args:
            belief: The belief statement
            evidence: New evidence to consider
            evidence_strength: "weak", "medium", "strong"

        Returns:
            Dict with update result and new confidence
        """
        # Get or create belief
        current_confidence = self.beliefs.get(belief, 0.5)

        # Determine update type based on evidence
        # (Simplified - real implementation would analyze evidence)
        update_type = "strengthen"  # Default assumption

        # Build context
        context = {
            "belief": belief,
            "belief_type": "factual",  # Could classify this
            "current_confidence": current_confidence,
            "evidence": evidence,
            "evidence_strength": evidence_strength,
            "update_type": update_type
        }

        # Learn update patterns (if needed)
        result = self._maybe_learn_from_teacher(context)

        if result:
            # Apply learned patterns to compute new confidence
            # (Simplified - would parse and apply patterns in real implementation)

            patterns = result.get("patterns", [])

            # Example logic based on learned patterns
            confidence_delta = {
                "weak": 0.1,
                "medium": 0.2,
                "strong": 0.3
            }.get(evidence_strength, 0.1)

            if update_type == "strengthen":
                new_confidence = min(1.0, current_confidence + confidence_delta)
            elif update_type == "weaken":
                new_confidence = max(0.0, current_confidence - confidence_delta)
            else:
                new_confidence = current_confidence

            # Store updated belief
            self.beliefs[belief] = new_confidence

            return {
                "ok": True,
                "belief": belief,
                "old_confidence": current_confidence,
                "new_confidence": new_confidence,
                "patterns_used": len(patterns),
                "source": result["source"]
            }
        else:
            # Fallback: simple update
            new_confidence = current_confidence + 0.1
            self.beliefs[belief] = new_confidence

            return {
                "ok": True,
                "belief": belief,
                "old_confidence": current_confidence,
                "new_confidence": new_confidence,
                "patterns_used": 0,
                "source": "fallback"
            }

    def revise_belief(
        self,
        belief: str,
        new_evidence: str,
        contradicts: bool = False
    ) -> Dict[str, Any]:
        """
        Revise a belief based on new contradictory evidence.

        Args:
            belief: The belief to revise
            new_evidence: Evidence that may contradict the belief
            contradicts: Whether evidence directly contradicts belief

        Returns:
            Dict with revision result
        """
        current_confidence = self.beliefs.get(belief, 0.5)

        context = {
            "belief": belief,
            "belief_type": "factual",
            "current_confidence": current_confidence,
            "evidence": new_evidence,
            "evidence_strength": "strong" if contradicts else "medium",
            "update_type": "contradict" if contradicts else "revise"
        }

        result = self._maybe_learn_from_teacher(context)

        if result:
            # Apply revision patterns
            if contradicts:
                new_confidence = max(0.0, current_confidence - 0.4)
            else:
                new_confidence = max(0.0, current_confidence - 0.2)

            self.beliefs[belief] = new_confidence

            return {
                "ok": True,
                "belief": belief,
                "revised": True,
                "old_confidence": current_confidence,
                "new_confidence": new_confidence,
                "patterns_used": result.get("pattern_count", 0)
            }
        else:
            return {
                "ok": False,
                "belief": belief,
                "revised": False,
                "reason": "No revision patterns available"
            }


# Example usage
if __name__ == "__main__":
    tracker = BeliefTrackerExample()

    # Example 1: Update a belief with supporting evidence
    result1 = tracker.update_belief(
        belief="Python is a popular programming language",
        evidence="Stack Overflow survey shows Python is #1",
        evidence_strength="strong"
    )

    print(f"\nBelief updated:")
    print(f"  Old confidence: {result1['old_confidence']:.2f}")
    print(f"  New confidence: {result1['new_confidence']:.2f}")
    print(f"  Source: {result1['source']}")
    print(f"  Patterns used: {result1['patterns_used']}")

    # Example 2: Update same belief type again (should use memory)
    result2 = tracker.update_belief(
        belief="JavaScript is widely used for web development",
        evidence="Most websites use JavaScript",
        evidence_strength="strong"  # Same strength as before
    )

    print(f"\nSecond update source: {result2['source']}")
    print("Should be 'memory' if first call succeeded!")

    # Example 3: Revise a belief with contradictory evidence
    tracker.beliefs["The Earth is flat"] = 0.3  # Start with low confidence

    result3 = tracker.revise_belief(
        belief="The Earth is flat",
        new_evidence="Photos from space show spherical Earth",
        contradicts=True
    )

    print(f"\nBelief revised:")
    print(f"  Old confidence: {result3.get('old_confidence', 0):.2f}")
    print(f"  New confidence: {result3.get('new_confidence', 0):.2f}")
    print(f"  Patterns used: {result3.get('patterns_used', 0)}")
