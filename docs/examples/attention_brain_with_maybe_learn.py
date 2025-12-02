"""
Example: Attention Brain with _maybe_learn_from_teacher() Pattern
===================================================================

This example shows how to implement the _maybe_learn_from_teacher()
pattern in the Attention brain using teach_for_brain() directly.

This is an EXAMPLE - the actual attention brain uses TeacherHelper.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from brains.memory.brain_memory import BrainMemory
from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from brains.cognitive.reasoning.truth_classifier import TruthClassifier


class AttentionBrainExample:
    """
    Attention allocation brain that learns focus strategies.

    Uses _maybe_learn_from_teacher() to learn:
    - Which tasks/inputs deserve focus
    - How to prioritize multiple competing inputs
    - When to shift attention vs. maintain focus
    """

    def __init__(self):
        self.name = "attention"
        self.memory = BrainMemory(self.name)

        # Current focus state
        self.current_focus = None
        self.focus_history = []

    def _build_situation_key(self, context: Dict[str, Any]) -> str:
        """Build unique key for this attention situation."""
        input_count = context.get("input_count", 1)
        urgency = context.get("urgency", "normal")
        input_types = context.get("input_types", [])

        # Create signature from input types
        type_signature = "-".join(sorted(set(input_types))) if input_types else "unknown"

        return f"inputs:{input_count}:urgency:{urgency}:types:{type_signature}"

    def _is_valid_teacher_result(self, teacher_result: Dict[str, Any]) -> bool:
        """Validate Teacher result."""
        if teacher_result.get("verdict") == "ERROR":
            return False

        if teacher_result.get("confidence", 0.0) < 0.5:
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

            if classification["type"] == "RANDOM" or not classification.get("allow_memory_write", True):
                return False

        return True

    def _maybe_learn_from_teacher(self, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Learn attention allocation rules from Teacher.

        Args:
            context: Dict with attention context
                - inputs: List of input dicts
                - input_count: Number of inputs
                - urgency: "low", "normal", "high"
                - input_types: List of input type strings

        Returns:
            List of attention allocation rules, or None
        """
        # 0. Build situation key
        situation_key = self._build_situation_key(context)

        # 1. Check own memory for existing pattern
        existing = self.memory.retrieve(
            query={
                "kind": "teacher_pattern",
                "situation_key": situation_key
            },
            limit=1,
            tiers=["stm", "mtm", "ltm"]
        )

        if existing:
            for record in existing:
                if record.get("confidence", 0) >= 0.7:
                    print(f"[{self.name.upper()}] Using learned attention rules from memory")
                    return record.get("content")

        # 2. Call Teacher
        print(f"[{self.name.upper()}] No attention rules found, calling Teacher...")

        inputs = context.get("inputs", [])
        input_preview = [
            {
                "type": inp.get("type"),
                "priority": inp.get("priority"),
                "content": str(inp.get("content", ""))[:50]
            }
            for inp in inputs[:5]  # Limit to first 5 for prompt
        ]

        teacher_result = teach_for_brain(
            brain_name=self.name,
            situation={
                "question": f"How should I allocate attention among {len(inputs)} inputs?",
                "input_count": len(inputs),
                "urgency": context.get("urgency", "normal"),
                "input_types": context.get("input_types", []),
                "inputs_preview": input_preview
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
                "rule_count": len(patterns),
                "input_count": context.get("input_count"),
                "urgency": context.get("urgency")
            }
        )

        print(f"[{self.name.upper()}] Stored {len(patterns)} attention allocation rules")

        return patterns

    def allocate_attention(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Allocate attention across multiple inputs using learned rules.

        Args:
            inputs: List of input dicts, each with:
                - type: str (input type)
                - content: Any (input content)
                - priority: str ("low", "medium", "high")
                - urgent: bool (is this urgent?)

        Returns:
            Dict with attention allocation decision
        """
        if not inputs:
            return {"focus": None, "reasoning": "No inputs"}

        # Build context
        input_types = [inp.get("type", "unknown") for inp in inputs]
        urgency = "high" if any(inp.get("urgent") for inp in inputs) else "normal"

        context = {
            "inputs": inputs,
            "input_count": len(inputs),
            "urgency": urgency,
            "input_types": input_types
        }

        # Get attention rules (learns from Teacher if needed)
        rules = self._maybe_learn_from_teacher(context)

        if rules:
            # Apply learned rules to pick focus
            # (Simplified - real implementation would parse and apply rules)

            # For this example, just pick highest priority
            focus = max(inputs, key=lambda x: {
                "high": 3,
                "medium": 2,
                "low": 1
            }.get(x.get("priority", "low"), 1))

            self.current_focus = focus
            self.focus_history.append(focus)

            return {
                "focus": focus,
                "rules_applied": len(rules),
                "source": "learned_rules",
                "confidence": 0.8
            }
        else:
            # Fallback strategy
            focus = inputs[0] if inputs else None
            self.current_focus = focus

            return {
                "focus": focus,
                "rules_applied": 0,
                "source": "fallback",
                "confidence": 0.5
            }

    def should_shift_focus(self, new_input: Dict[str, Any]) -> bool:
        """
        Decide whether to shift focus to a new input.

        Uses learned patterns about when to shift vs. maintain focus.

        Args:
            new_input: New input that's requesting attention

        Returns:
            True if should shift focus, False if should maintain current focus
        """
        if not self.current_focus:
            return True

        # Build context for shift decision
        context = {
            "inputs": [self.current_focus, new_input],
            "input_count": 2,
            "urgency": "high" if new_input.get("urgent") else "normal",
            "input_types": [
                self.current_focus.get("type", "unknown"),
                new_input.get("type", "unknown")
            ]
        }

        # Learn shift patterns
        patterns = self._maybe_learn_from_teacher(context)

        if patterns:
            # Apply patterns to make decision
            # (Simplified - would parse patterns in real implementation)

            # For this example, shift if new input is urgent
            should_shift = new_input.get("urgent", False)

            print(f"[{self.name.upper()}] Shift decision based on {len(patterns)} learned rules: {should_shift}")

            return should_shift
        else:
            # Fallback: maintain current focus unless new is urgent
            return new_input.get("urgent", False)


# Example usage
if __name__ == "__main__":
    brain = AttentionBrainExample()

    # Example 1: Allocate attention among multiple inputs
    inputs = [
        {"type": "user_query", "content": "What is Python?", "priority": "high", "urgent": False},
        {"type": "system_alert", "content": "Low memory", "priority": "medium", "urgent": False},
        {"type": "background_task", "content": "Cleanup logs", "priority": "low", "urgent": False}
    ]

    result = brain.allocate_attention(inputs)
    print(f"\nAttention allocated to: {result['focus']['type']}")
    print(f"Source: {result['source']}")
    print(f"Rules applied: {result['rules_applied']}")

    # Example 2: Should we shift focus?
    new_urgent_input = {
        "type": "error",
        "content": "Critical error occurred",
        "priority": "high",
        "urgent": True
    }

    should_shift = brain.should_shift_focus(new_urgent_input)
    print(f"\nShould shift to error? {should_shift}")

    # Example 3: Allocate attention again (should use memory)
    result2 = brain.allocate_attention(inputs)
    print(f"\nSecond allocation source: {result2['source']}")
    print("Should be 'learned_rules' if first call succeeded!")
