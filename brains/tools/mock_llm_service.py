"""Mock LLM Service for Testing

This module provides a mock LLM service that can be used for testing
when Ollama is not available. It returns predefined responses based
on prompt patterns.
"""
from __future__ import annotations
from typing import Dict, Any
import re


class MockLLMService:
    """Mock LLM service that returns predefined responses for testing."""

    def __init__(self) -> None:
        self.enabled = True
        self.call_count = 0
        self.last_prompt = None

        # Predefined responses for common patterns
        self.response_patterns = {
            # Identity questions
            r"who are you|what are you|tell me about yourself": {
                "answer": "I'm Maven, a living cognitive system conceived by Josh Hinkle.",
                "facts": [
                    "Maven is a cognitive AI system with multiple specialized brains",
                    "Maven was created by Josh Hinkle",
                    "Maven uses a pipeline-based architecture for processing queries",
                    "Maven has multi-tier memory system (STM, MTM, LTM, Archive)"
                ]
            },
            # Sensorium patterns
            r"casual_statement": {
                "patterns": [
                    "Detect informal conversational statements (e.g., 'hey there', 'what's up')",
                    "Mark as small_talk affect if no specific question is asked",
                    "Route to integrator with low attention priority"
                ]
            },
            # Affect priority patterns
            r"small_talk": {
                "patterns": [
                    "Assign low affect priority to casual greetings",
                    "Use brief, friendly responses",
                    "Don't escalate to reasoning unless user asks specific question"
                ]
            },
            # Context management
            r"default_decay|affect_decay|answer_decay": {
                "strategies": [
                    "Default decay: 0.9 decay rate per query for general context",
                    "Affect decay: 0.7 decay rate for emotional context",
                    "Answer decay: 0.5 decay rate for previous answers"
                ]
            },
            # Integration rules
            r"language_vs_memory": {
                "rules": [
                    "If memory confidence > 0.8, prefer memory over language",
                    "If memory confidence < 0.5, prefer language understanding",
                    "If both low, escalate to reasoning with teacher"
                ]
            },
            # High stress patterns
            r"high_stress": {
                "patterns": [
                    "Detect contradiction between memory and query",
                    "Detect repeated failed queries (>3 times)",
                    "Escalate to teacher for new learning"
                ]
            },
        }

    def call(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Mock LLM call that returns predefined responses.

        Args:
            prompt: The prompt to process
            max_tokens: Maximum tokens (ignored in mock)
            temperature: Temperature (ignored in mock)
            context: Additional context (ignored in mock)

        Returns:
            Dict with ok, text, source, and llm_used keys
        """
        self.call_count += 1
        self.last_prompt = prompt

        if not self.enabled:
            return {"ok": False, "error": "LLM disabled"}

        # Try to match prompt against patterns
        prompt_lower = prompt.lower()

        for pattern, response_data in self.response_patterns.items():
            if re.search(pattern, prompt_lower):
                # Build response based on response type
                if "answer" in response_data:
                    # Teacher TEACH format
                    text = f"ANSWER: {response_data['answer']}\n"
                    text += "FACTS:\n"
                    for fact in response_data.get("facts", []):
                        text += f"- {fact}\n"
                elif "patterns" in response_data:
                    # Teacher pattern format
                    text = "PATTERNS:\n"
                    for pattern_text in response_data["patterns"]:
                        text += f"- {pattern_text}\n"
                elif "strategies" in response_data:
                    # Strategy format
                    text = "STRATEGIES:\n"
                    for strategy in response_data["strategies"]:
                        text += f"- {strategy}\n"
                elif "rules" in response_data:
                    # Rules format
                    text = "RULES:\n"
                    for rule in response_data["rules"]:
                        text += f"- {rule}\n"
                else:
                    text = str(response_data)

                return {
                    "ok": True,
                    "text": text,
                    "source": "mock",
                    "llm_used": False,
                }

        # Default response if no pattern matches
        # Use ANSWER: format so extraction logic works properly
        return {
            "ok": True,
            "text": "ANSWER: I'll help with that based on the context provided.\nFACTS:\n- The mock LLM service is being used\n- This is a fallback response for unmatched patterns",
            "source": "mock_default",
            "llm_used": False,
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """Return mock learning statistics."""
        return {
            "templates": len(self.response_patterns),
            "interactions": self.call_count,
            "source": "mock"
        }


# Create global instance
mock_llm_service = MockLLMService()
