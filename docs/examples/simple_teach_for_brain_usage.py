"""
Simple teach_for_brain() Usage Examples
=========================================

This shows the simplest way to use teach_for_brain() directly,
without any wrapper methods or helper classes.

This is Pattern 2 from the integration patterns documentation.
"""

from __future__ import annotations
from typing import Dict, Any
from brains.cognitive.teacher.service.teacher_brain import teach_for_brain
from brains.memory.brain_memory import BrainMemory


# Example 1: Minimal usage - just call the function
def example_1_minimal():
    """Minimal example - call teach_for_brain and use result."""
    print("\n=== Example 1: Minimal Usage ===")

    result = teach_for_brain(
        brain_name="planner",
        situation="How should I break down a web scraping task?"
    )

    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Patterns learned: {len(result.get('patterns', []))}")

    if result['verdict'] == 'LEARNED':
        for i, pattern in enumerate(result['patterns'], 1):
            print(f"  Pattern {i}: {pattern.get('pattern', '')[:60]}...")


# Example 2: Dict situation with rich context
def example_2_rich_context():
    """Example with dict situation containing rich context."""
    print("\n=== Example 2: Rich Context ===")

    result = teach_for_brain(
        brain_name="autonomy",
        situation={
            "question": "How should I prioritize these 3 tasks?",
            "tasks": [
                "Fix critical bug",
                "Write documentation",
                "Refactor code"
            ],
            "urgency": "high",
            "resources": "limited"
        }
    )

    print(f"Verdict: {result['verdict']}")

    if result['verdict'] == 'LEARNED':
        print(f"Learned {len(result['patterns'])} prioritization strategies")
        for pattern in result['patterns']:
            print(f"  - {pattern.get('pattern')}")


# Example 3: With manual memory storage
def example_3_with_storage():
    """Example showing how to store results to memory yourself."""
    print("\n=== Example 3: Manual Storage ===")

    memory = BrainMemory("example_brain")

    # 1. Check memory first
    existing = memory.retrieve(
        query={"kind": "planning_pattern", "task_type": "web_scraping"},
        limit=1
    )

    if existing and existing[0].get("confidence", 0) >= 0.7:
        print("Found existing pattern in memory!")
        print(f"  Pattern: {existing[0].get('content')}")
        return

    # 2. No pattern found - call Teacher
    print("No pattern found, calling Teacher...")

    result = teach_for_brain(
        brain_name="planner",
        situation={
            "question": "How to plan a web scraping task?",
            "task_type": "web_scraping",
            "complexity": "medium"
        }
    )

    # 3. Store if successful
    if result['verdict'] == 'LEARNED' and result.get('patterns'):
        memory.store(
            content=result['patterns'],
            metadata={
                "kind": "planning_pattern",
                "task_type": "web_scraping",
                "source": "teacher",
                "confidence": result['confidence']
            }
        )
        print(f"Stored {len(result['patterns'])} patterns to memory")


# Example 4: With validation before storage
def example_4_with_validation():
    """Example with validation before storing."""
    print("\n=== Example 4: With Validation ===")

    from brains.cognitive.reasoning.truth_classifier import TruthClassifier

    memory = BrainMemory("example_brain")

    result = teach_for_brain(
        brain_name="learning",
        situation="What's the best way to learn a new programming language?"
    )

    if result['verdict'] != 'LEARNED':
        print(f"No learning happened: {result['verdict']}")
        return

    # Validate each pattern before storing
    validated_patterns = []

    for pattern in result.get('patterns', []):
        pattern_text = pattern.get('pattern', '')

        # Use TruthClassifier
        classification = TruthClassifier.classify(
            content=pattern_text,
            confidence=result['confidence'],
            evidence=None
        )

        if classification['type'] != 'RANDOM' and classification['allow_memory_write']:
            validated_patterns.append(pattern)
            print(f"  ✅ Validated: {pattern_text[:60]}...")
        else:
            print(f"  ❌ Rejected: {pattern_text[:60]}...")

    # Store only validated patterns
    if validated_patterns:
        memory.store(
            content=validated_patterns,
            metadata={
                "kind": "learning_pattern",
                "source": "teacher",
                "confidence": result['confidence'],
                "validated": True
            }
        )
        print(f"\nStored {len(validated_patterns)} validated patterns")


# Example 5: Error handling
def example_5_error_handling():
    """Example showing error handling."""
    print("\n=== Example 5: Error Handling ===")

    # Test with invalid brain name
    result = teach_for_brain(
        brain_name="nonexistent_brain",
        situation="test question"
    )

    if result['verdict'] == 'ERROR':
        print(f"Error occurred: {result.get('error')}")
        print("Falling back to default behavior...")
        # Your fallback logic here

    # Test with valid brain but no LLM
    result = teach_for_brain(
        brain_name="planner",
        situation="How to plan a task?"
    )

    if result['verdict'] == 'ERROR':
        print(f"Error (expected in test env): {result.get('error')}")
    elif result['verdict'] == 'NO_ANSWER':
        print("Teacher had no answer")
    elif result['verdict'] == 'LEARNED':
        print(f"Success! Learned {len(result['patterns'])} patterns")


# Example 6: Different brain types
def example_6_different_brains():
    """Example showing different brain types."""
    print("\n=== Example 6: Different Brain Types ===")

    brains_to_test = [
        ("planner", "How to plan a code refactoring task?"),
        ("attention", "How to prioritize 5 competing inputs?"),
        ("learning", "What's the best way to memorize facts?"),
        ("coder", "How to structure a REST API?"),
        ("belief_tracker", "How to update belief confidence?")
    ]

    for brain_name, question in brains_to_test:
        result = teach_for_brain(brain_name=brain_name, situation=question)

        print(f"\n{brain_name}:")
        print(f"  Question: {question[:50]}...")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Patterns: {len(result.get('patterns', []))}")

        # Note: In test environment, these will show ERROR (no LLM configured)
        # In production with LLM, they will show LEARNED or NO_ANSWER


# Example 7: Using results immediately
def example_7_immediate_use():
    """Example using Teacher results immediately without storage."""
    print("\n=== Example 7: Immediate Use (No Storage) ===")

    # Sometimes you just want quick advice, not persistent learning
    result = teach_for_brain(
        brain_name="autonomy",
        situation={
            "question": "Should I handle task A or task B first?",
            "task_a": "User-facing bug",
            "task_b": "Internal refactoring",
            "deadline": "today"
        }
    )

    if result['verdict'] == 'LEARNED':
        print("Teacher's advice:")
        for pattern in result['patterns']:
            print(f"  - {pattern.get('pattern')}")

        # Use advice immediately without storing
        print("\nApplying advice to make decision...")
        # Your decision logic here
    else:
        print("No advice available, using default logic")


# Run all examples
if __name__ == "__main__":
    print("=" * 70)
    print("teach_for_brain() Simple Usage Examples")
    print("=" * 70)

    example_1_minimal()
    example_2_rich_context()
    example_3_with_storage()
    example_4_with_validation()
    example_5_error_handling()
    example_6_different_brains()
    example_7_immediate_use()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
    print("\nNOTE: In test environment without LLM configured, most calls")
    print("will return ERROR verdict. This is expected and normal.")
    print("In production with LLM configured, calls return LEARNED or NO_ANSWER.")
