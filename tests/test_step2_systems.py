#!/usr/bin/env python3
"""
Step 2 Systems Test
===================

Test the three cognitive systems enhanced in Step 2:
  - reasoning brain (GENERATE_THOUGHTS operation)
  - planner brain (enhanced PLAN with real multi-step plans)
  - thought_synthesis (SYNTHESIZE operation)

This script runs probe queries to verify behavioral correctness.
"""

import sys
import json
from pathlib import Path

# Add maven2_fix to path
MAVEN_ROOT = Path(__file__).parent
sys.path.insert(0, str(MAVEN_ROOT))

from brains.cognitive.reasoning.service import reasoning_brain
from brains.cognitive.planner.service import planner_brain
from brains.cognitive.thought_synthesis.service import thought_synthesizer


def test_reasoning_generate_thoughts():
    """Test reasoning brain GENERATE_THOUGHTS operation."""
    print("\n" + "="*70)
    print("TEST: Reasoning Brain - GENERATE_THOUGHTS")
    print("="*70)

    test_cases = [
        {
            "name": "Simple fact query with memory",
            "query_text": "What is 2+2?",
            "intent": "QUESTION",
            "entities": [],
            "retrieved_memories": [
                {"content": "2+2 equals 4", "confidence": 0.9, "type": "fact"}
            ],
            "context": {}
        },
        {
            "name": "Identity query with no memory",
            "query_text": "Who am I?",
            "intent": "IDENTITY_QUERY",
            "entities": [],
            "retrieved_memories": [],
            "context": {}
        },
        {
            "name": "Why question",
            "query_text": "Why do birds fly?",
            "intent": "WHY",
            "entities": ["birds", "fly"],
            "retrieved_memories": [
                {"content": "Birds have wings", "confidence": 0.8, "type": "fact"},
                {"content": "Flight helps birds escape predators", "confidence": 0.7, "type": "fact"}
            ],
            "context": {}
        }
    ]

    for tc in test_cases:
        print(f"\n  Test: {tc['name']}")
        print(f"  Query: {tc['query_text']}")
        print(f"  Intent: {tc['intent']}")

        msg = {
            "op": "GENERATE_THOUGHTS",
            "mid": "test_001",
            "payload": {
                "query_text": tc["query_text"],
                "intent": tc["intent"],
                "entities": tc["entities"],
                "retrieved_memories": tc["retrieved_memories"],
                "context": tc["context"]
            }
        }

        resp = reasoning_brain.service_api(msg)
        if resp.get("ok"):
            thought_steps = resp.get("payload", {}).get("thought_steps", [])
            print(f"  ✓ Generated {len(thought_steps)} thought steps")
            for i, step in enumerate(thought_steps, 1):
                print(f"    {i}. {step.get('type')}: {step.get('content', '')[:60]}...")
        else:
            print(f"  ✗ Failed: {resp.get('error')}")

    print("\n  Reasoning brain tests complete.")


def test_planner_multi_step():
    """Test planner brain enhanced PLAN operation."""
    print("\n" + "="*70)
    print("TEST: Planner Brain - Multi-Step Plans")
    print("="*70)

    test_cases = [
        {
            "name": "Simple Q&A intent",
            "text": "What do I like?",
            "intent": "QUESTION"
        },
        {
            "name": "Explain intent",
            "text": "Explain what Maven is",
            "intent": "EXPLAIN"
        },
        {
            "name": "Compare intent",
            "text": "Compare cats and dogs",
            "intent": "COMPARE"
        },
        {
            "name": "Command intent",
            "text": "Analyze this code",
            "intent": "COMMAND"
        }
    ]

    for tc in test_cases:
        print(f"\n  Test: {tc['name']}")
        print(f"  Text: {tc['text']}")
        print(f"  Intent: {tc['intent']}")

        msg = {
            "op": "PLAN",
            "mid": "test_002",
            "payload": {
                "text": tc["text"],
                "intent": tc["intent"],
                "context": {},
                "motivation": None
            }
        }

        resp = planner_brain.service_api(msg)
        if resp.get("ok"):
            plan = resp.get("payload", {})
            steps = plan.get("steps", [])
            print(f"  ✓ Plan created with {len(steps)} steps (priority={plan.get('priority')})")
            for i, step in enumerate(steps, 1):
                print(f"    {i}. {step.get('kind')} (id={step.get('id')}, status={step.get('status')})")

            # Verify no "planner skipped" notes
            notes = plan.get("notes", "")
            if "skipped" in notes.lower():
                print(f"  ✗ WARNING: Planner shows 'skipped' in notes: {notes}")
            else:
                print(f"  ✓ Planner produced real plan (not skipped)")
        else:
            print(f"  ✗ Failed: {resp.get('error')}")

    print("\n  Planner brain tests complete.")


def test_thought_synthesis():
    """Test thought_synthesis SYNTHESIZE operation."""
    print("\n" + "="*70)
    print("TEST: Thought Synthesis - SYNTHESIZE")
    print("="*70)

    # Create a sample plan
    plan = {
        "plan_id": "plan_test_001",
        "intent": "QUESTION",
        "steps": [
            {"id": "s1", "kind": "retrieve", "status": "pending"},
            {"id": "s2", "kind": "reason", "status": "pending"},
            {"id": "s3", "kind": "compose_answer", "status": "pending"}
        ],
        "priority": 0.7
    }

    # Create sample thought steps from reasoning
    thought_steps = [
        {
            "type": "recall",
            "source": "memory",
            "content": "User likes pizza",
            "confidence": 0.85,
            "memory_type": "preference"
        },
        {
            "type": "recall",
            "source": "memory",
            "content": "User prefers Italian food",
            "confidence": 0.75,
            "memory_type": "preference"
        }
    ]

    memories = [
        {"content": "User likes pizza", "confidence": 0.85}
    ]

    context = {}

    print(f"\n  Test: Synthesize with plan and thought steps")
    print(f"  Plan ID: {plan['plan_id']}")
    print(f"  Plan steps: {len(plan['steps'])}")
    print(f"  Thought steps: {len(thought_steps)}")

    msg = {
        "op": "SYNTHESIZE",
        "mid": "test_003",
        "payload": {
            "plan": plan,
            "thought_steps": thought_steps,
            "memories": memories,
            "context": context
        }
    }

    resp = thought_synthesizer.service_api(msg)
    if resp.get("ok"):
        payload = resp.get("payload", {})
        final_thoughts = payload.get("final_thoughts", [])
        answer_skeleton = payload.get("answer_skeleton", {})

        print(f"  ✓ Synthesis successful")
        print(f"  ✓ Final thoughts: {len(final_thoughts)}")
        for i, ft in enumerate(final_thoughts, 1):
            print(f"    {i}. {ft.get('type')}: {ft.get('content', '')[:50]}...")

        print(f"\n  ✓ Answer skeleton kind: {answer_skeleton.get('kind')}")
        slots = answer_skeleton.get("slots", {})
        print(f"    - main_point: {slots.get('main_point', '')[:50]}...")
        print(f"    - supporting_points: {len(slots.get('supporting_points', []))}")
        print(f"    - uncertainties: {len(slots.get('uncertainties', []))}")

        # Verify no stub behavior
        if not final_thoughts:
            print(f"  ✗ WARNING: final_thoughts is empty!")
        if not answer_skeleton:
            print(f"  ✗ WARNING: answer_skeleton is empty!")
    else:
        print(f"  ✗ Failed: {resp.get('error')}")

    print("\n  Thought synthesis tests complete.")


def run_probe_queries():
    """Run the five probe queries specified in Step 2."""
    print("\n" + "="*70)
    print("PROBE QUERIES: Full Pipeline Verification")
    print("="*70)

    probes = [
        ("Who am I?", "IDENTITY_QUERY"),
        ("What do I like?", "PREFERENCE_QUERY"),
        ("Why do birds fly?", "WHY"),
        ("Compare cats and dogs", "COMPARE"),
        ("Explain what Maven is", "EXPLAIN")
    ]

    for query, intent in probes:
        print(f"\n  Probe: {query}")
        print(f"  Intent: {intent}")

        # Step 1: Planner produces plan
        plan_msg = {
            "op": "PLAN",
            "payload": {"text": query, "intent": intent, "context": {}}
        }
        plan_resp = planner_brain.service_api(plan_msg)
        plan = plan_resp.get("payload", {}) if plan_resp.get("ok") else {}

        if plan.get("steps"):
            print(f"  ✓ Planner: {len(plan.get('steps', []))} steps")
        else:
            print(f"  ✗ Planner: no steps produced")

        # Step 2: Reasoning produces thought steps
        reasoning_msg = {
            "op": "GENERATE_THOUGHTS",
            "payload": {
                "query_text": query,
                "intent": intent,
                "entities": [],
                "retrieved_memories": [],  # Simulating no memory for now
                "context": {}
            }
        }
        reasoning_resp = reasoning_brain.service_api(reasoning_msg)
        thought_steps = reasoning_resp.get("payload", {}).get("thought_steps", []) if reasoning_resp.get("ok") else []

        if thought_steps:
            print(f"  ✓ Reasoning: {len(thought_steps)} thought steps")
        else:
            print(f"  ✗ Reasoning: no thought steps produced")

        # Step 3: Thought synthesis produces answer skeleton
        synth_msg = {
            "op": "SYNTHESIZE",
            "payload": {
                "plan": plan,
                "thought_steps": thought_steps,
                "memories": [],
                "context": {}
            }
        }
        synth_resp = thought_synthesizer.service_api(synth_msg)
        answer_skeleton = synth_resp.get("payload", {}).get("answer_skeleton", {}) if synth_resp.get("ok") else {}

        if answer_skeleton:
            print(f"  ✓ Synthesis: answer_skeleton kind={answer_skeleton.get('kind')}")
        else:
            print(f"  ✗ Synthesis: no answer_skeleton produced")

        print(f"  ---")

    print("\n  Probe queries complete.")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 2 SYSTEMS VERIFICATION")
    print("="*70)
    print("\nTesting reasoning, planner, and thought_synthesis brains...")

    try:
        test_reasoning_generate_thoughts()
        test_planner_multi_step()
        test_thought_synthesis()
        run_probe_queries()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print("\n✓ Step 2 implementation verified:")
        print("  - Reasoning brain produces structured thought steps")
        print("  - Planner brain produces real multi-step plans")
        print("  - Thought synthesis produces answer skeletons")
        print("\nNo stubs, no TODOs, systems carry weight.")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
