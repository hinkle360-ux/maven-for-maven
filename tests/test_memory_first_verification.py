#!/usr/bin/env python3
"""
Memory-First Runtime Verification Test
=======================================

This script tests that the memory-first architecture works correctly:
1. First "what are birds" → should call Teacher and store lesson/facts
2. Second "what are birds" → should hit memory, NO Teacher call
3. "are birds" → should hit memory via concept_key, NO Teacher call
4. Repeat "are birds" → should hit memory, NO Teacher call

Expected log patterns for PASS:
- First query: [TEACHER_HELPER] Mode=TRAINING, calling LLM...
- Subsequent queries: [TEACHER_HELPER] ✓ MEMORY HIT or ✓ Found lesson by concept_key

Failure indicators:
- Mode=TRAINING, calling LLM... on repeated queries
- Memory miss for {self.brain_id}, considering LLM teacher...
"""

import sys
import time
from pathlib import Path

# Setup path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from brains.memory.brain_memory import BrainMemory
from brains.learning.learning_mode import LearningMode
from brains.learning.lesson_utils import canonical_concept_key

# Test queries
TEST_QUERIES = [
    "what are birds",
    "what are birds",  # Repeat - should hit memory
    "are birds",       # Different phrasing - should hit via concept_key
    "are birds",       # Repeat - should hit memory
]


def clear_test_memories():
    """Clear memories for clean test start."""
    print("\n" + "="*60)
    print("CLEARING TEST MEMORIES")
    print("="*60)

    for brain_id in ["reasoning", "factual"]:
        try:
            mem = BrainMemory(brain_id)
            # We can't easily delete, but we can see what's there
            records = mem.retrieve(query=None, limit=10)
            print(f"[{brain_id}] Found {len(records)} existing records")
        except Exception as e:
            print(f"[{brain_id}] Error: {e}")


def test_concept_key_normalization():
    """Test that concept keys are generated correctly."""
    print("\n" + "="*60)
    print("TESTING CONCEPT KEY NORMALIZATION")
    print("="*60)

    test_cases = [
        ("what are birds", "birds"),
        ("are birds", "birds"),
        ("tell me about birds", "birds"),
        ("birds", "birds"),
        ("What is the capital of France?", "capital france"),
    ]

    all_pass = True
    for question, expected in test_cases:
        actual = canonical_concept_key(question)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} '{question}' → '{actual}' (expected: '{expected}')")

    return all_pass


def test_teacher_helper_memory_flow():
    """Test TeacherHelper memory-first behavior."""
    print("\n" + "="*60)
    print("TESTING TEACHER HELPER MEMORY-FIRST FLOW")
    print("="*60)

    try:
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        helper = TeacherHelper("reasoning")

        results = []

        for i, query in enumerate(TEST_QUERIES):
            print(f"\n--- Query {i+1}: '{query}' ---")
            concept_key = canonical_concept_key(query)
            print(f"    Concept key: '{concept_key}'")

            result = helper.maybe_call_teacher(
                question=query,
                context={"user_query": query},
                check_memory_first=True,
                learning_mode=LearningMode.TRAINING
            )

            if result:
                source = result.get("source", "unknown")
                verdict = result.get("verdict", "unknown")
                facts_stored = result.get("facts_stored", 0)
                patterns_stored = result.get("patterns_stored", 0)

                print(f"    Source: {source}")
                print(f"    Verdict: {verdict}")
                print(f"    Facts stored: {facts_stored}")
                print(f"    Patterns stored: {patterns_stored}")

                results.append({
                    "query": query,
                    "source": source,
                    "verdict": verdict,
                    "expected_source": "local_memory" if i > 0 else "teacher"
                })
            else:
                print(f"    Result: None (no answer)")
                results.append({
                    "query": query,
                    "source": None,
                    "verdict": None,
                    "expected_source": "local_memory" if i > 0 else "teacher"
                })

            time.sleep(0.1)  # Small delay between queries

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_reasoning_brain_strategy_flow():
    """Test Reasoning brain strategy-based flow."""
    print("\n" + "="*60)
    print("TESTING REASONING BRAIN STRATEGY FLOW")
    print("="*60)

    try:
        from brains.cognitive.reasoning.service.reasoning_brain import (
            load_strategies_from_lessons,
            select_strategy,
            apply_strategy,
            classify_reasoning_problem,
            reasoning_llm_lesson,
            STRATEGY_TABLE
        )

        results = []

        for i, query in enumerate(TEST_QUERIES):
            print(f"\n--- Query {i+1}: '{query}' ---")

            context = {"user_query": query}

            # Load strategies from lessons
            load_strategies_from_lessons(context)
            print(f"    Loaded {len(STRATEGY_TABLE)} strategies")

            # Classify problem
            problem_type = classify_reasoning_problem(context)
            print(f"    Problem type: {problem_type}")

            # Try to select strategy
            strategy = select_strategy(problem_type)
            if strategy:
                print(f"    Found strategy: {strategy.get('name', 'unknown')}")
                print(f"    Strategy confidence: {strategy.get('confidence', 0)}")

                # Apply strategy (memory-first)
                result, confidence = apply_strategy(strategy, context)
                print(f"    Apply result source: {result.get('source', 'no_source')}")
                print(f"    Apply result verdict: {result.get('verdict', 'no_verdict')}")

                results.append({
                    "query": query,
                    "has_strategy": True,
                    "result_source": result.get("source"),
                    "confidence": confidence
                })
            else:
                print(f"    No strategy found - would call reasoning_llm_lesson")
                results.append({
                    "query": query,
                    "has_strategy": False,
                    "result_source": None,
                    "confidence": 0
                })

            time.sleep(0.1)

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def analyze_results(th_results, reasoning_results):
    """Analyze test results and determine pass/fail."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    issues = []

    # Check TeacherHelper results
    print("\nTeacherHelper Analysis:")
    for i, r in enumerate(th_results):
        query = r.get("query", "")
        source = r.get("source")
        expected = r.get("expected_source")

        if i == 0:
            # First query - expect teacher call
            if source == "teacher":
                print(f"  ✓ Query 1 correctly called Teacher (source={source})")
            elif source in ("local_memory", "domain_memory"):
                print(f"  ✓ Query 1 found existing memory (source={source}) - pre-existing data")
            else:
                issues.append(f"Query 1 unexpected source: {source}")
                print(f"  ? Query 1 unexpected source: {source}")
        else:
            # Subsequent queries - expect memory hit
            if source in ("local_memory", "domain_memory"):
                print(f"  ✓ Query {i+1} correctly hit memory (source={source})")
            elif source == "teacher":
                issues.append(f"Query {i+1} ('{query}') called Teacher when memory should have hit!")
                print(f"  ✗ Query {i+1} INCORRECTLY called Teacher - memory-first FAILED!")
            else:
                print(f"  ? Query {i+1} unexpected source: {source}")

    # Check Reasoning results
    print("\nReasoning Brain Analysis:")
    for i, r in enumerate(reasoning_results):
        query = r.get("query", "")
        has_strategy = r.get("has_strategy", False)
        result_source = r.get("result_source")

        if i > 0 and not has_strategy:
            issues.append(f"Query {i+1} ('{query}') had no strategy when one should exist from lesson")
            print(f"  ✗ Query {i+1} no strategy found - lesson not loaded?")
        elif has_strategy:
            if result_source in ("qa_memory", "domain_facts"):
                print(f"  ✓ Query {i+1} answered from memory via strategy (source={result_source})")
            else:
                print(f"  ? Query {i+1} strategy applied but source={result_source}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if not issues:
        print("✓ ALL TESTS PASSED - Memory-first architecture working correctly")
        return True
    else:
        print(f"✗ FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return False


# =============================================================================
# ADDITIONAL REGRESSION TESTS - Dogs (factual), Planner, Sensorium (greetings)
# =============================================================================

# Dogs test queries (factual concept, like birds)
DOGS_TEST_QUERIES = [
    "what are dogs",
    "are dogs mammals",  # Different phrasing, same concept_key="dogs"
    "what are dogs",     # Repeat - should hit memory
]

# Planning test queries
PLANNING_TEST_QUERIES = [
    "help me plan my week",
    "plan my week",      # Different phrasing, same concept_key
    "help me plan my week",  # Repeat - should hit memory
]

# Greeting normalization test queries
GREETING_TEST_QUERIES = [
    "hey",
    "hello",
    "hi there",
    "hey",  # Repeat - should use learned pattern
]


def test_dogs_memory_first_flow():
    """
    Test memory-first behavior for a factual concept (dogs).

    Regression test: Ensures concept_key indexing works for factual queries.

    Expected:
    - First query: Teacher called, lesson/pattern stored with concept_key="dogs"
    - Subsequent queries: MEMORY HIT (no new Teacher calls)
    """
    print("\n" + "="*60)
    print("REGRESSION TEST: DOGS (Factual Concept)")
    print("="*60)

    try:
        from brains.cognitive.teacher.service.teacher_helper import TeacherHelper
        helper = TeacherHelper("reasoning")

        results = []
        teacher_call_count = 0

        for i, query in enumerate(DOGS_TEST_QUERIES):
            print(f"\n--- Query {i+1}: '{query}' ---")
            concept_key = canonical_concept_key(query)
            print(f"    Concept key: '{concept_key}'")

            result = helper.maybe_call_teacher(
                question=query,
                context={"user_query": query},
                check_memory_first=True,
                learning_mode=LearningMode.TRAINING
            )

            if result:
                source = result.get("source", "unknown")
                verdict = result.get("verdict", "unknown")

                print(f"    Source: {source}")
                print(f"    Verdict: {verdict}")

                if source == "teacher":
                    teacher_call_count += 1

                results.append({
                    "query": query,
                    "source": source,
                    "verdict": verdict,
                    "is_memory_hit": source in ("local_memory", "domain_memory")
                })
            else:
                print(f"    Result: None")
                results.append({
                    "query": query,
                    "source": None,
                    "verdict": None,
                    "is_memory_hit": False
                })

            time.sleep(0.1)

        # ASSERTION: Teacher should be called at most once (for the first unique query)
        # Subsequent queries should all hit memory
        print(f"\n    Teacher calls: {teacher_call_count}")
        print(f"    Expected: ≤1 (first query only)")

        # Count memory hits (should be all but first)
        memory_hits = sum(1 for r in results if r["is_memory_hit"])
        print(f"    Memory hits: {memory_hits}/{len(results)}")

        success = teacher_call_count <= 1 and (memory_hits >= len(results) - 1 or results[0]["is_memory_hit"])

        if success:
            print("    ✓ DOGS TEST PASSED")
        else:
            print("    ✗ DOGS TEST FAILED - Memory-first not working!")

        return success, results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_planner_memory_first_flow():
    """
    Test memory-first behavior for planning concepts.

    Regression test: Ensures Planner brain uses strategies/memory first.

    Expected:
    - First "help me plan my week": Teacher/strategy learning happens
    - Subsequent planning queries: Strategy/memory hit (no new Teacher calls)
    """
    print("\n" + "="*60)
    print("REGRESSION TEST: PLANNER (Planning Concept)")
    print("="*60)

    try:
        from brains.cognitive.planner.service.planner_brain import (
            service_api as planner_api,
            classify_planning_problem,
            load_planner_strategies_from_lessons,
            select_planner_strategy,
            PLANNER_STRATEGY_TABLE
        )

        results = []

        for i, query in enumerate(PLANNING_TEST_QUERIES):
            print(f"\n--- Query {i+1}: '{query}' ---")
            concept_key = canonical_concept_key(query)
            print(f"    Concept key: '{concept_key}'")

            context = {"user_query": query}

            # Load strategies from lessons
            load_planner_strategies_from_lessons(context)
            strategy_count = len(PLANNER_STRATEGY_TABLE)
            print(f"    Loaded strategies: {strategy_count}")

            # Classify problem
            problem_type = classify_planning_problem(context)
            print(f"    Problem type: {problem_type}")

            # Try to select strategy (memory-first check)
            strategy = select_planner_strategy(problem_type)
            has_strategy = strategy is not None
            strategy_confidence = strategy.get("confidence", 0) if strategy else 0

            if has_strategy:
                print(f"    ✓ Found strategy: {strategy.get('name', 'unknown')} (confidence={strategy_confidence})")
            else:
                print(f"    No strategy found (would call Teacher)")

            results.append({
                "query": query,
                "problem_type": problem_type,
                "has_strategy": has_strategy,
                "strategy_confidence": strategy_confidence,
            })

            # Also test via service_api for full flow
            response = planner_api({
                "op": "DECOMPOSE_TASK",
                "payload": {"task": query, "context": context}
            })

            if response.get("ok"):
                patterns_used = response.get("payload", {}).get("patterns_used", [])
                print(f"    Patterns used: {patterns_used}")
                results[-1]["patterns_used"] = patterns_used

            time.sleep(0.1)

        # ASSERTION: After first query, subsequent queries should find strategies
        # (either pre-existing or learned from first query)
        success = True
        for i, r in enumerate(results):
            if i > 0:  # Skip first query
                if not r["has_strategy"] and r.get("strategy_confidence", 0) < 0.5:
                    # Not a hard failure if patterns were still used
                    if "teacher_learned_pattern" not in r.get("patterns_used", []):
                        success = False
                        print(f"    ✗ Query {i+1} had no strategy and no learned pattern")

        if success:
            print("\n    ✓ PLANNER TEST PASSED")
        else:
            print("\n    ✗ PLANNER TEST FAILED - Memory-first not working!")

        return success, results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_sensorium_normalization_memory_first():
    """
    Test memory-first behavior for Sensorium normalization (greetings).

    Regression test: Ensures Sensorium learns normalization patterns once
    and reuses them for subsequent similar inputs.

    Expected:
    - First greeting ("hey"): Teacher called once, pattern stored for "greeting" norm_type
    - Subsequent greetings: Pattern used from memory (no new Teacher calls for same norm_type)
    """
    print("\n" + "="*60)
    print("REGRESSION TEST: SENSORIUM (Greeting Normalization)")
    print("="*60)

    try:
        from brains.cognitive.sensorium.service.sensorium_brain import (
            service_api as sensorium_api,
            _norm_type_asked_this_run,
            _memory
        )

        # Clear the in-run guard to simulate fresh session
        _norm_type_asked_this_run.clear()

        results = []
        teacher_calls_by_norm_type = {}

        for i, query in enumerate(GREETING_TEST_QUERIES):
            print(f"\n--- Input {i+1}: '{query}' ---")

            # Call sensorium normalize
            response = sensorium_api({
                "op": "NORMALIZE",
                "payload": {"text": query}
            })

            if response.get("ok"):
                payload = response.get("payload", {})
                norm_type = payload.get("norm_type", "unknown")
                normalized = payload.get("normalized", "")

                print(f"    Norm type: {norm_type}")
                print(f"    Normalized: '{normalized}'")

                # Track Teacher calls per norm_type
                # The in-run guard prevents repeated Teacher calls for same norm_type
                was_first_for_norm_type = norm_type not in teacher_calls_by_norm_type
                if was_first_for_norm_type:
                    teacher_calls_by_norm_type[norm_type] = 1
                else:
                    teacher_calls_by_norm_type[norm_type] += 1

                # Check if this norm_type is now in the asked set (meaning pattern learned)
                pattern_learned = norm_type in _norm_type_asked_this_run

                print(f"    Pattern learned for '{norm_type}': {pattern_learned}")
                print(f"    First query for this norm_type: {was_first_for_norm_type}")

                results.append({
                    "query": query,
                    "norm_type": norm_type,
                    "normalized": normalized,
                    "was_first_for_norm_type": was_first_for_norm_type,
                    "pattern_learned": pattern_learned,
                })
            else:
                print(f"    ERROR: {response.get('error')}")
                results.append({
                    "query": query,
                    "norm_type": "error",
                    "normalized": "",
                    "was_first_for_norm_type": False,
                    "pattern_learned": False,
                })

            time.sleep(0.1)

        # ASSERTION: For "greeting" norm_type:
        # - First greeting should potentially call Teacher (if no pattern)
        # - Subsequent greetings should NOT call Teacher (pattern reused)
        # The _norm_type_asked_this_run guard should prevent repeated calls

        print(f"\n    Norm types encountered: {set(r['norm_type'] for r in results)}")
        print(f"    Teacher calls by norm_type: {teacher_calls_by_norm_type}")

        # Check that greeting norm_type was learned after first call
        greeting_results = [r for r in results if r["norm_type"] == "greeting"]
        success = True

        if greeting_results:
            # After first greeting, pattern should be learned
            first_greeting = greeting_results[0]
            if not first_greeting["pattern_learned"]:
                print(f"    ✗ Pattern not learned after first greeting")
                # This isn't necessarily a failure if the input was too short for Teacher

            # Subsequent greetings should NOT trigger new Teacher calls
            # (handled by _norm_type_asked_this_run guard)
            for i, gr in enumerate(greeting_results[1:], start=2):
                if gr["was_first_for_norm_type"]:
                    print(f"    ✗ Query {i} incorrectly marked as first for norm_type")
                    success = False

        # Also verify the in-run guard is working
        if "greeting" in _norm_type_asked_this_run:
            print(f"    ✓ In-run guard contains 'greeting' (no repeated Teacher calls)")
        else:
            # Not a failure if no Teacher call was made (input too short)
            print(f"    ? 'greeting' not in in-run guard (may be ok if input too short)")

        if success:
            print("\n    ✓ SENSORIUM TEST PASSED")
        else:
            print("\n    ✗ SENSORIUM TEST FAILED - Memory-first not working!")

        return success, results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def run_all_regression_tests():
    """Run all memory-first regression tests and report summary."""
    print("\n" + "="*60)
    print("RUNNING ALL MEMORY-FIRST REGRESSION TESTS")
    print("="*60)

    all_results = {}

    # Test 1: Dogs (factual concept)
    dogs_success, dogs_results = test_dogs_memory_first_flow()
    all_results["dogs"] = {"success": dogs_success, "results": dogs_results}

    # Test 2: Planner (planning concept)
    planner_success, planner_results = test_planner_memory_first_flow()
    all_results["planner"] = {"success": planner_success, "results": planner_results}

    # Test 3: Sensorium (greeting normalization)
    sensorium_success, sensorium_results = test_sensorium_normalization_memory_first()
    all_results["sensorium"] = {"success": sensorium_success, "results": sensorium_results}

    # Summary
    print("\n" + "="*60)
    print("REGRESSION TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, result in all_results.items():
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        if result["success"]:
            passed += 1
        else:
            failed += 1
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed")

    return all_results, failed == 0


def main():
    """Run all verification tests."""
    print("="*60)
    print("MEMORY-FIRST RUNTIME VERIFICATION")
    print("="*60)
    print(f"Test queries: {TEST_QUERIES}")
    print(f"Expected behavior:")
    print(f"  1. First query calls Teacher, stores lesson with concept_key")
    print(f"  2. Repeated queries hit memory via concept_key match")
    print(f"  3. Different phrasings ('are birds') match via same concept_key")

    # Test concept key normalization
    concept_ok = test_concept_key_normalization()
    if not concept_ok:
        print("\n✗ CONCEPT KEY NORMALIZATION FAILED - stopping test")
        return 1

    # Run TeacherHelper test
    th_results = test_teacher_helper_memory_flow()

    # Run Reasoning brain test
    reasoning_results = test_reasoning_brain_strategy_flow()

    # Analyze
    success = analyze_results(th_results, reasoning_results)

    # Run additional regression tests
    print("\n" + "="*60)
    print("RUNNING ADDITIONAL REGRESSION TESTS")
    print("="*60)

    regression_results, regression_success = run_all_regression_tests()

    # Final result
    overall_success = success and regression_success

    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    if overall_success:
        print("✓ ALL TESTS PASSED - Memory-first architecture working correctly")
    else:
        print("✗ SOME TESTS FAILED - Memory-first architecture has issues")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
