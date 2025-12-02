"""
Routing Test Harness
====================

A test harness for evaluating routing accuracy against gold-labelled examples.

This allows measuring:
- Grammar layer accuracy (should be 100%)
- Router LLM accuracy
- Learned patterns accuracy
- Overall routing accuracy

Usage:
    python -m tests.test_routing_harness

    # Or with specific test set:
    python -m tests.test_routing_harness --test-set browser_tools

    # Or with verbose output:
    python -m tests.test_routing_harness --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class GoldExample:
    """A gold-labelled routing example."""

    input_text: str
    expected_tools: List[str]
    expected_brains: List[str]
    expected_intent: str
    category: str = "general"
    notes: str = ""


@dataclass
class TestResult:
    """Result of testing a single example."""

    example: GoldExample
    actual_tools: List[str]
    actual_brains: List[str]
    actual_intent: str
    tools_correct: bool
    brains_correct: bool
    passed: bool
    error: Optional[str] = None


# =============================================================================
# GOLD TEST EXAMPLES
# =============================================================================

GOLD_EXAMPLES = [
    # =========================================================================
    # BROWSER TOOL COMMANDS
    # =========================================================================
    GoldExample(
        input_text="x grok hello from maven",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="Explicit grok command via x tool"
    ),
    GoldExample(
        input_text="x: grok what is the meaning of life",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="x:grok with colon syntax"
    ),
    GoldExample(
        input_text="grok hello",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="Standalone grok command"
    ),
    GoldExample(
        input_text="post to x: Hello world!",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="Post to X"
    ),
    GoldExample(
        input_text="x post testing 123",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="X post command"
    ),
    GoldExample(
        input_text="chatgpt what is python",
        expected_tools=["chatgpt"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="ChatGPT command"
    ),

    # =========================================================================
    # RESEARCH COMMANDS
    # =========================================================================
    GoldExample(
        input_text="research: AI safety developments",
        expected_tools=["web_search", "web_fetch"],
        expected_brains=["research_manager", "reasoning"],
        expected_intent="research",
        category="research",
        notes="Explicit research command"
    ),
    GoldExample(
        input_text='research: "climate change solutions" web:20',
        expected_tools=["web_search", "web_fetch"],
        expected_brains=["research_manager", "reasoning"],
        expected_intent="research",
        category="research",
        notes="Research with web limit"
    ),
    GoldExample(
        input_text="search: latest news on AI",
        expected_tools=["web_search"],
        expected_brains=["research_manager"],
        expected_intent="web_search",
        category="research",
        notes="Web search command"
    ),

    # =========================================================================
    # BROWSER OPEN COMMANDS
    # =========================================================================
    GoldExample(
        input_text="browser_open: https://example.com",
        expected_tools=["browser_open"],
        expected_brains=["language"],
        expected_intent="browser_open",
        category="browser_tools",
        notes="Browser open URL"
    ),
    GoldExample(
        input_text="open: https://github.com",
        expected_tools=["browser_open"],
        expected_brains=["language"],
        expected_intent="browser_open",
        category="browser_tools",
        notes="Open URL shorthand"
    ),

    # =========================================================================
    # EXPLICIT TOOL CALLS
    # =========================================================================
    GoldExample(
        input_text="use grok tool: explain quantum computing",
        expected_tools=["grok"],
        expected_brains=["language"],
        expected_intent="explicit_tool",
        category="explicit_tools",
        notes="Explicit tool call syntax"
    ),
    GoldExample(
        input_text="use web_search tool: python tutorials",
        expected_tools=["web_search"],
        expected_brains=["language"],
        expected_intent="explicit_tool",
        category="explicit_tools",
        notes="Explicit web_search call"
    ),

    # =========================================================================
    # SHELL COMMANDS
    # =========================================================================
    GoldExample(
        input_text="shell: ls -la",
        expected_tools=["shell"],
        expected_brains=["coder"],
        expected_intent="shell_execution",
        category="execution",
        notes="Shell command"
    ),
    GoldExample(
        input_text="run: pip install requests",
        expected_tools=["shell"],
        expected_brains=["coder"],
        expected_intent="shell_execution",
        category="execution",
        notes="Run command"
    ),

    # =========================================================================
    # PYTHON COMMANDS
    # =========================================================================
    GoldExample(
        input_text="python: print('hello world')",
        expected_tools=["python_sandbox"],
        expected_brains=["coder"],
        expected_intent="python_execution",
        category="execution",
        notes="Python execution"
    ),

    # =========================================================================
    # CODING REQUESTS
    # =========================================================================
    GoldExample(
        input_text="code: write a function to sort a list",
        expected_tools=[],
        expected_brains=["coder", "reasoning"],
        expected_intent="coding",
        category="coding",
        notes="Coding request"
    ),

    # =========================================================================
    # SELF COMMANDS
    # =========================================================================
    GoldExample(
        input_text="maven explain yourself",
        expected_tools=[],
        expected_brains=["self_model"],
        expected_intent="self_introduction",
        category="self_commands",
        notes="Self introduction command"
    ),
    GoldExample(
        input_text="maven whoami",
        expected_tools=[],
        expected_brains=["self_model"],
        expected_intent="self_introduction",
        category="self_commands",
        notes="Whoami command"
    ),
    GoldExample(
        input_text="maven scan self",
        expected_tools=[],
        expected_brains=["self_model"],
        expected_intent="self_scan",
        category="self_commands",
        notes="Self scan command"
    ),

    # =========================================================================
    # GENERIC X COMMANDS
    # =========================================================================
    GoldExample(
        input_text="x search python",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="X search command"
    ),
    GoldExample(
        input_text="x login",
        expected_tools=["x"],
        expected_brains=["language"],
        expected_intent="browser_tool",
        category="browser_tools",
        notes="X login command"
    ),
]


def run_grammar_test(example: GoldExample) -> TestResult:
    """
    Test an example against the routing engine (unified layered routing).

    Args:
        example: The gold example to test

    Returns:
        TestResult with actual vs expected
    """
    try:
        # Use the new unified routing engine
        from brains.routing import build_routing_plan

        decision = build_routing_plan(
            query=example.input_text,
            capability_snapshot={
                "execution_mode": "FULL",
                "web_research_enabled": True,
                "tools_available": ["x", "chatgpt", "browser_open", "web_search", "web_fetch", "shell", "python_sandbox"],
            },
            llm_router_enabled=False,  # Test grammar layer only
        )

        if decision is None or decision.source == "safe_default":
            return TestResult(
                example=example,
                actual_tools=[],
                actual_brains=[],
                actual_intent="",
                tools_correct=False,
                brains_correct=False,
                passed=False,
                error="Grammar did not match (fallback to safe_default)"
            )

        # Compare tools
        actual_tools = sorted(decision.tools)
        expected_tools = sorted(example.expected_tools)
        tools_correct = actual_tools == expected_tools

        # Compare brains
        actual_brains = sorted(decision.brains)
        expected_brains = sorted(example.expected_brains)
        brains_correct = actual_brains == expected_brains

        # Compare intent
        intent_correct = decision.intent == example.expected_intent

        return TestResult(
            example=example,
            actual_tools=decision.tools,
            actual_brains=decision.brains,
            actual_intent=decision.intent,
            tools_correct=tools_correct,
            brains_correct=brains_correct,
            passed=tools_correct and brains_correct and intent_correct,
        )

    except Exception as e:
        return TestResult(
            example=example,
            actual_tools=[],
            actual_brains=[],
            actual_intent="",
            tools_correct=False,
            brains_correct=False,
            passed=False,
            error=str(e)
        )


def run_all_tests(
    category_filter: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run all gold examples through the routing tests.

    Args:
        category_filter: Only run tests in this category
        verbose: Print detailed output

    Returns:
        Dict with test results and statistics
    """
    examples = GOLD_EXAMPLES

    if category_filter:
        examples = [e for e in examples if e.category == category_filter]

    results = []
    passed = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"ROUTING TEST HARNESS")
    print(f"{'='*60}")
    print(f"Testing {len(examples)} examples" + (f" in category '{category_filter}'" if category_filter else ""))
    print()

    for example in examples:
        result = run_grammar_test(example)
        results.append(result)

        if result.passed:
            passed += 1
            if verbose:
                print(f"  [PASS] {example.input_text[:40]}...")
        else:
            failed += 1
            print(f"  [FAIL] {example.input_text[:40]}...")
            if result.error:
                print(f"         Error: {result.error}")
            else:
                print(f"         Expected tools: {example.expected_tools}, Got: {result.actual_tools}")
                print(f"         Expected brains: {example.expected_brains}, Got: {result.actual_brains}")
                print(f"         Expected intent: {example.expected_intent}, Got: {result.actual_intent}")

    # Summary
    total = len(examples)
    accuracy = (passed / total * 100) if total > 0 else 0

    print()
    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{total} ({accuracy:.1f}%)")
    print(f"  Failed: {failed}/{total}")

    # By category
    categories: Dict[str, Dict[str, int]] = {}
    for result in results:
        cat = result.example.category
        if cat not in categories:
            categories[cat] = {"passed": 0, "failed": 0}
        if result.passed:
            categories[cat]["passed"] += 1
        else:
            categories[cat]["failed"] += 1

    print()
    print("By category:")
    for cat, counts in sorted(categories.items()):
        cat_total = counts["passed"] + counts["failed"]
        cat_acc = (counts["passed"] / cat_total * 100) if cat_total > 0 else 0
        print(f"  {cat}: {counts['passed']}/{cat_total} ({cat_acc:.1f}%)")

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": accuracy,
        "results": results,
        "by_category": categories,
    }


def main():
    """Run the routing test harness."""
    parser = argparse.ArgumentParser(description="Routing Test Harness")
    parser.add_argument(
        "--test-set",
        type=str,
        help="Only run tests in this category (browser_tools, research, execution, etc.)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for all tests"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available test categories"
    )

    args = parser.parse_args()

    if args.list_categories:
        categories = set(e.category for e in GOLD_EXAMPLES)
        print("Available categories:")
        for cat in sorted(categories):
            count = sum(1 for e in GOLD_EXAMPLES if e.category == cat)
            print(f"  {cat}: {count} examples")
        return

    results = run_all_tests(
        category_filter=args.test_set,
        verbose=args.verbose
    )

    # Exit with error code if tests failed
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
