"""
Browser Runtime Examples
========================

Collection of examples demonstrating browser automation capabilities.
"""

import asyncio
from maven_browser_client.client import BrowserClient
from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
from brains.agent.tools.browser.browser_tool import (
    simple_google_search,
    open_url,
    run_browser_task,
    execute_browser_plan,
)
from brains.agent.tools.browser.intent_resolver import resolve_intent
from brains.agent.tools.browser.pattern_store import get_pattern_store


# ============================================================================
# Example 1: Simple Google Search
# ============================================================================

def example_simple_google_search():
    """Demonstrate simple Google search using convenience function."""
    print("=== Example 1: Simple Google Search ===\n")

    result = simple_google_search("Python web scraping tutorial")

    print(f"Status: {result['status']}")
    print(f"Steps executed: {result['steps_executed']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")
    print(f"Final URL: {result['final_url']}")
    print(f"\nPage text (first 300 chars):\n{result['final_text'][:300]}...\n")


# ============================================================================
# Example 2: Open URL and Extract Content
# ============================================================================

def example_open_url():
    """Demonstrate opening a URL and extracting content."""
    print("=== Example 2: Open URL ===\n")

    result = open_url("https://news.ycombinator.com")

    print(f"Status: {result['status']}")
    print(f"Final URL: {result['final_url']}")
    print(f"Content length: {len(result['final_text'])} characters")
    print(f"\nFirst few headlines:\n{result['final_text'][:500]}...\n")


# ============================================================================
# Example 3: Custom Browser Plan
# ============================================================================

def example_custom_plan():
    """Demonstrate creating and executing a custom browser plan."""
    print("=== Example 3: Custom Browser Plan ===\n")

    # Create a plan to search Wikipedia
    plan = {
        "goal": "Search Wikipedia for 'Artificial Intelligence'",
        "max_steps": 10,
        "steps": [
            {
                "action": ActionType.OPEN.value,
                "params": {"url": "https://en.wikipedia.org"}
            },
            {
                "action": ActionType.TYPE.value,
                "params": {
                    "selector": "input[name=search]",
                    "text": "Artificial Intelligence",
                    "submit": True
                }
            },
            {
                "action": ActionType.WAIT_FOR.value,
                "params": {
                    "selector": "#firstHeading",
                    "timeout_ms": 10000
                }
            },
            {
                "action": ActionType.SCREENSHOT.value,
                "params": {"full_page": False}
            },
        ]
    }

    result = run_browser_task("Search Wikipedia", plan)

    print(f"Status: {result['status']}")
    print(f"Steps executed: {result['steps_executed']}")
    print(f"Screenshot: {result.get('screenshot_path', 'N/A')}")
    print(f"\nArticle preview:\n{result['final_text'][:400]}...\n")


# ============================================================================
# Example 4: Intent Resolution
# ============================================================================

def example_intent_resolution():
    """Demonstrate resolving natural language intents to browser plans."""
    print("=== Example 4: Intent Resolution ===\n")

    intents = [
        "search for best pizza recipes",
        "open https://github.com",
        "find documentation for pytest",
    ]

    for intent in intents:
        print(f"Intent: {intent}")
        plan = resolve_intent(intent)
        print(f"  → Resolved to {len(plan.steps)} steps")
        print(f"  → Goal: {plan.goal}")
        print()


# ============================================================================
# Example 5: Low-Level Client API
# ============================================================================

async def example_low_level_client():
    """Demonstrate using the low-level client API."""
    print("=== Example 5: Low-Level Client API ===\n")

    async with BrowserClient() as client:
        # Check server health
        health = await client.health_check()
        print(f"Server status: {health['status']}")
        print(f"Active pages: {health['active_pages']}\n")

        # Open a page
        print("Opening example.com...")
        page_id, snapshot = await client.open_url("https://example.com")
        print(f"  Page ID: {page_id}")
        print(f"  Title: {snapshot.title}")
        print(f"  URL: {snapshot.url}\n")

        # Scroll down
        print("Scrolling down...")
        await client.scroll(page_id, direction="down", amount=300)
        print("  Scrolled 300px\n")

        # Take screenshot
        print("Taking screenshot...")
        result = await client.screenshot(page_id, full_page=True)
        print(f"  Screenshot saved to: {result.snapshot.screenshot_path}\n")

        # Get text
        print("Extracting text...")
        text = await client.get_text(page_id)
        print(f"  Text length: {len(text)} characters")
        print(f"  Preview: {text[:200]}...\n")

        # Close page
        print("Closing page...")
        await client.close_page(page_id)
        print("  Page closed\n")


# ============================================================================
# Example 6: Pattern Learning
# ============================================================================

def example_pattern_learning():
    """Demonstrate pattern learning and reuse."""
    print("=== Example 6: Pattern Learning ===\n")

    store = get_pattern_store()

    # List available patterns
    print("Available patterns:")
    for pattern in store.list_patterns():
        success_rate = 0
        if pattern.success_count + pattern.failure_count > 0:
            success_rate = pattern.success_count / (pattern.success_count + pattern.failure_count)

        print(f"  - {pattern.name}: {pattern.description}")
        print(f"    Triggers: {', '.join(pattern.trigger_keywords)}")
        print(f"    Success rate: {success_rate:.1%} ({pattern.success_count} successes)")
        print()

    # Find pattern for a goal
    goal = "search for machine learning tutorials"
    print(f"Finding pattern for: '{goal}'")
    pattern = store.find_pattern(goal)
    if pattern:
        print(f"  → Matched pattern: {pattern.name}")
        print(f"  → Will execute {len(pattern.template_plan.steps)} steps")
    else:
        print("  → No pattern found, will generate custom plan")
    print()


# ============================================================================
# Example 7: Multi-Step Task with Error Handling
# ============================================================================

async def example_multi_step_task():
    """Demonstrate complex multi-step task with error handling."""
    print("=== Example 7: Multi-Step Task ===\n")

    plan = BrowserPlan(
        goal="Research Python async/await",
        max_steps=15,
        timeout_seconds=120,
        steps=[
            BrowserAction(
                action=ActionType.OPEN,
                params={"url": "https://www.google.com"}
            ),
            BrowserAction(
                action=ActionType.TYPE,
                params={
                    "selector": "textarea[name=q]",
                    "text": "Python async await tutorial",
                    "submit": True
                }
            ),
            BrowserAction(
                action=ActionType.WAIT_FOR,
                params={"selector": "#search", "timeout_ms": 10000}
            ),
            BrowserAction(
                action=ActionType.SCREENSHOT,
                params={"full_page": False}
            ),
            BrowserAction(
                action=ActionType.CLICK,
                params={"selector": "h3", "nth": 0}
            ),
            BrowserAction(
                action=ActionType.WAIT_FOR,
                params={"selector": "body", "timeout_ms": 10000}
            ),
            BrowserAction(
                action=ActionType.SCREENSHOT,
                params={"full_page": True}
            ),
        ]
    )

    try:
        result = await execute_browser_plan(plan)

        print(f"Status: {result.status.value}")
        print(f"Steps executed: {result.steps_executed}/{len(plan.steps)}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")

        if result.status.value == "completed":
            print(f"Final URL: {result.final_url}")
            print(f"Content length: {len(result.final_text)} characters")
        else:
            print(f"Error: {result.error_message}")

    except Exception as e:
        print(f"Exception occurred: {e}")

    print()


# ============================================================================
# Example 8: Taking and Analyzing Screenshots
# ============================================================================

async def example_screenshots():
    """Demonstrate taking screenshots at various points."""
    print("=== Example 8: Screenshots ===\n")

    async with BrowserClient() as client:
        # Open page
        page_id, _ = await client.open_url("https://example.com")

        # Screenshot 1: Initial load
        result = await client.screenshot(page_id, full_page=False)
        print(f"Initial screenshot: {result.snapshot.screenshot_path}")

        # Scroll down
        await client.scroll(page_id, direction="down", amount=500)

        # Screenshot 2: After scroll
        result = await client.screenshot(page_id, full_page=False)
        print(f"After scroll: {result.snapshot.screenshot_path}")

        # Screenshot 3: Full page
        result = await client.screenshot(page_id, full_page=True)
        print(f"Full page: {result.snapshot.screenshot_path}")

        await client.close_page(page_id)

    print()


# ============================================================================
# Main: Run All Examples
# ============================================================================

def run_all_examples():
    """Run all examples (requires browser runtime server to be running)."""
    print("\n" + "=" * 70)
    print("Browser Runtime Examples")
    print("=" * 70 + "\n")
    print("NOTE: These examples require the browser runtime server to be running.")
    print("Start it with: python -m browser_runtime.server\n")

    try:
        # Synchronous examples
        example_simple_google_search()
        example_open_url()
        example_custom_plan()
        example_intent_resolution()
        example_pattern_learning()

        # Asynchronous examples
        print("Running async examples...\n")
        asyncio.run(example_low_level_client())
        asyncio.run(example_multi_step_task())
        asyncio.run(example_screenshots())

        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure the browser runtime server is running!")


if __name__ == "__main__":
    run_all_examples()
