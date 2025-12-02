"""
Browser Integration Tests
=========================

End-to-end integration tests for the browser system.
These tests require the browser runtime server to be running.

NOTE: These tests require Playwright to be installed.
Skip with: pytest -m "not browser"
"""

import pytest
import asyncio
import subprocess
import time
from pathlib import Path

# Mark all tests in this module as browser tests (optional)
pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser_server():
    """Fixture to start and stop browser runtime server."""
    # Start server in background
    proc = subprocess.Popen(
        ["python", "-m", "optional.browser_runtime"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    yield proc

    # Shutdown server
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_google_search_flow(browser_server):
    """Test complete Google search flow."""
    from maven_browser_client.client import BrowserClient

    async with BrowserClient() as client:
        # Check server is healthy
        health = await client.health_check()
        assert health["status"] == "healthy"

        # Perform Google search
        snapshot = await client.search_google("Python tutorial")

        assert snapshot is not None
        assert "google" in snapshot.url.lower()
        assert len(snapshot.text) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_url_open_and_interact(browser_server):
    """Test opening URL and interacting with page."""
    from maven_browser_client.client import BrowserClient

    async with BrowserClient() as client:
        # Open example.com
        page_id, snapshot = await client.open_url("https://example.com")

        assert page_id is not None
        assert "example.com" in snapshot.url.lower()
        assert "example domain" in snapshot.text.lower()

        # Scroll down
        result = await client.scroll(page_id, direction="down", amount=300)
        assert result.status == "success"

        # Take screenshot
        result = await client.screenshot(page_id, full_page=True)
        assert result.status == "success"
        assert result.snapshot.screenshot_path is not None

        # Close page
        await client.close_page(page_id)


@pytest.mark.integration
def test_browser_tool_execution(browser_server):
    """Test browser tool with plan execution."""
    from brains.agent.tools.browser.browser_tool import run_browser_task
    from maven_browser_client.types import BrowserAction, ActionType

    plan = {
        "goal": "Open example.com",
        "max_steps": 5,
        "steps": [
            {"action": ActionType.OPEN.value, "params": {"url": "https://example.com"}},
            {"action": ActionType.SCREENSHOT.value, "params": {"full_page": False}},
        ],
    }

    result = run_browser_task("Open example.com", plan)

    assert result["status"] == "completed"
    assert result["steps_executed"] == 2
    assert result["final_url"] is not None


@pytest.mark.integration
def test_intent_to_execution(browser_server):
    """Test full flow from intent to execution."""
    from brains.agent.tools.browser.intent_resolver import resolve_intent
    from brains.agent.tools.browser.browser_tool import execute_browser_plan
    import asyncio

    # Resolve intent
    plan = resolve_intent("search for pytest documentation")

    # Execute plan
    result = asyncio.run(execute_browser_plan(plan))

    assert result.status.value == "completed"
    assert result.final_url is not None
    assert "google" in result.final_url.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
