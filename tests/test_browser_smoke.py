"""
Browser Smoke Tests
===================

Quick smoke tests to verify basic browser functionality.
These tests actually launch a browser and are slower.

NOTE: These tests require Playwright to be installed.
Skip with: pytest -m "not browser"
"""

import pytest
import asyncio

# Mark all tests in this module as browser tests (optional)
pytestmark = pytest.mark.browser


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_playwright_install():
    """Test that Playwright is installed and can launch browser."""
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://example.com")

            title = await page.title()
            assert title is not None

            await browser.close()

    except ImportError:
        pytest.fail("Playwright not installed. Run: pip install playwright && playwright install")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_open_example_com():
    """Smoke test: open example.com and verify content."""
    from optional.browser_runtime.session_manager import BrowserSessionManager
    from optional.browser_runtime.actions import action_open
    from optional.browser_runtime.models import OpenRequest

    manager = BrowserSessionManager()
    await manager.initialize()

    try:
        page_id, page = await manager.create_page()

        request = OpenRequest(url="https://example.com")
        result = await action_open(page, request)

        assert result.status == "success"
        assert result.snapshot is not None
        assert "Example Domain" in result.snapshot.text

    finally:
        await manager.shutdown()


@pytest.mark.smoke
def test_browser_config_loads():
    """Smoke test: verify browser configuration loads."""
    from optional.browser_runtime.config import get_config

    config = get_config()
    assert config is not None
    assert config.browser_type in ["chromium", "firefox", "webkit"]
    assert config.max_steps_per_task > 0


@pytest.mark.smoke
def test_pattern_store_has_defaults():
    """Smoke test: verify pattern store has default patterns."""
    from optional.browser_tools.pattern_store import get_pattern_store

    store = get_pattern_store()
    patterns = store.list_patterns()

    assert len(patterns) >= 2
    pattern_names = [p.name for p in patterns]
    assert "google_search" in pattern_names
    assert "open_url" in pattern_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
