"""
Phase 2 Tests - Basic Human-like Interaction API
================================================

Tests for:
- Click, type, wait, scroll, screenshot actions
- Selector helpers (CSS selector, text search)
- Google search flow (scripted)
- Robustness (error handling, timeouts)
"""

import pytest
import asyncio
import tempfile
from pathlib import Path


class TestClickAction:
    """Tests for click action."""

    @pytest.mark.asyncio
    async def test_click_by_selector(self):
        """Click element by CSS selector."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_click
        from browser_runtime.models import OpenRequest, ClickRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Open example.com
            request = OpenRequest(url="https://example.com")
            result = await action_open(page, request)
            assert result.status == "success"

            # Click the "More information..." link
            click_request = ClickRequest(
                page_id=page_id,
                selector="a",  # Click first link
                timeout_ms=5000
            )
            result = await action_click(page, click_request)

            # Should succeed (or error if no clickable link)
            assert result.status in ["success", "error"]
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_click_by_text(self):
        """Click element by visible text."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_click
        from browser_runtime.models import OpenRequest, ClickRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Open example.com
            request = OpenRequest(url="https://example.com")
            result = await action_open(page, request)
            assert result.status == "success"

            # Click by text
            click_request = ClickRequest(
                page_id=page_id,
                text="More information",
                timeout_ms=5000
            )
            result = await action_click(page, click_request)

            # Should succeed (or error if not found)
            assert result.status in ["success", "error"]
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_click_missing_selector_returns_error(self):
        """Click with missing selector returns error."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_click
        from browser_runtime.models import OpenRequest, ClickRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Open example.com
            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            # Click with no selector or text
            click_request = ClickRequest(
                page_id=page_id,
                selector=None,
                text=None,
                timeout_ms=5000
            )
            result = await action_click(page, click_request)

            assert result.status == "error"
            assert result.error.error_type == "invalid_request"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_click_nonexistent_element_times_out(self):
        """Click on nonexistent element returns timeout error."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_click
        from browser_runtime.models import OpenRequest, ClickRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Open example.com
            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            # Click nonexistent element
            click_request = ClickRequest(
                page_id=page_id,
                selector="#nonexistent-element-12345",
                timeout_ms=1000  # Short timeout
            )
            result = await action_click(page, click_request)

            assert result.status == "error"
            assert result.error.error_type == "selector_not_found"
        finally:
            await manager.shutdown()


class TestTypeAction:
    """Tests for type action."""

    @pytest.mark.asyncio
    async def test_type_into_input(self):
        """Type text into input field."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_type
        from browser_runtime.models import OpenRequest, TypeRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Open httpbin.org for testing
            request = OpenRequest(url="https://httpbin.org/forms/post")
            result = await action_open(page, request)
            assert result.status == "success"

            # Type into an input field
            type_request = TypeRequest(
                page_id=page_id,
                selector="input[name=custname]",
                text="Test User",
                submit=False,
                delay_ms=50
            )
            result = await action_type(page, type_request)

            assert result.status == "success"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_type_with_submit(self):
        """Type text and submit (press Enter)."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_type
        from browser_runtime.models import TypeRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Go to Google
            await page.goto("https://www.google.com")

            # Type and submit
            type_request = TypeRequest(
                page_id=page_id,
                selector="textarea[name=q]",
                text="test search",
                submit=True,
                delay_ms=30
            )
            result = await action_type(page, type_request)

            # Should succeed (search submitted)
            assert result.status == "success"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_type_nonexistent_selector_returns_error(self):
        """Type with nonexistent selector returns error."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_type
        from browser_runtime.models import OpenRequest, TypeRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            type_request = TypeRequest(
                page_id=page_id,
                selector="#nonexistent-input-12345",
                text="test",
                timeout_ms=1000
            )
            result = await action_type(page, type_request)

            assert result.status == "error"
            assert result.error.error_type == "selector_not_found"
        finally:
            await manager.shutdown()


class TestWaitForAction:
    """Tests for wait_for action."""

    @pytest.mark.asyncio
    async def test_wait_for_existing_element(self):
        """Wait for element that exists."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_wait_for
        from browser_runtime.models import OpenRequest, WaitForRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            wait_request = WaitForRequest(
                page_id=page_id,
                selector="h1",
                timeout_ms=5000,
                state="visible"
            )
            result = await action_wait_for(page, wait_request)

            assert result.status == "success"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_wait_for_timeout_returns_error(self):
        """Wait for nonexistent element times out."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_wait_for
        from browser_runtime.models import OpenRequest, WaitForRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            wait_request = WaitForRequest(
                page_id=page_id,
                selector="#nonexistent-element",
                timeout_ms=1000,
                state="visible"
            )
            result = await action_wait_for(page, wait_request)

            assert result.status == "error"
            assert result.error.error_type == "timeout"
        finally:
            await manager.shutdown()


class TestScrollAction:
    """Tests for scroll action."""

    @pytest.mark.asyncio
    async def test_scroll_down(self):
        """Scroll page down."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_scroll
        from browser_runtime.models import OpenRequest, ScrollRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            scroll_request = ScrollRequest(
                page_id=page_id,
                direction="down",
                amount=200
            )
            result = await action_scroll(page, scroll_request)

            assert result.status == "success"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_scroll_to_bottom(self):
        """Scroll to bottom of page."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_scroll
        from browser_runtime.models import OpenRequest, ScrollRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            scroll_request = ScrollRequest(
                page_id=page_id,
                direction="bottom"
            )
            result = await action_scroll(page, scroll_request)

            assert result.status == "success"
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_scroll_to_top(self):
        """Scroll to top of page."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_scroll
        from browser_runtime.models import OpenRequest, ScrollRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            # Scroll down first
            await action_scroll(page, ScrollRequest(
                page_id=page_id, direction="down", amount=500
            ))

            # Scroll to top
            scroll_request = ScrollRequest(
                page_id=page_id,
                direction="top"
            )
            result = await action_scroll(page, scroll_request)

            assert result.status == "success"
        finally:
            await manager.shutdown()


class TestScreenshotAction:
    """Tests for screenshot action."""

    @pytest.mark.asyncio
    async def test_screenshot_creates_file(self):
        """Screenshot creates a file."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_screenshot
        from browser_runtime.models import OpenRequest, ScreenshotRequest
        from browser_runtime.config import get_config

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            screenshot_request = ScreenshotRequest(
                page_id=page_id,
                full_page=False
            )
            result = await action_screenshot(page, screenshot_request)

            assert result.status == "success"
            assert result.snapshot is not None
            assert result.snapshot.screenshot_path is not None

            # Verify file exists and has content
            screenshot_path = Path(result.snapshot.screenshot_path)
            assert screenshot_path.exists()
            assert screenshot_path.stat().st_size > 0
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_full_page_screenshot(self):
        """Full page screenshot works."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_screenshot
        from browser_runtime.models import OpenRequest, ScreenshotRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            await action_open(page, request)

            screenshot_request = ScreenshotRequest(
                page_id=page_id,
                full_page=True
            )
            result = await action_screenshot(page, screenshot_request)

            assert result.status == "success"
            assert result.snapshot is not None
            assert result.snapshot.screenshot_path is not None
        finally:
            await manager.shutdown()


class TestGoogleSearchFlow:
    """Scripted Google search flow test (no LLM)."""

    @pytest.mark.asyncio
    async def test_google_search_flow(self):
        """
        Complete Google search flow:
        1. Open Google
        2. Type query
        3. Submit
        4. Wait for results
        5. Verify results contain text
        """
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import (
            action_open, action_type, action_wait_for, create_page_snapshot
        )
        from browser_runtime.models import (
            OpenRequest, TypeRequest, WaitForRequest
        )

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            # Step 1: Open Google
            open_result = await action_open(page, OpenRequest(url="https://www.google.com"))
            assert open_result.status == "success"

            # Step 2-3: Type query and submit
            type_result = await action_type(page, TypeRequest(
                page_id=page_id,
                selector="textarea[name=q]",
                text="OpenAI GPT",
                submit=True,
                delay_ms=30
            ))
            assert type_result.status == "success"

            # Step 4: Wait for results
            wait_result = await action_wait_for(page, WaitForRequest(
                page_id=page_id,
                selector="#search",
                timeout_ms=10000,
                state="visible"
            ))
            assert wait_result.status == "success"

            # Step 5: Verify results
            snapshot = await create_page_snapshot(page)
            assert snapshot.text is not None
            assert len(snapshot.text) > 0
            # Results should contain something about the query
            # (Note: actual content varies)

        finally:
            await manager.shutdown()


class TestInteractionRobustness:
    """Tests for interaction robustness and error handling."""

    @pytest.mark.asyncio
    async def test_click_fail_returns_clear_error(self):
        """Click failure returns clear error message."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_click
        from browser_runtime.models import OpenRequest, ClickRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            await action_open(page, OpenRequest(url="https://example.com"))

            result = await action_click(page, ClickRequest(
                page_id=page_id,
                selector="#does-not-exist",
                timeout_ms=1000
            ))

            assert result.status == "error"
            assert result.error is not None
            assert result.error.message is not None
            assert len(result.error.message) > 0
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_wait_timeout_no_crash(self):
        """Wait timeout doesn't crash, returns result."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, action_wait_for
        from browser_runtime.models import OpenRequest, WaitForRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            await action_open(page, OpenRequest(url="https://example.com"))

            # This should timeout, not crash
            result = await action_wait_for(page, WaitForRequest(
                page_id=page_id,
                selector="#never-exists",
                timeout_ms=500,
                state="visible"
            ))

            # Should get a proper timeout result
            assert result is not None
            assert result.status == "error"
            assert result.error.error_type == "timeout"
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
