"""
Phase 1 Tests - Bare-bones Browser Runtime
==========================================

Tests for:
- Opening pages and returning HTML/text
- Page snapshot creation
- Error handling for invalid URLs
- Process cleanup
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestPageOpen:
    """Tests for page opening functionality."""

    @pytest.mark.asyncio
    async def test_open_example_com_returns_snapshot(self):
        """Open example.com and verify snapshot contains expected data."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open, create_page_snapshot
        from browser_runtime.models import OpenRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://example.com")
            result = await action_open(page, request)

            assert result.status == "success"
            assert result.snapshot is not None
            assert result.snapshot.url == "https://example.com/"
            assert "Example Domain" in result.snapshot.text
            assert len(result.snapshot.html) > 0
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_open_invalid_url_returns_error(self):
        """Opening invalid URL returns structured error."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open
        from browser_runtime.models import OpenRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://this-domain-definitely-does-not-exist-12345.com")
            result = await action_open(page, request)

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type in ["navigation_error", "timeout"]
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_page_snapshot(self):
        """Test page snapshot creation."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import create_page_snapshot
        from browser_runtime.models import OpenRequest

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()
            await page.goto("https://example.com")

            snapshot = await create_page_snapshot(page)

            assert snapshot.url == "https://example.com/"
            assert snapshot.title == "Example Domain"
            assert "Example Domain" in snapshot.text
            assert "<html" in snapshot.html.lower()
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_snapshot_include_options(self):
        """Test page snapshot with include options."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import create_page_snapshot

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()
            await page.goto("https://example.com")

            # Test with html only
            snapshot = await create_page_snapshot(page, include_html=True, include_text=False)
            assert len(snapshot.html) > 0
            assert snapshot.text == ""

            # Test with text only
            snapshot = await create_page_snapshot(page, include_html=False, include_text=True)
            assert snapshot.html == ""
            assert len(snapshot.text) > 0
        finally:
            await manager.shutdown()


class TestDomainAllowBlockLists:
    """Tests for domain allow/block list functionality."""

    def test_domain_allowed_with_empty_lists(self):
        """Domain is allowed when both lists are empty."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(allowed_domains=[], blocked_domains=[])
        assert config.is_domain_allowed("google.com") is True

    def test_domain_blocked_when_in_block_list(self):
        """Domain is blocked when in block list."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(allowed_domains=[], blocked_domains=["evil.com"])
        assert config.is_domain_allowed("evil.com") is False
        assert config.is_domain_allowed("good.com") is True

    def test_domain_allowed_only_when_in_allow_list(self):
        """Domain is allowed only when in allow list."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(allowed_domains=["good.com"], blocked_domains=[])
        assert config.is_domain_allowed("good.com") is True
        assert config.is_domain_allowed("other.com") is False

    @pytest.mark.asyncio
    async def test_blocked_domain_returns_error(self):
        """Opening blocked domain returns domain_blocked error."""
        from browser_runtime.session_manager import BrowserSessionManager
        from browser_runtime.actions import action_open
        from browser_runtime.models import OpenRequest
        from browser_runtime.config import set_config, BrowserConfig

        # Set config with blocked domain
        config = BrowserConfig(blocked_domains=["blocked.example.com"])
        set_config(config)

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()

            request = OpenRequest(url="https://blocked.example.com/path")
            result = await action_open(page, request)

            assert result.status == "error"
            assert result.error is not None
            assert result.error.error_type == "domain_blocked"
        finally:
            await manager.shutdown()
            # Reset config
            set_config(BrowserConfig())


class TestPageManagement:
    """Tests for page lifecycle management."""

    @pytest.mark.asyncio
    async def test_create_page_returns_unique_id(self):
        """Creating pages returns unique IDs."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id1, page1 = await manager.create_page()
            page_id2, page2 = await manager.create_page()

            assert page_id1 != page_id2
            assert page1 != page2
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_page_by_id(self):
        """Can retrieve page by ID."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, created_page = await manager.create_page()
            retrieved_page = await manager.get_page(page_id)

            assert retrieved_page == created_page
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_nonexistent_page_raises_error(self):
        """Getting nonexistent page raises ValueError."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            with pytest.raises(ValueError, match="not found"):
                await manager.get_page("nonexistent-id")
        finally:
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_close_page_removes_from_tracking(self):
        """Closing page removes it from tracking."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()
        await manager.initialize()

        try:
            page_id, page = await manager.create_page()
            assert page_id in manager.pages

            await manager.close_page(page_id)
            assert page_id not in manager.pages
        finally:
            await manager.shutdown()


class TestErrorStructure:
    """Tests for error structure and handling."""

    def test_browser_error_model(self):
        """Test BrowserError model structure."""
        from browser_runtime.models import BrowserError

        error = BrowserError(
            error_type="test_error",
            message="Test message",
            details={"key": "value"}
        )

        assert error.error_type == "test_error"
        assert error.message == "Test message"
        assert error.details["key"] == "value"

    def test_action_result_error_model(self):
        """Test ActionResult with error."""
        from browser_runtime.models import ActionResult, BrowserError

        error = BrowserError(error_type="test", message="Test")
        result = ActionResult(status="error", error=error)

        assert result.status == "error"
        assert result.error is not None
        assert result.error.error_type == "test"

    def test_action_result_success_model(self):
        """Test ActionResult with success."""
        from browser_runtime.models import ActionResult, PageSnapshot

        snapshot = PageSnapshot(url="https://example.com", title="Test")
        result = ActionResult(status="success", snapshot=snapshot)

        assert result.status == "success"
        assert result.snapshot is not None
        assert result.snapshot.url == "https://example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
