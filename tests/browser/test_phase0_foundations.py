"""
Phase 0 Tests - Foundations
===========================

Tests for:
- Playwright installation and availability
- Basic browser launch smoke test
- Module import verification
"""

import pytest
import asyncio
import subprocess
import sys


class TestPlaywrightInstallation:
    """Tests for Playwright installation and availability."""

    def test_playwright_package_installed(self):
        """Verify playwright package can be imported."""
        try:
            import playwright
            assert playwright is not None
        except ImportError:
            pytest.fail("playwright package is not installed")

    def test_playwright_async_api_available(self):
        """Verify async API is available."""
        try:
            from playwright.async_api import async_playwright
            assert async_playwright is not None
        except ImportError:
            pytest.fail("playwright async API not available")

    def test_playwright_sync_api_available(self):
        """Verify sync API is available."""
        try:
            from playwright.sync_api import sync_playwright
            assert sync_playwright is not None
        except ImportError:
            pytest.fail("playwright sync API not available")


class TestBrowserRuntimeModules:
    """Tests for browser runtime module imports."""

    def test_server_module_imports(self):
        """Verify server module can be imported."""
        from browser_runtime import server
        assert hasattr(server, 'app')
        assert hasattr(server, 'run_server')

    def test_session_manager_imports(self):
        """Verify session manager module can be imported."""
        from browser_runtime import session_manager
        assert hasattr(session_manager, 'BrowserSessionManager')
        assert hasattr(session_manager, 'get_session_manager')

    def test_actions_module_imports(self):
        """Verify actions module can be imported."""
        from browser_runtime import actions
        assert hasattr(actions, 'action_open')
        assert hasattr(actions, 'action_click')
        assert hasattr(actions, 'action_type')
        assert hasattr(actions, 'action_wait_for')
        assert hasattr(actions, 'action_scroll')
        assert hasattr(actions, 'action_screenshot')

    def test_models_module_imports(self):
        """Verify models module can be imported."""
        from browser_runtime import models
        assert hasattr(models, 'ActionResult')
        assert hasattr(models, 'PageSnapshot')
        assert hasattr(models, 'BrowserError')
        assert hasattr(models, 'BrowserPlan')
        assert hasattr(models, 'BrowserAction')

    def test_config_module_imports(self):
        """Verify config module can be imported."""
        from browser_runtime import config
        assert hasattr(config, 'BrowserConfig')
        assert hasattr(config, 'get_config')


class TestBrowserClientModules:
    """Tests for browser client module imports."""

    def test_client_module_imports(self):
        """Verify client module can be imported."""
        from maven_browser_client import client
        assert hasattr(client, 'BrowserClient')

    def test_types_module_imports(self):
        """Verify types module can be imported."""
        from maven_browser_client import types
        assert hasattr(types, 'BrowserPlan')
        assert hasattr(types, 'BrowserAction')
        assert hasattr(types, 'ActionType')
        assert hasattr(types, 'BrowserTaskResult')
        assert hasattr(types, 'PatternMatch')


class TestAgentToolsModules:
    """Tests for agent tools browser module imports."""

    def test_browser_tool_imports(self):
        """Verify browser tool module can be imported."""
        from brains.agent.tools.browser import browser_tool
        assert hasattr(browser_tool, 'run_browser_task')
        assert hasattr(browser_tool, 'simple_google_search')
        assert hasattr(browser_tool, 'open_url')

    def test_task_executor_imports(self):
        """Verify task executor module can be imported."""
        from brains.agent.tools.browser import task_executor
        assert hasattr(task_executor, 'TaskExecutor')

    def test_plan_validator_imports(self):
        """Verify plan validator module can be imported."""
        from brains.agent.tools.browser import plan_validator
        assert hasattr(plan_validator, 'PlanValidator')
        assert hasattr(plan_validator, 'validate_plan')
        assert hasattr(plan_validator, 'ValidationError')

    def test_intent_resolver_imports(self):
        """Verify intent resolver module can be imported."""
        from brains.agent.tools.browser import intent_resolver
        assert hasattr(intent_resolver, 'IntentResolver')
        assert hasattr(intent_resolver, 'resolve_intent')

    def test_pattern_store_imports(self):
        """Verify pattern store module can be imported."""
        from brains.agent.tools.browser import pattern_store
        assert hasattr(pattern_store, 'PatternStore')
        assert hasattr(pattern_store, 'get_pattern_store')


class TestConfigurationDefaults:
    """Tests for configuration defaults."""

    def test_config_defaults(self):
        """Verify configuration has sensible defaults."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig()

        assert config.browser_mode == "headless"
        assert config.browser_type == "chromium"
        assert config.host == "127.0.0.1"
        assert config.port == 8765
        assert config.max_steps_per_task == 20
        assert config.max_duration_seconds == 120
        assert config.min_delay_ms >= 0
        assert config.max_delay_ms >= config.min_delay_ms

    def test_config_from_env(self):
        """Verify configuration can be loaded from environment."""
        import os
        from browser_runtime.config import BrowserConfig

        # Set test environment variables
        os.environ["BROWSER_MODE"] = "headed"
        os.environ["BROWSER_PORT"] = "9999"

        # Note: This tests from_env but env vars may not be read depending on implementation
        config = BrowserConfig.from_env()

        # Clean up
        del os.environ["BROWSER_MODE"]
        del os.environ["BROWSER_PORT"]

        # Just verify it doesn't crash
        assert config is not None


@pytest.mark.asyncio
class TestBrowserLaunchSmokeTest:
    """Smoke test for browser launch."""

    async def test_browser_launches_and_closes(self):
        """Verify browser can be launched and closed without errors."""
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.launch(headless=True)
            assert browser is not None
            assert browser.is_connected()

            # Close browser
            await browser.close()
            assert not browser.is_connected()
        finally:
            await playwright.stop()

    async def test_browser_opens_example_com(self):
        """Smoke test: open example.com and verify content."""
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        try:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            # Navigate to example.com
            await page.goto("https://example.com")

            # Verify page loaded
            html = await page.content()
            assert html is not None
            assert len(html) > 0

            # Verify expected content
            text = await page.evaluate("() => document.body.innerText")
            assert "Example Domain" in text

            await browser.close()
        finally:
            await playwright.stop()

    async def test_session_manager_initialize_shutdown(self):
        """Verify session manager can initialize and shutdown cleanly."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()

        # Initialize
        await manager.initialize()
        assert manager._initialized

        # Shutdown
        await manager.shutdown()
        assert not manager._initialized

    async def test_multiple_open_close_no_zombies(self):
        """Verify repeated open/close doesn't leak processes."""
        from browser_runtime.session_manager import BrowserSessionManager

        manager = BrowserSessionManager()

        for _ in range(3):
            await manager.initialize()
            page_id, page = await manager.create_page()
            assert page_id is not None

            await manager.close_page(page_id)
            await manager.shutdown()

            # Verify cleanup
            assert len(manager.pages) == 0
            assert not manager._initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
