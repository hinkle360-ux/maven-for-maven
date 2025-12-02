"""
Browser Runtime Tests
=====================

Comprehensive tests for the browser automation system.

NOTE: These tests require Playwright to be installed.
Skip with: pytest -m "not browser"
"""

import pytest
import asyncio
from pathlib import Path

# Mark all tests in this module as browser tests (optional)
pytestmark = pytest.mark.browser

# Phase 0 Tests - Foundations
def test_config_from_env(monkeypatch):
    """Test configuration loading from environment variables."""
    monkeypatch.setenv("BROWSER_MODE", "headed")
    monkeypatch.setenv("BROWSER_TYPE", "firefox")
    monkeypatch.setenv("BROWSER_MAX_STEPS_PER_TASK", "30")

    from browser_runtime.config import BrowserConfig

    config = BrowserConfig.from_env()

    assert config.browser_mode == "headed"
    assert config.browser_type == "firefox"
    assert config.max_steps_per_task == 30


def test_domain_allowed():
    """Test domain allow/block list logic."""
    from browser_runtime.config import BrowserConfig

    # Test with allow list
    config = BrowserConfig(allowed_domains=["example.com", "test.com"])
    assert config.is_domain_allowed("example.com") is True
    assert config.is_domain_allowed("blocked.com") is False

    # Test with block list
    config = BrowserConfig(blocked_domains=["evil.com"])
    assert config.is_domain_allowed("example.com") is True
    assert config.is_domain_allowed("evil.com") is False


# Phase 1 Tests - Basic Runtime
@pytest.mark.asyncio
async def test_session_manager_initialization():
    """Test browser session manager initialization."""
    from browser_runtime.session_manager import BrowserSessionManager

    manager = BrowserSessionManager()
    await manager.initialize()

    assert manager._initialized is True
    assert manager.browser is not None
    assert manager.context is not None

    await manager.shutdown()
    assert manager._initialized is False


@pytest.mark.asyncio
async def test_create_and_close_page():
    """Test creating and closing pages."""
    from browser_runtime.session_manager import BrowserSessionManager

    manager = BrowserSessionManager()
    await manager.initialize()

    # Create page
    page_id, page = await manager.create_page()
    assert page_id is not None
    assert page is not None
    assert page_id in manager.pages

    # Close page
    await manager.close_page(page_id)
    assert page_id not in manager.pages

    await manager.shutdown()


# Phase 2 Tests - Browser Actions
@pytest.mark.asyncio
async def test_action_open():
    """Test opening a URL."""
    from browser_runtime.session_manager import BrowserSessionManager
    from browser_runtime.actions import action_open
    from browser_runtime.models import OpenRequest

    manager = BrowserSessionManager()
    await manager.initialize()
    page_id, page = await manager.create_page()

    # Open example.com
    request = OpenRequest(url="https://example.com")
    result = await action_open(page, request)

    assert result.status == "success"
    assert result.snapshot is not None
    assert "example.com" in result.snapshot.url.lower()
    assert len(result.snapshot.text) > 0

    await manager.shutdown()


@pytest.mark.asyncio
async def test_action_open_domain_blocked():
    """Test that blocked domains are rejected."""
    from browser_runtime.session_manager import BrowserSessionManager
    from browser_runtime.actions import action_open
    from browser_runtime.models import OpenRequest
    from browser_runtime.config import BrowserConfig, set_config

    # Set config with blocked domain
    config = BrowserConfig(blocked_domains=["evil.com"])
    set_config(config)

    manager = BrowserSessionManager()
    await manager.initialize()
    page_id, page = await manager.create_page()

    # Try to open blocked domain
    request = OpenRequest(url="https://evil.com")
    result = await action_open(page, request)

    assert result.status == "error"
    assert result.error.error_type == "domain_blocked"

    await manager.shutdown()


# Phase 3 Tests - Maven Client
@pytest.mark.asyncio
async def test_browser_client_open():
    """Test browser client open method."""
    from maven_browser_client.client import BrowserClient

    # This test requires the server to be running
    # Skipping for now - would need to start server in test fixture
    pytest.skip("Requires running browser runtime server")


# Phase 4 Tests - Plan Validation
def test_plan_validator_max_steps():
    """Test plan validation - max steps constraint."""
    from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
    from brains.agent.tools.browser.plan_validator import PlanValidator, ValidationError
    from browser_runtime.config import BrowserConfig, set_config

    config = BrowserConfig(max_steps_per_task=10)
    set_config(config)

    validator = PlanValidator()

    # Create plan with too many steps
    plan = BrowserPlan(
        goal="Test",
        max_steps=20,  # Exceeds limit of 10
        steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})]
    )

    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "exceeds limit" in error


def test_plan_validator_domains():
    """Test plan validation - domain restrictions."""
    from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
    from brains.agent.tools.browser.plan_validator import PlanValidator
    from browser_runtime.config import BrowserConfig, set_config

    config = BrowserConfig(allowed_domains=["example.com"])
    set_config(config)

    validator = PlanValidator()

    # Create plan with disallowed domain
    plan = BrowserPlan(
        goal="Test",
        max_steps=5,
        steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://blocked.com"})]
    )

    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "not allowed" in error


def test_plan_validator_action_params():
    """Test plan validation - action parameters."""
    from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
    from brains.agent.tools.browser.plan_validator import PlanValidator

    validator = PlanValidator()

    # Create plan with OPEN action missing URL
    plan = BrowserPlan(
        goal="Test",
        max_steps=5,
        steps=[BrowserAction(action=ActionType.OPEN, params={})]  # Missing url
    )

    is_valid, error = validator.validate(plan)
    assert is_valid is False
    assert "missing 'url'" in error


# Phase 5 Tests - Logging
def test_task_executor_logging(tmp_path):
    """Test that task executor logs tasks."""
    from brains.agent.tools.browser.task_executor import TaskExecutor
    from browser_runtime.config import BrowserConfig, set_config

    config = BrowserConfig(log_dir=tmp_path)
    set_config(config)

    executor = TaskExecutor()
    assert executor.log_dir.exists()


# Phase 6 Tests - Pattern Learning
def test_pattern_store_initialization(tmp_path):
    """Test pattern store initialization."""
    from brains.agent.tools.browser.pattern_store import PatternStore

    store_path = tmp_path / "patterns.json"
    store = PatternStore(store_path=store_path)

    # Should have default patterns
    assert len(store.patterns) > 0

    # Should have saved to disk
    assert store_path.exists()


def test_pattern_store_find_pattern():
    """Test finding patterns by goal."""
    from brains.agent.tools.browser.pattern_store import PatternStore

    store = PatternStore()

    # Find Google search pattern
    pattern = store.find_pattern("search for python tutorial")
    assert pattern is not None
    assert pattern.name == "google_search"

    # Find open URL pattern
    pattern = store.find_pattern("open https://example.com")
    assert pattern is not None
    assert pattern.name == "open_url"


def test_pattern_store_record_success():
    """Test recording pattern success."""
    from brains.agent.tools.browser.pattern_store import PatternStore

    store = PatternStore()

    initial_count = 0
    for pattern in store.patterns:
        if pattern.name == "google_search":
            initial_count = pattern.success_count
            break

    store.record_success("google_search")

    for pattern in store.patterns:
        if pattern.name == "google_search":
            assert pattern.success_count == initial_count + 1
            assert pattern.last_used is not None
            break


def test_intent_resolver_google_search():
    """Test intent resolver for Google search."""
    from brains.agent.tools.browser.intent_resolver import IntentResolver

    resolver = IntentResolver()
    plan = resolver.resolve("search for python tutorial")

    assert plan.goal == "search for python tutorial"
    assert len(plan.steps) > 0

    # Check that query was extracted
    found_query = False
    for step in plan.steps:
        if step.action.value == "type" and "python tutorial" in step.params.get("text", ""):
            found_query = True
            break

    assert found_query is True


def test_intent_resolver_open_url():
    """Test intent resolver for opening URLs."""
    from brains.agent.tools.browser.intent_resolver import IntentResolver

    resolver = IntentResolver()
    plan = resolver.resolve("open https://example.com")

    assert len(plan.steps) > 0
    assert plan.steps[0].action.value == "open"
    assert "example.com" in plan.steps[0].params.get("url", "")


# Integration Tests
def test_browser_tool_simple_search():
    """Test browser tool simple Google search."""
    from brains.agent.tools.browser.browser_tool import simple_google_search

    # This test requires the server to be running
    # Skipping for now
    pytest.skip("Requires running browser runtime server")


def test_browser_tool_open_url():
    """Test browser tool open URL."""
    from brains.agent.tools.browser.browser_tool import open_url

    # This test requires the server to be running
    # Skipping for now
    pytest.skip("Requires running browser runtime server")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
