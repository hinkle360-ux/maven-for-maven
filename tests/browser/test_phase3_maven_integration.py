"""
Phase 3 Tests - Maven-side Client + Tool Wiring
===============================================

Tests for:
- BrowserClient HTTP API wrapper
- Browser tool Maven integration
- Intent resolver for natural language
- End-to-end tests through Maven interface
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestBrowserClient:
    """Tests for BrowserClient API wrapper."""

    @pytest.mark.asyncio
    async def test_client_open_url(self):
        """Test client.open() method."""
        from maven_browser_client.client import BrowserClient
        from browser_runtime.models import ActionResult, PageSnapshot

        # Mock the HTTP response
        mock_response = {
            "status": "success",
            "page_id": "test-page-id",
            "snapshot": {
                "url": "https://example.com",
                "title": "Example Domain",
                "html": "<html></html>",
                "text": "Example Domain"
            }
        }

        with patch.object(BrowserClient, '__init__', lambda x, **kwargs: None):
            client = BrowserClient()
            client.base_url = "http://localhost:8765"
            client.timeout = 30.0
            client.client = MagicMock()

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()

            client.client.post = AsyncMock(return_value=mock_http_response)

            result = await client.open("https://example.com")

            assert result.status == "success"
            assert result.page_id == "test-page-id"
            assert result.snapshot.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_client_health_check(self):
        """Test client.health_check() method."""
        from maven_browser_client.client import BrowserClient

        mock_response = {
            "status": "healthy",
            "initialized": True,
            "active_pages": 0
        }

        with patch.object(BrowserClient, '__init__', lambda x, **kwargs: None):
            client = BrowserClient()
            client.base_url = "http://localhost:8765"
            client.timeout = 30.0
            client.client = MagicMock()

            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()

            client.client.get = AsyncMock(return_value=mock_http_response)

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["initialized"] is True


class TestBrowserPlanTypes:
    """Tests for browser plan type definitions."""

    def test_browser_action_types(self):
        """Test ActionType enum values."""
        from maven_browser_client.types import ActionType

        assert ActionType.OPEN == "open"
        assert ActionType.CLICK == "click"
        assert ActionType.TYPE == "type"
        assert ActionType.WAIT_FOR == "wait_for"
        assert ActionType.SCROLL == "scroll"
        assert ActionType.SCREENSHOT == "screenshot"
        assert ActionType.EXTRACT_TEXT == "extract_text"

    def test_browser_plan_creation(self):
        """Test BrowserPlan model creation."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        plan = BrowserPlan(
            goal="Test search",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://google.com"}),
                BrowserAction(action=ActionType.TYPE, params={"selector": "input", "text": "test"})
            ]
        )

        assert plan.goal == "Test search"
        assert plan.max_steps == 5
        assert len(plan.steps) == 2
        assert plan.steps[0].action == ActionType.OPEN

    def test_browser_task_result(self):
        """Test BrowserTaskResult model."""
        from maven_browser_client.types import BrowserTaskResult, TaskStatus

        result = BrowserTaskResult(
            task_id="test-123",
            goal="Test goal",
            status=TaskStatus.COMPLETED,
            steps_executed=3,
            duration_seconds=5.5,
            final_url="https://example.com",
            final_text="Example text"
        )

        assert result.task_id == "test-123"
        assert result.status == TaskStatus.COMPLETED
        assert result.steps_executed == 3


class TestPlanValidator:
    """Tests for plan validator."""

    def test_validate_valid_plan(self):
        """Valid plan passes validation."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        plan = BrowserPlan(
            goal="Test",
            max_steps=10,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"}),
                BrowserAction(action=ActionType.CLICK, params={"selector": "#btn"}),
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is True
        assert error is None

    def test_validate_exceeds_max_steps(self):
        """Plan exceeding max steps fails validation."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        # Create plan with too many steps
        steps = [
            BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
        ]
        for i in range(50):  # Exceed default 20 step limit
            steps.append(BrowserAction(action=ActionType.SCROLL, params={}))

        plan = BrowserPlan(
            goal="Test",
            max_steps=100,  # This exceeds config limit
            steps=steps
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "exceeds limit" in error.lower()

    def test_validate_missing_url_in_open(self):
        """OPEN action without URL fails validation."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        plan = BrowserPlan(
            goal="Test",
            max_steps=10,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={}),  # Missing URL
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "url" in error.lower()

    def test_validate_click_needs_selector_or_text(self):
        """CLICK action needs selector or text."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        plan = BrowserPlan(
            goal="Test",
            max_steps=10,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"}),
                BrowserAction(action=ActionType.CLICK, params={}),  # Missing selector/text
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "selector" in error.lower() or "text" in error.lower()


class TestIntentResolver:
    """Tests for intent resolver."""

    def test_resolve_google_search_intent(self):
        """Resolve 'search for X' into Google search plan."""
        from brains.agent.tools.browser.intent_resolver import IntentResolver

        resolver = IntentResolver()

        plan = resolver.resolve("search for OpenAI news")

        assert plan is not None
        assert plan.goal == "search for OpenAI news"
        assert len(plan.steps) > 0
        # Should have an open action for Google
        assert any(
            step.action.value == "open" and "google" in step.params.get("url", "").lower()
            for step in plan.steps
        )

    def test_resolve_open_url_intent(self):
        """Resolve 'open https://...' into open URL plan."""
        from brains.agent.tools.browser.intent_resolver import IntentResolver

        resolver = IntentResolver()

        plan = resolver.resolve("open https://example.com")

        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.steps[0].action.value == "open"
        assert plan.steps[0].params.get("url") == "https://example.com"

    def test_resolve_visit_domain(self):
        """Resolve 'visit example.com' into open URL plan."""
        from brains.agent.tools.browser.intent_resolver import IntentResolver

        resolver = IntentResolver()

        plan = resolver.resolve("visit example.com")

        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.steps[0].action.value == "open"
        assert "example.com" in plan.steps[0].params.get("url", "")

    def test_extract_search_query(self):
        """Test search query extraction."""
        from brains.agent.tools.browser.intent_resolver import IntentResolver

        resolver = IntentResolver()

        assert resolver._extract_search_query("search for cats") == "cats"
        assert resolver._extract_search_query("google python tutorials") == "python tutorials"
        assert resolver._extract_search_query("find best restaurants") == "best restaurants"


class TestPatternStore:
    """Tests for pattern store."""

    def test_default_patterns_loaded(self):
        """Default patterns are available."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        import tempfile
        from pathlib import Path

        # Use temp directory to avoid affecting real store
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            patterns = store.list_patterns()
            pattern_names = [p.name for p in patterns]

            assert "google_search" in pattern_names
            assert "open_url" in pattern_names

    def test_find_pattern_by_keyword(self):
        """Find pattern matching keywords."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            pattern = store.find_pattern("search for something")
            assert pattern is not None
            assert pattern.name == "google_search"

            pattern = store.find_pattern("open a website")
            assert pattern is not None
            assert pattern.name == "open_url"

    def test_record_success_failure(self):
        """Record success/failure updates pattern counts."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Get initial counts
            pattern = store.get_pattern("google_search")
            initial_success = pattern.success_count
            initial_failure = pattern.failure_count

            # Record success
            store.record_success("google_search")
            pattern = store.get_pattern("google_search")
            assert pattern.success_count == initial_success + 1

            # Record failure
            store.record_failure("google_search")
            pattern = store.get_pattern("google_search")
            assert pattern.failure_count == initial_failure + 1


class TestBrowserToolInterface:
    """Tests for browser tool interface."""

    def test_run_browser_task_returns_dict(self):
        """run_browser_task returns dictionary result."""
        from brains.agent.tools.browser.browser_tool import run_browser_task
        from maven_browser_client.types import ActionType

        # Create a simple plan dict
        plan = {
            "goal": "Test",
            "max_steps": 1,
            "steps": [
                {"action": "open", "params": {"url": "https://example.com"}}
            ]
        }

        # This will actually try to connect to browser runtime
        # In unit test environment, it should fail gracefully
        result = run_browser_task("Test goal", plan)

        assert isinstance(result, dict)
        assert "task_id" in result
        assert "status" in result

    def test_simple_google_search_returns_dict(self):
        """simple_google_search returns dictionary result."""
        from brains.agent.tools.browser.browser_tool import simple_google_search

        # This will fail if browser runtime isn't running, but should return dict
        result = simple_google_search("test query")

        assert isinstance(result, dict)
        assert "task_id" in result
        assert "status" in result

    def test_open_url_returns_dict(self):
        """open_url returns dictionary result."""
        from brains.agent.tools.browser.browser_tool import open_url

        result = open_url("https://example.com")

        assert isinstance(result, dict)
        assert "task_id" in result
        assert "status" in result


class TestTaskExecutor:
    """Tests for task executor."""

    def test_task_executor_initializes(self):
        """TaskExecutor initializes without error."""
        from brains.agent.tools.browser.task_executor import TaskExecutor

        executor = TaskExecutor()
        assert executor is not None
        assert executor.log_dir.exists()

    @pytest.mark.asyncio
    async def test_task_executor_validates_plan(self):
        """TaskExecutor validates plan before execution."""
        from brains.agent.tools.browser.task_executor import TaskExecutor
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        executor = TaskExecutor()

        # Create invalid plan (missing URL in OPEN)
        plan = BrowserPlan(
            goal="Test",
            max_steps=10,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={}),  # Missing URL
            ]
        )

        result = await executor.execute(plan)

        assert result.status.value == "failed"
        assert "validation" in result.error_message.lower()


class TestEndToEndMavenInterface:
    """End-to-end tests through Maven interface."""

    def test_resolve_and_validate_search_plan(self):
        """Resolve intent and validate resulting plan."""
        from brains.agent.tools.browser.intent_resolver import resolve_intent
        from brains.agent.tools.browser.plan_validator import validate_plan

        # Resolve intent
        plan = resolve_intent("search for weather forecast")

        # Validate plan
        try:
            validate_plan(plan)
            valid = True
        except Exception:
            valid = False

        assert valid is True
        assert plan.goal == "search for weather forecast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
