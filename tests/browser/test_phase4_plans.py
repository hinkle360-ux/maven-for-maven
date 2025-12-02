"""
Phase 4 Tests - General-purpose Browser Plans
==============================================

Tests for:
- Browser plan schema validation
- Multi-step task execution
- Plan execution with state management
- Domain restriction enforcement
- Max steps guard
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestBrowserPlanSchema:
    """Tests for browser plan schema."""

    def test_browser_plan_schema_required_fields(self):
        """BrowserPlan requires goal and steps."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        # Valid plan
        plan = BrowserPlan(
            goal="Test goal",
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
            ]
        )
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1

    def test_browser_plan_defaults(self):
        """BrowserPlan has sensible defaults."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        plan = BrowserPlan(
            goal="Test",
            steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})]
        )

        assert plan.max_steps == 20
        assert plan.timeout_seconds == 120
        assert plan.allowed_domains is None

    def test_browser_action_schema(self):
        """BrowserAction schema is correct."""
        from maven_browser_client.types import BrowserAction, ActionType

        action = BrowserAction(
            action=ActionType.CLICK,
            params={"selector": "#button", "text": "Click me"}
        )

        assert action.action == ActionType.CLICK
        assert action.params["selector"] == "#button"
        assert action.params["text"] == "Click me"

    def test_browser_plan_json_serialization(self):
        """BrowserPlan can be serialized to JSON."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
        import json

        plan = BrowserPlan(
            goal="Search for something",
            max_steps=10,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://google.com"}),
                BrowserAction(action=ActionType.TYPE, params={
                    "selector": "input[name=q]",
                    "text": "test query",
                    "submit": True
                }),
            ],
            allowed_domains=["google.com"]
        )

        # Should be able to dump and load
        json_str = plan.model_dump_json()
        loaded = json.loads(json_str)

        assert loaded["goal"] == "Search for something"
        assert loaded["max_steps"] == 10
        assert len(loaded["steps"]) == 2


class TestMultiStepTaskExecution:
    """Tests for multi-step task execution."""

    def test_multi_step_plan_structure(self):
        """Create a multi-step plan for complex task."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        # Plan for "Search Google for 'NYC weather' and open the first result"
        plan = BrowserPlan(
            goal="Search Google for 'NYC weather' and open the first result",
            max_steps=12,
            steps=[
                # Step 1: Open Google
                BrowserAction(action=ActionType.OPEN, params={"url": "https://www.google.com"}),
                # Step 2: Type search query
                BrowserAction(action=ActionType.TYPE, params={
                    "selector": "textarea[name=q]",
                    "text": "NYC weather",
                    "submit": True
                }),
                # Step 3: Wait for results
                BrowserAction(action=ActionType.WAIT_FOR, params={
                    "selector": "#search",
                    "timeout_ms": 10000
                }),
                # Step 4: Click first result
                BrowserAction(action=ActionType.CLICK, params={
                    "selector": "#search .g a",  # First result link
                    "nth": 0
                }),
            ]
        )

        assert len(plan.steps) == 4
        assert plan.steps[0].action == ActionType.OPEN
        assert plan.steps[1].action == ActionType.TYPE
        assert plan.steps[2].action == ActionType.WAIT_FOR
        assert plan.steps[3].action == ActionType.CLICK


class TestDomainRestrictions:
    """Tests for domain restriction enforcement."""

    def test_plan_allowed_domains_validation(self):
        """Plan with allowed_domains restricts domains."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        # Plan tries to access blocked domain
        plan = BrowserPlan(
            goal="Test",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://evil.com/malware"})
            ],
            allowed_domains=["google.com", "example.com"]  # evil.com not in list
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "not in plan's allowed domains" in error

    def test_plan_global_config_domain_validation(self):
        """Plan respects global config domain restrictions."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
        from browser_runtime.config import set_config, BrowserConfig

        # Set config with blocked domain
        config = BrowserConfig(blocked_domains=["blocked-domain.com"])
        set_config(config)

        validator = PlanValidator()

        plan = BrowserPlan(
            goal="Test",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://blocked-domain.com/"})
            ]
            # No allowed_domains specified, should use global config
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "not allowed" in error.lower()

        # Reset config
        set_config(BrowserConfig())


class TestMaxStepsGuard:
    """Tests for max steps guard."""

    def test_plan_exceeds_config_max_steps(self):
        """Plan max_steps exceeding config limit fails validation."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
        from browser_runtime.config import get_config

        config = get_config()
        validator = PlanValidator()

        # Create plan with max_steps exceeding config limit
        plan = BrowserPlan(
            goal="Test",
            max_steps=config.max_steps_per_task + 10,  # Exceed limit
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "exceeds limit" in error

    def test_plan_steps_exceed_max_steps(self):
        """Plan with more steps than max_steps fails validation."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        # Create plan with more steps than max_steps
        steps = [
            BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
        ]
        for i in range(10):
            steps.append(BrowserAction(action=ActionType.SCROLL, params={"direction": "down"}))

        plan = BrowserPlan(
            goal="Test",
            max_steps=5,  # Only 5 allowed
            steps=steps  # 11 steps
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "steps" in error.lower()


class TestPlanExampleFormats:
    """Tests for plan format from the spec."""

    def test_flights_search_plan_format(self):
        """Test the flights search plan format from spec."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType
        from brains.agent.tools.browser.plan_validator import validate_plan

        # Example from spec
        plan = BrowserPlan(
            goal="Find cheapest flights NYC to Tokyo next month",
            max_steps=12,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://www.google.com"}),
                BrowserAction(action=ActionType.TYPE, params={
                    "selector": "textarea[name=q]",
                    "text": "NYC to Tokyo flights",
                    "submit": True
                }),
                BrowserAction(action=ActionType.WAIT_FOR, params={
                    "selector": "#search",
                    "timeout_ms": 10000
                }),
                BrowserAction(action=ActionType.EXTRACT_TEXT, params={}),
            ]
        )

        # Should be valid
        try:
            validate_plan(plan)
            valid = True
        except Exception as e:
            valid = False

        assert valid is True

    def test_plan_from_dict(self):
        """Create plan from dictionary (like JSON input)."""
        from maven_browser_client.types import BrowserPlan

        plan_dict = {
            "goal": "Test goal",
            "max_steps": 5,
            "steps": [
                {"action": "open", "params": {"url": "https://example.com"}},
                {"action": "click", "params": {"selector": "#button"}},
                {"action": "wait_for", "params": {"selector": "#result", "timeout_ms": 5000}},
            ],
            "allowed_domains": ["example.com"],
            "timeout_seconds": 60
        }

        plan = BrowserPlan(**plan_dict)

        assert plan.goal == "Test goal"
        assert plan.max_steps == 5
        assert len(plan.steps) == 3
        assert plan.steps[0].action.value == "open"


class TestActionValidation:
    """Tests for action-specific validation."""

    def test_type_action_requires_selector_and_text(self):
        """TYPE action requires selector and text."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        # Missing text
        plan = BrowserPlan(
            goal="Test",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"}),
                BrowserAction(action=ActionType.TYPE, params={"selector": "input"}),  # No text
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "text" in error.lower()

    def test_wait_for_action_requires_selector(self):
        """WAIT_FOR action requires selector."""
        from brains.agent.tools.browser.plan_validator import PlanValidator
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        validator = PlanValidator()

        plan = BrowserPlan(
            goal="Test",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"}),
                BrowserAction(action=ActionType.WAIT_FOR, params={"timeout_ms": 5000}),  # No selector
            ]
        )

        is_valid, error = validator.validate(plan)
        assert is_valid is False
        assert "selector" in error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
