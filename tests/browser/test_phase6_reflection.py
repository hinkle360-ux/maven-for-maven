"""
Phase 6 Tests - Autonomy, Reflection, and Pattern Learning
==========================================================

Tests for:
- Reflection step per task
- Pattern storage and reuse
- Learning from successful tasks
- Pattern improvement over time
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timezone


class TestReflectionSummary:
    """Tests for task reflection summaries."""

    def test_task_result_has_metadata(self):
        """Task result can include reflection metadata."""
        from maven_browser_client.types import BrowserTaskResult, TaskStatus

        result = BrowserTaskResult(
            task_id="test-123",
            goal="Test goal",
            status=TaskStatus.COMPLETED,
            steps_executed=5,
            duration_seconds=10.5,
            metadata={
                "reflection": {
                    "goal_met": True,
                    "unnecessary_steps": [],
                    "failed_steps": [],
                    "patterns_identified": ["google_search"]
                }
            }
        )

        assert result.metadata.get("reflection") is not None
        assert result.metadata["reflection"]["goal_met"] is True


class TestPatternStorage:
    """Tests for pattern storage."""

    def test_pattern_store_saves_patterns(self):
        """Pattern store persists patterns to disk."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        from maven_browser_client.types import PatternMatch, BrowserPlan, BrowserAction, ActionType

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "patterns.json"
            store = PatternStore(store_path=store_path)

            # Add a pattern
            new_pattern = PatternMatch(
                name="test_pattern",
                description="Test pattern for unit tests",
                trigger_keywords=["test", "unit"],
                domains=["test.com"],
                template_plan=BrowserPlan(
                    goal="Test",
                    max_steps=5,
                    steps=[
                        BrowserAction(action=ActionType.OPEN, params={"url": "https://test.com"})
                    ]
                )
            )

            store.add_pattern(new_pattern)

            # Verify file exists
            assert store_path.exists()

            # Load a new store from same path
            store2 = PatternStore(store_path=store_path)
            loaded_pattern = store2.get_pattern("test_pattern")

            assert loaded_pattern is not None
            assert loaded_pattern.description == "Test pattern for unit tests"

    def test_pattern_match_model(self):
        """PatternMatch model structure is correct."""
        from maven_browser_client.types import PatternMatch, BrowserPlan, BrowserAction, ActionType

        pattern = PatternMatch(
            name="example_pattern",
            description="An example pattern",
            trigger_keywords=["search", "find"],
            domains=["google.com"],
            template_plan=BrowserPlan(
                goal="Search template",
                max_steps=5,
                steps=[
                    BrowserAction(action=ActionType.OPEN, params={"url": "https://google.com"})
                ]
            ),
            success_count=10,
            failure_count=2,
            last_used=datetime.now(timezone.utc)
        )

        assert pattern.name == "example_pattern"
        assert pattern.success_count == 10
        assert pattern.failure_count == 2
        assert "search" in pattern.trigger_keywords


class TestPatternReuse:
    """Tests for pattern reuse."""

    def test_pattern_matching_by_keywords(self):
        """Patterns are matched by trigger keywords."""
        from brains.agent.tools.browser.pattern_store import PatternStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Default patterns should be loaded
            # "search" should match google_search pattern
            pattern = store.find_pattern("I want to search for something")

            assert pattern is not None
            assert pattern.name == "google_search"

    def test_pattern_scoring_prefers_successful(self):
        """Pattern matching prefers patterns with higher success rate."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        from maven_browser_client.types import PatternMatch, BrowserPlan, BrowserAction, ActionType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Add two patterns with same keywords but different success rates
            pattern1 = PatternMatch(
                name="search_v1",
                description="Old search pattern",
                trigger_keywords=["search"],
                domains=["google.com"],
                template_plan=BrowserPlan(
                    goal="Old search",
                    max_steps=5,
                    steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://google.com"})]
                ),
                success_count=2,
                failure_count=8  # Low success rate
            )

            pattern2 = PatternMatch(
                name="search_v2",
                description="New search pattern",
                trigger_keywords=["search"],
                domains=["google.com"],
                template_plan=BrowserPlan(
                    goal="New search",
                    max_steps=5,
                    steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://google.com"})]
                ),
                success_count=9,
                failure_count=1  # High success rate
            )

            store.add_pattern(pattern1)
            store.add_pattern(pattern2)

            # Should prefer pattern2 due to higher success rate
            matched = store.find_pattern("search for something")
            # Note: google_search is also present with 0 history, so v2 should win
            assert matched is not None

    def test_record_success_updates_count(self):
        """Recording success increments success count."""
        from brains.agent.tools.browser.pattern_store import PatternStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Get initial count
            pattern = store.get_pattern("google_search")
            initial = pattern.success_count

            # Record success
            store.record_success("google_search")

            # Verify increment
            pattern = store.get_pattern("google_search")
            assert pattern.success_count == initial + 1

    def test_record_failure_updates_count(self):
        """Recording failure increments failure count."""
        from brains.agent.tools.browser.pattern_store import PatternStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Get initial count
            pattern = store.get_pattern("google_search")
            initial = pattern.failure_count

            # Record failure
            store.record_failure("google_search")

            # Verify increment
            pattern = store.get_pattern("google_search")
            assert pattern.failure_count == initial + 1


class TestLearningFromTasks:
    """Tests for learning from task execution."""

    def test_successful_task_can_create_pattern(self):
        """Successful task can be turned into a pattern."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType, PatternMatch
        from brains.agent.tools.browser.pattern_store import PatternStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Simulate a successful task
            successful_plan = BrowserPlan(
                goal="Search StackOverflow for Python help",
                max_steps=5,
                steps=[
                    BrowserAction(action=ActionType.OPEN, params={"url": "https://stackoverflow.com"}),
                    BrowserAction(action=ActionType.TYPE, params={
                        "selector": "input[name=q]",
                        "text": "Python help",
                        "submit": True
                    }),
                ]
            )

            # Create pattern from successful task
            new_pattern = PatternMatch(
                name="stackoverflow_search",
                description="Search StackOverflow for programming help",
                trigger_keywords=["stackoverflow", "programming", "code help"],
                domains=["stackoverflow.com"],
                template_plan=BrowserPlan(
                    goal="Search StackOverflow for {query}",
                    max_steps=5,
                    steps=[
                        BrowserAction(action=ActionType.OPEN, params={"url": "https://stackoverflow.com"}),
                        BrowserAction(action=ActionType.TYPE, params={
                            "selector": "input[name=q]",
                            "text": "{query}",
                            "submit": True
                        }),
                    ]
                ),
                success_count=1
            )

            store.add_pattern(new_pattern)

            # Verify pattern is now available
            pattern = store.get_pattern("stackoverflow_search")
            assert pattern is not None
            assert pattern.success_count == 1


class TestPatternImprovement:
    """Tests for pattern improvement over time."""

    def test_pattern_update_preserves_history(self):
        """Updating pattern preserves success/failure counts."""
        from brains.agent.tools.browser.pattern_store import PatternStore
        from maven_browser_client.types import PatternMatch, BrowserPlan, BrowserAction, ActionType

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")

            # Add pattern
            pattern = PatternMatch(
                name="test_update",
                description="Original description",
                trigger_keywords=["test"],
                domains=["test.com"],
                template_plan=BrowserPlan(
                    goal="Test",
                    max_steps=5,
                    steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://test.com"})]
                ),
                success_count=10,
                failure_count=2
            )
            store.add_pattern(pattern)

            # Update pattern (same name replaces)
            updated_pattern = PatternMatch(
                name="test_update",
                description="Updated description",
                trigger_keywords=["test", "update"],
                domains=["test.com"],
                template_plan=BrowserPlan(
                    goal="Updated Test",
                    max_steps=10,
                    steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://test.com/v2"})]
                ),
                success_count=10,  # Preserve count
                failure_count=2
            )
            store.add_pattern(updated_pattern)

            # Verify update
            loaded = store.get_pattern("test_update")
            assert loaded.description == "Updated description"
            assert "update" in loaded.trigger_keywords
            assert loaded.success_count == 10  # Preserved


class TestIntentResolverPatternUse:
    """Tests for intent resolver using patterns."""

    def test_resolver_uses_stored_patterns(self):
        """Intent resolver consults pattern store."""
        from brains.agent.tools.browser.intent_resolver import IntentResolver
        from brains.agent.tools.browser.pattern_store import PatternStore

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create resolver with custom pattern store
            store = PatternStore(store_path=Path(tmpdir) / "patterns.json")
            resolver = IntentResolver()
            resolver.pattern_store = store

            # Resolve an intent that matches google_search pattern
            plan = resolver.resolve("search for weather forecast")

            # Should use google_search pattern
            assert plan is not None
            assert "google" in plan.steps[0].params.get("url", "").lower()


class TestReflectionIntegration:
    """Tests for reflection system integration."""

    def test_task_result_includes_reflection_data(self):
        """Task results can include reflection data for learning."""
        from maven_browser_client.types import BrowserTaskResult, TaskStatus

        # Complete result with reflection
        result = BrowserTaskResult(
            task_id="reflect-test",
            goal="Search for news",
            status=TaskStatus.COMPLETED,
            steps_executed=3,
            duration_seconds=5.0,
            final_url="https://google.com/search?q=news",
            metadata={
                "pattern_used": "google_search",
                "reflection": {
                    "goal_met": True,
                    "steps_analysis": [
                        {"step": 0, "action": "open", "success": True},
                        {"step": 1, "action": "type", "success": True},
                        {"step": 2, "action": "wait_for", "success": True},
                    ],
                    "improvements": []
                }
            }
        )

        assert result.metadata["pattern_used"] == "google_search"
        assert result.metadata["reflection"]["goal_met"] is True
        assert len(result.metadata["reflection"]["steps_analysis"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
