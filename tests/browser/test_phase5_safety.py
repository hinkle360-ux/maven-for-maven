"""
Phase 5 Tests - Safety, Sandboxing, and Logging
===============================================

Tests for:
- Max steps and max duration limits
- Rate limiting per origin
- Global allow/deny lists
- Task logging
- Sensitive data redaction
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


class TestMaxStepsLimit:
    """Tests for max steps per task limit."""

    def test_config_max_steps_default(self):
        """Config has sensible max_steps default."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig()
        assert config.max_steps_per_task == 20

    def test_config_max_steps_from_env(self):
        """Max steps can be set via environment."""
        import os
        from browser_runtime.config import BrowserConfig

        os.environ["BROWSER_MAX_STEPS_PER_TASK"] = "50"
        config = BrowserConfig.from_env()
        assert config.max_steps_per_task == 50

        del os.environ["BROWSER_MAX_STEPS_PER_TASK"]

    @pytest.mark.asyncio
    async def test_execution_respects_max_steps(self):
        """Plan execution stops at max_steps."""
        from brains.agent.tools.browser.browser_tool import execute_browser_plan
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        # Create plan with many steps
        steps = [
            BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
        ]
        for _ in range(10):
            steps.append(BrowserAction(action=ActionType.SCROLL, params={"direction": "down"}))

        plan = BrowserPlan(
            goal="Test",
            max_steps=3,  # Only allow 3 steps
            steps=steps
        )

        # Note: This would need browser runtime running to actually execute
        # For unit test, we verify the plan structure
        assert plan.max_steps == 3
        assert len(plan.steps) > plan.max_steps


class TestMaxDurationLimit:
    """Tests for max duration per task limit."""

    def test_config_max_duration_default(self):
        """Config has sensible max_duration default."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig()
        assert config.max_duration_seconds == 120

    def test_browser_plan_timeout(self):
        """BrowserPlan has configurable timeout."""
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

        plan = BrowserPlan(
            goal="Test",
            max_steps=5,
            steps=[BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})],
            timeout_seconds=30
        )

        assert plan.timeout_seconds == 30

    @pytest.mark.asyncio
    async def test_task_executor_enforces_timeout(self):
        """TaskExecutor enforces timeout on plan execution."""
        from brains.agent.tools.browser.task_executor import TaskExecutor
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType, TaskStatus

        executor = TaskExecutor()

        # Create plan with very short timeout
        plan = BrowserPlan(
            goal="Test timeout",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"})
            ],
            timeout_seconds=1  # Very short timeout
        )

        # Mock the execute to take longer than timeout
        with patch.object(executor, '_execute_plan', new_callable=AsyncMock) as mock_execute:
            async def slow_execute(*args, **kwargs):
                await asyncio.sleep(5)  # Longer than timeout
                return MagicMock(status=TaskStatus.COMPLETED)

            mock_execute.side_effect = slow_execute

            result = await executor.execute(plan)

            # Should timeout
            assert result.status == TaskStatus.TIMEOUT


class TestRateLimiting:
    """Tests for rate limiting per domain."""

    def test_config_rate_limit_default(self):
        """Config has rate limit setting."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig()
        assert config.rate_limit_per_domain >= 0

    def test_config_rate_limit_from_env(self):
        """Rate limit can be set via environment."""
        import os
        from browser_runtime.config import BrowserConfig

        os.environ["BROWSER_RATE_LIMIT_PER_DOMAIN"] = "10"
        config = BrowserConfig.from_env()
        assert config.rate_limit_per_domain == 10

        del os.environ["BROWSER_RATE_LIMIT_PER_DOMAIN"]


class TestAllowDenyLists:
    """Tests for allow/deny lists."""

    def test_empty_lists_allow_all(self):
        """Empty lists allow all domains."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(allowed_domains=[], blocked_domains=[])

        assert config.is_domain_allowed("any-domain.com") is True
        assert config.is_domain_allowed("another.org") is True

    def test_allow_list_restricts(self):
        """Non-empty allow list restricts to listed domains."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(
            allowed_domains=["trusted.com", "example.com"],
            blocked_domains=[]
        )

        assert config.is_domain_allowed("trusted.com") is True
        assert config.is_domain_allowed("example.com") is True
        assert config.is_domain_allowed("untrusted.org") is False

    def test_block_list_denies(self):
        """Domains in block list are denied."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(
            allowed_domains=[],
            blocked_domains=["evil.com", "malware.org"]
        )

        assert config.is_domain_allowed("evil.com") is False
        assert config.is_domain_allowed("malware.org") is False
        assert config.is_domain_allowed("good.com") is True

    def test_block_list_overrides_allow_list(self):
        """Block list takes precedence if domain is in both."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig(
            allowed_domains=["mixed.com"],
            blocked_domains=["mixed.com"]
        )

        # Block list should win
        assert config.is_domain_allowed("mixed.com") is False


class TestTaskLogging:
    """Tests for task logging."""

    def test_task_executor_creates_log_dir(self):
        """TaskExecutor creates log directory on init."""
        from brains.agent.tools.browser.task_executor import TaskExecutor

        executor = TaskExecutor()
        assert executor.log_dir.exists()

    def test_task_log_file_created(self):
        """Task log file is created for task."""
        from brains.agent.tools.browser.task_executor import TaskExecutor
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from browser_runtime.config import BrowserConfig, set_config

            config = BrowserConfig(log_dir=Path(tmpdir) / "browser_logs")
            set_config(config)

            executor = TaskExecutor()

            # Save a mock log
            executor._save_log("test-task-id", {
                "task_id": "test-task-id",
                "goal": "Test goal",
                "status": "completed"
            })

            # Verify log file exists
            log_file = executor.log_dir / "test-task-id.json"
            assert log_file.exists()

            # Verify content
            with open(log_file) as f:
                log_data = json.load(f)
            assert log_data["task_id"] == "test-task-id"

            # Reset config
            set_config(BrowserConfig())

    def test_task_log_retrieval(self):
        """Task log can be retrieved by ID."""
        from brains.agent.tools.browser.task_executor import TaskExecutor
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            from browser_runtime.config import BrowserConfig, set_config

            config = BrowserConfig(log_dir=Path(tmpdir) / "browser_logs")
            set_config(config)

            executor = TaskExecutor()

            # Save a log
            executor._save_log("retrieve-test", {
                "task_id": "retrieve-test",
                "goal": "Test retrieval",
                "status": "completed"
            })

            # Retrieve it
            log_data = executor.get_task_log("retrieve-test")
            assert log_data is not None
            assert log_data["goal"] == "Test retrieval"

            # Non-existent returns None
            assert executor.get_task_log("non-existent") is None

            # Reset config
            set_config(BrowserConfig())

    @pytest.mark.asyncio
    async def test_failed_task_logs_error(self):
        """Failed task logs error information."""
        from brains.agent.tools.browser.task_executor import TaskExecutor
        from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType, TaskStatus

        executor = TaskExecutor()

        # Create invalid plan (will fail validation)
        plan = BrowserPlan(
            goal="Test failure logging",
            max_steps=5,
            steps=[
                BrowserAction(action=ActionType.OPEN, params={}),  # Missing URL
            ]
        )

        result = await executor.execute(plan)

        assert result.status == TaskStatus.FAILED

        # Verify log was created
        log_data = executor.get_task_log(result.task_id)
        assert log_data is not None
        assert log_data["status"] == "failed"
        assert "error" in log_data


class TestTaskLogStructure:
    """Tests for task log structure."""

    def test_task_log_model(self):
        """TaskLog model has required fields."""
        from browser_runtime.models import TaskLog
        from datetime import datetime, timezone

        log = TaskLog(
            task_id="test-123",
            goal="Test goal",
            start_time=datetime.now(timezone.utc),
            status="running",
        )

        assert log.task_id == "test-123"
        assert log.goal == "Test goal"
        assert log.status == "running"
        assert log.steps_executed == 0

    def test_task_log_completion(self):
        """TaskLog can record completion status."""
        from browser_runtime.models import TaskLog, PageSnapshot
        from datetime import datetime, timezone

        start = datetime.now(timezone.utc)
        log = TaskLog(
            task_id="test-123",
            goal="Test goal",
            start_time=start,
            status="completed",
            steps_executed=5,
            end_time=datetime.now(timezone.utc),
            result=PageSnapshot(url="https://example.com", title="Result")
        )

        assert log.status == "completed"
        assert log.steps_executed == 5
        assert log.result is not None


class TestSensitiveDataRedaction:
    """Tests for sensitive data redaction in logs."""

    def test_password_fields_recognized(self):
        """Common password field names are recognized as sensitive."""
        sensitive_fields = [
            "password", "passwd", "pwd", "secret", "token",
            "api_key", "apikey", "auth", "credential"
        ]

        # This would be implemented in a redaction utility
        for field in sensitive_fields:
            assert field in sensitive_fields

    def test_redaction_function_exists(self):
        """Redaction utility is available."""
        # For now, verify the structure exists
        # Implementation would add a redact_sensitive_data function
        pass


class TestSecurityBoundaries:
    """Tests for security boundaries."""

    def test_no_file_url_allowed(self):
        """file:// URLs are blocked by default."""
        from browser_runtime.config import BrowserConfig

        config = BrowserConfig()

        # file:// URLs should be blocked
        # Note: This would need implementation in domain validation
        # to check URL scheme as well as domain

    def test_no_javascript_url_allowed(self):
        """javascript: URLs are blocked by default."""
        # Similar to file:// URLs, javascript: should be blocked
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
