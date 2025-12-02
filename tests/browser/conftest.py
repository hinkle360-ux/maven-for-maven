"""
Browser Tests Configuration
===========================

Pytest configuration and fixtures for browser module tests.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add maven2_fix to path for imports
maven_root = Path(__file__).parent.parent.parent / "maven2_fix"
if str(maven_root) not in sys.path:
    sys.path.insert(0, str(maven_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def browser_config():
    """Provide a test browser configuration."""
    from browser_runtime.config import BrowserConfig

    return BrowserConfig(
        browser_mode="headless",
        browser_type="chromium",
        max_steps_per_task=20,
        max_duration_seconds=60,
        rate_limit_per_domain=10,
        min_delay_ms=50,
        max_delay_ms=100,
    )


@pytest.fixture
def sample_plan():
    """Provide a sample browser plan for testing."""
    from maven_browser_client.types import BrowserPlan, BrowserAction, ActionType

    return BrowserPlan(
        goal="Test plan",
        max_steps=5,
        steps=[
            BrowserAction(action=ActionType.OPEN, params={"url": "https://example.com"}),
            BrowserAction(action=ActionType.WAIT_FOR, params={"selector": "h1", "timeout_ms": 5000}),
        ],
    )


@pytest.fixture
def temp_log_dir(tmp_path):
    """Provide a temporary directory for test logs."""
    log_dir = tmp_path / "browser_logs"
    log_dir.mkdir(parents=True)
    return log_dir


@pytest.fixture
def reset_config():
    """Reset browser config after test."""
    from browser_runtime.config import set_config, BrowserConfig

    yield

    set_config(BrowserConfig())


@pytest.fixture
def reset_rate_limiter():
    """Reset rate limiter after test."""
    from browser_runtime.rate_limiter import reset_rate_limiter

    yield

    reset_rate_limiter()
