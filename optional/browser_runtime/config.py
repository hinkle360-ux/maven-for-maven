"""
Browser Runtime Configuration
=============================

Centralizes all configuration for the browser runtime, including
safety limits, allowed domains, and behavior settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class BrowserConfig:
    """Configuration for the browser runtime."""

    # Browser settings
    browser_mode: str = "headless"  # "headless" or "headed"
    browser_type: str = "chromium"  # "chromium", "firefox", or "webkit"
    stealth_mode: bool = True  # Enable stealth mode to avoid bot detection (requires headed mode)

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8765

    # Safety limits
    max_steps_per_task: int = 20
    max_duration_seconds: int = 120
    rate_limit_per_domain: int = 5

    # Domain restrictions
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)

    # Human-like behavior
    min_delay_ms: int = 100
    max_delay_ms: int = 500

    # Logging
    log_level: str = "INFO"
    log_dir: Path = field(default_factory=lambda: Path("./logs/browser"))

    # CAPTCHA solving settings
    captcha_enabled: bool = True
    captcha_timeout_seconds: int = 180
    captcha_poll_interval_seconds: int = 5

    @classmethod
    def from_env(cls) -> BrowserConfig:
        """Load configuration from environment variables."""

        def parse_list(val: str) -> List[str]:
            """Parse comma-separated string into list."""
            if not val or not val.strip():
                return []
            return [x.strip() for x in val.split(",") if x.strip()]

        return cls(
            browser_mode=os.getenv("BROWSER_MODE", "headless"),
            browser_type=os.getenv("BROWSER_TYPE", "chromium"),
            stealth_mode=os.getenv("BROWSER_STEALTH_MODE", "true").lower() == "true",
            host=os.getenv("BROWSER_RUNTIME_HOST", "127.0.0.1"),
            port=int(os.getenv("BROWSER_RUNTIME_PORT", "8765")),
            max_steps_per_task=int(os.getenv("BROWSER_MAX_STEPS_PER_TASK", "20")),
            max_duration_seconds=int(os.getenv("BROWSER_MAX_DURATION_SECONDS", "120")),
            rate_limit_per_domain=int(os.getenv("BROWSER_RATE_LIMIT_PER_DOMAIN", "5")),
            allowed_domains=parse_list(os.getenv("BROWSER_ALLOWED_DOMAINS", "")),
            blocked_domains=parse_list(os.getenv("BROWSER_BLOCKED_DOMAINS", "")),
            min_delay_ms=int(os.getenv("BROWSER_MIN_DELAY_MS", "100")),
            max_delay_ms=int(os.getenv("BROWSER_MAX_DELAY_MS", "500")),
            log_level=os.getenv("BROWSER_LOG_LEVEL", "INFO"),
            log_dir=Path(os.getenv("BROWSER_LOG_DIR", "./logs/browser")),
            captcha_enabled=os.getenv("CAPTCHA_ENABLED", "true").lower() == "true",
            captcha_timeout_seconds=int(os.getenv("CAPTCHA_TIMEOUT_SECONDS", "180")),
            captcha_poll_interval_seconds=int(os.getenv("CAPTCHA_POLL_INTERVAL_SECONDS", "5")),
        )

    def is_domain_allowed(self, domain: str) -> bool:
        """Check if a domain is allowed based on allow/block lists."""
        # If allowed list exists and domain not in it, reject
        if self.allowed_domains and domain not in self.allowed_domains:
            return False

        # If domain is in block list, reject
        if domain in self.blocked_domains:
            return False

        return True


# Global config instance
_config: Optional[BrowserConfig] = None


def get_config() -> BrowserConfig:
    """Get the global browser configuration instance."""
    global _config
    if _config is None:
        _config = BrowserConfig.from_env()
    return _config


def set_config(config: BrowserConfig) -> None:
    """Set the global browser configuration instance."""
    global _config
    _config = config
