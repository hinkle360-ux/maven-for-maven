"""
Resource Monitor
================

Monitor and manage computational resources for the autonomous agent.
This module provides checks for memory usage, CPU load, available
storage and API rate limits.  The implementation uses the ``psutil``
library if available.  If ``psutil`` is not installed, the checks
return True for all resources.
"""

from __future__ import annotations

from typing import Dict, Any

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore


class ResourceMonitor:
    """Check resource usage and rate limits."""

    def __init__(self, rate_limit_threshold: int = 1000) -> None:
        self.rate_limit_threshold = rate_limit_threshold
        # Store timestamps of API calls if tracking rate limits
        self._api_calls: list[float] = []

    def check_resources(self) -> Dict[str, bool]:
        """Return a dict indicating whether each resource is within limits."""
        return {
            "memory_ok": self._check_memory(),
            "cpu_ok": self._check_cpu(),
            "storage_ok": self._check_storage(),
            "rate_limits_ok": self._check_rate_limits(),
        }

    def _check_memory(self) -> bool:
        if psutil is None:
            return True
        mem = psutil.virtual_memory()
        return mem.percent < 85.0

    def _check_cpu(self) -> bool:
        if psutil is None:
            return True
        # Use a short interval to avoid blocking
        cpu_usage = psutil.cpu_percent(interval=0.1)
        return cpu_usage < 90.0

    def _check_storage(self) -> bool:
        # Simplistic check: always return True.  Extend to examine
        # specific mount points or file systems as needed.
        return True

    def record_api_call(self) -> None:
        """Record an API call timestamp for rate limit tracking."""
        import time
        self._api_calls.append(time.time())
        # Remove calls older than one minute
        cutoff = time.time() - 60.0
        self._api_calls = [t for t in self._api_calls if t > cutoff]

    def _check_rate_limits(self) -> bool:
        """Check whether the agent is within rate limits."""
        return len(self._api_calls) < self.rate_limit_threshold

    def throttle_if_needed(self) -> bool:
        """Slow down execution if rate limits are exceeded.

        Returns:
            True if throttling was applied; False otherwise.
        """
        if not self._check_rate_limits():
            # Simple backoff: sleep a short time to reduce call rate
            import time
            time.sleep(1.0)
            return True
        return False