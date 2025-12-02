"""
Retry Strategy
==============

This module implements a simple exponential backoff retry strategy
for task execution.  Given a task and an executor callable, it
attempts to execute the task multiple times with increasing delays
between attempts.  If all attempts fail, it returns the last error.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Any


class RetryStrategy:
    """Execute tasks with retry and exponential backoff."""

    def __init__(self, max_attempts: int = 3, backoff_multiplier: float = 2.0) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_multiplier = float(backoff_multiplier)

    def execute_with_retry(self, task: Dict[str, Any], executor: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Run the given executor with retries on failure.

        Args:
            task: The task dictionary to pass to the executor.
            executor: A callable that takes a task and returns a
                result dictionary with a ``success`` key.

        Returns:
            The result dictionary from the successful execution, or
            an error dict after exhausting all attempts.
        """
        last_error: str | None = None
        attempt = 0
        while attempt < self.max_attempts:
            try:
                result = executor(task)
                if result.get("success"):
                    return result
                last_error = result.get("error") or "Unknown error"
            except Exception as e:
                last_error = str(e)
            # Exponential backoff
            wait_time = (self.backoff_multiplier ** attempt) * 2.0
            time.sleep(wait_time)
            attempt += 1
        return {
            "success": False,
            "error": f"Failed after {self.max_attempts} attempts: {last_error}",
            "attempts": attempt,
        }