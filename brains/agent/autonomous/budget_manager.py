"""Budget Manager
==================

This module provides a simple execution budget tracker for the
autonomous agent.  The budget is specified in arbitrary units (e.g.
API call count, token count, or CPU time).  Each task execution
consumes some budget.  The manager supports checking whether a task
can be executed given an estimated cost, recording actual cost, and
computing the remaining budget.  Budgets reset at midnight UTC each
day.

The implementation is intentionally lightweight: it does not
persist state across process restarts.  In future iterations, the
budget state could be saved to disk.
"""

from __future__ import annotations

import datetime
from typing import Optional


class BudgetManager:
    """Track and enforce a daily execution budget."""

    def __init__(self, daily_budget: float) -> None:
        #: Maximum budget allowed per UTC day
        self.daily_budget: float = float(daily_budget)
        #: Amount spent so far today
        self.spent_today: float = 0.0
        #: Date of last reset (UTC)
        self.last_reset: datetime.date = datetime.datetime.utcnow().date()

    def _reset_if_new_day(self) -> None:
        """Reset the budget counters when a new UTC day begins."""
        current_day = datetime.datetime.utcnow().date()
        if current_day != self.last_reset:
            self.spent_today = 0.0
            self.last_reset = current_day

    def can_execute(self, estimated_cost: float) -> tuple[bool, Optional[str]]:
        """Check if a task can be executed given its estimated cost.

        Args:
            estimated_cost: The estimated cost of executing the task.

        Returns:
            Tuple (can_execute, reason).  If ``can_execute`` is False,
            ``reason`` contains a descriptive message.  If True, reason
            is None.
        """
        self._reset_if_new_day()
        if estimated_cost <= 0:
            return True, None
        if self.spent_today + estimated_cost > self.daily_budget:
            remaining = self.daily_budget - self.spent_today
            return False, f"Daily budget exceeded. Remaining: {remaining:.2f}, required: {estimated_cost:.2f}"
        return True, None

    def record_execution(self, actual_cost: float) -> None:
        """Record the actual cost of a completed task.

        This updates the spent_today counter.  Negative costs are ignored.
        """
        if actual_cost <= 0:
            return
        self._reset_if_new_day()
        self.spent_today += float(actual_cost)

    def get_remaining_budget(self) -> float:
        """Return the remaining budget for the current UTC day."""
        self._reset_if_new_day()
        return max(0.0, self.daily_budget - self.spent_today)