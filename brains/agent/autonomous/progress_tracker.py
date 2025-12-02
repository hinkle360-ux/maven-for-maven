"""
Progress Tracker
================

This module implements a simple progress tracker for long‑running
goals.  It calculates progress based on completed versus total tasks
and stores progress information for reporting.  Milestone
notifications are stubbed out for future implementation.
"""

from __future__ import annotations

import time
from typing import Dict, Any


class ProgressTracker:
    """Track and report progress of goals."""

    def __init__(self) -> None:
        self.progress_store: Dict[str, Dict[str, Any]] = {}

    def update_progress(self, goal_id: str, completed_tasks: int, total_tasks: int) -> None:
        """Update progress metrics for a goal.

        Args:
            goal_id: Identifier of the goal.
            completed_tasks: Number of completed sub‑tasks.
            total_tasks: Total number of sub‑tasks.
        """
        if total_tasks <= 0:
            progress = 1.0
        else:
            progress = completed_tasks / float(total_tasks)
        record = {
            "goal_id": goal_id,
            "progress": progress,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "last_update": time.time(),
        }
        self.progress_store[goal_id] = record
        # Milestone notifications can be implemented here.

    def get_progress(self, goal_id: str) -> Dict[str, Any]:
        """Retrieve progress information for a goal."""
        return self.progress_store.get(goal_id, {
            "goal_id": goal_id,
            "progress": 0.0,
            "completed_tasks": 0,
            "total_tasks": 0,
            "last_update": None,
        })

    def generate_report(self, goal_id: str, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a progress report for a goal.

        Args:
            goal_id: Identifier of the goal.
            goal: The goal record itself (for title and created_at).

        Returns:
            A dictionary summarising the current progress.
        """
        prog = self.get_progress(goal_id)
        title = str(goal.get("title", ""))
        created_at = goal.get("created_at", time.time())
        elapsed = time.time() - created_at
        # Format progress percentage without nesting quotes inside f-string
        pct = prog.get("progress", 0.0) * 100.0
        pct_str = f"{pct:.1f}%"
        return {
            "goal": title,
            "status": "in_progress" if prog.get("progress", 0.0) < 1.0 else "completed",
            "progress_pct": pct_str,
            "completed": prog.get("completed_tasks"),
            "total": prog.get("total_tasks"),
            "elapsed_time": elapsed,
        }