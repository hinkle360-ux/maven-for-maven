"""
teacher_budget.py
~~~~~~~~~~~~~~~~~

Budget tracking and enforcement for Teacher LLM usage.

This module provides governance over Teacher usage across all brains:
- Tracks Teacher calls per brain
- Enforces per-brain and global limits
- Prevents budget overruns
- Provides usage statistics

IMPORTANT: This is a P0 governance requirement. All Teacher calls
must go through budget checking to prevent excessive LLM usage.

Usage:
    from brains.governance.teacher_budget import can_call_teacher, record_teacher_call

    # Before calling Teacher
    if can_call_teacher("planner"):
        # ... call Teacher ...
        record_teacher_call("planner", success=True)
    else:
        # Budget exceeded, skip Teacher call
        pass
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

from brains.maven_paths import (
    get_runtime_memory_root,
    validate_path_confinement,
)


class TeacherBudget:
    """
    Budget tracker for Teacher LLM usage.

    This class tracks and enforces limits on Teacher calls:
    - Per-brain limits (prevent single brain from dominating)
    - Global limits (prevent total overuse)
    - Session-based tracking (resets per session)
    """

    def __init__(
        self,
        max_calls_per_brain: int = 10,
        max_total_calls: int = 50,
        session_id: Optional[str] = None
    ):
        """
        Initialize budget tracker.

        Args:
            max_calls_per_brain: Maximum calls per brain per session
            max_total_calls: Maximum total calls per session
            session_id: Optional session identifier
        """
        self.max_calls_per_brain = max_calls_per_brain
        self.max_total_calls = max_total_calls
        self.session_id = session_id or self._generate_session_id()

        # Tracking dictionaries
        self.calls_per_brain: Dict[str, int] = {}
        self.successes_per_brain: Dict[str, int] = {}
        self.failures_per_brain: Dict[str, int] = {}

        # Session start time
        self.session_start = datetime.now()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a session ID based on timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def can_call(self, brain_id: str) -> bool:
        """
        Check if a brain is within budget to call Teacher.

        Args:
            brain_id: Name of the brain

        Returns:
            True if within budget, False if budget exceeded
        """
        # Check per-brain limit
        brain_calls = self.calls_per_brain.get(brain_id, 0)
        if brain_calls >= self.max_calls_per_brain:
            print(f"[BUDGET_EXCEEDED] Brain '{brain_id}' has reached limit ({self.max_calls_per_brain} calls)")
            return False

        # Check global limit
        total_calls = sum(self.calls_per_brain.values())
        if total_calls >= self.max_total_calls:
            print(f"[BUDGET_EXCEEDED] Global limit reached ({self.max_total_calls} calls)")
            return False

        return True

    def record_call(self, brain_id: str, success: bool = True):
        """
        Record a Teacher call.

        Args:
            brain_id: Name of the brain that made the call
            success: Whether the call was successful
        """
        # Increment call counter
        self.calls_per_brain[brain_id] = self.calls_per_brain.get(brain_id, 0) + 1

        # Track success/failure
        if success:
            self.successes_per_brain[brain_id] = self.successes_per_brain.get(brain_id, 0) + 1
        else:
            self.failures_per_brain[brain_id] = self.failures_per_brain.get(brain_id, 0) + 1

        # Log
        total = sum(self.calls_per_brain.values())
        print(f"[BUDGET_TRACKING] {brain_id}: {self.calls_per_brain[brain_id]}/{self.max_calls_per_brain} (global: {total}/{self.max_total_calls})")

    def get_usage(self, brain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics.

        Args:
            brain_id: Optional brain to get stats for (None = all)

        Returns:
            Dict with usage statistics
        """
        if brain_id:
            # Stats for specific brain
            return {
                "brain_id": brain_id,
                "calls": self.calls_per_brain.get(brain_id, 0),
                "successes": self.successes_per_brain.get(brain_id, 0),
                "failures": self.failures_per_brain.get(brain_id, 0),
                "remaining": self.max_calls_per_brain - self.calls_per_brain.get(brain_id, 0),
                "limit": self.max_calls_per_brain
            }
        else:
            # Global stats
            total_calls = sum(self.calls_per_brain.values())
            total_successes = sum(self.successes_per_brain.values())
            total_failures = sum(self.failures_per_brain.values())

            return {
                "session_id": self.session_id,
                "total_calls": total_calls,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "remaining": self.max_total_calls - total_calls,
                "limit": self.max_total_calls,
                "per_brain": {
                    brain_id: {
                        "calls": self.calls_per_brain[brain_id],
                        "successes": self.successes_per_brain.get(brain_id, 0),
                        "failures": self.failures_per_brain.get(brain_id, 0)
                    }
                    for brain_id in self.calls_per_brain.keys()
                },
                "session_start": self.session_start.isoformat()
            }

    def reset(self):
        """Reset all counters (start new session)."""
        self.calls_per_brain.clear()
        self.successes_per_brain.clear()
        self.failures_per_brain.clear()
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        print(f"[BUDGET_RESET] New session: {self.session_id}")

    def save_stats(self, filepath: Optional[Path] = None):
        """
        Save usage statistics to file.

        Args:
            filepath: Optional path to save to (default: brains/runtime_memory/teacher_budget.json)
        """
        if not filepath:
            filepath = get_runtime_memory_root() / "teacher_budget.json"

        filepath = validate_path_confinement(filepath, "teacher budget save")

        try:
            # Get full stats
            stats = self.get_usage()

            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save to JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

            print(f"[BUDGET_SAVED] Stats saved to {filepath}")

        except Exception as e:
            print(f"[BUDGET_SAVE_ERROR] {str(e)[:100]}")

    def load_stats(self, filepath: Optional[Path] = None):
        """
        Load usage statistics from file.

        Args:
            filepath: Optional path to load from (default: brains/runtime_memory/teacher_budget.json)
        """
        if not filepath:
            filepath = get_runtime_memory_root() / "teacher_budget.json"

        filepath = validate_path_confinement(filepath, "teacher budget load")

        try:
            if not filepath.exists():
                print(f"[BUDGET_LOAD] No saved stats found at {filepath}")
                return

            # Load from JSON
            with open(filepath, "r", encoding="utf-8") as f:
                stats = json.load(f)

            # Restore state
            self.session_id = stats.get("session_id", self.session_id)
            per_brain = stats.get("per_brain", {})

            for brain_id, brain_stats in per_brain.items():
                self.calls_per_brain[brain_id] = brain_stats.get("calls", 0)
                self.successes_per_brain[brain_id] = brain_stats.get("successes", 0)
                self.failures_per_brain[brain_id] = brain_stats.get("failures", 0)

            # Parse session start
            session_start_str = stats.get("session_start")
            if session_start_str:
                try:
                    self.session_start = datetime.fromisoformat(session_start_str)
                except Exception:
                    pass

            print(f"[BUDGET_LOADED] Stats loaded from {filepath}")

        except Exception as e:
            print(f"[BUDGET_LOAD_ERROR] {str(e)[:100]}")


# Global singleton budget tracker
_global_budget: Optional[TeacherBudget] = None


def get_budget() -> TeacherBudget:
    """
    Get the global budget tracker.

    Returns:
        Global TeacherBudget instance
    """
    global _global_budget
    if _global_budget is None:
        _global_budget = TeacherBudget(
            max_calls_per_brain=10,  # 10 calls per brain per session
            max_total_calls=50       # 50 total calls per session
        )
    return _global_budget


def set_budget(budget: TeacherBudget):
    """
    Set a custom global budget tracker.

    Args:
        budget: TeacherBudget instance to use
    """
    global _global_budget
    _global_budget = budget


# Convenience functions for common operations

def can_call_teacher(brain_id: str) -> bool:
    """
    Check if a brain can call Teacher within budget.

    Args:
        brain_id: Name of the brain

    Returns:
        True if within budget, False otherwise
    """
    return get_budget().can_call(brain_id)


def record_teacher_call(brain_id: str, success: bool = True):
    """
    Record a Teacher call for budget tracking.

    Args:
        brain_id: Name of the brain that made the call
        success: Whether the call was successful
    """
    get_budget().record_call(brain_id, success=success)


def get_teacher_usage(brain_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Teacher usage statistics.

    Args:
        brain_id: Optional brain ID (None = all brains)

    Returns:
        Usage statistics dict
    """
    return get_budget().get_usage(brain_id)


def reset_budget():
    """Reset the budget tracker (start new session)."""
    get_budget().reset()


def save_budget_stats(filepath: Optional[Path] = None):
    """
    Save budget statistics to file.

    Args:
        filepath: Optional path to save to
    """
    get_budget().save_stats(filepath)


def load_budget_stats(filepath: Optional[Path] = None):
    """
    Load budget statistics from file.

    Args:
        filepath: Optional path to load from
    """
    get_budget().load_stats(filepath)


# Budget enforcement decorator
def enforce_teacher_budget(brain_id: str):
    """
    Decorator to enforce Teacher budget on a function.

    Usage:
        @enforce_teacher_budget("planner")
        def call_teacher_for_planning(question):
            # ... Teacher call ...
            pass

    Args:
        brain_id: Name of the brain making the call
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check budget before call
            if not can_call_teacher(brain_id):
                print(f"[BUDGET_BLOCKED] Teacher call blocked for {brain_id}")
                return None

            # Call function
            try:
                result = func(*args, **kwargs)
                record_teacher_call(brain_id, success=True)
                return result
            except Exception as e:
                record_teacher_call(brain_id, success=False)
                raise e

        return wrapper
    return decorator


# Export public API
__all__ = [
    "TeacherBudget",
    "get_budget",
    "set_budget",
    "can_call_teacher",
    "record_teacher_call",
    "get_teacher_usage",
    "reset_budget",
    "save_budget_stats",
    "load_budget_stats",
    "enforce_teacher_budget",
]
