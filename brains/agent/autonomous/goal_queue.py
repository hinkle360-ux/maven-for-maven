"""
Goal Queue Manager
==================

This module implements a simple goal queue for the autonomous agent.
It interfaces with the existing personal goal memory to fetch tasks,
determine their eligibility for execution, and mark them as completed.
This abstraction separates scheduling from execution logic and can be
extended with priority ordering, dependency resolution and progress
tracking in future phases.
"""

from __future__ import annotations

from typing import List, Dict, Any

try:
    # Import goal memory using fully qualified package path.  When the
    # agent modules are loaded as part of the ``maven_update.maven``
    # package hierarchy, the goal memory resides under
    # ``maven_update.maven.brains.personal.memory``.
    from maven_update.maven.brains.personal.memory import goal_memory  # type: ignore
except Exception:
    try:
        # Secondary import: in some contexts the package may be
        # registered under just ``maven`` (e.g. if sys.path inserts the
        # ``maven_update/maven`` directory as a top‑level entry).  In
        # that case, import from ``maven.brains.personal.memory``.
        from maven.brains.personal.memory import goal_memory  # type: ignore
    except Exception:
        # Final fallback stub if goal_memory cannot be imported
        goal_memory = None  # type: ignore


class GoalQueue:
    """Manage retrieval and scheduling of goals for the agent."""

    def load_goals(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Load goals from the personal goal memory.

        Args:
            active_only: When True, return only incomplete goals.

        Returns:
            A list of goal dictionaries.  If the goal memory is not
            available, returns an empty list.
        """
        if goal_memory is None:
            return []
        try:
            return goal_memory.get_goals(active_only=active_only)  # type: ignore[arg-type]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Goal management helpers
    def add_goal(self, title: str, description: str | None = None, *, depends_on: List[str] | None = None,
                 condition: str | None = None, parent_id: str | None = None,
                 deadline_ts: float | None = None, priority: float | None = None) -> Dict[str, Any]:
        """Add a new goal to the personal goal memory.

        This is a thin wrapper around ``goal_memory.add_goal``.  It returns
        the created record so that callers can inspect its ``goal_id``.

        Args:
            title: Short description of the goal.
            description: Optional longer description.
            depends_on: Optional list of goal IDs that must complete before
                this one can run.
            condition: Optional condition flag ("success" or "failure").
            parent_id: Optional parent goal identifier.
            deadline_ts: Optional absolute deadline (epoch seconds).
            priority: Optional priority weight (0..1).  This value is not
                persisted by the goal memory but may be stored in the
                ``metrics`` field for later use by the scheduler.

        Returns:
            The goal record created or, if a duplicate exists, the existing
            record as returned by ``goal_memory.add_goal``.
        """
        if goal_memory is None:
            return {}
        try:
            metrics = {"priority": float(priority) if priority is not None else None}
            rec = goal_memory.add_goal(
                title,
                description,
                depends_on=depends_on,
                condition=condition,
                parent_id=parent_id,
                deadline_ts=deadline_ts,
                metrics=metrics,
            )  # type: ignore[arg-type]
            # Validate that dependency IDs exist.  If any listed dependencies are
            # not present in the goal memory, record them for diagnostic
            # purposes.  Invalid dependencies do not block creation but are
            # surfaced in the returned record.
            try:
                invalid: List[str] = []
                if depends_on:
                    # Fetch current goals once to avoid repeated calls
                    existing = {g.get("goal_id") for g in goal_memory.get_goals(active_only=False)}  # type: ignore[attr-defined]
                    for dep in depends_on:
                        if dep not in existing:
                            invalid.append(dep)
                if invalid:
                    rec["invalid_dependencies"] = invalid
            except Exception:
                # Ignore validation errors silently
                pass
            return rec
        except Exception:
            return {}

    def list_goals(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Return the list of goals stored in memory.

        Args:
            active_only: When True, only return goals that are not completed.

        Returns:
            A list of goal dictionaries sorted by creation time (oldest first).
        """
        goals = self.load_goals(active_only=active_only)
        try:
            return sorted(goals, key=lambda g: g.get("created_at", 0.0))
        except Exception:
            return goals

    def get_executable_goals(self) -> List[Dict[str, Any]]:
        """Return a list of goals that are ready to execute now.

        A goal is executable if it is not completed and all of its
        dependencies have been completed.  Goals with unmet conditions
        are skipped.  Currently, conditions are not evaluated and are
        assumed satisfied.

        Returns:
            List of goal dictionaries that are ready for execution.
        """
        all_goals = self.load_goals(active_only=True)
        executable: List[Dict[str, Any]] = []
        for goal in all_goals:
            if self._is_executable(goal, all_goals):
                executable.append(goal)
        return executable

    def _is_executable(self, goal: Dict[str, Any], all_goals: List[Dict[str, Any]]) -> bool:
        """Determine whether a goal can be executed now.

        Args:
            goal: Goal record to evaluate.
            all_goals: The full list of goals for dependency lookup.

        Returns:
            True if the goal is ready for execution; False otherwise.
        """
        # Skip goals that are already completed or have been flagged with
        # errors (e.g. cyclic dependencies).  The goal_memory module
        # annotates records with 'cycle_error' and 'error_message' when
        # detection occurs.  Such goals should never be executed.
        if goal.get("completed"):
            return False
        if goal.get("cycle_error") or goal.get("error_message"):
            return False
        deps = goal.get("depends_on") or []
        if deps:
            # A goal is not executable until all dependencies are completed.
            for dep_id in deps:
                dep_goal = next((g for g in all_goals if g.get("goal_id") == dep_id), None)
                # If dependency is missing or incomplete, block execution
                if not dep_goal or not dep_goal.get("completed"):
                    return False
                # Prevent simple cycles where a goal depends on itself via direct chain
                if dep_id == goal.get("goal_id"):
                    return False
        # Conditions are not currently evaluated; assume true
        return True

    def select_next_goal(self, executable: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the next goal to execute based on priority and age.

        This implementation orders executable goals by their associated
        priority (descending) and creation timestamp (ascending).  The
        priority is read from the goal's ``metrics['priority']`` field.  A
        missing or invalid priority is treated as zero.  When multiple
        goals share the same priority, the oldest goal is chosen first.

        Args:
            executable: List of executable goals.

        Returns:
            The goal record chosen for execution, or an empty dict if
            none are available.
        """
        if not executable:
            return {}
        try:
            # Sort by priority (high to low) then by creation time (low to high)
            sorted_goals = sorted(
                executable,
                key=lambda g: (
                    -float((g.get("metrics", {}) or {}).get("priority") or 0.0),
                    float(g.get("created_at", 0.0) or 0.0),
                ),
            )
            return sorted_goals[0] if sorted_goals else {}
        except Exception:
            # Fallback to the first goal if sorting fails
            return executable[0] if executable else {}

    def complete_goal(self, goal_id: str) -> None:
        """Mark a goal as completed.

        Args:
            goal_id: Identifier of the goal to complete.

        If the goal memory is unavailable, this method does nothing.
        """
        if goal_memory is None:
            return
        try:
            goal_memory.complete_goal(goal_id)  # type: ignore[attr-defined]
        except Exception:
            pass