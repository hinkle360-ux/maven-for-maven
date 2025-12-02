"""
Agent Daemon
============

This module provides the ``AgentDaemon`` class which runs Maven in
autonomous agent mode.  The daemon continuously monitors the goal
queue and executes eligible goals in the background.  It is designed
to operate across multiple sessions, handling failures and providing
graceful start/stop semantics.

The implementation is intentionally lightweight: it delegates goal
scheduling logic to ``GoalQueue`` and execution persistence to
``ExecutionState``.  It should be started from a control interface
such as an agent CLI or a long‑running process.

Example usage::

    from brains.agent.autonomous import AgentDaemon
    agent = AgentDaemon()
    agent.start()
    # ... later ...
    agent.stop()

At present, the execution loop simply logs that a goal would be
executed.  Integration with the cognitive brains and tool
orchestration will occur in future phases.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, List

from .goal_queue import GoalQueue
from .execution_state import ExecutionState
from .task_decomposer import TaskDecomposer
from .tool_orchestrator import ToolOrchestrator
from .retry_strategy import RetryStrategy
from .strategy_adapter import StrategyAdapter
from .failure_tracker import FailureTracker
from .progress_tracker import ProgressTracker
from .activity_log import ActivityLog
from .resource_monitor import ResourceMonitor
from .budget_manager import BudgetManager


class AgentDaemon:
    """Run Maven autonomously in the background.

    The daemon manages a dedicated thread that periodically checks the
    goal queue for executable tasks.  When it finds a task ready for
    execution, it delegates to future execution logic (to be
    implemented in later phases).  It also maintains an instance of
    ``ExecutionState`` so that progress can be resumed after a crash
    or shutdown.
    """

    def __init__(self, sleep_interval: float = 5.0) -> None:
        #: Whether the execution loop should be running
        self.running: bool = False
        #: Thread object used to run the execution loop
        self._thread: Optional[threading.Thread] = None
        #: Seconds to sleep between polling cycles when there is no work
        self.sleep_interval: float = float(sleep_interval)
        #: Goal queue abstraction for retrieving executable goals
        self.goal_queue = GoalQueue()
        #: Execution state persistence
        self.state = ExecutionState()

        # Initialise supporting components for autonomous execution
        self.decomposer = TaskDecomposer()
        self.orchestrator = ToolOrchestrator()
        self.retry_strategy = RetryStrategy()
        self.strategy_adapter = StrategyAdapter()
        self.failure_tracker = FailureTracker()
        self.progress_tracker = ProgressTracker()
        self.activity_log = ActivityLog()
        self.resource_monitor = ResourceMonitor()
        # Set a generous default daily budget (arbitrary units)
        self.budget_manager = BudgetManager(daily_budget=1000.0)

    def start(self) -> None:
        """Start the autonomous execution loop in a background thread.

        If the daemon is already running, this call does nothing.
        """
        if self.running:
            return
        self.running = True
        # Use daemon thread so that it exits when the main program
        # terminates unexpectedly
        self._thread = threading.Thread(
            target=self._execution_loop, name="maven_agent_daemon", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the execution loop and wait for the thread to terminate.

        Args:
            timeout: Maximum time to wait for the thread to join (in
                seconds).  If the thread does not finish within this
                period, it will be left running in the background but
                ``self.running`` will be set to False.
        """
        if not self.running:
            return
        self.running = False
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    #
    def _execution_loop(self) -> None:
        """Continuously poll the goal queue and dispatch work.

        This method runs in a separate thread.  It retrieves executable
        goals from the ``GoalQueue`` and would call into an executor
        engine to perform work.  Since the executor is not yet
        implemented, it currently marks the goal as completed and
        records checkpoints for demonstration purposes.  The loop
        sleeps for ``sleep_interval`` seconds when there are no
        executable goals or after handling a goal.
        """
        while self.running:
            # Respect rate limits and system resources
            try:
                if self.resource_monitor.throttle_if_needed():
                    continue
                ready_goals: List[dict] = self.goal_queue.get_executable_goals()
                if ready_goals:
                    # Select the next goal to execute.  A future scheduler could
                    # incorporate priority or deadlines.  Here we pick the first.
                    goal = self.goal_queue.select_next_goal(ready_goals)
                    gid = goal.get("goal_id")
                    if gid:
                        # Check if we can execute tasks within budget
                        # Estimate total cost as number of sub‑tasks (each costs 1)
                        tasks = self.decomposer.decompose(goal)
                        est_cost = float(len(tasks))
                        can_run, reason = self.budget_manager.can_execute(est_cost)
                        if not can_run:
                            # Cannot execute due to budget; postpone
                            # Record checkpoint and skip this cycle
                            self.state.save_checkpoint(gid, {"status": "paused", "reason": reason})
                            # Sleep briefly before next poll
                            time.sleep(self.sleep_interval)
                            continue
                        # Persist starting checkpoint
                        self.state.save_checkpoint(gid, {"status": "started"})
                        self.activity_log.log_action(
                            "goal_start",
                            {"goal_id": gid, "title": goal.get("title", ""), "task_count": len(tasks)},
                            {"goal": goal}
                        )
                        # Execute tasks sequentially
                        completed = 0
                        total_tasks = len(tasks)
                        for idx, task in enumerate(tasks):
                            # Check resources before each task
                            if self.resource_monitor.throttle_if_needed():
                                # If throttled, wait then recheck
                                continue
                            # Check budget per task (cost 1)
                            can_task, reason_task = self.budget_manager.can_execute(1.0)
                            if not can_task:
                                # Out of budget mid‑goal: record progress and break
                                self.state.save_checkpoint(gid, {
                                    "status": "paused_budget",
                                    "completed_tasks": completed,
                                    "total_tasks": total_tasks,
                                    "reason": reason_task,
                                })
                                break
                            # Execute with retry
                            result = self.retry_strategy.execute_with_retry(
                                task, self.orchestrator.execute_task
                            )
                            # Each execution counts as an API call
                            self.resource_monitor.record_api_call()
                            # Record cost in budget
                            self.budget_manager.record_execution(1.0)
                            if result.get("success"):
                                completed += 1
                                self.activity_log.log_action(
                                    "task_completed",
                                    {
                                        "goal_id": gid,
                                        "task_index": idx,
                                        "task": task,
                                        "result": result.get("result"),
                                    },
                                    {"goal": goal}
                                )
                            else:
                                error = result.get("error", "Unknown error")
                                # Record failure
                                self.failure_tracker.record_failure(gid, task, error)
                                # Attempt alternative strategies
                                alternatives = self.strategy_adapter.generate_alternative(task, error)
                                alt_success = False
                                for alt in alternatives:
                                    alt_result = self.retry_strategy.execute_with_retry(
                                        alt, self.orchestrator.execute_task
                                    )
                                    self.resource_monitor.record_api_call()
                                    self.budget_manager.record_execution(1.0)
                                    if alt_result.get("success"):
                                        completed += 1
                                        alt_success = True
                                        self.activity_log.log_action(
                                            "task_completed_alt",
                                            {
                                                "goal_id": gid,
                                                "task_index": idx,
                                                "task": alt,
                                                "result": alt_result.get("result"),
                                            },
                                            {"goal": goal}
                                        )
                                        break
                                    else:
                                        # Record alt failure
                                        alt_error = alt_result.get("error", "Unknown error")
                                        self.failure_tracker.record_failure(gid, alt, alt_error)
                                if not alt_success:
                                    # Could not complete this task; continue to next
                                    self.activity_log.log_action(
                                        "task_failed",
                                        {
                                            "goal_id": gid,
                                            "task_index": idx,
                                            "task": task,
                                            "error": error,
                                        },
                                        {"goal": goal}
                                    )
                            # Update progress after each task
                            self.progress_tracker.update_progress(gid, completed, total_tasks)
                        # Determine completion
                        if completed >= total_tasks:
                            # Mark goal completed successfully
                            self.goal_queue.complete_goal(gid)
                            self.progress_tracker.update_progress(gid, total_tasks, total_tasks)
                            self.state.save_checkpoint(gid, {"status": "completed"})
                            self.activity_log.log_action(
                                "goal_completed",
                                {
                                    "goal_id": gid,
                                    "title": goal.get("title", ""),
                                    "completed_tasks": completed,
                                    "total_tasks": total_tasks,
                                },
                                {"goal": goal}
                            )
                        # If goal not fully completed, leave it active; progress is stored
                    else:
                        # If no goal_id, skip this goal
                        pass
                    # After handling a goal, sleep a short interval to yield
                    time.sleep(0.1)
                else:
                    # No ready goals; sleep until next poll
                    time.sleep(self.sleep_interval)
            except Exception as e:
                # On any unexpected error, log and sleep to avoid tight loop
                try:
                    self.activity_log.log_action(
                        "daemon_error",
                        {"error": str(e)},
                        None,
                    )
                except Exception:
                    pass
                time.sleep(self.sleep_interval)