"""
Browser Task Executor
=====================

Manages execution of browser tasks with timeout, logging, and state tracking.
Includes rate limiting per domain and sensitive data redaction.
"""

from __future__ import annotations

import asyncio
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from urllib.parse import urlparse

from optional.maven_browser_client.client import BrowserClient
from optional.maven_browser_client.types import BrowserPlan, BrowserTaskResult, TaskStatus, ActionType
from optional.browser_runtime.config import get_config
from optional.browser_runtime.rate_limiter import get_rate_limiter, DomainRateLimiter
from optional.browser_runtime.redaction import redact_task_log, LogRedactor
from optional.browser_tools.plan_validator import validate_plan, ValidationError


class TaskExecutor:
    """Executes browser tasks with safety constraints and logging."""

    def __init__(self):
        self.config = get_config()
        self.log_dir = self.config.log_dir / "tasks"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = get_rate_limiter()
        self.log_redactor = LogRedactor()
        self._step_results: List[Dict[str, Any]] = []

    def _extract_domains_from_plan(self, plan: BrowserPlan) -> List[str]:
        """Extract all domains that will be accessed by the plan."""
        domains = []
        for step in plan.steps:
            action = step.action.value if hasattr(step.action, 'value') else str(step.action)
            if action == "open":
                url = step.params.get("url", "")
                if url:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        domains.append(parsed.netloc)
        return list(set(domains))

    async def _check_rate_limits(self, plan: BrowserPlan) -> Optional[str]:
        """
        Check rate limits for all domains in plan.

        Returns error message if rate limited, None if OK.
        """
        domains = self._extract_domains_from_plan(plan)

        for domain in domains:
            if not self.rate_limiter.check_rate_limit(domain):
                wait_time = self.rate_limiter.get_wait_time(domain)
                return f"Rate limited for domain {domain}. Wait {wait_time:.1f}s"

        return None

    async def execute(self, plan: BrowserPlan) -> BrowserTaskResult:
        """
        Execute a browser plan with timeout and logging.

        Args:
            plan: BrowserPlan to execute

        Returns:
            BrowserTaskResult with execution outcome
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        self._step_results = []

        # Create task log entry
        task_log = {
            "task_id": task_id,
            "goal": plan.goal,
            "plan": plan.model_dump(),
            "start_time": start_time.isoformat(),
            "status": "running",
            "steps": [],
        }

        try:
            # Check rate limits
            rate_limit_error = await self._check_rate_limits(plan)
            if rate_limit_error:
                task_log["status"] = "failed"
                task_log["error"] = rate_limit_error
                self._save_log(task_id, task_log)

                return BrowserTaskResult(
                    task_id=task_id,
                    goal=plan.goal,
                    status=TaskStatus.FAILED,
                    error_message=rate_limit_error,
                    duration_seconds=0,
                )

            # Validate plan
            try:
                validate_plan(plan)
            except ValidationError as e:
                task_log["status"] = "failed"
                task_log["error"] = str(e)
                self._save_log(task_id, task_log)

                return BrowserTaskResult(
                    task_id=task_id,
                    goal=plan.goal,
                    status=TaskStatus.FAILED,
                    error_message=f"Plan validation failed: {str(e)}",
                    duration_seconds=0,
                )

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_plan(plan, task_log),
                    timeout=plan.timeout_seconds
                )

                task_log["status"] = result.status.value
                task_log["end_time"] = datetime.now(timezone.utc).isoformat()
                task_log["duration_seconds"] = result.duration_seconds
                task_log["steps_executed"] = result.steps_executed

                if result.error_message:
                    task_log["error"] = result.error_message

                self._save_log(task_id, task_log)
                return result

            except asyncio.TimeoutError:
                task_log["status"] = "timeout"
                task_log["end_time"] = datetime.now(timezone.utc).isoformat()
                task_log["error"] = f"Task exceeded timeout of {plan.timeout_seconds} seconds"
                self._save_log(task_id, task_log)

                return BrowserTaskResult(
                    task_id=task_id,
                    goal=plan.goal,
                    status=TaskStatus.TIMEOUT,
                    error_message=f"Task exceeded timeout of {plan.timeout_seconds} seconds",
                    duration_seconds=plan.timeout_seconds,
                )

        except Exception as e:
            task_log["status"] = "failed"
            task_log["end_time"] = datetime.now(timezone.utc).isoformat()
            task_log["error"] = str(e)
            self._save_log(task_id, task_log)

            return BrowserTaskResult(
                task_id=task_id,
                goal=plan.goal,
                status=TaskStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

    async def _execute_plan(self, plan: BrowserPlan, task_log: Dict[str, Any]) -> BrowserTaskResult:
        """
        Execute the browser plan.

        Args:
            plan: BrowserPlan to execute
            task_log: Task log dictionary to update

        Returns:
            BrowserTaskResult
        """
        from brains.agent.tools.browser.browser_tool import execute_browser_plan

        # Execute the plan
        result = await execute_browser_plan(plan)

        return result

    def _save_log(self, task_id: str, task_log: Dict[str, Any]) -> None:
        """
        Save task log to file with sensitive data redaction.

        Args:
            task_id: Task ID
            task_log: Task log dictionary
        """
        log_file = self.log_dir / f"{task_id}.json"

        try:
            # Redact sensitive data before saving
            redacted_log = redact_task_log(task_log)

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(redacted_log, f, indent=2, default=str)
        except Exception as e:
            # Log error but don't fail the task
            print(f"Failed to save task log: {e}")

    def get_step_results(self) -> List[Dict[str, Any]]:
        """Get results from the last executed plan's steps."""
        return self._step_results.copy()

    def get_task_log(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task log by ID.

        Args:
            task_id: Task ID

        Returns:
            Task log dictionary or None if not found
        """
        log_file = self.log_dir / f"{task_id}.json"

        if not log_file.exists():
            return None

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
