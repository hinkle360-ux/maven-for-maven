"""
Execution Engine - High-Level Agent Task Coordinator
====================================================

This module provides the execution engine for autonomous agent tasks.
It orchestrates the full lifecycle of a task:
1. Decomposition - Break goal into executable steps
2. Tool execution - Run steps via tool orchestrator
3. Self-critique - Evaluate result quality
4. Optional retries - Retry on failure if budget allows

Input:
    AgentTask with:
    - goal: str - The high-level goal
    - constraints: dict - Max steps, risk budget, time limit
    - time_budget_ms: int - Maximum execution time
    - risk_budget: int - Maximum risk score

Output:
    - final_answer: The result of the task
    - trace: List of steps and tool calls
    - errors: Any errors encountered
    - success: Whether the task completed successfully
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List

from brains.agent.autonomous.task_decomposer import TaskDecomposer, DecompositionResult, DecomposedStep
from brains.agent.autonomous.tool_orchestrator import ToolOrchestrator, ExecutionContext, OrchestratorResult


@dataclass
class AgentTask:
    """A high-level task for the execution engine."""
    goal: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    time_budget_ms: int = 60000  # 1 minute default
    risk_budget: int = 30
    max_retries: int = 2


@dataclass
class TraceStep:
    """A step in the execution trace."""
    step_id: int
    tool: str
    action: str
    params: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionResult:
    """Result of executing an agent task."""
    task_goal: str
    success: bool
    final_answer: Any = None
    trace: List[TraceStep] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    total_duration_ms: float = 0.0
    steps_executed: int = 0
    retries_used: int = 0
    self_critique_score: float = 0.0
    self_critique_feedback: str = ""


class ExecutionEngine:
    """
    High-level execution engine for autonomous agent tasks.

    This engine orchestrates:
    1. Goal decomposition via TaskDecomposer
    2. Step execution via ToolOrchestrator
    3. Self-critique evaluation of results
    4. Optional retries on failure

    The engine respects time and risk budgets, stopping execution
    if either is exhausted.
    """

    def __init__(self):
        """Initialize the execution engine."""
        self.decomposer = TaskDecomposer()
        self.orchestrator = ToolOrchestrator()
        self._self_critique_available = self._load_self_critique()

    def _load_self_critique(self) -> bool:
        """Load self-critique module if available."""
        try:
            from brains.self_critique_v2 import SelfCritic
            self._critic = SelfCritic()
            return True
        except ImportError:
            self._critic = None
            return False

    def execute_task(self, task: AgentTask) -> ExecutionResult:
        """
        Execute a complete agent task.

        This is the main entry point for the execution engine.

        Args:
            task: AgentTask with goal, constraints, and budgets

        Returns:
            ExecutionResult with outcome, trace, and metrics
        """
        start_time = time.time()
        trace: List[TraceStep] = []
        errors: List[str] = []
        retries_used = 0

        # Step 1: Decompose the goal
        decomposition = self._decompose_goal(task)

        if not decomposition.valid:
            return ExecutionResult(
                task_goal=task.goal,
                success=False,
                errors=decomposition.validation_errors,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

        # Step 2: Execute steps with retry logic
        final_result = None
        last_orchestrator_result = None

        for attempt in range(task.max_retries + 1):
            # Check time budget
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > task.time_budget_ms:
                errors.append(f"Time budget ({task.time_budget_ms}ms) exceeded")
                break

            # Execute all steps
            steps_as_dicts = [s.to_dict() for s in decomposition.steps]
            orchestrator_result = self.orchestrator.execute_steps(steps_as_dicts)
            last_orchestrator_result = orchestrator_result

            # Build trace
            for step_result in orchestrator_result.step_results:
                step_info = next(
                    (s for s in decomposition.steps if s.step_id == step_result.step_id),
                    None
                )
                trace.append(TraceStep(
                    step_id=step_result.step_id,
                    tool=step_info.tool if step_info else "",
                    action=step_info.action if step_info else "",
                    params=step_info.params if step_info else {},
                    success=step_result.success,
                    result=step_result.result,
                    error=step_result.error,
                    duration_ms=step_result.duration_ms,
                ))

            # Check if execution succeeded
            if orchestrator_result.success:
                # Extract final result from context
                final_result = self._extract_final_result(orchestrator_result.context)
                break
            else:
                retries_used += 1
                if orchestrator_result.abort_reason:
                    errors.append(orchestrator_result.abort_reason)

                # Don't retry on the last attempt
                if attempt >= task.max_retries:
                    break

                # Analyze failure for retry strategy
                should_retry, retry_reason = self._should_retry(orchestrator_result)
                if not should_retry:
                    errors.append(f"Not retrying: {retry_reason}")
                    break

        # Step 3: Self-critique the result
        critique_score = 0.0
        critique_feedback = ""

        if final_result and self._self_critique_available:
            critique_score, critique_feedback = self._self_critique(task.goal, final_result)

        total_duration = (time.time() - start_time) * 1000

        success = (
            last_orchestrator_result is not None and
            last_orchestrator_result.success and
            critique_score >= 0.5  # Accept if critique score is reasonable
        )

        return ExecutionResult(
            task_goal=task.goal,
            success=success,
            final_answer=final_result,
            trace=trace,
            errors=errors,
            total_duration_ms=total_duration,
            steps_executed=len(trace),
            retries_used=retries_used,
            self_critique_score=critique_score,
            self_critique_feedback=critique_feedback,
        )

    def _decompose_goal(self, task: AgentTask) -> DecompositionResult:
        """Decompose a task goal into executable steps."""
        goal_dict = {
            "title": task.goal,
            "description": task.goal,
        }

        constraints = task.constraints.copy()
        constraints["risk_budget"] = task.risk_budget

        return self.decomposer.decompose(goal_dict, constraints)

    def _extract_final_result(self, context: ExecutionContext) -> Any:
        """Extract the final result from execution context."""
        # Get the last step's output as the final result
        if context.outputs:
            last_step_id = max(context.outputs.keys())
            return context.outputs.get(last_step_id)

        # Or check for specific named variables
        if context.variables:
            for key in ["result", "output", "answer", "summary"]:
                if key in context.variables:
                    return context.variables[key]

        return None

    def _should_retry(self, result: OrchestratorResult) -> tuple[bool, str]:
        """
        Determine if execution should be retried.

        Args:
            result: The orchestrator result to analyze

        Returns:
            Tuple of (should_retry, reason)
        """
        # Don't retry if all critical steps passed
        failed_critical = sum(
            1 for r in result.step_results
            if not r.success and not r.skipped
        )

        if failed_critical == 0:
            return False, "No critical failures"

        # Don't retry if failure was due to policy
        if result.abort_reason and "policy" in result.abort_reason.lower():
            return False, "Policy violation"

        # Don't retry if too many steps failed
        failure_rate = result.steps_failed / max(1, result.steps_executed)
        if failure_rate > 0.5:
            return False, f"Too many failures ({failure_rate:.0%})"

        return True, "Retrying due to recoverable failures"

    def _self_critique(self, goal: str, result: Any) -> tuple[float, str]:
        """
        Evaluate the result quality using self-critique.

        Args:
            goal: The original goal
            result: The result to evaluate

        Returns:
            Tuple of (score, feedback) where score is 0-1
        """
        if not self._self_critique_available or not self._critic:
            return 1.0, "Self-critique not available"

        try:
            # Build a simple evaluation
            result_str = str(result)[:500] if result else "No result"

            # Basic heuristics for self-critique
            score = 0.5  # Start neutral

            # Check if result is not empty
            if result:
                score += 0.2

            # Check if result is not an error
            if result and not (isinstance(result, dict) and result.get("error")):
                score += 0.2

            # Check if result contains relevant content
            if result_str and len(result_str) > 10:
                score += 0.1

            feedback = f"Result evaluation: {score:.2f}"

            return min(1.0, score), feedback

        except Exception as e:
            return 0.5, f"Self-critique failed: {e}"

    def decompose_goal(self, goal: str) -> List[str]:
        """
        Legacy interface: decompose a goal string into subtasks.

        Args:
            goal: The goal string

        Returns:
            List of subtask descriptions
        """
        result = self.decomposer.decompose({"title": goal, "description": goal})
        return [s.description for s in result.steps]

    def plan_execution(self, goal: str) -> Dict[str, Any]:
        """
        Legacy interface: plan execution for a goal.

        Args:
            goal: The goal string

        Returns:
            Execution plan dictionary
        """
        result = self.decomposer.decompose({"title": goal, "description": goal})

        return {
            "goal": goal,
            "tasks": [
                {"id": s.step_id, "description": s.description, "status": "PENDING"}
                for s in result.steps
            ],
            "estimated_time_minutes": result.estimated_duration_ms / 60000,
            "risk_level": result.max_risk_level,
        }

    def execute_goal(
        self,
        goal: str,
        budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Legacy interface: execute a goal with optional budget.

        Args:
            goal: The goal string
            budget: Optional maximum number of tasks

        Returns:
            Execution result dictionary
        """
        task = AgentTask(
            goal=goal,
            constraints={"max_steps": budget} if budget else {},
            time_budget_ms=120000,  # 2 minutes
        )

        result = self.execute_task(task)

        # Convert to legacy format
        return {
            "goal": goal,
            "tasks": [
                {
                    "id": t.step_id,
                    "description": f"{t.tool}:{t.action}",
                    "status": "COMPLETED" if t.success else ("SKIPPED" if t.error else "FAILED"),
                }
                for t in result.trace
            ],
            "completed": result.success,
            "reason": result.errors[0] if result.errors else None,
            "final_answer": result.final_answer,
        }


def handle(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Service API entry point for the execution engine.

    Supported operations:
    - EXECUTE: Execute a task
    - PLAN: Plan execution without running
    - DECOMPOSE: Decompose a goal into steps
    """
    op = (msg or {}).get("op", "").upper()
    payload = msg.get("payload", {}) or {}
    mid = msg.get("mid")

    engine = ExecutionEngine()

    if op == "EXECUTE":
        goal = payload.get("goal", "")
        constraints = payload.get("constraints", {})
        time_budget = payload.get("time_budget_ms", 60000)
        risk_budget = payload.get("risk_budget", 30)

        task = AgentTask(
            goal=goal,
            constraints=constraints,
            time_budget_ms=time_budget,
            risk_budget=risk_budget,
        )

        result = engine.execute_task(task)

        return {
            "ok": result.success,
            "op": op,
            "mid": mid,
            "payload": {
                "success": result.success,
                "final_answer": result.final_answer,
                "steps_executed": result.steps_executed,
                "errors": result.errors,
                "duration_ms": result.total_duration_ms,
                "self_critique_score": result.self_critique_score,
            }
        }

    if op == "PLAN":
        goal = payload.get("goal", "")
        plan = engine.plan_execution(goal)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": plan
        }

    if op == "DECOMPOSE":
        goal = payload.get("goal", "")
        subtasks = engine.decompose_goal(goal)

        return {
            "ok": True,
            "op": op,
            "mid": mid,
            "payload": {"subtasks": subtasks}
        }

    return {
        "ok": False,
        "op": op,
        "mid": mid,
        "error": {"code": "UNSUPPORTED_OP", "message": f"Unsupported operation: {op}"}
    }


# Standard service contract
service_api = handle
