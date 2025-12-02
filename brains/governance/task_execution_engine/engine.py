"""
Task Execution Engine (TEE)
Phase 8 - Deterministic multi-step task execution with full reasoning traces.

This module orchestrates:
1. Task decomposition (via planner_brain)
2. Step-by-step routing to specialist brains
3. Pattern-based execution
4. Deterministic aggregation
5. Full execution trace generation
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from api.utils import generate_mid, success_response, error_response


class StepCounter:
    """Deterministic step counter that resets per execution."""

    def __init__(self):
        self._counter = 0

    def next(self) -> int:
        """Get next step ID."""
        self._counter += 1
        return self._counter

    def reset(self) -> None:
        """Reset counter to 0."""
        self._counter = 0

    def current(self) -> int:
        """Get current counter value."""
        return self._counter


class TaskExecutionEngine:
    """
    Main Task Execution Engine.

    Responsibilities:
    - Decompose tasks into deterministic steps
    - Route steps to appropriate specialist brains
    - Execute steps with pattern application
    - Aggregate results deterministically
    - Build full reasoning traces
    - Handle errors and rollback
    """

    def __init__(self):
        self.step_counter = StepCounter()
        self._trace_entries: List[Dict[str, Any]] = []

    def execute_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        with_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a task with deterministic step-by-step processing.

        Args:
            task_description: The task to execute
            context: Optional context for task execution
            with_trace: Whether to include full reasoning trace

        Returns:
            Dict with:
                - output: Final aggregated result
                - trace: Full reasoning trace (if with_trace=True)
                - steps_executed: Number of steps executed
                - success: Whether execution succeeded
        """
        # Reset state for new execution
        self.step_counter.reset()
        self._trace_entries = []

        try:
            # Step 1: Decompose task into steps
            decomposition_result = self._decompose_task(task_description, context)

            if not decomposition_result.get("success"):
                return self._build_error_result(
                    "DECOMPOSITION_FAILED",
                    decomposition_result.get("error", "Unknown decomposition error")
                )

            steps = decomposition_result.get("steps", [])

            if not steps:
                return self._build_error_result(
                    "NO_STEPS_GENERATED",
                    "Task decomposition produced no steps"
                )

            # Add decomposition to trace
            self._add_trace_entry({
                "step_id": 0,
                "step_type": "decomposition",
                "description": "Task decomposition",
                "input": task_description,
                "output": steps,
                "patterns_used": decomposition_result.get("patterns_used", []),
                "success": True
            })

            # Step 2: Execute each step sequentially
            step_results = []

            for step in steps:
                step_id = self.step_counter.next()

                # Route step to appropriate brain
                execution_result = self._execute_step(step, step_id, context)

                if not execution_result.get("success"):
                    # Rollback on error
                    return self._rollback(
                        step_id,
                        execution_result.get("error", "Step execution failed"),
                        step_results,
                        with_trace
                    )

                step_results.append(execution_result)

                # Add to trace
                self._add_trace_entry({
                    "step_id": step_id,
                    "step_type": step.get("type", "unknown"),
                    "description": step.get("description", ""),
                    "input": step.get("input"),
                    "output": execution_result.get("output"),
                    "brain": execution_result.get("brain"),
                    "patterns_used": execution_result.get("patterns_used", []),
                    "success": True
                })

            # Step 3: Aggregate results
            aggregation_result = self._aggregate_results(step_results)

            # Add aggregation to trace
            self._add_trace_entry({
                "step_id": self.step_counter.next(),
                "step_type": "aggregation",
                "description": "Result aggregation",
                "input": step_results,
                "output": aggregation_result.get("output"),
                "success": True
            })

            # Build final result
            result = {
                "output": aggregation_result.get("output"),
                "steps_executed": len(steps),
                "success": True
            }

            if with_trace:
                result["trace"] = self._build_trace()

            return result

        except Exception as e:
            return self._build_error_result(
                "EXECUTION_ERROR",
                f"Unexpected error during execution: {str(e)}"
            )

    def _decompose_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decompose task into deterministic steps using planner_brain.

        Returns:
            Dict with:
                - success: bool
                - steps: List[Dict] - List of step dictionaries
                - patterns_used: List[str] - Planning patterns used
                - error: str (if success=False)
        """
        from brains.cognitive.planner.service.planner_brain import service_api as planner_api

        try:
            # Call planner to decompose task
            plan_msg = {
                "op": "DECOMPOSE_TASK",
                "mid": generate_mid(),
                "payload": {
                    "task": task_description,
                    "context": context or {}
                }
            }

            plan_response = planner_api(plan_msg)

            # Normalize response (planner uses "ok" format)
            if not plan_response.get("ok"):
                error_info = plan_response.get("error", {})
                error_msg = error_info.get("message", "Planning failed") if isinstance(error_info, dict) else str(error_info)
                return {
                    "success": False,
                    "error": error_msg
                }

            result = plan_response.get("payload", {})

            return {
                "success": True,
                "steps": result.get("steps", []),
                "patterns_used": result.get("patterns_used", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Exception during task decomposition: {str(e)}"
            }

    def _execute_step(
        self,
        step: Dict[str, Any],
        step_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a single step by routing to appropriate brain.

        Args:
            step: Step dictionary with type, description, tags, etc.
            step_id: Deterministic step ID
            context: Execution context

        Returns:
            Dict with:
                - success: bool
                - output: Any - Step result
                - brain: str - Brain that executed step
                - patterns_used: List[str]
                - error: str (if success=False)
        """
        from brains.governance.task_execution_engine.step_router import route_step

        try:
            # Import router and route step
            brain_name = route_step(step)

            # Execute step with routed brain
            step_msg = {
                "op": "EXECUTE_STEP",
                "mid": generate_mid(),
                "payload": {
                    "step": step,
                    "step_id": step_id,
                    "context": context or {}
                }
            }

            # Import and call appropriate brain
            brain_result = self._call_brain(brain_name, step_msg)

            # Normalize response format (handle both {"ok": ...} and {"status": ...})
            is_success, result_data, error_msg = self._normalize_brain_response(brain_result)

            if not is_success:
                return {
                    "success": False,
                    "error": error_msg,
                    "brain": brain_name
                }

            result = result_data

            return {
                "success": True,
                "output": result.get("output"),
                "brain": brain_name,
                "patterns_used": result.get("patterns_used", [])
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Exception during step execution: {str(e)}"
            }

    def _call_brain(self, brain_name: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specialist brain's service API."""

        brain_map = {
            "planner": "brains.cognitive.planner.service.planner_brain",
            "coder": "brains.cognitive.coder.service.coder_brain",
            "imaginer": "brains.cognitive.imaginer.service.imaginer_brain",
            "committee": "brains.cognitive.committee.service.committee_brain",
            "language": "brains.cognitive.language.service.language_brain",
            "reasoning": "brains.cognitive.reasoning.service.reasoning_brain"
        }

        module_path = brain_map.get(brain_name)

        if not module_path:
            return error_response(
                msg.get("op", ""),
                msg.get("mid", ""),
                "UNKNOWN_BRAIN",
                f"Brain '{brain_name}' not found"
            )

        try:
            # Dynamic import
            module_parts = module_path.split('.')
            module = __import__(module_path, fromlist=[module_parts[-1]])
            brain_api = getattr(module, 'service_api')

            return brain_api(msg)

        except Exception as e:
            return error_response(
                msg.get("op", ""),
                msg.get("mid", ""),
                "BRAIN_CALL_ERROR",
                f"Error calling {brain_name}: {str(e)}"
            )

    def _normalize_brain_response(
        self,
        response: Dict[str, Any]
    ) -> tuple[bool, Dict[str, Any], str]:
        """
        Normalize brain response to standard format.

        Handles two formats:
        1. {"ok": True/False, "payload": {...}}
        2. {"status": "success"/"error", "result": {...}}

        Returns:
            (is_success, result_data, error_message)
        """
        # Format 1: {"ok": True/False, "payload": {...}}
        if "ok" in response:
            is_success = response.get("ok", False)
            if is_success:
                payload = response.get("payload", {})
                return (True, payload, "")
            else:
                error_info = response.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", str(error_info))
                else:
                    error_msg = str(error_info)
                return (False, {}, error_msg)

        # Format 2: {"status": "success"/"error", "result": {...}}
        if "status" in response:
            is_success = response.get("status") == "success"
            if is_success:
                result = response.get("result", {})
                return (True, result, "")
            else:
                error_info = response.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", str(error_info))
                else:
                    error_msg = str(error_info)
                return (False, {}, error_msg)

        # Unknown format - assume failure
        return (False, {}, "Unknown response format")

    def _aggregate_results(self, step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate step results deterministically.

        No fuzzy merging - deterministic concatenation in step order.
        """
        outputs = []
        all_patterns = []

        for result in step_results:
            output = result.get("output")
            if output is not None:
                outputs.append(output)

            patterns = result.get("patterns_used", [])
            all_patterns.extend(patterns)

        # Deterministic aggregation: join outputs in order
        if len(outputs) == 1:
            final_output = outputs[0]
        elif all(isinstance(o, str) for o in outputs):
            final_output = "\n".join(outputs)
        else:
            final_output = outputs

        return {
            "output": final_output,
            "patterns_used": sorted(set(all_patterns))  # Deterministic unique patterns
        }

    def _rollback(
        self,
        failed_step_id: int,
        error_message: str,
        partial_results: List[Dict[str, Any]],
        with_trace: bool
    ) -> Dict[str, Any]:
        """
        Handle rollback when a step fails.

        Returns deterministic error result with trace of what was attempted.
        """
        # Add failure to trace
        self._add_trace_entry({
            "step_id": failed_step_id,
            "step_type": "failure",
            "description": "Step execution failed",
            "error": error_message,
            "success": False
        })

        result = {
            "output": None,
            "steps_executed": failed_step_id - 1,
            "success": False,
            "error": error_message,
            "failed_at_step": failed_step_id
        }

        if with_trace:
            result["trace"] = self._build_trace()

        return result

    def _add_trace_entry(self, entry: Dict[str, Any]) -> None:
        """Add entry to execution trace."""
        self._trace_entries.append(entry)

    def _build_trace(self) -> Dict[str, Any]:
        """
        Build final reasoning trace.

        Returns:
            Dict with:
                - entries: List of trace entries
                - total_steps: Total steps executed
                - deterministic: Always True
        """
        return {
            "entries": self._trace_entries.copy(),
            "total_steps": self.step_counter.current(),
            "deterministic": True
        }

    def _build_error_result(self, error_code: str, error_message: str) -> Dict[str, Any]:
        """Build standardized error result."""
        return {
            "output": None,
            "steps_executed": 0,
            "success": False,
            "error": error_message,
            "error_code": error_code
        }


# Module-level singleton
_engine_instance: Optional[TaskExecutionEngine] = None


def get_engine() -> TaskExecutionEngine:
    """Get singleton TEE instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TaskExecutionEngine()
    return _engine_instance
