"""
pipeline_executor.py
~~~~~~~~~~~~~~~~~~~~

Single pipeline executor for Maven's cognition system.

P1 PIPELINE REQUIREMENT:
This module provides the ONLY authorized way to execute the cognition pipeline.
It enforces the canonical stage order, manages the blackboard context between
stages, and ensures all bypass paths are explicitly whitelisted.

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import time

from brains.pipeline.pipeline_definition import (
    CanonicalPipeline,
    PipelineStage,
    StageDefinition,
    PipelineBypassError
)


class PipelineBlackboard:
    """
    Shared context (blackboard) that flows between pipeline stages.

    Each stage reads from and writes to this blackboard, allowing data
    to flow through the pipeline without tight coupling between stages.
    """

    def __init__(self, initial_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the blackboard.

        Args:
            initial_context: Optional initial context
        """
        self._data: Dict[str, Any] = initial_context or {}
        self._stage_outputs: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the blackboard."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value on the blackboard."""
        self._data[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all blackboard data."""
        return self._data.copy()

    def update(self, data: Dict[str, Any]) -> None:
        """Update blackboard with multiple values."""
        self._data.update(data)

    def set_stage_output(self, stage: str, output: Any) -> None:
        """Record the output of a stage."""
        self._stage_outputs[stage] = output

    def get_stage_output(self, stage: str) -> Any:
        """Get the output of a previous stage."""
        return self._stage_outputs.get(stage)

    def get_all_stage_outputs(self) -> Dict[str, Any]:
        """Get all stage outputs."""
        return self._stage_outputs.copy()


class PipelineExecutor:
    """
    Single executor for the canonical Maven pipeline.

    This executor:
    1. Takes the canonical stage list
    2. Calls each brain service in order
    3. Manages the blackboard between stages
    4. Controls bypass paths
    5. Logs execution for governance auditing
    """

    def __init__(self, pipeline: Optional[CanonicalPipeline] = None):
        """
        Initialize the pipeline executor.

        Args:
            pipeline: Optional pipeline instance (creates new if None)
        """
        self.pipeline = pipeline or CanonicalPipeline()
        self._service_registry: Dict[str, Callable] = {}

    def register_service(self, service_path: str, service_callable: Callable) -> None:
        """
        Register a brain service for pipeline execution.

        Args:
            service_path: Service path (e.g., "reasoning.service.reasoning_brain")
            service_callable: The service function to call
        """
        self._service_registry[service_path] = service_callable

    def execute(
        self,
        initial_context: Dict[str, Any],
        bypass_reason: Optional[str] = None,
        skip_stages: Optional[List[PipelineStage]] = None
    ) -> Dict[str, Any]:
        """
        Execute the canonical pipeline.

        Args:
            initial_context: Initial context/input for the pipeline
            bypass_reason: Optional reason for bypassing stages
            skip_stages: Optional list of stages to skip (must be authorized)

        Returns:
            Final pipeline output with blackboard state and execution log

        Raises:
            PipelineBypassError: If bypass is not authorized
        """
        # Validate bypass if requested
        if skip_stages and bypass_reason:
            self.pipeline.is_bypass_authorized(bypass_reason, skip_stages)
            self.pipeline.log_bypass_attempt(bypass_reason, skip_stages, authorized=True)
        elif skip_stages:
            raise PipelineBypassError("Cannot skip stages without bypass_reason")

        # Initialize blackboard
        blackboard = PipelineBlackboard(initial_context)

        # Execute each stage in order
        for stage_def in self.pipeline.STAGES:
            # Check if stage should be skipped
            if skip_stages and stage_def.stage in skip_stages:
                continue

            # Execute stage
            start_time = time.time()
            try:
                stage_output = self._execute_stage(stage_def, blackboard)
                duration_ms = (time.time() - start_time) * 1000

                # Log successful execution
                self.pipeline.log_stage_execution(
                    stage=stage_def.stage,
                    success=True,
                    duration_ms=duration_ms,
                    output=stage_output
                )

                # Store stage output on blackboard
                blackboard.set_stage_output(stage_def.stage.value, stage_output)

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Log failed execution
                self.pipeline.log_stage_execution(
                    stage=stage_def.stage,
                    success=False,
                    duration_ms=duration_ms,
                    output={"error": str(e)}
                )

                # For required stages, fail the pipeline
                if stage_def.required:
                    return {
                        "ok": False,
                        "error": f"Required stage {stage_def.stage.value} failed: {e}",
                        "blackboard": blackboard.get_all(),
                        "execution_log": self.pipeline.get_execution_log()
                    }

        # Return final result
        return {
            "ok": True,
            "blackboard": blackboard.get_all(),
            "stage_outputs": blackboard.get_all_stage_outputs(),
            "execution_log": self.pipeline.get_execution_log(),
            "bypass_attempts": self.pipeline.get_bypass_attempts()
        }

    def _execute_stage(
        self,
        stage_def: StageDefinition,
        blackboard: PipelineBlackboard
    ) -> Any:
        """
        Execute a single pipeline stage.

        Args:
            stage_def: Stage definition
            blackboard: Shared blackboard

        Returns:
            Stage output

        Raises:
            Exception: If stage execution fails
        """
        # Get service from registry
        service = self._service_registry.get(stage_def.brain_service)

        if not service:
            # If service not registered, return a stub (for testing)
            return {
                "stage": stage_def.stage.value,
                "status": "no_service_registered",
                "blackboard_snapshot": blackboard.get_all()
            }

        # Build stage input from blackboard
        stage_input = {
            "stage": stage_def.stage.value,
            "context": blackboard.get_all(),
            "previous_outputs": blackboard.get_all_stage_outputs()
        }

        # Call the service
        stage_output = service(stage_input)

        # Update blackboard if service returned updates
        if isinstance(stage_output, dict):
            if "blackboard_updates" in stage_output:
                blackboard.update(stage_output["blackboard_updates"])

        return stage_output


def create_pipeline_executor(
    pipeline: Optional[CanonicalPipeline] = None
) -> PipelineExecutor:
    """
    Factory function to create a pipeline executor.

    Args:
        pipeline: Optional pipeline instance

    Returns:
        PipelineExecutor instance
    """
    return PipelineExecutor(pipeline=pipeline)


# Export key components
__all__ = [
    "PipelineBlackboard",
    "PipelineExecutor",
    "create_pipeline_executor"
]
