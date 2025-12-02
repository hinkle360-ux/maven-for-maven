"""
pipeline_definition.py
~~~~~~~~~~~~~~~~~~~~~~

Canonical pipeline declaration for Maven's cognition system.

P1 PIPELINE REQUIREMENT:
This module defines the ONLY authorized cognition pipeline. All request
processing must follow this pipeline unless explicitly whitelisted bypass
paths are declared.

PIPELINE STAGES:
1. NLU - Natural Language Understanding
2. PATTERN_RECOGNITION - Pattern detection and classification
3. MEMORY - Memory retrieval and context building
4. REASONING - Fact evaluation and inference
5. VALIDATION - Truth validation and safety checks
6. GENERATION - Response generation
7. FINALIZATION - Output formatting and cleanup
8. HISTORY - Conversation history update
9. AUTONOMY - Autonomous action consideration

STDLIB ONLY, OFFLINE, WINDOWS 10 COMPATIBLE
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass


class PipelineStage(Enum):
    """Canonical pipeline stages in execution order."""
    NLU = "NLU"
    PATTERN_RECOGNITION = "PATTERN_RECOGNITION"
    MEMORY = "MEMORY"
    REASONING = "REASONING"
    VALIDATION = "VALIDATION"
    GENERATION = "GENERATION"
    FINALIZATION = "FINALIZATION"
    HISTORY = "HISTORY"
    AUTONOMY = "AUTONOMY"


@dataclass
class StageDefinition:
    """Definition of a single pipeline stage."""
    stage: PipelineStage
    brain_service: str  # Service name to invoke (e.g., "language_brain")
    required: bool = True  # If False, stage can be skipped
    timeout_ms: int = 30000  # Stage timeout in milliseconds
    description: str = ""


class PipelineBypassError(Exception):
    """Raised when an unauthorized pipeline bypass is attempted."""
    pass


class CanonicalPipeline:
    """
    The canonical Maven cognition pipeline.

    This class defines the standard flow of cognitive processing and
    enforces that all requests follow the defined stages unless explicitly
    whitelisted bypass paths are used.
    """

    # The canonical pipeline definition
    STAGES: List[StageDefinition] = [
        StageDefinition(
            stage=PipelineStage.NLU,
            brain_service="language.service.language_brain",
            required=True,
            description="Parse user intent and extract entities"
        ),
        StageDefinition(
            stage=PipelineStage.PATTERN_RECOGNITION,
            brain_service="pattern_recognition.service.pattern_recognition_brain",
            required=False,
            description="Detect patterns in input"
        ),
        StageDefinition(
            stage=PipelineStage.MEMORY,
            brain_service="memory_librarian.service.memory_librarian",
            required=True,
            description="Retrieve relevant memories and context"
        ),
        StageDefinition(
            stage=PipelineStage.REASONING,
            brain_service="reasoning.service.reasoning_brain",
            required=True,
            description="Evaluate facts and perform inference"
        ),
        StageDefinition(
            stage=PipelineStage.VALIDATION,
            brain_service="governance.council.service.council_brain",
            required=True,
            description="Validate truth tags and check governance rules"
        ),
        StageDefinition(
            stage=PipelineStage.GENERATION,
            brain_service="language.service.language_brain",
            required=True,
            description="Generate final response"
        ),
        StageDefinition(
            stage=PipelineStage.FINALIZATION,
            brain_service="integrator.service.integrator_brain",
            required=False,
            description="Format and finalize output"
        ),
        StageDefinition(
            stage=PipelineStage.HISTORY,
            brain_service="system_history.service.system_history_brain",
            required=False,
            description="Update conversation history"
        ),
        StageDefinition(
            stage=PipelineStage.AUTONOMY,
            brain_service="autonomy.service.autonomy_brain",
            required=False,
            description="Consider autonomous actions"
        ),
    ]

    # Whitelisted bypass paths
    # These are the ONLY authorized ways to skip pipeline stages
    AUTHORIZED_BYPASSES: Dict[str, List[PipelineStage]] = {
        "cache_hit": [
            PipelineStage.PATTERN_RECOGNITION,
            PipelineStage.REASONING,
        ],
        "simple_greeting": [
            PipelineStage.PATTERN_RECOGNITION,
            PipelineStage.MEMORY,
            PipelineStage.REASONING,
            PipelineStage.AUTONOMY,
        ],
        "health_check": [
            PipelineStage.PATTERN_RECOGNITION,
            PipelineStage.MEMORY,
            PipelineStage.REASONING,
            PipelineStage.VALIDATION,
            PipelineStage.GENERATION,
            PipelineStage.AUTONOMY,
        ],
    }

    def __init__(self):
        """Initialize the canonical pipeline."""
        self._execution_log: List[Dict[str, Any]] = []
        self._bypass_attempts: List[Dict[str, Any]] = []

    @classmethod
    def get_stages(cls) -> List[StageDefinition]:
        """
        Get the canonical pipeline stages.

        Returns:
            List of stage definitions in execution order
        """
        return cls.STAGES.copy()

    @classmethod
    def get_stage_names(cls) -> List[str]:
        """
        Get the names of all pipeline stages.

        Returns:
            List of stage names in execution order
        """
        return [stage.stage.value for stage in cls.STAGES]

    @classmethod
    def is_bypass_authorized(cls, bypass_reason: str, skipped_stages: List[PipelineStage]) -> bool:
        """
        Check if a pipeline bypass is authorized.

        Args:
            bypass_reason: Reason for bypass (must be in AUTHORIZED_BYPASSES)
            skipped_stages: Stages that will be skipped

        Returns:
            True if this bypass is authorized

        Raises:
            PipelineBypassError: If bypass is not authorized
        """
        if bypass_reason not in cls.AUTHORIZED_BYPASSES:
            raise PipelineBypassError(
                f"Unauthorized bypass reason: {bypass_reason}. "
                f"Allowed reasons: {list(cls.AUTHORIZED_BYPASSES.keys())}"
            )

        authorized_skips = cls.AUTHORIZED_BYPASSES[bypass_reason]

        for stage in skipped_stages:
            if stage not in authorized_skips:
                raise PipelineBypassError(
                    f"Bypass '{bypass_reason}' not authorized to skip stage {stage.value}. "
                    f"Authorized skips for '{bypass_reason}': {[s.value for s in authorized_skips]}"
                )

        return True

    def log_stage_execution(
        self,
        stage: PipelineStage,
        success: bool,
        duration_ms: float,
        output: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log execution of a pipeline stage.

        Args:
            stage: The stage that was executed
            success: Whether execution succeeded
            duration_ms: Execution time in milliseconds
            output: Optional stage output
        """
        log_entry = {
            "stage": stage.value,
            "success": success,
            "duration_ms": duration_ms,
            "output_summary": str(output)[:100] if output else None
        }
        self._execution_log.append(log_entry)

    def log_bypass_attempt(
        self,
        bypass_reason: str,
        skipped_stages: List[PipelineStage],
        authorized: bool
    ) -> None:
        """
        Log a pipeline bypass attempt.

        Args:
            bypass_reason: Reason for bypass
            skipped_stages: Stages that were/would be skipped
            authorized: Whether the bypass was authorized
        """
        bypass_entry = {
            "bypass_reason": bypass_reason,
            "skipped_stages": [s.value for s in skipped_stages],
            "authorized": authorized
        }
        self._bypass_attempts.append(bypass_entry)

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the execution log for this pipeline run."""
        return self._execution_log.copy()

    def get_bypass_attempts(self) -> List[Dict[str, Any]]:
        """Get logged bypass attempts."""
        return self._bypass_attempts.copy()

    def clear_logs(self) -> None:
        """Clear execution and bypass logs."""
        self._execution_log.clear()
        self._bypass_attempts.clear()


def get_canonical_pipeline() -> CanonicalPipeline:
    """
    Factory function to get the canonical pipeline instance.

    Returns:
        CanonicalPipeline instance
    """
    return CanonicalPipeline()


def validate_pipeline_execution(
    executed_stages: List[str],
    skipped_stages: Optional[List[str]] = None,
    bypass_reason: Optional[str] = None
) -> bool:
    """
    Validate that a pipeline execution followed the canonical definition.

    Args:
        executed_stages: List of stage names that were executed
        skipped_stages: Optional list of stage names that were skipped
        bypass_reason: Optional reason for skipping stages

    Returns:
        True if execution is valid

    Raises:
        PipelineBypassError: If execution violated pipeline rules
    """
    canonical_stages = CanonicalPipeline.get_stage_names()

    # If stages were skipped, validate the bypass
    if skipped_stages and bypass_reason:
        skipped_enum = [PipelineStage(s) for s in skipped_stages]
        CanonicalPipeline.is_bypass_authorized(bypass_reason, skipped_enum)

    # Verify all required stages were executed (unless authorized bypass)
    required_stages = [
        s.stage.value for s in CanonicalPipeline.STAGES if s.required
    ]

    authorized_skips = []
    if bypass_reason and bypass_reason in CanonicalPipeline.AUTHORIZED_BYPASSES:
        authorized_skips = [
            s.value for s in CanonicalPipeline.AUTHORIZED_BYPASSES[bypass_reason]
        ]

    for required_stage in required_stages:
        if required_stage not in executed_stages and required_stage not in authorized_skips:
            raise PipelineBypassError(
                f"Required stage '{required_stage}' was not executed and has no bypass authorization"
            )

    return True


# Export key components
__all__ = [
    "PipelineStage",
    "StageDefinition",
    "CanonicalPipeline",
    "PipelineBypassError",
    "get_canonical_pipeline",
    "validate_pipeline_execution"
]
